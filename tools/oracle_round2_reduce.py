"""Stage 0 Triton oracle 第二轮（docs/v0.5-reduce.md §8.1，桥接假说 H2）。

问题：Tila 的边缘化规则声称 Product(L0,L1) --reduce_k--> L_(1-k)——
"幸存轴保留自身分布"。Triton 的 tl.sum/tl.max 结果 encoding 是否与之相容？

实验域（评审 §3 采纳的强化矩阵）：
    op ∈ {sum, max} × axis ∈ {0, 1} × num_warps ∈ {1, 2, 4, 8} × dtype ∈ {fp16, fp32}
    2D tile shape = (64, 128)（BLOCK_R=64, BLOCK_C=128）

三个 kernel：
    A  reduce        ：load 2D tile → tl.sum/tl.max(x, axis) → store 1D
                       （观察 tt.reduce 的结果 layout）
    B  independent   ：tl.arange(0, 幸存长度) → store（幸存轴"独立构造"的 1D
                       encoding，即 Tila 语义中该轴自身的分布）
    C  consumer      ：softmax 轨迹形状——s = reduce(x,1); s2 = expand(s,1);
                       y = x * s2 → store 2D（统计 convert_layout 次数：
                       轨迹闭环是否需要显式分布转换）

三分类（docs/v0.5-reduce.md §8.1）：
    Strong  A 的 reduce 结果 layout 与 B 的独立 1D layout 结构相同（无 convert 预测）
    Weak    不同族，但 C 的消费链最多一次 convert_layout 即到达语义等价分布
    False   编译失败 / 无法消费

用法（GPU；MSVC 环境由 scripts/run_oracle_round2.bat 提供）：
    python tools/oracle_round2_reduce.py [--out build/oracle_round2]
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import tempfile
from pathlib import Path

BLOCK_R, BLOCK_C = 64, 128
WARPS = (1, 2, 4, 8)
DTYPES = ("fp16", "fp32")
OPS = ("sum", "max")
AXES = (0, 1)

KERNEL_SRC = '''\
import triton
import triton.language as tl


@triton.jit
def oracle_reduce(a, out, R, C, OP: tl.constexpr, AXIS: tl.constexpr,
                  BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.arange(0, BLOCK_C)
    rows2 = tl.expand_dims(offs_r, 1)
    cols2 = tl.expand_dims(offs_c, 0)
    m2 = (rows2 < R) & (cols2 < C)
    x = tl.load(a + rows2 * BLOCK_C + cols2, mask=m2, other=0.0)
    if OP == 0:
        s = tl.sum(x, AXIS)
    else:
        s = tl.max(x, AXIS)
    if AXIS == 0:
        offs = offs_c
        n = C
    else:
        offs = offs_r
        n = R
    tl.store(out + offs, s, mask=offs < n)


@triton.jit
def oracle_reduce_consume(a, out, R, C, OP: tl.constexpr, AXIS: tl.constexpr,
                          BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.arange(0, BLOCK_C)
    rows2 = tl.expand_dims(offs_r, 1)
    cols2 = tl.expand_dims(offs_c, 0)
    m2 = (rows2 < R) & (cols2 < C)
    x = tl.load(a + rows2 * BLOCK_C + cols2, mask=m2, other=0.0)
    if OP == 0:
        s = tl.sum(x, AXIS)
    else:
        s = tl.max(x, AXIS)
    s2 = tl.expand_dims(s, AXIS)
    y = x * s2
    tl.store(out + rows2 * BLOCK_C + cols2, y, mask=m2)


@triton.jit
def oracle_1d(out, N, L: tl.constexpr):
    offs = tl.arange(0, L)
    tl.store(out + offs, offs, mask=offs < N)
'''

_TORCH_DTYPE = {"fp16": "float16", "fp32": "float32"}
# tl.max 对 <32 位 dtype 提升返回（实测 3.7.1）：结果 buffer 必须按提升后 dtype
_RESULT_DTYPE = {("sum", "fp16"): "fp16", ("sum", "fp32"): "fp32",
                 ("max", "fp16"): "fp32", ("max", "fp32"): "fp32"}


def _build_kernels():
    import importlib.util

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tila_oracle_round2.py"
        path.write_text(KERNEL_SRC, encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location("tila_oracle_round2", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _layout_defs(ttgir: str) -> dict:
    """`#blocked = #ttg.blocked<{...}>` 定义行 → 名 → 定义体。"""
    defs = {}
    for line in ttgir.splitlines():
        line = line.strip()
        if " = #" in line and line.startswith("#"):
            name, _, body = line.partition(" = #")
            defs[f"#{name}"] = f"#{body}"
    return defs


def _type_on_line(line: str) -> str | None:
    """`... : tensor<64xf32, #ttg.slice<...>> loc(...)` → 完整类型文本。"""
    m = re.search(r":\s+(tensor<.*?>)(?:\s+loc\()", line)
    return m.group(1) if m else None


def _layout_of_type(tensor_type: str) -> str | None:
    """`tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>` → layout 文本
    （布局含嵌套 <>，取 dtype 逗号之后、最外层 > 之前整段）。"""
    m = re.match(r"^tensor<\S+,\s*(.*)>$", tensor_type)
    return m.group(1) if m else None


def _reduce_result_layout(ttgir: str) -> str | None:
    """tt.reduce 是多行 region：结果类型在 `}) : (…) -> tensor<…>` 行。"""
    lines = ttgir.splitlines()
    for i, line in enumerate(lines):
        if '"tt.reduce"' not in line:
            continue
        for tail in lines[i + 1:i + 8]:
            if "}) :" in tail:
                m = re.search(r"->\s*(tensor<.*?>)(?:\s+loc\()", tail)
                if m:
                    return _layout_of_type(m.group(1))
        return None
    return None


def _arange_layout(ttgir: str) -> str | None:
    """独立 1D kernel 的 tt.make_range 结果 layout。"""
    for line in ttgir.splitlines():
        if "tt.make_range" in line and "#ttg.slice" not in line:
            ty = _type_on_line(line)
            if ty:
                return _layout_of_type(ty)
    return None


def _find_line(ttgir: str, marker: str) -> str | None:
    for line in ttgir.splitlines():
        if marker in line:
            return line
    return None


def run_combo(mod, warps: int, dtype: str, op: str, axis: int, out_dir: Path):
    import torch

    torch_dtype = getattr(torch, _TORCH_DTYPE[dtype])
    result_dtype = getattr(torch, _TORCH_DTYPE[_RESULT_DTYPE[(op, dtype)]])
    r, c = BLOCK_R, BLOCK_C
    a = torch.randn(r, c, device="cuda", dtype=torch_dtype)
    op_idx = OPS.index(op)
    surviving = c if axis == 0 else r

    out1 = torch.empty(surviving, device="cuda", dtype=result_dtype)
    out2 = torch.empty(surviving, device="cuda", dtype=torch_dtype)

    common = dict(R=r, C=c, OP=op_idx, AXIS=axis,
                  BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, grid=(1,), num_warps=warps)
    compiled_a = mod.oracle_reduce.warmup(a, out1, **common)
    compiled_c = mod.oracle_reduce_consume.warmup(a, out2, **common)
    compiled_b = mod.oracle_1d.warmup(out1, surviving, L=surviving,
                                      grid=(1,), num_warps=warps)

    ttgir_a = compiled_a.asm["ttgir"]
    ttgir_c = compiled_c.asm["ttgir"]
    ttgir_b = compiled_b.asm["ttgir"]

    if out_dir:
        tag = f"{op}_axis{axis}_w{warps}_{dtype}"
        (out_dir / f"A_{tag}.mlir").write_text(ttgir_a, encoding="utf-8", newline="\n")
        (out_dir / f"B_{tag}.mlir").write_text(ttgir_b, encoding="utf-8", newline="\n")
        (out_dir / f"C_{tag}.mlir").write_text(ttgir_c, encoding="utf-8", newline="\n")

    defs_a, defs_b = _layout_defs(ttgir_a), _layout_defs(ttgir_b)

    def resolve(lay, defs):
        if lay is None:
            return None
        return defs.get(lay, lay)          # 名字 → 定义体；内联 slice 串原样

    reduce_def = resolve(_reduce_result_layout(ttgir_a), defs_a)
    arange_def = resolve(_arange_layout(ttgir_b), defs_b)
    if reduce_def is None or arange_def is None:
        return {"status": "parse-failed",
                "reduce_layout_found": reduce_def is not None,
                "arange_layout_found": arange_def is not None}
    converts_c = ttgir_c.count("convert_layout")
    converts_a = ttgir_a.count("convert_layout")
    layout_equal = reduce_def == arange_def

    if layout_equal:
        cls = "Strong"
    elif converts_c <= 1:
        cls = "Weak"
    else:
        cls = "False"

    return {
        "status": "ok",
        "classification": cls,
        "reduce_layout": reduce_def,
        "independent_layout": arange_def,
        "reduce_layout_kind": reduce_def.split("<", 1)[0],
        "converts_in_reduce_kernel": converts_a,
        "converts_in_consumer_kernel": converts_c,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=Path("build/oracle_round2"))
    ap.add_argument("--json", type=Path, default=None,
                    help="汇总 JSON 输出路径（默认 build/oracle_round2/summary.json）")
    args = ap.parse_args()

    try:
        import torch

        import triton  # noqa: F401

        assert torch.cuda.is_available()
    except (ImportError, AssertionError) as e:
        print(f"需要 triton + CUDA 设备：{e}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    mod = _build_kernels()
    results = {}
    counts = {"Strong": 0, "Weak": 0, "False": 0, "parse-failed": 0}
    for op, axis, warps, dtype in itertools.product(OPS, AXES, WARPS, DTYPES):
        key = f"{op}/axis{axis}/w{warps}/{dtype}"
        try:
            res = run_combo(mod, warps, dtype, op, axis, args.out)
        except Exception as e:  # noqa: BLE001 - 编译失败本身是实验结果
            res = {"status": "compile-failed", "error": f"{type(e).__name__}: {e}"}
        results[key] = res
        cls = res.get("classification", res["status"])
        counts[cls] = counts.get(cls, 0) + 1
        print(f"{key:28s} -> {res.get('classification', res['status'])}"
              + (f"  converts(A/C)={res.get('converts_in_reduce_kernel')}"
                 f"/{res.get('converts_in_consumer_kernel')}"
                 if res.get("status") == "ok" else
                 f"  {res.get('error', '')[:100]}"))

    print("\n汇总：", json.dumps(counts))
    out_json = args.json or (args.out / "summary.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8", newline="\n")
    print(f"明细：{out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
