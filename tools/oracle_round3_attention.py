"""Stage 0 Triton oracle 第三轮（docs/v0.6-attention.md §8.1，桥接假说 H3-S/H3-C）。

问题：Tila 声称 marginal(Mma, k) = Slice(Mma, k)（v0.6 分支六），且
where-cond / 广播侧 / rescale 乘数"读透明"。Triton 对 dot 结果的
tl.max/tl.sum 与 attention 轨迹是否与之相容？

判据分离（评审 §9/§10 采纳）：
    H3-S（语义判定）：reduce-over-dot 结果 encoding 与亲代切片同构
        （#ttg.slice<{dim=k, parent=<dot-encoding>}>），expand_dims 还原
        亲代；跨实验矩阵稳定。
    H3-C（代价判定）：attention 正则轨迹在 dot 操作数入场之外的
        convert_layout 计数有界（C_max = 4/kernel；本机基线 = 每 dot
        1 次操作数入场 convert，H1 实测）。

实验域：kernel {A,B,C,D} × 形状 (BM,BN,D) ∈ {(64,64,64),(64,128,64),
(128,128,64)} × num_warps {1,4}；f16 入 / f32 累加。

四个 kernel（手写 Triton，docs/v0.6-attention.md §1.1/§1.2 的逐行同构）：
    A  reduce-over-dot ：dot → tl.max(qk, 1) → store 1D（H3-S①）
    B  expand-join     ：A + expand_dims → qk * s2 → store 2D（H3-S②/R-BT 位）
    C  fragment        ：where(-inf) → max → exp → sum → div → cast →
                         第二 dot → store（H3-C：R-PT/R-BT 定量）
    D  flash body      ：for n0: maximum/α 重定标（acc *= α₂）、
                         acc += dot(p,v)；循环外终归一化除法（H3-C：R-ST/
                         跨 Mma 位）

附带核实：tl.full 接受 float("-inf")；tl.maximum 的 dtype 行为。
SM 6.1 无 Tensor Core（dot 以 FMA 下探）——结构结论按 encoding 无关设计，
mma_layout 的 encoding 级复核待 TC 机器（与 H1 同款限定）。

用法（GPU；MSVC 环境由 scripts/run_oracle_round3.bat 提供）：
    python tools/oracle_round3_attention.py [--out build/oracle_round3]
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import tempfile
from pathlib import Path

SHAPES = ((64, 64, 64), (64, 128, 64), (128, 64, 64))
# 注：初始矩阵的第三组 (128,128,64) 在本机（SM 6.1 FMA 下探）对 D（flash
# 循环，双 128 维 dot）编译超 1 小时未完成——换 (128,64,64) 覆盖 BM=128
# 方向；BM=BN=128 的值级正确性由 GPU 差分覆盖（tests/test_gpu_integration
# 的 (128,128,64,128,64) 组合）。
WARPS = (1, 4)
C_MAX = 4          # H3-C 阈值：每 kernel 在 dot 入场之外的 convert 上限
DOT_ENTRY_BASELINE = 1   # 本机基线：每 dot 1 次操作数入场 convert（H1 实测）

KERNEL_SRC = '''\
import triton
import triton.language as tl


@triton.jit
def oracle_reduce_dot(q, k, out, M, N, D,
                      BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rm2 = tl.expand_dims(rm, 1)
    rn2 = tl.expand_dims(tl.arange(0, BN), 0)
    rd2 = tl.expand_dims(tl.arange(0, BD), 0)
    rd3 = tl.expand_dims(tl.arange(0, BD), 1)
    q2 = tl.load(q + rm2 * D + rd2, mask=(rm2 < M) & (rd2 < D), other=0.0)
    k2 = tl.load(k + rn2 * D + rd3, mask=(rn2 < N) & (rd3 < D), other=0.0)
    qk = tl.dot(q2, k2)
    s = tl.max(qk, 1)
    tl.store(out + rm, s, mask=rm < M)


@triton.jit
def oracle_expand_join(q, k, out, M, N, D,
                       BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rm2 = tl.expand_dims(pid * BM + tl.arange(0, BM), 1)
    rn2 = tl.expand_dims(tl.arange(0, BN), 0)
    rd2 = tl.expand_dims(tl.arange(0, BD), 0)
    rd3 = tl.expand_dims(tl.arange(0, BD), 1)
    q2 = tl.load(q + rm2 * D + rd2, mask=(rm2 < M) & (rd2 < D), other=0.0)
    k2 = tl.load(k + rn2 * D + rd3, mask=(rn2 < N) & (rd3 < D), other=0.0)
    qk = tl.dot(q2, k2)
    s = tl.max(qk, 1)
    s2 = tl.expand_dims(s, 1)
    y = qk * s2
    tl.store(out + rm2 * BN + rn2, y, mask=(rm2 < M) & (rn2 < N))


@triton.jit
def oracle_attention(q, k, v, o, M, N, D,
                     BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rm2 = tl.expand_dims(pid * BM + tl.arange(0, BM), 1)
    rn2 = tl.expand_dims(tl.arange(0, BN), 0)
    rn3 = tl.expand_dims(tl.arange(0, BN), 1)
    rd2 = tl.expand_dims(tl.arange(0, BD), 0)
    rd3 = tl.expand_dims(tl.arange(0, BD), 1)
    m_qk = rn2 < N
    q2 = tl.load(q + rm2 * D + rd2, mask=(rm2 < M) & (rd2 < D), other=0.0)
    k2 = tl.load(k + rn2 * D + rd3, mask=(rn2 < N) & (rd3 < D), other=0.0)
    v2 = tl.load(v + rn3 * D + rd2, mask=(rn3 < N) & (rd2 < D), other=0.0)
    qk = tl.dot(q2, k2)
    neg = tl.where(m_qk, qk, float("-inf"))
    mx = tl.max(neg, 1)
    mx2 = tl.expand_dims(mx, 1)
    p = tl.exp(neg - mx2)
    s = tl.sum(p, 1)
    s2 = tl.expand_dims(s, 1)
    w = p / s2
    y = tl.dot(w.to(tl.float16), v2)
    tl.store(o + rm2 * D + rd2, y.to(tl.float16), mask=(rm2 < M) & (rd2 < D))


@triton.jit
def oracle_flash(q, k, v, o, M, N, D,
                 BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rm2 = tl.expand_dims(pid * BM + tl.arange(0, BM), 1)
    rn2 = tl.expand_dims(tl.arange(0, BN), 0)
    rn3 = tl.expand_dims(tl.arange(0, BN), 1)
    rd2 = tl.expand_dims(tl.arange(0, BD), 0)
    rd3 = tl.expand_dims(tl.arange(0, BD), 1)
    q2 = tl.load(q + rm2 * D + rd2, mask=(rm2 < M) & (rd2 < D), other=0.0)
    m = tl.full((BM,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, BD), dtype=tl.float32)
    for n0 in range(0, N, BN):
        rkn = rn2 + n0
        rvn = rn3 + n0
        k2 = tl.load(k + rkn * D + rd3, mask=(rkn < N) & (rd3 < D), other=0.0)
        v2 = tl.load(v + rvn * D + rd2, mask=(rvn < N) & (rd2 < D), other=0.0)
        qk = tl.dot(q2, k2)
        neg = tl.where(rkn < N, qk, float("-inf"))
        rowmax = tl.max(neg, 1)
        m1 = tl.maximum(m, rowmax)
        alpha = tl.exp(m - m1)
        p = tl.exp(neg - tl.expand_dims(m1, 1))
        l = l * alpha + tl.sum(p, 1)
        acc = acc * tl.expand_dims(alpha, 1)
        acc = acc + tl.dot(p.to(tl.float16), v2)
        m = m1
    y = acc / tl.expand_dims(l, 1)
    tl.store(o + rm2 * D + rd2, y.to(tl.float16), mask=(rm2 < M) & (rd2 < D))
'''


def _build_kernels():
    import importlib.util

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tila_oracle_round3.py"
        path.write_text(KERNEL_SRC, encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location("tila_oracle_round3", path)
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
            defs[f"#{name.lstrip('#')}"] = f"#{body}"
    return defs


def _resolve(layout: str | None, defs: dict) -> str | None:
    """名字 → 定义体（一次；嵌套引用返回原样，族判定足够）。"""
    if layout is None:
        return None
    return defs.get(layout, layout)


def _type_after_arrow(lines: list[str], start: int) -> str | None:
    """自 start 行起找 `-> tensor<…>` 的完整类型（跨最多 6 行）。"""
    for line in lines[start:start + 6]:
        m = re.search(r"->\s*(tensor<.*?>)(?:\s+loc\()", line)
        if m:
            return m.group(1)
    return None


def _layout_of_type(tensor_type: str | None) -> str | None:
    """`tensor<64xf32, LAY>` → LAY（布局含嵌套 <>，取最外层）。"""
    if tensor_type is None:
        return None
    m = re.match(r"^tensor<\S+,\s*(.*)>$", tensor_type)
    return m.group(1) if m else None


def _dot_result_layout(ttgir: str) -> str | None:
    """tt.dot 的结果 layout（tt.dot 可跨行：结果类型在 `-> tensor<…>`）。"""
    lines = ttgir.splitlines()
    for i, line in enumerate(lines):
        if "tt.dot" in line:
            return _layout_of_type(_type_after_arrow(lines, i))
    return None


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


def _expand_result_layout(ttgir: str, operand_layout: str | None = None) -> str | None:
    """tt.expand_dims 的结果 layout。operand_layout 给出时，只匹配操作数
    类型含该布局的 expand（排除坐标构造的 expand_dims——它们先于 reduce
    出现且操作数是 i32 slice）。"""
    lines = ttgir.splitlines()
    for i, line in enumerate(lines):
        if "tt.expand_dims" not in line:
            continue
        if operand_layout is not None and operand_layout not in line.split("->")[0]:
            continue
        return _layout_of_type(_type_after_arrow(lines, i))
    return None


def _convert_lines(ttgir: str, limit: int = 8) -> list[str]:
    out = []
    for line in ttgir.splitlines():
        s = line.strip()
        if "convert_layout" in s:
            out.append(s[:160])
            if len(out) >= limit:
                break
    return out


def _slice_parent(layout: str | None) -> str | None:
    """`#ttg.slice<{dim = 1, parent = #blocked2}>` → parent 引用。"""
    if layout is None:
        return None
    m = re.search(r"#ttg\.slice<\{[^}]*parent\s*=\s*(#\w+)", layout)
    return m.group(1) if m else None


def run_combo(mod, warps: int, shape: tuple[int, int, int], out_dir: Path):
    import torch

    bm, bn, d = shape
    q = torch.randn(bm, d, device="cuda", dtype=torch.float16)
    k = torch.randn(bn, d, device="cuda", dtype=torch.float16)
    v = torch.randn(bn, d, device="cuda", dtype=torch.float16)
    o16 = torch.empty(bm, d, device="cuda", dtype=torch.float16)
    out1 = torch.empty(bm, device="cuda", dtype=torch.float32)
    out2 = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    n_flash = 4 * bn                       # 多块循环（TTGIR 单体不随块数变化）

    ce = dict(BM=bm, BN=bn, BD=d, grid=(1,), num_warps=warps)
    jobs = {
        "A": (mod.oracle_reduce_dot, (q, k, out1, bm, bn, d), ce),
        "B": (mod.oracle_expand_join, (q, k, out2, bm, bn, d), ce),
        "C": (mod.oracle_attention, (q, k, v, o16, bm, bn, d), ce),
        "D": (mod.oracle_flash, (q, k, v, o16, bm, n_flash, d), ce),
    }

    res: dict = {}
    ttgirs: dict = {}
    for letter, (fn, args, ce_) in jobs.items():
        try:
            compiled = fn.warmup(*args, **ce_)
            ttgir = compiled.asm["ttgir"]
            ttgirs[letter] = ttgir
            res[f"{letter}_status"] = "ok"
            res[f"{letter}_converts"] = ttgir.count("convert_layout")
            res[f"{letter}_convert_lines"] = _convert_lines(ttgir)
            res[f"{letter}_dots"] = ttgir.count("tt.dot")
        except Exception as e:  # noqa: BLE001 - 编译失败本身是实验结果
            res[f"{letter}_status"] = "compile-failed"
            res[f"{letter}_error"] = f"{type(e).__name__}: {e}"

    if out_dir:
        tag = f"bm{bm}_bn{bn}_d{d}_w{warps}"
        for letter, ttgir in ttgirs.items():
            (out_dir / f"{letter}_{tag}.mlir").write_text(
                ttgir, encoding="utf-8", newline="\n")

    # ---- H3-S：A 的 reduce 结果是否亲代切片；B 的 expand 是否还原亲代 ----
    h3s = None
    if res.get("A_status") == "ok" and res.get("B_status") == "ok":
        defs_a = _layout_defs(ttgirs["A"])
        defs_b = _layout_defs(ttgirs["B"])
        dot_a = _resolve(_dot_result_layout(ttgirs["A"]), defs_a)
        red_a = _resolve(_reduce_result_layout(ttgirs["A"]), defs_a)
        dot_b = _resolve(_dot_result_layout(ttgirs["B"]), defs_b)
        red_b = _resolve(_reduce_result_layout(ttgirs["B"]), defs_b)
        # 只认 reduce 之后的那个 expand（操作数类型 = reduce 结果布局，f32 slice）
        exp_b = _resolve(_expand_result_layout(ttgirs["B"], _reduce_result_layout(ttgirs["B"])),
                          defs_b)

        parent_ref = _slice_parent(_reduce_result_layout(ttgirs["A"]))
        parent_a = _resolve(parent_ref, defs_a) if parent_ref else None
        slice_of_dot = (parent_a is not None
                        and dot_a is not None
                        and parent_a.split("<", 1)[0] == dot_a.split("<", 1)[0]
                        and parent_a == dot_a)
        # B 的 expand 结果（应带 parent 的 rank-2 族）：与 dot 同族即还原
        expand_returns_parent = (exp_b is not None and dot_b is not None
                                 and exp_b == dot_b)
        res.update({
            "A_dot_layout": dot_a, "A_reduce_layout": red_a,
            "A_reduce_is_slice_of_dot": slice_of_dot,
            "B_dot_layout": dot_b, "B_reduce_layout": red_b,
            "B_expand_layout": exp_b,
            "B_expand_returns_parent": expand_returns_parent,
        })
        h3s = "pass" if (slice_of_dot and expand_returns_parent) else "fail"
    res["H3S"] = h3s if h3s is not None else "inconclusive"

    # ---- H3-C：C/D 在 dot 入场之外的 convert 计数 ----
    def extra(letter):
        if res.get(f"{letter}_status") != "ok":
            return None
        return res[f"{letter}_converts"] - DOT_ENTRY_BASELINE * res[f"{letter}_dots"]

    extra_c, extra_d = extra("C"), extra("D")
    res["C_extra_converts"] = extra_c
    res["D_extra_converts"] = extra_d
    if extra_c is not None and extra_d is not None:
        res["H3C"] = "strong" if (extra_c <= C_MAX and extra_d <= C_MAX) else "weak"
    else:
        res["H3C"] = "inconclusive"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=Path("build/oracle_round3"))
    ap.add_argument("--json", type=Path, default=None,
                    help="汇总 JSON 输出路径（默认 build/oracle_round3/summary.json）")
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
    tally = {"H3S_pass": 0, "H3S_fail": 0, "H3S_inconclusive": 0,
             "H3C_strong": 0, "H3C_weak": 0, "H3C_inconclusive": 0}
    for shape, warps in itertools.product(SHAPES, WARPS):
        key = f"bm{shape[0]}/bn{shape[1]}/d{shape[2]}/w{warps}"
        try:
            res = run_combo(mod, warps, shape, args.out)
        except Exception as e:  # noqa: BLE001
            res = {"H3S": "inconclusive", "H3C": "inconclusive",
                   "error": f"{type(e).__name__}: {e}"}
        results[key] = res
        tally[f"H3S_{res['H3S']}"] = tally.get(f"H3S_{res['H3S']}", 0) + 1
        tally[f"H3C_{res['H3C']}"] = tally.get(f"H3C_{res['H3C']}", 0) + 1
        print(f"{key:22s} H3-S={res['H3S']:<13s} H3-C={res['H3C']:<8s}"
              f"  converts(A/B/C/D)={res.get('A_converts')}/{res.get('B_converts')}"
              f"/{res.get('C_converts')}/{res.get('D_converts')}"
              f"  dots(C/D)={res.get('C_dots')}/{res.get('D_dots')}"
              + ("" if res.get("D_status") == "ok"
                 else f"  D:{res.get('D_error', '')[:80]}"))

    print("\n汇总：", json.dumps(tally))
    out_json = args.json or (args.out / "summary.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8",
                        newline="\n")
    print(f"明细：{out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
