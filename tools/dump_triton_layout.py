"""Stage 0 Triton oracle（docs/development-plan.md §2）。

不写 Tila 编译器代码，只用现成 Triton 回答一个问题：
"Tila 的 layout 抽象（v0.1 只有 identity）是否够用？"

手写 add 的 Triton kernel（与 examples/add.lowered.py 同形），对
BLOCK × num_warps × dtype 的实验域 dump TTIR 与 TritonGPU IR，观察
arange / load / elementwise / mask 的 concrete encoding 是否一致。

用法：
    python tools/dump_triton_layout.py --list            # 查看组合清单（无需 GPU）
    python tools/dump_triton_layout.py --ir BLOCK=128 warps=4 dtype=fp32
    python tools/dump_triton_layout.py --all --out build/oracle/   # 批量收集
"""

from __future__ import annotations

import argparse
import itertools
import tempfile
from pathlib import Path

BLOCKS = (32, 64, 128, 256)
WARPS = (1, 2, 4, 8)
DTYPES = ("fp16", "bf16", "fp32")

KERNEL_TEMPLATE = '''\
import triton
import triton.language as tl


@triton.jit
def oracle_add(a, b, c, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(a + offs, mask=mask)
    y = tl.load(b + offs, mask=mask)
    z = x + y
    tl.store(c + offs, z, mask=mask)
'''

_TL_DTYPE = {"fp16": "tl.float16", "bf16": "tl.bfloat16", "fp32": "tl.float32"}
_TORCH_DTYPE = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def combinations():
    return itertools.product(BLOCKS, WARPS, DTYPES)


def _build_kernel():
    """@triton.jit 需要真实文件：写临时模块并导入。"""
    import importlib.util

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tila_oracle_add.py"
        path.write_text(KERNEL_TEMPLATE, encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location("tila_oracle_add", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.oracle_add


def dump_ir(block: int, warps: int, dtype: str) -> str:
    """编译一次并返回 TTIR / TTGIR 文本（无需真正 launch；需要 GPU 设备）。"""
    import torch

    kernel = _build_kernel()
    n = block * 4
    a = torch.randn(n, device="cuda", dtype=getattr(torch, _TORCH_DTYPE[dtype]))
    compiled = kernel.warmup(a, a, a, n, BLOCK=block, grid=(1,), num_warps=warps)
    ptx_like = []
    for opt in ("ttir", "ttgir"):
        try:
            ptx_like.append(f"==== {opt} ====")
            ptx_like.append(compiled.asm[opt])
        except Exception as e:  # noqa: BLE001
            ptx_like.append(f"({opt} unavailable: {e})")
    return "\n".join(ptx_like)


def observe_encodings(ir_text: str) -> dict:
    """从 TTGIR 文本里抽取 encoding 观察点（oracle 笔记的原始素材）。"""
    notes = {}
    for kind in ("ttg.dot_layout", "#blocked", "#linear", "blocked:", "encode"):
        count = ir_text.count(kind)
        if count:
            notes[kind] = count
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--list", action="store_true", help="只列出实验域组合")
    ap.add_argument("--ir", nargs=3, metavar=("BLOCK=..", "warps=..", "dtype=.."),
                    help="dump 单个组合的 IR")
    ap.add_argument("--all", action="store_true", help="批量收集全部组合")
    ap.add_argument("--out", type=Path, default=Path("build/oracle"))
    args = ap.parse_args()

    if args.list or (not args.ir and not args.all):
        print("实验域：BLOCK ∈ {32, 64, 128, 256} × num_warps ∈ {1, 2, 4, 8} "
              "× dtype ∈ {fp16, bf16, fp32}")
        for b, w, d in combinations():
            print(f"  BLOCK={b:4d} warps={w} dtype={d}")
        return 0

    try:
        import triton  # noqa: F401
        import torch

        assert torch.cuda.is_available()
    except (ImportError, AssertionError) as e:
        print(f"需要 triton + CUDA 设备：{e}")
        return 1

    if args.ir:
        spec = dict(item.split("=") for item in args.ir)
        text = dump_ir(int(spec["BLOCK"]), int(spec["warps"]), spec["dtype"])
        print(text)
        print("\n// encoding 观察点：", observe_encodings(text))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for b, w, d in combinations():
        try:
            text = dump_ir(b, w, d)
            out = args.out / f"add_b{b}_w{w}_{d}.mlir"
            out.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {out}  encodings={observe_encodings(text)}")
        except Exception as e:  # noqa: BLE001
            print(f"BLOCK={b} warps={w} dtype={d}: FAILED ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
