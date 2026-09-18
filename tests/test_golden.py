"""Golden：lowered Triton 源码与 TIR dump 逐字节比对（roadmap §8）。"""

import os

import pytest

import tila as ti

N = ti.Dim("N")


@ti.jit
def add_kernel(
    x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    y: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    BLOCK: ti.Const[int, ti.PowerOfTwo] = 128,
):
    pid = ti.program_id(0)
    offs = pid * BLOCK + ti.arange(0, BLOCK)
    mask = offs < N
    a = ti.load(x, offs, mask=mask)
    b = ti.load(y, offs, mask=mask)
    ti.store(out, offs, a + b, mask=mask)


GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def test_add_triton_source_golden():
    src, tir = add_kernel.materialize({})
    with open(os.path.join(GOLDEN_DIR, "add_kernel.triton.py"),
              encoding="utf-8") as f:
        assert src == f.read()


def test_add_tir_dump_golden():
    src, tir = add_kernel.materialize({})
    with open(os.path.join(GOLDEN_DIR, "add_kernel.tir.txt"),
              encoding="utf-8") as f:
        assert tir == f.read()


def test_golden_source_contains_expected_markers():
    src, _ = add_kernel.materialize({})
    assert "@triton.jit" in src
    assert "tl.max_contiguous(offs, BLOCK)" in src   # 事实自动发射
    assert "tl.multiple_of(offs_base, BLOCK)" in src  # 整除事实（§4）
    assert "mask=mask" in src
    assert "other=0.0" in src                        # masked other 缺省零
    assert "BLOCK: tl.constexpr" in src
