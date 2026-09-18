"""FP8 存储型 dtype 不参与算术（type-system.md §2 FloatStorage、
design-principles.md §4）：load → cast → compute → cast → store。

错误 kernel 定义在测试函数内（装饰即 Stage 1 拒绝）。
注：numpy 无原生 f8 dtype，合法流水线仅验证 Stage 1 通过，不上 interp。
"""

import pytest

import tila as ti
from tila.errors import TilaError

N = ti.Dim("N")


def test_f8e4m3fn_binop_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f8e4m3fn, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            b = ti.load(x, offs, mask=m)
            ti.store(x, offs, a + b, mask=m)
    assert ei.value.code == "TILA-TYPE-036"
    assert "f8e4m3fn" in str(ei.value)
    assert "storage-only" in str(ei.value)
    # details 命名两侧操作数 dtype
    assert "f8e4m3fn" in ei.value.render()


def test_f8e5m2_sub_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f8e5m2, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            b = ti.load(x, offs, mask=m)
            ti.store(x, offs, a - b, mask=m)
    assert ei.value.code == "TILA-TYPE-036"
    assert "f8e5m2" in str(ei.value)


def test_f8_unary_minus_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f8e4m3fn, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            ti.store(x, offs, -a, mask=m)
    assert ei.value.code == "TILA-TYPE-036"
    assert "f8e4m3fn" in str(ei.value)


def test_f8_exp_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f8e4m3fn, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            ti.store(x, offs, ti.exp(a), mask=m)
    assert ei.value.code == "TILA-TYPE-033"


def test_f8_legit_pipeline_passes_stage1():
    """合法 FP8 范式：load f8 → cast f32 → 算术 → cast f8 → store（仅 Stage 1）。"""
    @ti.jit
    def k(x: ti.Buffer[ti.f8e4m3fn, (N,), ti.ReadWrite],
          BLOCK: ti.Const[int] = 64):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        a = ti.load(x, offs, mask=m)
        b = ti.load(x, offs, mask=m)
        a32 = ti.cast[ti.f32](a)
        b32 = ti.cast[ti.f32](b)
        s32 = a32 + b32
        ti.store(x, offs, ti.cast[ti.f8e4m3fn](s32), mask=m)


def test_f8_comparison_still_allowed():
    """比较不受算术限制（type-system.md §2 仅禁止算术）。"""
    @ti.jit
    def k(x: ti.Buffer[ti.f8e4m3fn, (N,), ti.ReadWrite],
          BLOCK: ti.Const[int] = 64):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        a = ti.load(x, offs, mask=m)
        b = ti.load(x, offs, mask=m)
        w = a < b
        ti.store(x, offs, ti.cast[ti.f8e4m3fn](ti.where(w, 1.0, 0.0)), mask=m)
