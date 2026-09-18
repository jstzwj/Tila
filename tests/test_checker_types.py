"""类型规则矩阵：dtype 混算/字面量/控制流/Mask 语义（type-system.md §6、§10）。

错误 kernel 定义在测试函数内（装饰即 Stage 1 拒绝）。
"""

import pytest

import tila as ti
from tila.errors import TilaError

N = ti.Dim("N")


def test_signed_unsigned_mix_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(a: ti.i32, b: ti.u32):
            c = a + b
    assert ei.value.code == "TILA-TYPE-012"
    assert "i32" in str(ei.value) and "u32" in str(ei.value)


def test_int_float_mix_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(a: ti.i32, b: ti.f32):
            c = a + b
    assert ei.value.code == "TILA-TYPE-012"


def test_widening_ok():
    @ti.jit
    def k(a: ti.i16, b: ti.i32):
        c = a + b          # i16 → i32 widening ✓
        d = b * 2          # 字面量语境化 ✓

    # f16 语境下的 int 字面量
    @ti.jit
    def k2(x: ti.Buffer[ti.f16, (N,), ti.ReadWrite], BLOCK: ti.Const[int] = 64):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        v = ti.load(x, offs, mask=m)
        ti.store(x, offs, v + 1, mask=m)   # 1 实例化为 f16


def test_literal_out_of_range():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.u8, (N,), ti.ReadWrite], BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            v = ti.load(x, offs, mask=offs < N)
            ti.store(x, offs, v + 300, mask=offs < N)
    assert ei.value.code == "TILA-TYPE-013"
    assert "300" in str(ei.value) and "u8" in str(ei.value)


def test_f16_literal_no_silent_rounding():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f16, (N,), ti.ReadWrite], BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            v = ti.load(x, offs, mask=offs < N)
            ti.store(x, offs, v + 0.1, mask=offs < N)
    assert ei.value.code == "TILA-TYPE-013"


def test_int_division_is_float_only_error():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(a: ti.i32, b: ti.i32):
            c = a / b
    assert ei.value.code == "TILA-TYPE-014"


def test_store_dtype_must_match_exactly():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              y: ti.Buffer[ti.f16, (N,), ti.ReadOnly],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(y, offs, mask=m)
            ti.store(x, offs, v, mask=m)    # f16 → f32：拒绝
    assert ei.value.code == "TILA-TYPE-016"


def test_store_to_readonly_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m)
            ti.store(x, offs, v, mask=m)
    assert ei.value.code == "TILA-MEM-001"


def test_arange_runtime_bound_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(n: ti.i32):
            offs = ti.arange(0, n)
    assert ei.value.code == "TILA-CONST-001"


def test_mask_as_if_condition_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            if m:
                pass
    assert ei.value.code == "TILA-TYPE-022"


def test_mask_not_arithmetic():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            y = m + 1
    assert ei.value.code == "TILA-TYPE-026"


def test_branch_merge_type_mismatch():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], c: ti.i32,
              BLOCK: ti.Const[int] = 64):
            if c > 0:
                a = ti.zeros((BLOCK,), ti.f32)
            else:
                a = ti.zeros((32,), ti.f32)
            offs = ti.arange(0, BLOCK)
            ti.store(x, offs, a, mask=offs < N)
    assert ei.value.code == "TILA-TYPE-020"


def test_loop_carried_type_stability():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], K: ti.i32):
            a = ti.zeros((8,), ti.f32)
            for i in ti.range(0, K, 2):
                a = ti.zeros((16,), ti.f32)
            ti.store(x, ti.arange(0, 8), a)
    assert ei.value.code == "TILA-TYPE-021"


def test_where_branch_dtype_mismatch():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              y: ti.Buffer[ti.f16, (N,), ti.ReadOnly],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            b = ti.load(y, offs, mask=m)
            c = ti.where(m, a, b)
    assert ei.value.code == "TILA-TYPE-015"


def test_broadcast_incompatible():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(BLOCK: ti.Const[int] = 64):
            a = ti.zeros((64, 32), ti.f32)
            b = ti.zeros((16,), ti.f32)
            c = a + b
    assert ei.value.code == "TILA-SHAPE-003"


def test_dot_inner_dim_const_mismatch_stage1():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(BM: ti.Const[int] = 64, BN: ti.Const[int] = 64):
            a = ti.zeros((BM, 32), ti.f16)
            b = ti.zeros((64, BN), ti.f16)
            c = ti.dot(a, b)
    assert ei.value.code == "TILA-SHAPE-004"


def test_dot_operand_dtype_gate():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(BM: ti.Const[int] = 64, BN: ti.Const[int] = 64, BK: ti.Const[int] = 32):
            a = ti.zeros((BM, BK), ti.f32)
            b = ti.zeros((BK, BN), ti.f32)
            c = ti.dot(a, b)
    assert ei.value.code == "TILA-TYPE-031"


def test_dot_inner_dim_deferred_to_spec():
    @ti.jit
    def k(BM: ti.Const[int] = 64, BN: ti.Const[int] = 64, BK: ti.Const[int] = 32):
        a = ti.zeros((BM, BK), ti.f16)
        b = ti.zeros((64, BN), ti.f16)     # 64 与 BK 的等价延迟到特化期
        c = ti.dot(a, b)

    # BK=64 时 K1 == K2 == 64，通过
    src, _ = k.materialize({"BK": 64})
    assert "tl.dot" in src

    with pytest.raises(TilaError) as ei:
        k.materialize({"BK": 32})
    assert ei.value.code == "TILA-SHAPE-004"


FLAG = True          # 模块级常量（kernel 内可见的 Const 语境之一）


def test_constexpr_if_static_resolution():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
        if FLAG:
            offs = ti.arange(0, BLOCK)
        else:
            offs = ti.arange(0, 32)
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=offs < N)
    notes = "\n".join(k.tk.notes)
    assert "static branch resolved to then" in notes


def test_store_returns_unit_cannot_assign():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            y = ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
    assert ei.value.code == "TILA-TYPE-017"
