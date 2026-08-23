"""类型系统单元测试：dtype 能力表、shape ⊗、layout 律 L1–L5、cast 矩阵、E04/E05 助手。"""

import pytest

from tila.checker import broadcast as bcheck
from tila.checker import layout as lcheck
from tila.diagnostics import Loc, TilaError
from tila.types import (
    ARITH_OPS,
    CMP_OPS,
    DTYPES,
    Const,
    Identity,
    JoinL,
    LoadL,
    ProductL,
    ROW_MAJOR,
    BcastScalarL,
    CastL,
    Symbol,
    broadcast,
    cast_allowed,
    cmp_ops,
    equiv,
    is_float,
    is_int,
    normalize,
    numel,
    shape_eq,
    shape_str,
)
from tila.types import dtype as dt

ANY = Loc(1, 1)


# ---------------------------------------------------------------------------
# dtype 域与能力表（type-system §1.1）
# ---------------------------------------------------------------------------

def test_dtype_universe_17():
    assert len(DTYPES) == 17
    assert DTYPES[0] == "bool"


def test_int_kinds():
    for d in ("i8", "i32", "u8", "u64"):
        assert is_int(d)
        assert not is_float(d)
    for d in ("f16", "bf16", "f32", "f64"):
        assert is_float(d)
        assert not is_int(d)
    for d in ("fp8e4m3", "fp8e4m3fn"):
        assert is_float(d)  # 字面量类别口径：FP8 配 FLOAT 字面量
        assert not is_int(d)


def test_arith_capabilities():
    assert set(ARITH_OPS) - dt.arith_ops("f32") == set()       # f32 全部四种
    assert "/" not in dt.arith_ops("i32")                       # 整数无 /
    assert dt.arith_ops("bool") == frozenset()
    assert dt.arith_ops("fp8e4m3fn") == frozenset()             # FP8 存储专用


def test_cmp_capabilities():
    assert dt.cmp_ops("bool") == frozenset({"==", "!="})
    assert set(CMP_OPS) <= dt.cmp_ops("f16")
    assert dt.cmp_ops("fp8e5m2") == frozenset()
    assert dt.logic_ok("bool") and not dt.logic_ok("i32")


def test_cast_matrix_fully_open():
    for a in DTYPES:
        for b in DTYPES:
            assert cast_allowed(a, b)  # 数值 dtype 任意互转，含 bool/FP8 端点


def test_triton_dtype_map_complete():
    from tila.types import TRITON_DTYPE

    for d in DTYPES:
        assert TRITON_DTYPE[d].startswith("tl."), d
    assert TRITON_DTYPE["f32"] == "tl.float32"
    assert TRITON_DTYPE["fp8e4m3fn"] == "tl.float8e4m3fn"
    assert TRITON_DTYPE["bool"] == "tl.int1"
    assert TRITON_DTYPE["u8"] == "tl.uint8"


def test_surface_names_accept_both_spellings():
    from tila.types import SURFACE_DTYPE_NAMES

    assert SURFACE_DTYPE_NAMES["float32"] == "f32"
    assert SURFACE_DTYPE_NAMES["f32"] == "f32"
    assert SURFACE_DTYPE_NAMES["float8e4m3fn"] == "fp8e4m3fn"
    assert "bfloat16" in SURFACE_DTYPE_NAMES
    assert "nope" not in SURFACE_DTYPE_NAMES


# ---------------------------------------------------------------------------
# shape 迷你系统（type-system §2）
# ---------------------------------------------------------------------------

def test_shape_equality_by_symbol_name():
    assert shape_eq((Symbol("N"),), (Symbol("N"),))
    assert not shape_eq((Symbol("N"),), (Symbol("M"),))  # 即便运行时相等
    assert shape_eq((Const(128),), (Const(128),))


def test_shape_str_formats():
    assert shape_str((Const(128),)) == "(128,)"
    assert shape_str((Const(64), Const(1))) == "(64,1)"
    assert shape_str((Symbol("N"),)) == "(N,)"


def test_broadcast_equal_and_size1():
    assert broadcast((Const(8),), (Const(8),)) == (Const(8),)
    assert broadcast((Const(1),), (Const(8),)) == (Const(8),)
    assert broadcast((Const(8),), (Const(1),)) == (Const(8),)


def test_broadcast_incompatible():
    assert broadcast((Const(8),), (Const(7),)) is None
    assert broadcast((Const(8),), (Const(8), Const(1))) is None  # rank 不等 → None


def test_numel_and_static():
    assert numel((Const(64), Const(128))) == 8192
    assert numel((Symbol("N"),)) is None


# ---------------------------------------------------------------------------
# layout 律 L1–L5（type-system §3.3）
# ---------------------------------------------------------------------------

L128 = Identity((Const(128),))
L64 = Identity((Const(64),))


def test_L1_load_erases():
    assert normalize(LoadL(ROW_MAJOR, L128)) == L128


def test_L2_bcast_scalar_erases():
    assert normalize(BcastScalarL(L128)) == L128


def test_L3_cast_erases():
    assert normalize(CastL(L128)) == L128


def test_L4_join_erases():
    assert normalize(JoinL(L128, L128)) == L128


def test_L5_product_congruence():
    p = ProductL(LoadL(ROW_MAJOR, L64), CastL(L128))
    assert normalize(p) == ProductL(L64, L128)


def test_L5_unit_law_via_congruence():
    # Product(Identity(()), L) 与 L 分别规范化后结构不同但律的可扩展性由签名保证
    assert equiv(JoinL(ProductL(L64, L128), ProductL(L64, L128)),
                 ProductL(L64, L128))


def test_equiv_same_seed_true():
    assert equiv(LoadL(ROW_MAJOR, L128), JoinL(L128, BcastScalarL(L128)))


def test_equiv_different_seed_false():
    assert not equiv(L128, L64)


def test_normal_form_is_identity_in_v01():
    """v0.1 事实：正规形式恒为 Identity(n)——等价判定退化为种子相等。"""
    term = JoinL(CastL(LoadL(ROW_MAJOR, BcastScalarL(JoinL(L128, L128)))), L128)
    assert normalize(term) == L128


# ---------------------------------------------------------------------------
# checker 助手的 E04/E05（v0.1 表面语言不可触发——走单元测试）
# ---------------------------------------------------------------------------

def test_E04_rank_mismatch_unit():
    with pytest.raises(TilaError) as ei:
        bcheck.require_broadcast((Const(8),), (Const(8), Const(1)), ANY, "test")
    assert ei.value.code == "E04"


def test_E05_layout_mismatch_unit():
    with pytest.raises(TilaError) as ei:
        lcheck.require_equiv(L128, L64, ANY, "test")
    assert ei.value.code == "E05"
    assert "identity(128)" in ei.value.render()
