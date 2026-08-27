"""类型系统单元测试：dtype 能力表、shape ⊗、DistExpr 律 L1–L7、cast 矩阵、E04/E05 助手。"""

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
    Lift,
    Mma,
    NODIST,
    NoDist,
    Product,
    Seed,
    Slice,
    Symbol,
    broadcast,
    cast_allowed,
    cmp_ops,
    equiv_dist,
    is_float,
    is_int,
    normalize_dist,
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
# DistExpr 律 L1–L7（type-system §3；v0.6a DistExpr 重构——评审 D 冻结的
# 五 term 规范：v0.1 的擦除包装 term（LoadL/BcastScalarL/CastL/JoinL）退役）
# ---------------------------------------------------------------------------

L128 = Identity((Const(128),))
L64 = Identity((Const(64),))


def test_L1_structural_equivalence():
    assert normalize_dist(L128) == L128
    assert normalize_dist(Mma(64, 128, 64)) == Mma(64, 128, 64)


def test_L2_product_equiv_and_flatten():
    p = Product((L64, Product((L128,))))
    # D2：嵌套 Product 拍平为 n-ary
    assert normalize_dist(p) == Product((L64, L128))
    assert equiv_dist(Product((L64, L128)), Product((L64, L128)))
    assert not equiv_dist(Product((L64, L128)), Product((L128, L64)))


def test_L3_lift_provenance():
    a = Lift(L64, 1)
    b = Lift(L64, 1)
    c = Lift(L64, 0)
    assert equiv_dist(a, b)
    assert not equiv_dist(a, c)          # 轴不同 → 不等
    assert not equiv_dist(a, L64)        # Lift ≢ 内层（Lift 不裂回）
    # 同轴幂等（v0.2 L6 继承）
    assert normalize_dist(Lift(a, 1)) == a


def test_L4_slice_provenance():
    m = Mma(64, 128, 64)
    assert equiv_dist(Slice(m, 1), Slice(m, 1))
    assert not equiv_dist(Slice(m, 1), Slice(Mma(64, 128, 32), 1))
    assert not equiv_dist(Slice(m, 1), L64)
    assert normalize_dist(Slice(m, 1)) == Slice(m, 1)


def test_L5_strict_join_unit():
    # checker 层 strict_join 的纯语义：同形双侧 owner → equiv 放行 / E05
    assert equiv_dist(L128, L128)
    assert not equiv_dist(L128, L64)
    with pytest.raises(TilaError) as ei:
        lcheck.strict_join(L128, L64, ANY, "test")
    assert ei.value.code == "E05"
    assert "strict-join failure" in ei.value.render()


def test_L6_read_join_pure():
    s2 = (Const(64), Const(128))
    owner = Product((L64, L128))
    # one-sided：reader (64,1) 真广播投影 → owner
    assert lcheck.read_join(owner, s2, L64, (Const(64), Const(1)),
                            ANY, "t") == owner
    # NoDist 标量面恒成立
    assert lcheck.read_join(owner, s2, NODIST, None, ANY, "t") == owner
    # 同形 → 非 one-sided（proper 排除）→ None（落 strict）
    assert lcheck.read_join(owner, s2, owner, s2, ANY, "t") is None
    # 非投影（(128,1) 读 (64,128)）→ None（落逐轴路径）
    assert lcheck.read_join(owner, s2, L128, (Const(128), Const(1)),
                            ANY, "t") is None


def test_L7_marginal_product_and_mma():
    # checker 层 marginal 的纯语义（R20）：Product → 幸存因子；Mma → Slice
    from tila.types.shape import Const as C

    s2 = (C(64), C(128))
    assert lcheck.marginal(Product((L64, L128)), s2, 1, ANY, "t") == L64
    assert lcheck.marginal(Product((L64, L128)), s2, 0, ANY, "t") == L128
    m = Mma(64, 128, 64)
    assert lcheck.marginal(m, s2, 1, ANY, "t") == Slice(m, 1)
    assert lcheck.marginal(m, s2, 0, ANY, "t") == Slice(m, 0)


def test_seed_not_in_distexpr_world():
    # 评审 §38：Seed 是累加器状态载体，不是 DistExpr——两种子态互相"相等"
    # 是唯一成立情形
    z = Seed((Const(64), Const(128)))
    assert equiv_dist(z, z)
    assert not equiv_dist(z, Mma(64, 128, 64))
    assert not equiv_dist(z, L64)


def test_seed_materialization_is_checker_behavior():
    # 旧 L7（join(Zeros, L) ≡ L）不再是纯层律——种子物化是 checker 累加
    # 状态机（Seed + Tile[D] → Materialized(D)）。此处固定"值侧获胜"的
    # 纯语义由 check_augassign 路径覆盖（test_v04_kloop / matmul_loop），
    # 纯层只断言 Seed 不被 normalize 化简：
    z = Seed((Const(64),))
    assert normalize_dist(z) == z


def test_nodist_scalar_face():
    assert isinstance(NODIST, NoDist)
    s = (Const(64),)
    assert lcheck.read_join(L64, s, NODIST, None, ANY, "t") == L64


def test_equiv_same_seed_true():
    assert equiv_dist(L128, Lift(L128, 1).of)


def test_equiv_different_seed_false():
    assert not equiv_dist(L128, L64)


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
