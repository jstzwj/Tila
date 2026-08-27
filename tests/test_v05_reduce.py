"""v0.5 归约与逐元素内建（docs/v0.5-reduce.md）：R20 边缘化（marginal 三分支）、
R21 一元族（L8 记法）、R22 where（三方广播 + merge_nontrivial）、R23
num_programs（observational）、R24 语境字面量/neg_inf、E21 矩阵、
softmax 差分（interpreter 主场）与归约 × K-loop 累加组合。"""

import re
import numpy as np
import pytest
import torch

from tila.driver import compile_kernel
from tila.diagnostics import TilaError
from tila.interp import run_kernel
from tila.types.dist import Identity, Mma, Product, Seed, Slice, equiv_dist
from tila.types.shape import Const
from tila.checker.layout import marginal
from tila.diagnostics import Loc, TilaError as TE

SOFTMAX = open("examples/softmax.tila", encoding="utf-8").read()

# 2D 前奏：rows/cols 坐标 + 联合掩码（Product(L0,L1) 的标准构造点）
_PRELUDE_2D = """    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
"""

SIG_2D = ("a: tila.Tensor[tila.float32, M, N], "
          "BM: tila.constexpr = 64, BN: tila.constexpr = 128")


def _c(body, params=SIG_2D, overrides=None):
    return compile_kernel(
        f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n", overrides)


def _expect(body, params=SIG_2D, subcode=None, code="E21", overrides=None):
    with pytest.raises(TilaError) as ei:
        _c(body, params, overrides)
    assert ei.value.code == code, ei.value.render()
    if subcode is not None:
        assert ei.value.subcode == subcode, ei.value.render()
    return ei.value


def _load_2d(dtype="tila.float32"):
    return f"    x = tila.load(a, (rm2, rn2), mask=m2)\n"


# ---------------------------------------------------------------------------
# marginal 单元（layout 代数层）
# ---------------------------------------------------------------------------

_L0 = Identity((Const(64),))
_L1 = Identity((Const(128),))
_S2 = (Const(64), Const(128))


def test_marginal_product_takes_factor():
    assert marginal(Product((_L0, _L1)), _S2, 1, Loc(1, 1), "t") == _L0
    assert marginal(Product((_L0, _L1)), _S2, 0, Loc(1, 1), "t") == _L1


def test_marginal_size1_axis_leaves_layout():
    # (64,1)：归约平凡轴 → layout 原样
    assert marginal(_L0, (Const(64), Const(1)), 1, Loc(1, 1), "t") == _L0
    # (1,128)：归约平凡轴 0 → layout 原样
    assert marginal(_L1, (Const(1), Const(128)), 0, Loc(1, 1), "t") == _L1


def test_marginal_lift_axis_dropped():
    # expand_dim 的 Lift 产物被归约到无所有权轴 → Lift 剥离（评审 §31）
    from tila.types import Lift

    assert marginal(Lift(_L0, 1), (Const(64), Const(1)), 1, Loc(1, 1), "t") == _L0
    assert marginal(Lift(_L1, 0), (Const(1), Const(128)), 0, Loc(1, 1), "t") == _L1


def test_marginal_only_nontrivial_axis_reduced_to_trivial():
    # (64,1) 归约轴 0：结果 (1,) 全平凡
    assert marginal(_L0, (Const(64), Const(1)), 0, Loc(1, 1), "t") == Identity(())


def test_marginal_seed_closed_under_axis_drop():
    z = Seed(_S2)
    assert marginal(z, _S2, 1, Loc(1, 1), "t") == Seed((Const(64),))


def test_marginal_mma_branch6_slice():
    # v0.6 分支六（docs/v0.6-attention.md §2.1）：Mma → Slice(Mma, k)——
    # 边缘 = 亲代切片（v0.5 的 E21 ReduceLayout 对合法输入成为空位）
    assert marginal(Mma(64, 128, 64), _S2, 1, Loc(1, 1), "t") == Slice(Mma(64, 128, 64), 1)
    assert marginal(Mma(64, 128, 64), _S2, 0, Loc(1, 1), "t") == Slice(Mma(64, 128, 64), 0)
    # equiv 仅同亲代；Slice ≢ Identity（能力边界维持）
    assert equiv_dist(Slice(Mma(64, 128, 64), 1), Slice(Mma(64, 128, 64), 1))
    assert not equiv_dist(Slice(Mma(64, 128, 64), 1), Slice(Mma(64, 128, 32), 1))
    assert not equiv_dist(Slice(Mma(64, 128, 64), 1), Identity((Const(64),)))


# ---------------------------------------------------------------------------
# R20 sum/max：规则与诊断
# ---------------------------------------------------------------------------

def test_sum_max_tir_and_triton():
    body = _PRELUDE_2D + _load_2d() + """    s = tila.sum(x, axis=1)
    m = tila.max(x, axis=0)
"""
    res = _c(body)
    assert "%s = sum %x 1 : Tile<f32, (64,), L0> [#s]" in res.tir_dump
    assert "%m = max %x 0 : Tile<f32, (128,), L2> [#m]" in res.tir_dump
    assert "tl.sum(s_x, 1, dtype=tl.float32)" in res.triton_source or \
           "tl.sum(x, 1, dtype=tl.float32)" in res.triton_source
    assert "tl.max(x, 0)" in res.triton_source


def test_sum_axis_positional_and_constexpr():
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, 1)\n"
    assert "%s = sum %x 1" in _c(body).tir_dump
    sig = SIG_2D + ", AX: tila.constexpr = 0"
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, axis=AX)\n"
    res = _c(body, params=sig)
    assert "%s = sum %x 0" in res.tir_dump


def test_sum_dtype_preserved_no_promotion():
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, 1)\n"
    res = _c(body)
    assert "tl.sum(x, 1, dtype=tl.float32)" in res.triton_source


def test_max_narrow_dtype_cast_recovery():
    # f16 tile：tl.max 内部提升 f32 返回——lowering 以 cast 恢复 Tila dtype
    sig = ("a: tila.Tensor[tila.float16, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    m = tila.max(x, 1)\n"
    res = _c(body, params=sig)
    assert "tl.cast(tl.max(x, 1), tl.float16)" in res.triton_source
    assert "%m = max %x 1 : Tile<f16, (64,), L0>" in res.tir_dump


def test_int_sum_dtype_preserved():
    sig = ("a: tila.Tensor[tila.int32, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, 1)\n"
    res = _c(body, params=sig)
    assert "%s = sum %x 1 : Tile<i32, (64,), L0>" in res.tir_dump
    assert "tl.sum(x, 1, dtype=tl.int32)" in res.triton_source


def test_reduce_zeros_layout_closed():
    body = _PRELUDE_2D + "    z = tila.zeros((BM, BN), tila.float32)\n    s = tila.sum(z, 1)\n"
    res = _c(body)
    assert "seed(64)" in res.tir_dump  # Seed((64,)) 正规形式


def test_e21_rank1_full_reduce():
    body = ("    pid = tila.program_id(0)\n"
            "    offs = pid * 64 + tila.arange(0, 64)\n"
            "    m = offs < N\n"
            "    x = tila.load(a, (offs,), mask=m)\n"
            "    s = tila.sum(x, 0)\n")
    _expect(body, params="a: tila.Tensor[tila.float32, N], "
                         "B: tila.constexpr = 64", subcode="ReduceRank")


def test_e13_axis_form():
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, 2)\n"
    _expect(body, code="E13")
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, M)\n"   # 运行期符号
    _expect(body, code="E13")
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x)\n"
    _expect(body, code="E13")
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, 1, axis=1)\n"
    _expect(body, code="E13")


def test_e16_reduce_capability():
    sig = ("a: tila.Tensor[tila.float8e4m3fn, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    s = tila.sum(x, 1)\n"
    _expect(body, params=sig, code="E16")
    # bool 归约：掩码不是有序值
    body = _PRELUDE_2D + "    s = tila.max(m2, 1)\n"
    _expect(body, code="E16")


def test_mma_reduce_branch6_positive():
    # v0.6 分支六（docs/v0.6-attention.md §2.1）：dot 结果的归约合法——
    # 结果布局 = Slice(Mma, k)（亲代切片）；v0.5 的 E21 ReduceLayout 对合法
    # 输入成为空位。TIR 布局定义行出现 slice(L?, 1)。
    sig = ("a: tila.Tensor[tila.float16, M, K], b: tila.Tensor[tila.float16, K, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128, BK: tila.constexpr = 64")
    body = ("    pid = tila.program_id(0)\n"
            "    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)\n"
            "    rn2 = tila.expand_dim(tila.arange(0, BN), 0)\n"
            "    rk2 = tila.expand_dim(tila.arange(0, BK), 0)\n"
            "    rk3 = tila.expand_dim(tila.arange(0, BK), 1)\n"
            "    x = tila.load(a, (rm2, rk2), mask=(rm2 < M) & (rk2 < K), other=0.0)\n"
            "    y = tila.load(b, (rk3, rn2), mask=(rk3 < K) & (rn2 < N), other=0.0)\n"
            "    d = tila.dot(x, y)\n"
            "    s = tila.sum(d, 1)\n")
    res = _c(body, params=sig)
    assert re.search(r"^L\d+ = slice\(L\d+, 1\)$", res.tir_dump, re.M), res.tir_dump
    assert "tl.sum(d, 1, dtype=tl.float32)" in res.triton_source


# ---------------------------------------------------------------------------
# R21 一元族：L8（记法）与能力矩阵
# ---------------------------------------------------------------------------

def test_elem_layout_preserved_and_lowering():
    body = _PRELUDE_2D + _load_2d() + "    e = tila.exp(x)\n    r = tila.sqrt(x)\n    q = tila.exp2(x)\n"
    res = _c(body)
    assert "%e = exp %x : Tile<f32, (64,128), L4>" in res.tir_dump
    assert "%r = sqrt %x : Tile<f32, (64,128), L4>" in res.tir_dump
    assert "%q = exp2 %x : Tile<f32, (64,128), L4>" in res.tir_dump
    assert "tl.exp(x)" in res.triton_source
    assert "tl.sqrt(x)" in res.triton_source
    assert "tl.exp2(x)" in res.triton_source


def test_abs_int_domain():
    sig = ("a: tila.Tensor[tila.int32, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    v = tila.abs(x)\n"
    res = _c(body, params=sig)
    assert "%v = abs %x : Tile<i32, (64,128), L4>" in res.tir_dump
    assert "tl.abs(x)" in res.triton_source


def test_e16_elem_capability():
    sig = ("a: tila.Tensor[tila.int32, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    e = tila.exp(x)\n"
    _expect(body, params=sig, code="E16")
    body = _PRELUDE_2D + "    v = tila.abs(m2)\n"          # bool
    _expect(body, code="E16")
    sig8 = ("a: tila.Tensor[tila.float8e4m3fn, M, N], "
            "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    e = tila.sqrt(x)\n"
    _expect(body, params=sig8, code="E16")


def test_e13_elem_form():
    body = _PRELUDE_2D + _load_2d() + "    e = tila.exp(x, 1)\n"
    _expect(body, code="E13", overrides=None)


def test_elem_narrow_float_f32_compute_lowering():
    # 数学一元在 triton 3.7.1 只接受 fp32/fp64：f16/bf16 经 f32 计算 + cast 回
    sig = ("a: tila.Tensor[tila.float16, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    e = tila.exp(x)\n    v = tila.abs(x)\n"
    res = _c(body, params=sig)
    assert "tl.cast(tl.exp(tl.cast(x, tl.float32)), tl.float16)" in res.triton_source
    assert "tl.abs(x)" in res.triton_source


# ---------------------------------------------------------------------------
# R22 where：三方广播、merge_nontrivial、字面量分支
# ---------------------------------------------------------------------------

def test_where_literal_branches_and_neg_inf():
    body = _PRELUDE_2D + _load_2d() + """    neg = tila.where(m2, x, tila.neg_inf)
    z = tila.where(m2, x, 0.0)
"""
    res = _c(body)
    assert "%t2 = const -inf : Scalar(f32)" in res.tir_dump
    assert "%neg = where %m2 %x %t2 : Tile<f32, (64,128), L4>" in res.tir_dump
    assert 'tl.where(m2, x, float("-inf"))' in res.triton_source
    assert "tl.where(m2, x, 0.0)" in res.triton_source


def test_where_three_way_broadcast():
    # cond (BM,1)、a (BM,BN)、b 字面量 → (BM,BN)
    body = _PRELUDE_2D + _load_2d() + """    rm = pid * BM + tila.arange(0, BM)
    rowmask = tila.expand_dim(rm < M, 1)
    w = tila.where(rowmask, x, 0.0)
"""
    res = _c(body)
    assert "%w = where %rowmask %x %t" in res.tir_dump
    assert ": Tile<f32, (64,128), L4> [#w]" in res.tir_dump


def test_where_both_tiles_dtype_mismatch():
    body = _PRELUDE_2D + """    x = tila.load(a, (rm2, rn2), mask=m2)
    y = tila.cast(x, tila.float64)
    w = tila.where(m2, x, y)
"""
    _expect(body, code="E02")


def test_where_cond_not_bool():
    body = _PRELUDE_2D + """    x = tila.load(a, (rm2, rn2), mask=m2)
    w = tila.where(x, x, 0.0)
"""
    _expect(body, code="E02")


def test_where_both_literals():
    body = _PRELUDE_2D + _load_2d() + "    w = tila.where(m2, 0.0, 1.0)\n"
    _expect(body, code="E13")


def test_where_scalar_name_branch():
    body = _PRELUDE_2D + _load_2d() + "    w = tila.where(m2, x, M)\n"
    _expect(body, code="E13")


def test_where_literal_category():
    body = _PRELUDE_2D + _load_2d() + "    w = tila.where(m2, x, 0)\n"   # int 字面量进 float
    _expect(body, code="E02")


def test_where_mma_vs_product_layout_e05():
    sig = ("a: tila.Tensor[tila.float16, M, K], b: tila.Tensor[tila.float16, K, N], "
           "c: tila.Tensor[tila.float32, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128, BK: tila.constexpr = 64")
    body = ("    pid = tila.program_id(0)\n"
            "    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)\n"
            "    rn2 = tila.expand_dim(tila.arange(0, BN), 0)\n"
            "    m2 = (rm2 < M) & (rn2 < N)\n"
            "    x = tila.load(a, (rm2, tila.expand_dim(tila.arange(0, BK), 0)))\n"
            "    y = tila.load(b, (tila.expand_dim(tila.arange(0, BK), 1), rn2))\n"
            "    d = tila.dot(x, y)\n"
            "    f = tila.load(c, (rm2, rn2), mask=m2)\n"
            "    w = tila.where(m2, d, f)\n")
    _expect(body, params=sig, code="E05")


def test_where_fp8_e16():
    sig = ("a: tila.Tensor[tila.float8e4m3fn, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    w = tila.where(m2, x, 0.0)\n"
    _expect(body, params=sig, code="E16")


def test_e13_where_form():
    body = _PRELUDE_2D + _load_2d() + "    w = tila.where(m2, x)\n"
    _expect(body, code="E13")
    body = _PRELUDE_2D + _load_2d() + "    w = tila.where(m2, x, 0.0, 1.0)\n"
    _expect(body, code="E13")


# ---------------------------------------------------------------------------
# R23 num_programs：observational
# ---------------------------------------------------------------------------

def test_num_programs_tir_triton():
    body = _PRELUDE_2D + _load_2d() + \
        "    nblk = tila.num_programs(0)\n    y = x + tila.cast(nblk, tila.float32)\n"
    res = _c(body)
    assert "%nblk = num_programs 0 : Scalar(i32) [#nblk]" in res.tir_dump
    assert "tl.num_programs(0)" in res.triton_source


def test_num_programs_does_not_widen_grid():
    # 只读不建轴：grid 仍由 program_id 惯用法推导（M 一维）
    body = _PRELUDE_2D + _load_2d() + \
        "    nblk = tila.num_programs(1)\n    y = x + tila.cast(nblk, tila.float32)\n"
    res = _c(body)
    assert "grid = (triton.cdiv(M, BM),)" in res.triton_source


def test_num_programs_axis_form():
    body = _PRELUDE_2D + _load_2d() + "    nblk = tila.num_programs(2)\n"
    _expect(body, code="E13")
    body = _PRELUDE_2D + _load_2d() + "    nblk = tila.num_programs(M)\n"
    _expect(body, code="E13")


def test_num_programs_single_program_kernel():
    # 无 program_id 的 kernel：num_programs 仍是合法观测（grid 退化 1 program）
    body = ("    offs = tila.arange(0, 64)\n"
            "    m = offs < N\n"
            "    x = tila.load(a, (offs,), mask=m)\n"
            "    n = tila.num_programs(0)\n"
            "    y = x + tila.cast(n, tila.float32)\n"
            "    tila.store(a, (offs,), y, mask=m)\n")
    res = _c(body, params="a: tila.Tensor[tila.float32, N], "
                          "B: tila.constexpr = 64")
    assert "grid = (1,)" in res.triton_source


# ---------------------------------------------------------------------------
# R24 语境字面量 / neg_inf 表面形态
# ---------------------------------------------------------------------------

def test_neg_inf_call_form_rejected():
    body = _PRELUDE_2D + _load_2d() + "    w = tila.where(m2, x, tila.neg_inf())\n"
    _expect(body, code="E13")


def test_neg_inf_standalone_binding():
    body = _PRELUDE_2D + _load_2d() + "    v = tila.neg_inf\n"
    res = _c(body)
    assert "%v = const -inf : Scalar(f32) [#v]" in res.tir_dump


def test_r24_binop_tile_vs_float_literal():
    body = _PRELUDE_2D + _load_2d() + "    y = x * 0.5\n"
    res = _c(body)
    assert "%y = mul %x %t2 : Tile<f32, (64,128), L4> [#y]" in res.tir_dump
    assert "y = x * 0.5" in res.triton_source


def test_r24_cross_domain_rejected():
    body = _PRELUDE_2D + _load_2d() + "    y = x + 1\n"     # int 字面量进 f32 语境
    _expect(body, code="E02")
    sig = ("a: tila.Tensor[tila.int32, M, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + _load_2d() + "    y = x + 1.5\n"    # float 字面量进 i32 语境
    _expect(body, params=sig, code="E02")


# ---------------------------------------------------------------------------
# 差分（interpreter 主场）：softmax / 一元族 / 归约 / num_programs
# ---------------------------------------------------------------------------

def _softmax_case(M, N, BN=128, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, N)).astype(np.float16)
    a[0, 0] = -65504.0                 # f16 最低值：哨兵（−∞）不被污染
    a[M // 2] = -60000.0               # 全负行
    a[0, min(1, N - 1)] = 30000.0      # 大正值：max-减法防 exp 溢出
    return a


@pytest.mark.parametrize("M,N,BN", [
    (1, 1, 128), (63, 127, 128), (64, 128, 128), (65, 129, 256),
    (100, 140, 256), (130, 300, 512), (128, 128, 128),
])
def test_softmax_interpreter_vs_torch(M, N, BN):
    ov = {"BN": BN} if BN != 128 else None
    res = compile_kernel(SOFTMAX, ov)
    a = _softmax_case(M, N, seed=M + N)
    o = np.zeros((M, N), dtype=np.float16)
    run_kernel(res.kernel, {"a": a, "o": o}, {}, ov)
    ref = torch.softmax(torch.from_numpy(a.astype(np.float32)), dim=-1) \
        .numpy().astype(np.float16)
    assert np.abs(o.astype(np.float32) - ref.astype(np.float32)).max() <= 2e-3


@pytest.mark.parametrize("op,np_fn", [
    ("exp", np.exp), ("exp2", np.exp2), ("sqrt", np.sqrt), ("abs", np.abs),
])
def test_elem_interpreter_vs_numpy(op, np_fn):
    body = _PRELUDE_2D + _load_2d() + f"    y = tila.{op}(x)\n" + \
        "    tila.store(a, (rm2, rn2), y, mask=m2)\n"
    res = _c(body)
    rng = np.random.default_rng(7)
    M, N = 64, 128
    a = np.abs(rng.standard_normal((M, N)).astype(np.float32))  # sqrt 域非负
    src = a.copy()
    run_kernel(res.kernel, {"a": src}, {}, None)
    assert np.allclose(src, np_fn(a), rtol=1e-5, atol=1e-6)


def test_reduce_interpreter_vs_numpy_axis1():
    # (BM,BN) → (BM,)：o 是一维 buffer，逐 program 块写回
    sig = ("a: tila.Tensor[tila.float32, M, N], o: tila.Tensor[tila.float32, M], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + """    rm = pid * BM + tila.arange(0, BM)
    x = tila.load(a, (rm2, rn2), mask=m2)
    s = tila.sum(x, axis=1)
    tila.store(o, (rm,), s, mask=rm < M)
"""
    res = _c(body, params=sig)
    rng = np.random.default_rng(11)
    M, N = 100, 128
    a = rng.standard_normal((M, N)).astype(np.float32)
    o = np.zeros((M,), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "o": o}, {}, None)
    assert np.allclose(o, a.sum(axis=1), rtol=1e-5)

    body = _PRELUDE_2D + """    rm = pid * BM + tila.arange(0, BM)
    x = tila.load(a, (rm2, rn2), mask=m2)
    mx = tila.max(x, axis=1)
    tila.store(o, (rm,), mx, mask=rm < M)
"""
    res = _c(body, params=sig)
    o2 = np.zeros((M,), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "o": o2}, {}, None)
    assert np.allclose(o2, a.max(axis=1))


def test_reduce_interpreter_vs_numpy_axis0():
    # (BM,BN) → (BN,)：每个行块都覆盖整行——各 program 重算同一列和
    sig = ("a: tila.Tensor[tila.float32, M, N], o: tila.Tensor[tila.float32, N], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + """    rn = tila.arange(0, BN)
    x = tila.load(a, (rm2, rn2), mask=m2)
    s = tila.sum(x, axis=0)
    tila.store(o, (rn,), s, mask=rn < N)
"""
    res = _c(body, params=sig)
    rng = np.random.default_rng(13)
    M, N = 64, 100
    a = rng.standard_normal((M, N)).astype(np.float32)
    o = np.zeros((N,), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "o": o}, {}, None)
    assert np.allclose(o, a.sum(axis=0), rtol=1e-5)

    body = _PRELUDE_2D + """    rn = tila.arange(0, BN)
    x = tila.load(a, (rm2, rn2), mask=m2)
    mx = tila.max(x, axis=0)
    tila.store(o, (rn,), mx, mask=rn < N)
"""
    res = _c(body, params=sig)
    o2 = np.zeros((N,), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "o": o2}, {}, None)
    assert np.allclose(o2, a.max(axis=0))


def test_reduce_int32_interpreter_dtype_and_values():
    sig = ("a: tila.Tensor[tila.int32, M, N], o: tila.Tensor[tila.int32, M], "
           "BM: tila.constexpr = 64, BN: tila.constexpr = 128")
    body = _PRELUDE_2D + """    rm = pid * BM + tila.arange(0, BM)
    x = tila.load(a, (rm2, rn2), mask=m2)
    s = tila.sum(x, 1)
    tila.store(o, (rm,), s, mask=rm < M)
"""
    res = _c(body, params=sig)
    rng = np.random.default_rng(3)
    M, N = 64, 128
    a = rng.integers(-1000, 1000, size=(M, N)).astype(np.int32)
    o = np.zeros((M,), dtype=np.int32)
    run_kernel(res.kernel, {"a": a, "o": o}, {}, None)
    assert o.dtype == np.int32
    assert np.array_equal(o, a.sum(axis=1, dtype=np.int32))


def test_where_interpreter_select_semantics():
    body = _PRELUDE_2D + _load_2d() + \
        "    w = tila.where(m2, x, tila.neg_inf)\n" + \
        "    tila.store(a, (rm2, rn2), w, mask=m2)\n"
    res = _c(body)
    rng = np.random.default_rng(5)
    M, N = 63, 100
    a = rng.standard_normal((M, N)).astype(np.float32)
    src = a.copy()
    run_kernel(res.kernel, {"a": src}, {}, None)
    assert np.array_equal(src, a)  # 掩码内全写回原值；掩码外不动


def test_num_programs_interpreter_value():
    body = _PRELUDE_2D + _load_2d() + "    nblk = tila.num_programs(0)\n" + \
        "    tila.store(a, (rm2, rn2), x + tila.cast(nblk, tila.float32), mask=m2)\n"
    res = _c(body)
    M, N = 130, 128
    a = np.zeros((M, N), dtype=np.float32)
    run_kernel(res.kernel, {"a": a}, {}, None)
    assert np.allclose(a, 3.0)  # cdiv(130,64)=3 blocks，每 program +3


# ---------------------------------------------------------------------------
# 组合：归约 × K-loop 累加（L7 物化边缘化分布）；where 在循环体内
# ---------------------------------------------------------------------------

ROWSUM_LOOP = """import tila


@tila.jit
def rowsum_loop(
    a: tila.Tensor[tila.float32, M, N],
    o: tila.Tensor[tila.float32, M],
    BM: tila.constexpr = 64,
    BN: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rm = pid * BM + tila.arange(0, BM)
    acc = tila.zeros((BM,), tila.float32)
    for k0 in tila.range(0, N, BN):
        cols2 = tila.expand_dim(k0 + tila.arange(0, BN), 0)
        m2 = (rm2 < M) & (cols2 < N)
        t = tila.load(a, (rm2, cols2), mask=m2)
        e = tila.where(m2, t, 0.0)
        acc += tila.sum(e, 1)
    tila.store(o, (rm,), acc, mask=rm < M)
"""


@pytest.mark.parametrize("M,N,BN", [(64, 128, 128), (100, 300, 64), (1, 1, 128),
                                    (63, 127, 128), (128, 256, 128)])
def test_rowsum_loop_interpreter(M, N, BN):
    ov = {"BN": BN} if BN != 128 else None
    res = compile_kernel(ROWSUM_LOOP, ov)
    rng = np.random.default_rng(M * 1000 + N)
    a = rng.standard_normal((M, N)).astype(np.float32)
    o = np.zeros((M,), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "o": o}, {}, ov)
    # 容差按累加顺序差异定标：逐 64 列块累加 vs numpy pairwise（f32）
    assert np.allclose(o, a.sum(axis=1), rtol=2e-4, atol=1e-4)


def test_rowsum_loop_structural():
    res = compile_kernel(ROWSUM_LOOP)
    # 边缘化结果经 L7 物化进累加器：φ 与 next 链照常
    assert "sum %e 1" in res.tir_dump
    assert "acc.loop = phi %acc %acc.next" in res.tir_dump
