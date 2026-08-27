"""v0.6a attention fragment 测试（docs/v0.6-attention.md §8.2–§8.4）。

覆盖：marginal 分支六与 Slice 能力边界、read_join one-sided 广播透明、
R-PT 谓词透明（含 v0.5 同族等价断言）、E05 管辖收窄的负例集（同形跨族
仍拒）、attention 差分（interpreter vs torch，§8.4 矩阵）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tila.checker.layout import join_distributions, marginal, strict_join
from tila.driver import compile_kernel
from tila.diagnostics import Loc, TilaError
from tila.types.dist import Identity, Lift, Mma, Product, Slice, equiv_dist
from tila.types.shape import Const

_L0 = Identity((Const(64),))
_L1 = Identity((Const(128),))
_S2 = (Const(64), Const(128))
_MMA = Mma(64, 128, 64)

# ---------------------------------------------------------------------------
# 单元：marginal 分支六与 Slice 能力边界（§8.2；marginal 构造层的用例在
# test_v05_reduce.py，此处补 equiv/能力边界与 join 侧）
# ---------------------------------------------------------------------------


def test_slice_same_parent_equiv_cross_parent_e05():
    assert equiv_dist(Slice(_MMA, 1), Slice(_MMA, 1))
    assert not equiv_dist(Slice(_MMA, 1), Slice(Mma(64, 128, 32), 1))
    assert not equiv_dist(Slice(_MMA, 1), _L0)


def test_slice_join_same_parent_ok_cross_family_e05():
    # 能力边界（评审 §3）：同亲代 Slice 同形 join ✓；Slice ⊕ Identity → E05
    assert equiv_dist(Slice(_MMA, 1), Slice(_MMA, 1))
    # (BM,1) 广播侧 = expand_dim 产物的 Lift 形态；形状判定忽略 reader 分布
    lay = join_distributions(Lift(Slice(_MMA, 1), 1), (Const(64), Const(1)),
                             Slice(_MMA, 1), (Const(64), Const(128)),
                             Loc(1, 1), "t")
    assert lay == Slice(_MMA, 1)  # one-sided read_join：全形侧获胜
    with pytest.raises(TilaError) as ei:
        # 同形双侧（(64,) ⊗ (64,)）：非广播 → strict_join → equiv 失败
        strict_join(Slice(_MMA, 1), _L0, Loc(1, 1), "t")
    assert ei.value.code == "E05"


# ---------------------------------------------------------------------------
# 单元：read_join one-sided（§8.2；收窄回归是本设计最需钉死的负例集）
# ---------------------------------------------------------------------------


def test_read_join_one_sided_broadcast_takes_full_owner():
    # (64,128) 全形 Mma × (64,1) 广播侧 → Mma（广播侧分布免检）
    out = join_distributions(_MMA, _S2, Lift(_L0, 1), (Const(64), Const(1)),
                             Loc(1, 1), "t")
    assert out == _MMA
    # 反向：广播侧在左
    out = join_distributions(Lift(Slice(_MMA, 1), 1), (Const(64), Const(1)),
                             _MMA, _S2, Loc(1, 1), "t")
    assert out == _MMA


def test_read_join_same_shape_still_strict_e05():
    # 同形双侧不走 read_join：Mma ⊕ Product 同形 → strict_join 等价失败 → E05
    with pytest.raises(TilaError) as ei:
        strict_join(_MMA, Product((_L0, _L1)), Loc(1, 1), "t")
    assert ei.value.code == "E05"


def test_read_join_dual_broadcast_stays_per_axis():
    # 双广播侧（(64,1)⊗(1,128)）：无全形持有者 → 逐轴段联合（掩码/坐标
    # 模式；评审 §32 的 infer_coordinate_dist 同款机制）
    out = join_distributions(Lift(Slice(_MMA, 1), 1), (Const(64), Const(1)),
                             Lift(_L1, 0), (Const(1), Const(128)),
                             Loc(1, 1), "t")
    assert out == Product((Slice(_MMA, 1), _L1))


def test_read_join_product_family_unchanged():
    # v0.5 语义在同族场景逐项一致：product × (64,1) 广播 → product（逐字节
    # 等价的机器保证——softmax 黄金稳定）
    out = join_distributions(Product((_L0, _L1)), _S2, Lift(_L0, 1),
                             (Const(64), Const(1)), Loc(1, 1), "t")
    assert out == Product((_L0, _L1))


# ---------------------------------------------------------------------------
# 单元：R-PT 谓词透明（§8.2；含 v0.5 等价断言）
# ---------------------------------------------------------------------------

_PRELUDE_2D = (
    "    pid = tila.program_id(0)\n"
    "    rm2 = tila.expand_dim(pid * 64 + tila.arange(0, 64), 1)\n"
    "    rn2 = tila.expand_dim(tila.arange(0, 128), 0)\n"
    "    m2 = (rm2 < M) & (rn2 < N)\n"
)
_SIG_2D = ("a: tila.Tensor[tila.float32, M, N], BM: tila.constexpr = 64, "
           "BN: tila.constexpr = 128")


def _c(body, params=_SIG_2D, overrides=None):
    return compile_kernel(
        f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n", overrides)


def test_rpt_where_cond_any_layout_ok():
    # R-PT：cond 布局完全不进推导——v0.5 同族结果不变（黄金等价的用例级证明）
    body = _PRELUDE_2D + (
        "    x = tila.load(a, (rm2, rn2), mask=m2)\n"
        "    y = tila.where(m2, x, 0.0)\n"
        "    tila.store(a, (rm2, rn2), y, mask=m2)\n")
    res = _c(body)
    # 值侧分布 = load 的 product —— 与 v0.5 三方归并结果一致
    assert "where %m2 %x" in res.tir_dump
    assert ": Tile<f32, (64,128), L4>" in res.tir_dump


def test_rpt_where_cond_grows_result_shape():
    # cond 撑大值侧：(BM,1) 值 × (1,BN) cond → (BM,BN)，布局 = 值侧广播
    body = _PRELUDE_2D + (
        "    x = tila.load(a, (rm2, rn2), mask=m2)\n"
        "    mrow = tila.expand_dim(tila.max(x, axis=1), 1)\n"
        "    y = tila.where(rn2 < N, mrow, 0.0)\n")
    res = _c(body)
    assert "%y = where %m3 %mrow %t4 : Tile<f32, (64,128), L1>" in res.tir_dump, \
        res.tir_dump


# ---------------------------------------------------------------------------
# E05 收窄回归（同形跨族 join 仍拒——attention 轨迹内的防线）
# ---------------------------------------------------------------------------

_SIG_ATTN = ("q: tila.Tensor[tila.float16, M, D], k: tila.Tensor[tila.float16, N, D], "
             "o: tila.Tensor[tila.float16, M, D], "
             "BM: tila.constexpr = 64, BN: tila.constexpr = 64, BD: tila.constexpr = 64")


def _attn_body(tail):
    return (
        "    pid = tila.program_id(0)\n"
        "    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)\n"
        "    rn2 = tila.expand_dim(tila.arange(0, BN), 0)\n"
        "    rd3 = tila.expand_dim(tila.arange(0, BD), 1)\n"
        "    q2 = tila.load(q, (rm2, tila.expand_dim(tila.arange(0, BD), 0)), "
        "mask=(rm2 < M) & (tila.expand_dim(tila.arange(0, BD), 0) < D), other=0.0)\n"
        "    k2 = tila.load(k, (rn2, rd3), mask=(rn2 < N) & (rd3 < D), other=0.0)\n"
        "    d = tila.dot(q2, k2)\n" + tail)


def test_e05_mma_vs_product_same_shape_still_rejected():
    # dot 结果（Mma）⊕ load 值（Product）同形 → E05（归属真决策，v0.6 不放行）
    body = _attn_body(
        "    y = d + tila.cast(q2, tila.float32)\n")
    with pytest.raises(TilaError) as ei:
        _c(body, params=_SIG_ATTN)
    assert ei.value.code == "E05"


def test_e05_slice_vs_identity_1d_same_shape_rejected():
    # Slice(Mma,1) ⊕ arange 派生 1D 同形 → E05（Slice 不是独立 join 族）
    body = _attn_body(
        "    s = tila.max(d, axis=1)\n"
        "    t = tila.sum(tila.cast(q2, tila.float32), axis=1)\n"
        "    z = s + t\n")
    with pytest.raises(TilaError) as ei:
        _c(body, params=_SIG_ATTN)
    assert ei.value.code == "E05"


def test_e05_cross_mma_terms_same_shape_rejected():
    # Mma(BM,BN,BD) ⊕ Mma(BM,BN,32) 同形 → E05（跨项）
    body = _attn_body(
        "    rz2 = tila.expand_dim(tila.arange(0, 32), 0)\n"
        "    rz3 = tila.expand_dim(tila.arange(0, 32), 1)\n"
        "    q3 = tila.load(q, (rm2, rz2), mask=(rm2 < M) & (rz2 < D), other=0.0)\n"
        "    k3 = tila.load(k, (rn2, rz3), mask=(rn2 < N) & (rz3 < D), other=0.0)\n"
        "    e = tila.dot(q3, k3)\n"
        "    z = d + e\n")
    with pytest.raises(TilaError) as ei:
        _c(body, params=_SIG_ATTN)
    assert ei.value.code == "E05"


# ---------------------------------------------------------------------------
# attention 差分（§8.4：interpreter vs torch；GPU 侧在 test_gpu_integration）
# ---------------------------------------------------------------------------


def _ref_attention(q, k, v):
    qf, kf, vf = [torch.tensor(np.ascontiguousarray(x)) for x in (q, k, v)]
    return (torch.softmax(qf.float() @ kf.float().T, dim=-1) @ vf.float()).numpy()


_ATTN_SRC = None


def _attn_kernel():
    global _ATTN_SRC
    if _ATTN_SRC is None:
        from pathlib import Path

        _ATTN_SRC = Path("examples/attention.tila").read_text(encoding="utf-8")
    return compile_kernel(_ATTN_SRC)


def _overrides_for(N, D):
    ov = {}
    if N > 64:
        ov["BN"] = 128
    if D > 64:
        ov["BD"] = 128
    return ov or None


@pytest.mark.parametrize("M,N,D", [
    (1, 1, 64), (63, 63, 64), (64, 64, 64), (64, 127, 128),
    (100, 127, 64), (33, 64, 16), (64, 33, 33), (128, 128, 64),
])
def test_attention_differential_cpu(M, N, D):
    from tila.interp import run_kernel

    res = _attn_kernel()
    rng = np.random.default_rng(42 + M * 7 + N * 3 + D)
    q = (rng.standard_normal((M, D)) * 1.5).astype(np.float16)
    k = (rng.standard_normal((N, D)) * 3.0).astype(np.float16)
    v = rng.standard_normal((N, D)).astype(np.float16)
    o = np.zeros((M, D), dtype=np.float16)
    run_kernel(res.kernel, {"q": q, "k": k, "v": v, "o": o}, {},
               _overrides_for(N, D))
    ref = _ref_attention(q, k, v).astype(np.float16)
    np.testing.assert_allclose(o.astype(np.float32), ref.astype(np.float32),
                               rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("M,N,D", [(64, 64, 64), (100, 127, 64)])
def test_attention_extreme_values(M, N, D):
    # 大 logits（±3e4 量级：exp 安全由构造保证）、全负行、f16 最低值行
    from tila.interp import run_kernel

    res = _attn_kernel()
    rng = np.random.default_rng(7)
    ov = _overrides_for(N, D)
    for scale, offset in ((300.0, 0.0), (-50.0, -1.0)):
        q = ((rng.standard_normal((M, D)) + offset) * scale).astype(np.float16)
        k = ((rng.standard_normal((N, D)) + offset) * scale).astype(np.float16)
        v = rng.standard_normal((N, D)).astype(np.float16)
        o = np.zeros((M, D), dtype=np.float16)
        run_kernel(res.kernel, {"q": q, "k": k, "v": v, "o": o}, {}, ov)
        ref = _ref_attention(q, k, v).astype(np.float16)
        np.testing.assert_allclose(o.astype(np.float32), ref.astype(np.float32),
                                   rtol=2e-2, atol=2e-3)
    # f16 最低值元素（哨兵不污染 max 的边界验证）
    q = np.full((M, D), -65504.0, dtype=np.float16)
    k = np.full((N, D), -65504.0, dtype=np.float16)
    v = rng.standard_normal((N, D)).astype(np.float16)
    o = np.zeros((M, D), dtype=np.float16)
    run_kernel(res.kernel, {"q": q, "k": k, "v": v, "o": o}, {}, ov)
    ref = _ref_attention(q, k, v).astype(np.float16)
    np.testing.assert_allclose(o.astype(np.float32), ref.astype(np.float32),
                               rtol=2e-2, atol=2e-3)


def test_attention_tir_walk():
    # §2.3 走查：softmax 块全程驻留 Mma 族；slice 定义行存在；Lift 记录
    # 大小轴（mx2/s2 的 (64,1) 形态）
    res = _attn_kernel()
    d = res.tir_dump
    assert "L4 = mma(64,64,64)" in d
    assert "L5 = slice(L4, 1)" in d
    assert "L6 = lift(L5, 1)" in d
    assert "%qk_ = where %m_qk %qk %t5 : Tile<f32, (64,64), L4>" in d
    assert "%mx = max %qk_ 1 : Tile<f32, (64,), L5>" in d
    assert "%s = sum %p 1 : Tile<f32, (64,), L5>" in d
    assert "%w = div %p %s2 : Tile<f32, (64,64), L4>" in d
    assert "tl.where(m_qk, qk, float(\"-inf\"))" in res.triton_source
