"""v0.2 二维预览（docs/v0.2-preview-2d.md）：R13 expand_dim、R14 size-1 广播、
R15 & 合取、Product 与律 L5、rank-2 解禁、program_id(1)、二维 grid。"""

import numpy as np
import pytest

from tila.driver import compile_kernel
from tila.diagnostics import TilaError
from tila.interp import run_kernel
from tila.types import Const, Identity, Product, normalize_dist

F32_2D = ("a: tila.Tensor[tila.float32, M, N], b: tila.Tensor[tila.float32, M, N], "
          "c: tila.Tensor[tila.float32, M, N]")

BATCHED = """import tila


@tila.jit
def batched_add(
    a: tila.Tensor[tila.float32, M, N],
    b: tila.Tensor[tila.float32, M, N],
    c: tila.Tensor[tila.float32, M, N],
    BM: tila.constexpr = 64,
    BN: tila.constexpr = 128,
):
    pid_m = tila.program_id(0)
    pid_n = tila.program_id(1)
    rows = pid_m * BM + tila.arange(0, BM)
    cols = pid_n * BN + tila.arange(0, BN)
    rows2 = tila.expand_dim(rows, 1)
    cols2 = tila.expand_dim(cols, 0)
    mask = (rows2 < M) & (cols2 < N)
    x = tila.load(a, (rows2, cols2), mask=mask)
    y = tila.load(b, (rows2, cols2), mask=mask)
    z = x + y
    tila.store(c, (rows2, cols2), z, mask=mask)
"""


def _c(body, params=F32_2D, overrides=None):
    return compile_kernel(f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n",
                          overrides)


# ---------------------------------------------------------------------------
# 文档一致性：TIR dump 与 v0.2-preview §4 走查块逐行对应
# ---------------------------------------------------------------------------

def test_batched_add_tir_matches_doc_walkthrough():
    res = compile_kernel(BATCHED)
    lines = res.tir_dump.splitlines()
    assert lines[0] == ("func @batched_add(a: Buffer<f32, (M,N)>, b: Buffer<f32, (M,N)>, "
                        "c: Buffer<f32, (M,N)>, M: Scalar(i32), N: Scalar(i32), "
                        "BM: Constexpr(i32)=64, BN: Constexpr(i32)=128)")
    assert lines[1:5] == ["L0 = identity(64)", "L1 = identity(128)",
                          "L2 = lift(L0, 1)", "L3 = lift(L1, 0)"]
    assert lines[5] == "L4 = product(L0,L1)"
    assert "%rows2 = expand_dim %rows 1 : Tile<i32, (64,1), L2> [#rows2]" in lines
    assert "%cols2 = expand_dim %cols 0 : Tile<i32, (1,128), L3> [#cols2]" in lines
    assert "%mask = and %m0 %m1 : Tile<bool, (64,128), L4> [#mask]" in lines
    # v0.3 坐标寻址：线性化不进 TIR，addptr 携带坐标元组（docs/v0.3-strides.md §3）
    assert "%p0 = addptr %a [%rows2, %cols2] : Address<f32, (64,128), L4>" in lines
    assert "%p2 = addptr %c [%rows2, %cols2] : Address<f32, (64,128), L4>" in lines


def test_batched_add_triton_kernel_body_matches_doc():
    res = compile_kernel(BATCHED)
    src = res.triton_source
    for expected in (
        "pid_m = tl.program_id(0)",
        "pid_n = tl.program_id(1)",
        "rows = pid_m * BM + tl.arange(0, BM)",
        "rows2 = tl.expand_dims(rows, 1)",
        "cols2 = tl.expand_dims(cols, 0)",
        "mask = (rows2 < M) & (cols2 < N)",
        "x = tl.load(a + (rows2 * N + cols2), mask=mask)",
        "tl.store(c + (rows2 * N + cols2), z, mask=mask)",
    ):
        assert expected in src


def test_batched_add_launcher_2d_grid():
    res = compile_kernel(BATCHED)
    src = res.triton_source
    assert "def batched_add_launch(a, b, c, BM: int = 64, BN: int = 128):" in src
    assert "M = a.shape[0]" in src and "N = a.shape[1]" in src
    # v0.3：RowMajor 默认的连续性契约（docs/v0.3-strides.md §4.2）；
    # v0.4：size-0 契约空真守卫（numel() == 0 or (…)，v0.4-kloop §12）
    assert ("assert (a.numel() == 0 or (a.stride(1) == 1 and a.stride(0) == a.shape[1])) and "
            "(b.numel() == 0 or (b.stride(1) == 1 and b.stride(0) == b.shape[1])) and "
            "(c.numel() == 0 or (c.stride(1) == 1 and c.stride(0) == c.shape[1]))") in src
    assert "grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))" in src
    assert "batched_add[grid](a, b, c, M, N, BM=BM, BN=BN)" in src


# ---------------------------------------------------------------------------
# R13 expand_dim
# ---------------------------------------------------------------------------

PRELUDE_1D = """    r = tila.arange(0, 64)
    r2 = tila.expand_dim(r, 1)
"""


def test_r13_axis_bounds():
    # rank 1 → axis ∈ {0, 1}（按结果张量轴编号）
    _c(PRELUDE_1D)
    for bad in (-1, 2):
        with pytest.raises(TilaError) as ei:
            _c(f"    r = tila.arange(0, 64)\n    r2 = tila.expand_dim(r, {bad})\n")
        assert ei.value.code == "E04"


def test_r13_axis_must_be_literal():
    with pytest.raises(TilaError) as ei:
        _c("    r = tila.arange(0, 64)\n    r2 = tila.expand_dim(r, N)\n")
    assert ei.value.code == "E04"


def test_r13_non_tile_operand():
    with pytest.raises(TilaError) as ei:
        _c("    r2 = tila.expand_dim(pid, 0)\n",
           params="a: tila.Tensor[tila.float32, M, N], pid, "
                  "BLOCK: tila.constexpr = 64")
    assert ei.value.code == "E07"


def test_r13_dist_lifted_and_shape_inserted():
    res = _c(PRELUDE_1D + "    m = r2 < 8\n")
    assert "Tile<i32, (64,1), L1>" in res.tir_dump
    assert "Tile<bool, (64,1), L1>" in res.tir_dump
    assert "L1 = lift(L0, 1)" in res.tir_dump


# ---------------------------------------------------------------------------
# R14 size-1 广播与 Product（律 L5 交互）
# ---------------------------------------------------------------------------

def test_r14_broadcast_produces_product():
    res = _c("""    r = tila.arange(0, 64)
    r2 = tila.expand_dim(r, 1)
    s = tila.arange(0, 128)
    s2 = tila.expand_dim(s, 0)
    z = r2 + s2
""")
    assert "L1 = lift(L0, 1)" in res.tir_dump
    assert "L3 = lift(L2, 0)" in res.tir_dump
    assert "L4 = product(L0,L2)" in res.tir_dump
    assert "%z = add %r2 %s2 : Tile<i32, (64,128), L4> [#z]" in res.tir_dump


def test_r14_broadcast_comparison_and_logic():
    res = _c("""    r = tila.arange(0, 64)
    r2 = tila.expand_dim(r, 1)
    s = tila.arange(0, 128)
    s2 = tila.expand_dim(s, 0)
    m = (r2 < M) & (s2 < N)
""")
    assert "%m = and %m0 %m1 : Tile<bool, (64,128), L4> [#m]" in res.tir_dump


def test_r14_same_shape_still_requires_equiv():
    # 严格同形走 strict_join（L5）：两个独立 arange(64) 种子相同 → 等价成立
    res = _c("""    r1 = tila.arange(0, 64)
    r2 = tila.arange(0, 64)
    z = r1 + r2
""")
    assert "%z = add %r1 %r2 : Tile<i32, (64,), L0> [#z]" in res.tir_dump


def test_product_l5_in_types():
    """乘 L5：Product 结构在正规形式中保持可见（诊断能说清行/列来源）。"""
    res = compile_kernel(BATCHED)
    assert "L4 = product(L0,L1)" in res.tir_dump
    # 正规形式真正分支：不再是 v0.1 的"恒为 Identity"
    assert normalize_dist(Product((Identity((Const(64),)), Identity((Const(128),))))) == \
        Product((Identity((Const(64),)), Identity((Const(128),))))


# ---------------------------------------------------------------------------
# interpreter 差分（2D）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N", [(1, 1), (3, 5), (64, 128), (65, 129), (200, 300)])
def test_batched_add_interp_matches_numpy(M, N):
    res = compile_kernel(BATCHED)
    rng = np.random.default_rng(M * 7 + N)
    a = rng.standard_normal((M, N)).astype(np.float32)
    b = rng.standard_normal((M, N)).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    np.testing.assert_allclose(c, a + b, rtol=1e-6)


def test_batched_add_constexpr_specialization():
    res = compile_kernel(BATCHED, {"BM": 8, "BN": 16})
    assert "Tile<i32, (8,1), L2>" in res.tir_dump
    assert "Tile<f32, (8,16), L4>" in res.tir_dump
    assert "tl.arange(0, BM)" in res.triton_source  # 发射保留名字（模型 B）
    rng = np.random.default_rng(2)
    a = rng.standard_normal((19, 45)).astype(np.float32)
    b = rng.standard_normal((19, 45)).astype(np.float32)
    c = np.zeros((19, 45), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c},
               constexpr={"BM": 8, "BN": 16})
    np.testing.assert_allclose(c, a + b, rtol=1e-6)


def test_2d_e17_when_axis_unbounded():
    # 2D tiling 但完全没有 `offs < dim` 界谓词（无 mask 的 load/store 是
    # 用户责任）→ launch analysis 推不出 grid → E17
    src = """import tila


@tila.jit
def bad(a: tila.Tensor[tila.float32, M, N], c: tila.Tensor[tila.float32, M, N],
        BM: tila.constexpr = 64, BN: tila.constexpr = 128):
    pid_m = tila.program_id(0)
    pid_n = tila.program_id(1)
    rows = pid_m * BM + tila.arange(0, BM)
    cols = pid_n * BN + tila.arange(0, BN)
    rows2 = tila.expand_dim(rows, 1)
    cols2 = tila.expand_dim(cols, 0)
    x = tila.load(a, (rows2, cols2))
    tila.store(c, (rows2, cols2), x)
"""
    with pytest.raises(TilaError) as ei:
        compile_kernel(src)
    assert ei.value.code == "E17"
    assert "bound predicate" in ei.value.message or "tiling" in ei.value.message


def test_address_arithmetic_rejected_e07():
    """v0.3 定稿：地址算术整体退场——旧写法 `tila.load(a + offs)` 在实参数量
    检查处即被拒（E13，消息含迁移提示）；独立出现的 `a + offs` 算术是 E07。
    坐标只能写进 tila.load/store 的实参（docs/v0.3-strides.md §1.2）。"""
    src = """import tila


@tila.jit
def bad(a: tila.Tensor[tila.float32, M, N], c: tila.Tensor[tila.float32, M, N],
        BM: tila.constexpr = 64):
    pid = tila.program_id(0)
    offs = pid * BM + tila.arange(0, BM)
    mask = offs < N
    x = tila.load(a + offs, mask=mask)
    tila.store(c, (offs,), x, mask=mask)
"""
    with pytest.raises(TilaError) as ei:
        compile_kernel(src)
    assert ei.value.code == "E13"
    assert "was removed" in ei.value.message


def test_e19_coord_arity_replaces_flat_check():
    """rank-2 buffer 配 1 元坐标元组 → E19 CoordinateArity（元组长度必须等于
    buffer rank；rank-1 的 `(offs,)` 单元素形态合法）。"""
    src = """import tila


@tila.jit
def bad(a: tila.Tensor[tila.float32, M, N], c: tila.Tensor[tila.float32, M, N],
        BM: tila.constexpr = 64):
    pid = tila.program_id(0)
    offs = pid * BM + tila.arange(0, BM)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    tila.store(c, (offs,), x, mask=mask)
"""
    with pytest.raises(TilaError) as ei:
        compile_kernel(src)
    assert ei.value.code == "E19"
    assert ei.value.subcode == "CoordinateArity"
