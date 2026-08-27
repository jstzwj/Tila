"""v0.3 strides（docs/v0.3-strides.md）：坐标寻址 R7'、Strided MemoryLayout、
E19/E12 矩阵、launcher stride 契约、interpreter 非连续差分（转置 / padded /
stride-2 gather）——最后一批是本设计的验收主场。"""

import numpy as np
import pytest

from tila.driver import compile_kernel
from tila.diagnostics import TilaError
from tila.interp import run_kernel

MATMUL = open("examples/matmul.tila", encoding="utf-8").read()
BATCHED = open("examples/batched_add.tila", encoding="utf-8").read()

# 2D 坐标公共前奏
RC = """    pid_m = tila.program_id(0)
    pid_n = tila.program_id(1)
    rows = pid_m * 64 + tila.arange(0, 64)
    cols = pid_n * 128 + tila.arange(0, 128)
    rows2 = tila.expand_dim(rows, 1)
    cols2 = tila.expand_dim(cols, 0)
"""


def _c(body, params, overrides=None):
    return compile_kernel(f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n",
                          overrides)


# ---------------------------------------------------------------------------
# 注解三形态：全符号 / dim 复用 / 静态
# ---------------------------------------------------------------------------

def test_annotation_full_symbolic_strides():
    res = compile_kernel(MATMUL)
    assert "b: Buffer<f16, (K,N), strides=(sb0,sb1)>" in res.tir_dump
    assert "sb0: Scalar(i32), sb1: Scalar(i32)" in res.tir_dump


def test_annotation_dim_reuse_no_extra_sym():
    src = MATMUL.replace(
        "b: tila.Tensor[tila.float16, K, N, (sb0, sb1)]",
        "b: tila.Tensor[tila.float16, K, N, (N, 1)]")
    res = compile_kernel(src)
    assert "sb0" not in res.tir_dump
    assert "b: Buffer<f16, (K,N), strides=(N,1)>" in res.tir_dump
    # dim 复用 → 断言而非绑定：stride(0) 必须等于已绑定的 N
    assert "b.stride(0) == N" in res.triton_source
    assert "b.stride(1) == 1" in res.triton_source


def test_annotation_static_strides():
    body = RC + """    m = (rows2 < M) & (cols2 < N)
    x = tila.load(b, (rows2, cols2), mask=m)
    tila.store(c, (rows2, cols2), x, mask=m)
"""
    res = _c(body, "b: tila.Tensor[tila.float32, M, N, (N, 2)], "
                   "c: tila.Tensor[tila.float32, M, N, (N, 2)]")
    assert "strides=(N,2)" in res.tir_dump
    assert "b.stride(0) == N" in res.triton_source
    assert "b.stride(1) == 2" in res.triton_source
    assert "b + (rows2 * N + cols2 * 2)" in res.triton_source


def test_rank1_strided_flat_form_kept():
    """rank-1 平面形式不变；Strided 时发射乘步长。"""
    body = """    pid = tila.program_id(0)
    offs = pid * 64 + tila.arange(0, 64)
    x = tila.load(b, (offs,), mask=offs < N)
    tila.store(c, (offs,), x, mask=offs < N)
"""
    res = _c(body, "b: tila.Tensor[tila.float32, N, (2,)], "
                   "c: tila.Tensor[tila.float32, N, (2,)]")
    assert "%p0 = addptr %b %offs" in res.tir_dump          # 单坐标：v0.2 形式
    assert "tl.load(b + offs * 2" in res.triton_source
    assert "c.stride(0) == 2" in res.triton_source


# ---------------------------------------------------------------------------
# E12 注解矩阵
# ---------------------------------------------------------------------------

def _ann_e12(strides_src):
    with pytest.raises(TilaError) as ei:
        _c(RC + "    x = tila.load(b, (rows2, cols2))\n", strides_src)
    assert ei.value.code == "E12"


def test_e12_strides_arity_mismatch():
    _ann_e12("b: tila.Tensor[tila.float32, M, N, (1,)]")


def test_e12_strides_zero():
    _ann_e12("b: tila.Tensor[tila.float32, M, N, (N, 0)]")


def test_e12_strides_negative():
    _ann_e12("b: tila.Tensor[tila.float32, M, N, (N, -1)]")


def test_e12_strides_bad_element():
    _ann_e12("b: tila.Tensor[tila.float32, M, N, (N, 1.5)]")


def test_e12_strides_empty_tuple():
    _ann_e12("b: tila.Tensor[tila.float32, M, N, ()]")


def test_e12_strides_symbol_conflicts_with_param():
    _ann_e12("b: tila.Tensor[tila.float32, M, N, (q, 1)], q: tila.Tensor[tila.float32, M]")


# ---------------------------------------------------------------------------
# E19 寻址形态矩阵
# ---------------------------------------------------------------------------

def test_e19_coord_arity_mismatch():
    with pytest.raises(TilaError) as ei:
        _c(RC + "    x = tila.load(b, (rows2, cols2, rows2))\n",
           "b: tila.Tensor[tila.float32, M, N]")
    assert ei.value.code == "E19"
    assert ei.value.subcode == "CoordinateArity"
    assert "rank 2" in ei.value.message


def test_e07_load_first_arg_not_buffer():
    with pytest.raises(TilaError) as ei:
        _c(RC + "    t = rows2 + cols2\n    x = tila.load(t, (rows2, cols2))\n",
           "b: tila.Tensor[tila.float32, M, N]")
    assert ei.value.code == "E07"
    assert "first argument" in ei.value.message


def test_e19_scalar_coord():
    with pytest.raises(TilaError) as ei:
        _c(RC + "    x = tila.load(b, (rows2, 5))\n",
           "b: tila.Tensor[tila.float32, M, N]")
    assert ei.value.code == "E19"
    assert ei.value.subcode == "CoordinateKind"
    assert "i32 index tile" in ei.value.message


def test_e02_coord_dtype():
    with pytest.raises(TilaError) as ei:
        _c(RC + """    f = tila.cast(rows2, tila.float32)
    x = tila.load(b, (f, cols2))
""", "b: tila.Tensor[tila.float32, M, N]")
    assert ei.value.code == "E02"


# ---------------------------------------------------------------------------
# launcher 生成物
# ---------------------------------------------------------------------------

def test_launcher_stride_bindings_and_call_order():
    res = compile_kernel(MATMUL)
    src = res.triton_source
    assert "sb0 = b.stride(0)" in src and "sb1 = b.stride(1)" in src
    # 符号序：维（M,K,N）先于 stride（sb0,sb1）——签名与调用一致
    assert "def matmul_launch(a, b, c, BM: int = 64, BN: int = 128, BK: int = 64):" in src
    assert "matmul[grid](a, b, c, M, K, N, sb0, sb1, BM=BM, BN=BN, BK=BK)" in src
    # Strided buffer 不发连续性断言；RowMajor 的 a/c 发
    assert "b.stride(1) == 1 and" not in src
    assert "a.stride(1) == 1" in src and "c.stride(0) == c.shape[1]" in src


def test_addptr_emission_forms():
    res = compile_kernel(MATMUL)
    assert "a + (rm2 * K + rk2)" in res.triton_source       # RowMajor：编译器线性化
    assert "b + (rk3 * sb0 + rn2 * sb1)" in res.triton_source  # Strided：声明步长
    assert "c + (rm2 * N + rn2)" in res.triton_source
    assert "addptr %b [%rk3, %rn2]" in res.tir_dump


# ---------------------------------------------------------------------------
# interpreter 差分：非连续张量（验收主场）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,K,N", [(1, 1, 1), (33, 50, 47), (64, 64, 128), (65, 60, 130)])
def test_matmul_noncontig_transposed_b(M, K, N):
    """b = base.T（strides (1, N_base)）：声明式 strides 下结果必须正确。"""
    res = compile_kernel(MATMUL)
    rng = np.random.default_rng(M * 17 + K + N)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((N, K)) * 0.5).astype(np.float16).T  # (K,N) 非连续
    if K > 1 and N > 1:
        assert not b.flags["C_CONTIGUOUS"]
    c = np.zeros((M, N), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(c, want, rtol=1e-3, atol=1e-3)


def test_matmul_padded_stride_b():
    """padded 存储：b 是大数组的视图（stride(0) = N_pad ≠ N）。"""
    res = compile_kernel(MATMUL)
    rng = np.random.default_rng(5)
    M, K, N, pad = 20, 40, 33, 48
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    base = (rng.standard_normal((K, pad)) * 0.5).astype(np.float16)
    b = base[:, :N]                       # (K,N) strides (pad,1)
    assert b.strides[0] // b.itemsize == pad
    c = np.zeros((M, N), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(c, want, rtol=1e-3, atol=1e-3)


def test_rank1_hop2_gather():
    src = """import tila


@tila.jit
def gather(
    b: tila.Tensor[tila.float32, N, (2,)],
    c: tila.Tensor[tila.float32, N, (2,)],
    BLOCK: tila.constexpr = 64,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(b, (offs,), mask=mask)
    tila.store(c, (offs,), x, mask=mask)
"""
    res = compile_kernel(src)
    rng = np.random.default_rng(9)
    base = rng.standard_normal(200).astype(np.float32)
    view = base[::2]                       # stride 2 的 rank-1 视图
    out = base[1::2]                       # 另一个 stride-2 视图作写回目标
    run_kernel(res.kernel, {"b": view, "c": out}, constexpr={})
    np.testing.assert_allclose(out, view)


def test_rowmajor_noncontig_rejected_by_interpreter():
    res = compile_kernel(BATCHED)
    rng = np.random.default_rng(4)
    a = rng.standard_normal((40, 60)).astype(np.float32)
    b = rng.standard_normal((40, 60)).astype(np.float32)
    with pytest.raises(AssertionError, match="row-major"):
        run_kernel(res.kernel, {"a": a, "b": b, "c": (a + b).T})


def test_strided_static_mismatch_rejected_by_interpreter():
    src = """import tila


@tila.jit
def k(
    b: tila.Tensor[tila.float32, N, (4,)],
    BLOCK: tila.constexpr = 64,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(b, (offs,), mask=mask)
"""
    res = compile_kernel(src)
    rng = np.random.default_rng(6)
    base = rng.standard_normal(400).astype(np.float32)
    view = base[::2]                       # 实际 stride 2 ≠ 声明 4
    with pytest.raises(AssertionError, match="declared 4"):
        run_kernel(res.kernel, {"b": view})


def test_batched_add_coord_form_matches_numpy():
    res = compile_kernel(BATCHED)
    rng = np.random.default_rng(11)
    M, N = 70, 140
    a = rng.standard_normal((M, N)).astype(np.float32)
    b = rng.standard_normal((M, N)).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    np.testing.assert_allclose(c, a + b, rtol=1e-6)


# ---------------------------------------------------------------------------
# storage offset（评审 §17 采纳）：stride 正确 ≠ storage offset 正确——
# base(b) = 逻辑张量原点（ABI 规则），非零偏移视图必须照样正确
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transposed", [False, True])
def test_matmul_storage_offset_views(transposed):
    """b 取自大数组的带偏移视图：连续偏移 / 非连续（转置）偏移各一。"""
    res = compile_kernel(MATMUL)
    rng = np.random.default_rng(21)
    M, K, N, pad, off = 24, 40, 33, 48, 64
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    big = (rng.standard_normal((off + N * K,)) * 0.5).astype(np.float16)
    seg = big[off:]                                   # storage offset = 64 的视图
    b = seg.reshape(N, K).T if transposed else seg.reshape(K, N)
    assert b.strides[0] // b.itemsize != 0 and (not transposed or
                                                b.strides[0] // b.itemsize == 1)
    c = np.zeros((M, N), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(c, want, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Address Function 系数（strides_of：lowering 与文档共享的事实源，评审 §22）
# ---------------------------------------------------------------------------

def test_strides_of_address_function_coefficients():
    from tila.types import Const, Symbol
    from tila.types.memory import ROW_MAJOR, Strided, strides_of
    from tila.types.shape import const_shape

    shape = (Symbol("K"), Symbol("N"))
    assert strides_of(ROW_MAJOR, shape) == (Symbol("N"), Const(1))
    assert strides_of(ROW_MAJOR, const_shape(8, 16)) == (Const(16), Const(1))
    assert strides_of(ROW_MAJOR, const_shape(64,)) == (Const(1),)
    assert strides_of(Strided((Symbol("s"), Const(2))), shape) == \
        (Symbol("s"), Const(2))
    with pytest.raises(NotImplementedError):
        strides_of(ROW_MAJOR, const_shape(2, 3, 4))
