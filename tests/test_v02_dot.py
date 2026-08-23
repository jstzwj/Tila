"""matmul fragment 测试（docs/v0.2-matmul-fragment.md）：R16、Mma term、
E18 矩阵、R9' 内存边界放宽、家族混算 E05、interpreter/GPU 差分。"""

import numpy as np
import pytest

from tila.driver import compile_kernel
from tila.diagnostics import TilaError
from tila.interp import run_kernel
from tila.types import Mma, equiv

ARGS = ("a: tila.Tensor[tila.float16, M, K], b: tila.Tensor[tila.float16, K, N], "
        "c: tila.Tensor[tila.float32, M, N]")


def _c(body, params=ARGS, overrides=None):
    return compile_kernel(f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n",
                          overrides)


# 2D 种子 + dot 的公共前奏（静态块：BM=64 BN=128 BK=64）
PRELUDE = """    pid_m = tila.program_id(0)
    pid_n = tila.program_id(1)
    rm = pid_m * 64 + tila.arange(0, 64)
    rn = pid_n * 128 + tila.arange(0, 128)
    rk = tila.arange(0, 64)
    rm2 = tila.expand_dim(rm, 1)
    rk2 = tila.expand_dim(rk, 0)
    rk3 = tila.expand_dim(rk, 1)
    rn2 = tila.expand_dim(rn, 0)
"""


# ---------------------------------------------------------------------------
# layout 代数：Mma 是新种子类
# ---------------------------------------------------------------------------

def test_mma_term_is_its_own_family():
    assert equiv(Mma(64, 128, 64), Mma(64, 128, 64))
    from tila.types import Identity, ProductL, Const

    l0 = Identity((Const(64),))
    assert not equiv(Mma(64, 128, 64), l0)
    assert not equiv(Mma(64, 128, 64), ProductL(l0, l0))
    assert not equiv(Mma(64, 128, 64), Mma(64, 128, 32))


def test_mma_normal_form_untouched_by_erasure_laws():
    from tila.types import CastL, JoinL, normalize

    m = Mma(64, 128, 64)
    assert normalize(JoinL(CastL(m), m)) == m


# ---------------------------------------------------------------------------
# R16 前提（E18 矩阵）
# ---------------------------------------------------------------------------

def _dot_body(x_expr="x", y_expr="y"):
    return (PRELUDE + f"""
    a_idx = rm2 * K + rk2
    b_idx = rk3 * K + rn2
    x = tila.load(a + a_idx, mask=(rm2 < M) & (rk2 < K), other=0.0)
    y = tila.load(b + b_idx, mask=(rk3 < K) & (rn2 < N), other=0.0)
    acc = tila.dot({x_expr}, {y_expr})
    c_idx = rm2 * N + rn2
    tila.store(c + c_idx, acc, mask=(rm2 < M) & (rn2 < N))
""")


def test_dot_happy_path_compiles():
    res = _c(_dot_body())
    assert "%acc = dot %x %y : Tile<f32, (64,128)," in res.tir_dump
    assert "mma(64,128,64)" in res.tir_dump
    assert "acc = tl.dot(x, y)" in res.triton_source


def test_E18_f32_operands_not_mma_able():
    args = ARGS.replace("tila.float16", "tila.float32").replace(
        "c: tila.Tensor[tila.float32", "c: tila.Tensor[tila.float32")
    body = _dot_body().replace("other=0.0", "other=0.0")
    with pytest.raises(TilaError) as ei:
        _c(body, params=args)
    assert ei.value.code == "E18"
    assert "MMA-able" in ei.value.message or "MMA" in ei.value.message


def test_E18_mixed_operand_dtypes():
    body = (_dot_body()
            .replace("y = tila.load(b + b_idx", "yb = tila.load(b + b_idx")
            .replace("acc = tila.dot(x, y)", "y = tila.cast(yb, tila.float32)\n"
                     "    acc = tila.dot(x, y)"))
    with pytest.raises(TilaError) as ei:
        _c(body)
    assert ei.value.code == "E18"
    assert "same dtype" in ei.value.message or "dtype" in ei.value.message


def test_E18_extent_below_mma_minimum():
    body = (_dot_body()
            .replace("rk = tila.arange(0, 64)", "rk = tila.arange(0, 8)")
            .replace("rk2 = tila.expand_dim(rk, 0)", "rk2 = tila.expand_dim(rk, 0)")
            .replace("acc = tila.dot(x, y)", "acc = tila.dot(x, y)"))
    # K=8 < 16：x 的 shape (64,8) 与 y 的 (8,128) 收缩维一致，但 K < 16 → E18
    with pytest.raises(TilaError) as ei:
        _c(body)
    assert ei.value.code == "E18"
    assert "MMA minimum" in ei.value.message


def test_E18_contraction_mismatch():
    # x: (64,32)（rk 取 32）；y: (64,128)（独立 64 种子）→ 收缩维 32 ≠ 64
    body = PRELUDE.replace("rk = tila.arange(0, 64)",
                           "rk = tila.arange(0, 32)") + """
    rk_b = tila.arange(0, 64)
    rk_b3 = tila.expand_dim(rk_b, 1)
    a_idx = rm2 * K + rk2
    b_idx = rk_b3 * N + rn2
    x = tila.load(a + a_idx, mask=(rm2 < M) & (rk2 < K), other=0.0)
    y = tila.load(b + b_idx, mask=(rk_b3 < K) & (rn2 < N), other=0.0)
    acc = tila.dot(x, y)
    c_idx = rm2 * N + rn2
    tila.store(c + c_idx, acc, mask=(rm2 < M) & (rn2 < N))
"""
    with pytest.raises(TilaError) as ei:
        _c(body)
    assert ei.value.code == "E18"
    assert "contraction" in ei.value.message


def test_E18_rank1_operands():
    body = """    x = tila.arange(0, 64)
    y = tila.expand_dim(x, 0)
    acc = tila.dot(x, y)
"""
    with pytest.raises(TilaError) as ei:
        _c(body)
    assert ei.value.code == "E18"


def test_E13_dot_wrong_arity():
    with pytest.raises(TilaError) as ei:
        _c(PRELUDE + "    acc = tila.dot(x)\n")
    assert ei.value.code == "E13"


# ---------------------------------------------------------------------------
# 家族纪律：寄存器内混族 → E05；内存边界自由（R9'）
# ---------------------------------------------------------------------------

def test_E05_mma_blocked_mixing_in_register():
    body = _dot_body().replace(
        "    c_idx = rm2 * N + rn2",
        """    c_idx = rm2 * N + rn2
    bias = tila.load(c + c_idx, mask=(rm2 < M) & (rn2 < N), other=0.0)
    acc2 = acc + bias""").replace(
        "tila.store(c + c_idx, acc, mask=", "tila.store(c + c_idx, acc2, mask=")
    assert "acc2 = acc + bias" in body  # 防字符串替换失配
    with pytest.raises(TilaError) as ei:
        _c(body)
    assert ei.value.code == "E05"
    assert "mma" in ei.value.render() and "product" in ei.value.render()


def test_mma_same_family_arith_ok():
    """同为 Mma 的值运算合法（JoinL 擦除回到 Mma）。"""
    body = _dot_body().replace(
        "    tila.store(c + c_idx, acc, mask=",
        "    acc2 = acc + acc\n    tila.store(c + c_idx, acc2, mask=")
    res = _c(body)
    assert "%acc2 = add %acc %acc : Tile<f32, (64,128)," in res.tir_dump
    assert "mma(64,128,64)" in res.tir_dump


def test_R9_prime_mma_stores_into_blocked_address():
    """R9'：MMA 结果直接 store 进 blocked 索引的地址（坐标语义）。"""
    res = compile_kernel(MATMUL_SRC)
    assert "%acc = dot %x %y : Tile<f32, (64,128)," in res.tir_dump
    assert res.tir_dump.rstrip().endswith("store %p2 %acc mask=%c_m\nreturn")


# ---------------------------------------------------------------------------
# interpreter 差分
# ---------------------------------------------------------------------------

MATMUL_SRC = open("examples/matmul.tila", encoding="utf-8").read()


@pytest.mark.parametrize("M,N", [(1, 1), (64, 128), (65, 130)])
def test_matmul_interp_matches_numpy(M, N):
    res = compile_kernel(MATMUL_SRC)
    rng = np.random.default_rng(M * 31 + N)
    K = 50  # K ≤ BK=64 且非 2 的幂：验证 masked other=0.0 包络
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    c = np.zeros((M, N), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(c, want, rtol=1e-3, atol=1e-3)


def test_matmul_constexpr_specialization():
    res = compile_kernel(MATMUL_SRC, {"BM": 32, "BN": 32, "BK": 32})
    assert "mma(32,32,32)" in res.tir_dump
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((40, 30)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((30, 40)) * 0.5).astype(np.float16)
    c = np.zeros((40, 40), dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c},
               constexpr={"BM": 32, "BN": 32, "BK": 32})
    want = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(c, want, rtol=1e-3, atol=1e-3)
