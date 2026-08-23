"""差分测试（development-plan §8 / Stage 6）：TIR interpreter vs NumPy 参考。

无 GPU 环境下的可执行 oracle：同一 TIR 走 interpreter 与手写参考计算对拍。
"""

import numpy as np
import pytest

from tila.driver import compile_kernel
from tila.interp import run_kernel

EXAMPLES = {
    name: compile_kernel(
        open(f"examples/{name}.tila", encoding="utf-8").read())
    for name in ("add", "saxpy", "masked_add")
}

try:
    import ml_dtypes  # noqa: F401

    HAS_ML_DTYPES = True
except ImportError:  # pragma: no cover
    HAS_ML_DTYPES = False

if HAS_ML_DTYPES:
    EXAMPLES["fp8_add"] = compile_kernel(
        open("examples/fp8_add.tila", encoding="utf-8").read())

NS = [1, 2, 127, 128, 129, 1000]
BLOCKS = [32, 128]


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("BLOCK", BLOCKS)
def test_add_matches_numpy(N, BLOCK):
    res = compile_kernel(open("examples/add.tila", encoding="utf-8").read(),
                         {"BLOCK": BLOCK})
    rng = np.random.default_rng(N * 31 + BLOCK)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c}, constexpr={"BLOCK": BLOCK})
    np.testing.assert_allclose(c, a + b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("N", [1, 127, 129])
def test_saxpy_matches_numpy(N):
    res = EXAMPLES["saxpy"]
    rng = np.random.default_rng(N)
    x = rng.standard_normal(N).astype(np.float32)
    y = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)
    alpha = 7
    run_kernel(res.kernel, {"x": x, "y": y, "out": out}, scalars={"alpha": alpha})
    np.testing.assert_allclose(out, alpha * x + y, rtol=1e-6)


def test_masked_add_other_semantics():
    """masked-out 通道取 other（0.0），未写回的通道保持零。"""
    res = EXAMPLES["masked_add"]
    N = 130  # BLOCK=128 → 最后 2 个通道 masked-out
    rng = np.random.default_rng(3)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    np.testing.assert_allclose(c, a + b, rtol=1e-6)


@pytest.mark.skipif(not HAS_ML_DTYPES, reason="ml_dtypes 未安装（fp8 解释器支持）")
@pytest.mark.parametrize("N", [64, 130])
def test_fp8_roundtrip_matches_reference(N):
    """fp8 load→f16 计算→fp8 store 与 NumPy 走 ml_dtypes 的同路径参考一致。"""
    import ml_dtypes

    res = EXAMPLES["fp8_add"]
    rng = np.random.default_rng(5)
    a = (rng.standard_normal(N) * 0.5).astype(ml_dtypes.float8_e4m3fn)
    b = (rng.standard_normal(N) * 0.5).astype(ml_dtypes.float8_e4m3fn)
    c = np.zeros(N, dtype=ml_dtypes.float8_e4m3fn)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    expected = ((a.astype(np.float16) + b.astype(np.float16))
                .astype(ml_dtypes.float8_e4m3fn))
    np.testing.assert_array_equal(c, expected)


def test_unmasked_load_requires_in_bounds_contract():
    """无 mask 的 load/store：边界安全由用户负责（与 Triton 一致）。"""
    src = """import tila


@tila.jit
def fill(a: tila.Tensor[tila.float32, 128], c: tila.Tensor[tila.float32, 128]):
    offs = tila.arange(0, 128)
    x = tila.load(a + offs)
    y = x * 2.0
    tila.store(c + offs, y)
"""
    res = compile_kernel(src)
    a = np.arange(128, dtype=np.float32)
    c = np.zeros(128, dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "c": c})
    np.testing.assert_array_equal(c, a * 2.0)


def test_r12_semantics_in_interpreter():
    """f16 tile + FLOAT 字面量按元素 dtype 解释（值语义不受影响，验证通路）。"""
    src = """import tila


@tila.jit
def bias(a: tila.Tensor[tila.float16, 128], c: tila.Tensor[tila.float16, 128]):
    offs = tila.arange(0, 128)
    x = tila.load(a + offs)
    y = x + 1.0
    tila.store(c + offs, y)
"""
    res = compile_kernel(src)
    a = np.arange(128, dtype=np.float16)
    c = np.zeros(128, dtype=np.float16)
    run_kernel(res.kernel, {"a": a, "c": c})
    np.testing.assert_array_equal(c, a + np.float16(1.0))


def test_bool_ops_and_cast_bool():
    src = """import tila


@tila.jit
def clamp(a: tila.Tensor[tila.float32, 128], c: tila.Tensor[tila.float32, 128]):
    offs = tila.arange(0, 128)
    x = tila.load(a + offs)
    m = x > 0.5
    b = tila.cast(x, tila.bool)
    m2 = m | b
    y = x + 0.0
    tila.store(c + offs, y, mask=m2)
"""
    res = compile_kernel(src)
    rng = np.random.default_rng(9)
    a = rng.standard_normal(128).astype(np.float32)
    c = np.zeros(128, dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "c": c})
    expected = np.where((a > 0.5) | (a != 0), a, 0.0)
    np.testing.assert_allclose(c, expected, rtol=1e-6)
