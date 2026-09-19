"""M3 official example differential, fixed seed and algorithm-specific bounds."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tila as ti
from tila.errors import TilaError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from add_kernel import add_kernel
from softmax import softmax_kernel
from matmul import matmul_kernel
from self_attention import self_attention_kernel
from fused_attention import fused_attention_kernel


def execute(kernel, inputs, shape, grid, warps, *scalars, **consts):
    cpu = np.zeros(shape, np.float32)
    gpu = torch.zeros(shape, dtype=torch.float32, device="cuda")
    launch = kernel[grid].with_options(num_warps=warps)
    launch(*inputs, cpu, *scalars, **consts)
    launch(*(torch.from_numpy(x).cuda() for x in inputs), gpu, *scalars, **consts)
    return cpu, gpu.cpu().numpy()


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("n", [127, 128, 129])
def test_add(n, warps):
    rng = np.random.default_rng(301)
    x, y = [rng.normal(size=n).astype(np.float32) for _ in range(2)]
    results = execute(add_kernel, [x, y], (n,), (ti.cdiv(n, 128),), warps)
    for out in results:
        np.testing.assert_array_equal(out, x + y)


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("n", [31, 32, 33])
def test_softmax(n, warps):
    x = np.random.default_rng(302).normal(size=(3, n)).astype(np.float32)
    ref = np.exp(x.astype(np.float64) - x.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    for out in execute(softmax_kernel, [x], x.shape, (3,), warps, BN=64):
        np.testing.assert_allclose(out, ref, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("shape", [(16, 16, 16), (19, 23, 35)])
@pytest.mark.parametrize("strided", [False, True])
def test_matmul(shape, strided, warps):
    m, n, k = shape
    rng = np.random.default_rng(303)
    a = rng.normal(size=(m, k)).astype(np.float16)
    b = rng.normal(size=(k, n)).astype(np.float16)
    if strided:
        a = np.ascontiguousarray(a.T).T
        b = np.ascontiguousarray(b.T).T
    ref = a.astype(np.float64) @ b.astype(np.float64)
    # f16 products are exact in f32; bound f32 accumulation against f64.
    bound = (k * np.finfo(np.float32).eps) * (abs(a.astype(np.float64)) @ abs(b.astype(np.float64)))
    for out in execute(matmul_kernel, [a, b], (m, n), (ti.cdiv(m, 16), ti.cdiv(n, 16)), warps,
                       BM=16, BN=16, BK=32):
        assert np.all(abs(out - ref) <= bound), (shape, warps, out, ref, bound)


def attention_reference(q, k, v, causal):
    scores = q.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    if causal:
        scores = np.where(np.tril(np.ones(scores.shape[-2:], bool)), scores, -np.inf)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v.astype(np.float64)


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("n", [32, 65])
def test_self_attention(n, warps):
    rng = np.random.default_rng(304)
    inputs = [rng.normal(size=(n, 32)).astype(np.float16) for _ in range(3)]
    ref = attention_reference(*inputs, True)
    for out in execute(self_attention_kernel, inputs, (n, 32), (ti.cdiv(n, 32),), warps,
                       1 / np.sqrt(32), BM=32, BN=32, BD=32):
        # p is explicitly rounded to f16 before p@v in this example.
        np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("n", [32, 65])
@pytest.mark.parametrize("causal", [0, 1])
def test_fused_attention(n, causal, warps):
    rng = np.random.default_rng(305)
    inputs = [rng.normal(size=(2, n, 32)).astype(np.float16) for _ in range(3)]
    ref = attention_reference(*inputs, bool(causal))
    for out in execute(fused_attention_kernel, inputs, (2, n, 32), (ti.cdiv(n, 32), 2), warps,
                       1 / np.sqrt(32), BM=32, BN=32, BD=32, CAUSAL=causal):
        np.testing.assert_allclose(out, ref, rtol=2e-3, atol=2e-3)


def test_mixed_devices_rejected():
    x = torch.zeros(128, device="cuda")
    cpu = np.zeros(128, np.float32)
    with pytest.raises(TilaError) as exc:
        add_kernel[(1,)](x, cpu, torch.zeros_like(x))
    assert exc.value.code == "TILA-TARGET-008"


def test_hint_emission_preserves_results(monkeypatch):
    from tila.lowering import Lowering
    kernel = ti.jit(add_kernel.fn)
    source = Lowering(kernel.tk).kernel_source()
    assert "tl.multiple_of(" in source and "tl.max_contiguous(" in source
    emitter = Lowering(kernel.tk)
    emitter.kernel_source()
    assert emitter.hint_audit and all(origins for _, origins in emitter.hint_audit)
    x = np.arange(257, dtype=np.float32)[::-1].copy()
    hinted = execute(kernel, [x, x], x.shape, (3,), 4)[1]
    original = Lowering.kernel_source
    def without_hints(self):
        return "\n".join(line for line in original(self).splitlines()
                         if not line.strip().startswith(("tl.multiple_of(", "tl.max_contiguous("))) + "\n"
    monkeypatch.setattr(Lowering, "kernel_source", without_hints)
    unhinted = execute(ti.jit(add_kernel.fn), [x, x], x.shape, (3,), 4)[1]
    np.testing.assert_array_equal(hinted, unhinted)
    np.testing.assert_array_equal(hinted, x + x)
