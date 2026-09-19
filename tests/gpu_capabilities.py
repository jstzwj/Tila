"""M3-04 fixed-target operation/dtype evidence. No GPU means failure."""
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
import torch
import tila as ti

from m3_capability_support import elementwise
from tila.errors import TilaError
from tila.runtime import _Launcher

FLOATS = ("f16", "bf16", "f32", "f64")
NP = dict(f16=np.float16, bf16=ml_dtypes.bfloat16, f32=np.float32, f64=np.float64)
TORCH = dict(f16=torch.float16, bf16=torch.bfloat16, f32=torch.float32, f64=torch.float64)


def upload(x, dtype):
    return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16).cuda() if dtype == "bf16" else torch.from_numpy(x).cuda()


def download(x):
    return x.cpu().view(torch.int16).numpy().view(ml_dtypes.bfloat16) if x.dtype == torch.bfloat16 else x.cpu().numpy()


def directory(tmp_path):
    return Path(os.environ.get("TILA_GPU_KERNELS", tmp_path))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("operation", ["exp", "exp2"])
def test_exponential(dtype, operation, tmp_path):
    kernel = elementwise(directory(tmp_path), dtype, operation)
    # Finite values, signed zero, infinities, NaN, and a tail block.
    x = np.array([-4, -1, -0., 0., .125, .5, 1, 4, -np.inf, np.inf, np.nan], dtype=NP[dtype])
    cpu = np.zeros_like(x)
    gpu = torch.zeros(x.shape, dtype=TORCH[dtype], device="cuda")
    kernel[(2,)](x, cpu)
    kernel[(2,)](upload(x, dtype), gpu)
    reference = getattr(np, operation)(x.astype(np.float64))
    tolerance = dict(f16=2**-9, bf16=2**-6, f32=2e-6, f64=2e-14)[dtype]
    for actual in (cpu.astype(np.float64), download(gpu).astype(np.float64)):
        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=0, equal_nan=True)


@pytest.mark.parametrize("source", FLOATS)
@pytest.mark.parametrize("destination", FLOATS)
def test_float_cast_matrix(source, destination, tmp_path):
    kernel = elementwise(directory(tmp_path), source, "cast", destination)
    x = np.array([-np.inf, -3.5, -0., 0., .125, 1, 1.00390625, 1.00048828125, 65504, np.inf, np.nan], dtype=NP[source])
    cpu = np.zeros(x.shape, dtype=NP[destination])
    gpu = torch.zeros(x.shape, dtype=TORCH[destination], device="cuda")
    kernel[(2,)](x, cpu)
    kernel[(2,)](upload(x, source), gpu)
    reference = x.astype(NP[destination])
    for actual in (cpu, download(gpu)):
        np.testing.assert_equal(actual.astype(np.float64), reference.astype(np.float64))
        np.testing.assert_array_equal(np.signbit(actual[2:4]), [True, False])


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("operation", ["exp", "exp2"])
def test_narrow_exp_intermediate_rounding(dtype, operation, tmp_path):
    kernel = elementwise(directory(tmp_path), dtype, operation + "_chain", "f32")
    x = np.array([.125, .25, .5, .75, 1, 1.25, 1.5, 2], dtype=NP[dtype])
    cpu = np.zeros(8, np.float32)
    gpu = torch.zeros(8, dtype=torch.float32, device="cuda")
    kernel[(1,)](x, cpu)
    kernel[(1,)](upload(x, dtype), gpu)
    reference = getattr(np, operation)(x.astype(np.float32)).astype(NP[dtype]).astype(np.float32)
    np.testing.assert_array_equal(cpu, reference)
    np.testing.assert_array_equal(gpu.cpu().numpy(), reference)


@pytest.mark.parametrize("dtype,torch_dtype", [("f8e4m3fn", torch.float8_e4m3fn), ("f8e5m2", torch.float8_e5m2)])
@pytest.mark.parametrize("operation", ["copy", "cast"])
def test_fp8_storage_decode_rejected_before_execution(dtype, torch_dtype, operation, tmp_path, monkeypatch):
    output = dtype if operation == "copy" else "f32"
    kernel = elementwise(directory(tmp_path), dtype, operation, output)
    x = torch.zeros(8, dtype=torch.uint8, device="cuda").view(torch_dtype)
    out = torch.zeros(8, dtype=torch.uint8, device="cuda").view(torch_dtype) if operation == "copy" else torch.zeros(8, device="cuda")
    monkeypatch.setattr(_Launcher, "_execute", lambda *a: pytest.fail("unsupported FP8 executed"))
    with pytest.raises(TilaError, match="TILA-TARGET-009"):
        kernel[(1,)](x, out)


@pytest.mark.parametrize("operation", ["fp8_e4", "fp8_e5"])
def test_fp8_intermediate_rejected_before_execution(operation, tmp_path, monkeypatch):
    kernel = elementwise(directory(tmp_path), "f32", operation)
    x = torch.zeros(8, device="cuda")
    out = torch.zeros_like(x)
    monkeypatch.setattr(_Launcher, "_execute", lambda *a: pytest.fail("unsupported cast executed"))
    with pytest.raises(TilaError, match="TILA-TARGET-009"):
        kernel[(1,)](x, out)


@ti.jit
def bf16_broadcast(x: ti.Buffer[ti.bf16, (4,), ti.ReadOnly],
                   y: ti.Buffer[ti.bf16, (8,), ti.ReadOnly],
                   out: ti.Buffer[ti.bf16, (4, 8), ti.WriteOnly]):
    r = ti.arange(0, 4)
    c = ti.arange(0, 8)
    a = ti.load(x, r)[:, None]
    b = ti.load(y, c)[None, :]
    v = ti.where(a > b, a, b)
    ti.store(out, (r[:, None], c[None, :]), v)


def test_bf16_outer_broadcast():
    a = np.array([-2, -1, 0, 1], dtype=ml_dtypes.bfloat16)
    b = np.arange(8).astype(ml_dtypes.bfloat16)
    cpu = np.zeros((4, 8), dtype=ml_dtypes.bfloat16)
    gpu = torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda")
    bf16_broadcast[(1,)](a, b, cpu)
    bf16_broadcast[(1,)](upload(a, "bf16"), upload(b, "bf16"), gpu)
    expected = np.maximum(a.astype(np.float32)[:, None], b.astype(np.float32)[None, :])
    np.testing.assert_array_equal(cpu.astype(np.float32), expected)
    np.testing.assert_array_equal(download(gpu).astype(np.float32), expected)


N = ti.Dim("N")


@ti.jit
def structural(x: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
               out: ti.Buffer[ti.i32, (N,), ti.WriteOnly]):
    ti.static_assert(8 > 0)
    ti.assume(N >= 16)
    i = ti.program_id(0) * 8 + ti.arange(0, 8)
    a = ti.unsafe_load(x, i)
    b = ti.reshape(a, (2, 4))
    c = ti.reshape(b, (8,))
    for k in ti.range(0, 2):
        c = c + ti.num_programs(0)
    ti.unsafe_store(out, i, c)


def test_structural_and_explicit_unsafe_intrinsics(monkeypatch):
    monkeypatch.setenv("TILA_DEBUG", "1")
    x = np.arange(16, dtype=np.int32)
    cpu = np.zeros_like(x)
    gpu = torch.zeros(16, dtype=torch.int32, device="cuda")
    structural[(2,)](x, cpu)
    structural[(2,)](torch.from_numpy(x).cuda(), gpu)
    np.testing.assert_array_equal(cpu, x + 4)
    np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)
