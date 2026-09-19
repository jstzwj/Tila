"""Explicit GPU suite; run via tools/gpu_audit.py, never CPU auto-discovery."""
import math
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
import torch
import tila as ti

from reduction_support import reduction_kernel
from test_dataflow_interp import copy_view

SEED = 208009
DTYPES = ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64"]
TORCH_DTYPES = dict(zip(DTYPES, [torch.int8, torch.int16, torch.int32, torch.int64,
                               torch.uint8, torch.uint16, torch.uint32, torch.uint64,
                               torch.float16, torch.bfloat16, torch.float32, torch.float64]))


@pytest.mark.parametrize("suite", ["integer", "boolean", "const_bool", "constant"])
def test_existing_suite(suite):
    import importlib
    importlib.import_module(f"gpu_{suite}_smoke").main()


def numpy_value(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.cpu().view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return tensor.cpu().numpy()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("axis", [0, 1])
def test_reduction(dtype, op, axis, tmp_path):
    directory = Path(os.environ.get("TILA_GPU_KERNELS", tmp_path))
    kernel = reduction_kernel(directory, dtype, op, axis)
    ndtype = ml_dtypes.bfloat16 if dtype == "bf16" else getattr(np, {
        "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
        "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
        "f16": "float16", "f32": "float32", "f64": "float64"}[dtype])
    rng = np.random.default_rng(SEED)
    # Integer cases overflow at multiple intermediate tree nodes. Floating
    # cases include cancellation and f64 values that lose bits in f32.
    if dtype[0] in "iu":
        info = np.iinfo(ndtype)
        x = np.array([info.max, info.min, 1, 0, info.max, 1, 1, 1] * 4, dtype=ndtype).reshape(4, 8)
    else:
        x = rng.normal(size=(4, 8)).astype(ndtype)
        x[0] = np.array([256, 1, -256, 2, 0.5, 0.5, -1, 0], dtype=ndtype)
        if dtype == "f64":
            x[0, :4] = [2**40, 1, -2**40, 2]
    gpu_x = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16).cuda() if dtype == "bf16" else torch.from_numpy(x).cuda()
    cpu = np.zeros(x.shape[1 - axis], dtype=ndtype)
    gpu = torch.zeros(cpu.shape, dtype=TORCH_DTYPES[dtype], device="cuda")
    kernel[(1,)](x, cpu)
    kernel[(1,)](gpu_x, gpu)
    actual = numpy_value(gpu)
    if dtype[0] in "iu":
        expected = np.max(x, axis=axis) if op == "max" else np.sum(x.astype(object), axis=axis)
        if op == "sum":
            bits = np.iinfo(ndtype).bits
            expected = np.array([int(v) % (1 << bits) for v in expected], dtype=object)
            if dtype[0] == "i":
                expected = np.array([v - (1 << bits) if v >= 1 << (bits-1) else v for v in expected], dtype=object)
        np.testing.assert_array_equal(cpu, expected)
        np.testing.assert_array_equal(actual, expected)
    elif op == "max":
        np.testing.assert_array_equal(cpu, np.max(x, axis=axis))
        np.testing.assert_array_equal(actual, cpu)
    else:
        values = np.moveaxis(x.astype(np.float64), axis, -1)
        reference = np.array([math.fsum(row) for row in values])
        n = x.shape[axis]
        eps_acc = np.finfo(np.float64 if dtype == "f64" else np.float32).eps
        eps_out = 2**-7 if dtype == "bf16" else np.finfo(ndtype).eps
        bound = (n - 1) * eps_acc / (1 - (n - 1) * eps_acc) * np.sum(abs(values), axis=-1)
        bound += eps_out * (abs(reference) + bound)  # final output rounding
        for result in (cpu, actual):
            assert np.all(np.isfinite(result.astype(np.float64)))
            assert np.all(abs(result.astype(np.float64) - reference) <= bound), (dtype, axis, result, reference, bound, x)


@pytest.mark.parametrize("layout", ["sliced", "broadcast"])
def test_strided_views(layout):
    base = np.arange(80, dtype=np.float32).reshape(8, 10)
    gpu_base = torch.from_numpy(base).cuda()
    source = base[1:7:2, 1:9:2].T if layout == "sliced" else np.broadcast_to(base[0, :3], (4, 3))
    gpu_source = gpu_base[1:7:2, 1:9:2].T if layout == "sliced" else gpu_base[0, :3].expand(4, 3)
    backing = np.full((8, 10), -77, np.float32)
    gpu_backing = torch.from_numpy(backing).cuda()
    out = backing[1:7:2, 1:9:2].T
    gpu_out = gpu_backing[1:7:2, 1:9:2].T
    copy_view[(1,)](source, out)
    copy_view[(1,)](gpu_source, gpu_out)
    expected = np.full((8, 10), -77, np.float32)
    expected[1:7:2, 1:9:2].T[:] = source
    np.testing.assert_array_equal(backing, expected)
    np.testing.assert_array_equal(gpu_backing.cpu().numpy(), expected)


N = ti.Dim("N")


@ti.jit
def tail_cast(x: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
              out: ti.Buffer[ti.i16, (N,), ti.WriteOnly]):
    i = ti.program_id(0) * 8 + ti.arange(0, 8)
    value = ti.load(x, i, mask=i < N, other=0)
    ti.store(out, i, ti.cast[ti.i16](value), mask=i < N)


@pytest.mark.parametrize("n", [1, 7, 8, 9, 17])
def test_tail_cast(n):
    values = np.resize(np.array([-32769, -32768, -1, 0, 1, 32767, 32768], np.int32), n)
    cpu = np.zeros(n, np.int16)
    gpu = torch.zeros(n, dtype=torch.int16, device="cuda")
    tail_cast[(ti.cdiv(n, 8),)](values, cpu)
    tail_cast[(ti.cdiv(n, 8),)](torch.from_numpy(values).cuda(), gpu)
    np.testing.assert_array_equal(cpu, values.astype(np.int16))
    np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_reduction_specials(dtype, op, tmp_path):
    kernel = reduction_kernel(Path(os.environ.get("TILA_GPU_KERNELS", tmp_path)), dtype, op)
    ndtype = ml_dtypes.bfloat16 if dtype == "bf16" else {"f16": np.float16, "f32": np.float32, "f64": np.float64}[dtype]
    x = np.array([[float("nan"), 1, 2, 3, 4, 5, 6, 7],
                  [float("nan")] * 8,
                  [float("inf"), -float("inf"), 1, 2, 3, 4, 5, 6],
                  [-0.0, 0.0] * 4], dtype=ndtype)
    gpu_x = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16).cuda() if dtype == "bf16" else torch.from_numpy(x).cuda()
    cpu = np.zeros(4, ndtype)
    gpu = torch.zeros(4, dtype=TORCH_DTYPES[dtype], device="cuda")
    with np.errstate(invalid="ignore"):
        kernel[(1,)](x, cpu)
    kernel[(1,)](gpu_x, gpu)
    expected = [float("nan"), float("nan"), float("nan") if op == "sum" else float("inf"), 0]
    np.testing.assert_equal(cpu.astype(np.float64), expected)
    np.testing.assert_equal(numpy_value(gpu).astype(np.float64), expected)


@ti.jit
def masked_reduce(x: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
                  out: ti.Buffer[ti.i32, (2,), ti.WriteOnly]):
    i = ti.arange(0, 8)
    a = ti.load(x, i, mask=i < N, other=0)
    b = ti.load(x, i, mask=i < N, other=-2147483648)
    ti.store(out, 0, ti.sum(a, 0))
    ti.store(out, 1, ti.max(b, 0))


@pytest.mark.parametrize("n", [1, 7, 8])
def test_masked_reduction(n):
    x = -np.arange(1, n + 1, dtype=np.int32)
    cpu = np.zeros(2, np.int32)
    gpu = torch.zeros(2, dtype=torch.int32, device="cuda")
    masked_reduce[(1,)](x, cpu)
    masked_reduce[(1,)](torch.from_numpy(x).cuda(), gpu)
    np.testing.assert_array_equal(cpu, [sum(map(int, x)), max(x)])
    np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)
