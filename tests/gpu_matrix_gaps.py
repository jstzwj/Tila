"""M3 exit follow-up: accepted dtype evidence, not a new public feature."""
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
import torch

from m3_capability_support import dot_kernel, tile_kernel, typed_add
from gpu_capabilities import upload, download

DTYPES = ("bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64")
NP = dict(zip(DTYPES, [np.bool_, np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64,
                      np.float16, ml_dtypes.bfloat16, np.float32, np.float64]))


def directory(tmp_path):
    return Path(os.environ.get("TILA_GPU_KERNELS", tmp_path))


def values(dtype, size):
    if dtype == "bool":
        base = [False, True]
    elif dtype[0] in "iu":
        info = np.iinfo(NP[dtype])
        base = [info.min, info.max, 0, 1]
    else:
        base = [-np.inf, -3.5, -0., 0., .125, 1., np.inf, np.nan]
    return np.resize(np.array(base, dtype=NP[dtype]), size)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("operation", ["zeros", "reshape"])
def test_tile_dtype_matrix(dtype, rank, operation, tmp_path):
    kernel = tile_kernel(directory(tmp_path), dtype, operation, rank)
    x = values(dtype, 16)
    cpu = np.ones(16, dtype=NP[dtype])
    gpu = upload(cpu, dtype)
    kernel[(1,)](x, cpu)
    kernel[(1,)](upload(x, dtype), gpu)
    expected = np.zeros_like(x) if operation == "zeros" else x
    # Copy/reshape must preserve signed zero, NaN payload and integer high bits.
    for actual in (cpu, download(gpu)):
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


@pytest.mark.parametrize("dtype", DTYPES[1:])
@pytest.mark.parametrize("n", [127, 128, 129])
@pytest.mark.parametrize("warps", [4, 8])
def test_official_add_dtype_matrix(dtype, n, warps, tmp_path):
    kernel = typed_add(directory(tmp_path), dtype)
    x = values(dtype, n)
    y = np.ones(n, dtype=NP[dtype])
    cpu = np.zeros_like(x)
    gpu = upload(cpu, dtype)
    launch = kernel[((n + 127) // 128,)].with_options(num_warps=warps)
    launch(x, y, cpu)
    launch(upload(x, dtype), upload(y, dtype), gpu)
    if dtype[0] in "iu":
        bits = np.iinfo(NP[dtype]).bits
        result = [(int(v) + 1) % (1 << bits) for v in x]
        if dtype[0] == "i":
            result = [v - (1 << bits) if v >= 1 << (bits - 1) else v for v in result]
        expected = np.array(result, dtype=NP[dtype])
    else:
        # Exact f64 additions for selected values, then explicit output rounding.
        expected = (x.astype(np.float64) + 1).astype(NP[dtype])
    for actual in (cpu, download(gpu)):
        if dtype[0] in "iu":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(actual.astype(np.float64), expected.astype(np.float64))


@pytest.mark.parametrize("accumulator", ["f16", "f32"])
@pytest.mark.parametrize("case", ["rounding", "random"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("warps", [4, 8])
def test_dot_accumulator_dtype(accumulator, case, strided, warps, tmp_path):
    kernel = dot_kernel(directory(tmp_path), accumulator)
    rng = np.random.default_rng(306)
    a = rng.normal(size=(16, 32)).astype(np.float16)
    b = rng.normal(size=(32, 16)).astype(np.float16)
    c = rng.normal(size=(16, 16)).astype(NP[accumulator])
    if case == "rounding":
        a[:] = 0
        b[:] = 0
        a[:, 0] = 1
        b[0, :] = np.resize(np.array([2**-11, 3 * 2**-11, -2**-12, 2**-10], np.float16), 16)
        c[:] = 1
    if strided:
        a, b, c = [np.ascontiguousarray(v.T).T for v in (a, b, c)]
    reference = a.astype(np.float64) @ b.astype(np.float64) + c.astype(np.float64)
    bound = 33 * np.finfo(np.float32).eps * (abs(a.astype(np.float64)) @ abs(b.astype(np.float64)) + abs(c.astype(np.float64)))
    if accumulator == "f16":
        bound += 2**-11 * (abs(reference) + bound) + 2**-24
    cpu = np.zeros((16, 16), np.float32)
    gpu = torch.zeros((16, 16), dtype=torch.float32, device="cuda")
    launch = kernel[(1,)].with_options(num_warps=warps)
    launch(a, b, c, cpu)
    launch(*(torch.from_numpy(v).cuda() for v in (a, b, c)), gpu)
    for actual in (cpu, gpu.cpu().numpy()):
        assert np.all(abs(actual - reference) <= bound), (actual, reference, bound)
        if case == "rounding":
            expected = reference.astype(NP[accumulator]).astype(np.float32)
            np.testing.assert_array_equal(actual, expected)
        if accumulator == "f16":
            np.testing.assert_array_equal(actual, actual.astype(np.float16).astype(np.float32))
