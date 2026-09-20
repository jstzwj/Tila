"""Correctness closure: no invalid kernel execution; actual CPU/CUDA ABI parity."""
import numpy as np
import pytest
import torch
import tila as ti

from tila.errors import TilaError, TilaLaunchContractError
from tila.runtime import _Launcher
from test_soundness_closure import signed_mask, positive_f32, pointer_roundtrip
from test_c1_review import (mask_cast_values, mask_cast_tail, ptr_view_copy,
                            static_return_loop, runtime_return, loop_return)


@pytest.mark.parametrize('case', ['signed-mask', 'rounded-refinement'])
def test_invalid_launch_stops_before_backend(case, monkeypatch):
    out = torch.full((4,) if case == 'signed-mask' else (1,), -17,
                     dtype=torch.int32 if case == 'signed-mask' else torch.float32, device='cuda')

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid launch reached execution backend')

    monkeypatch.setattr(_Launcher, '_execute', forbidden)
    with pytest.raises((TilaError, TilaLaunchContractError)):
        if case == 'signed-mask':
            signed_mask[(1,)](out)
        else:
            positive_f32[(1,)](out, 1e-50)
    assert torch.all(out == -17)


def test_pointer_i32_steps_i64_roundtrip():
    x = np.arange(4, dtype=np.int32) + 31
    cpu = np.zeros_like(x)
    gpu = torch.zeros(4, dtype=torch.int32, device='cuda')
    pointer_roundtrip[(1,)](x, cpu)
    pointer_roundtrip[(1,)](torch.from_numpy(x).cuda(), gpu)
    np.testing.assert_array_equal(cpu, x)
    np.testing.assert_array_equal(gpu.cpu().numpy(), x)


@ti.jit
def scalar_f16(out: ti.Buffer[ti.f16, (1,), ti.WriteOnly], value: ti.f16):
    ti.store(out, 0, value)


@ti.jit
def scalar_bf16(out: ti.Buffer[ti.bf16, (1,), ti.WriteOnly], value: ti.bf16):
    ti.store(out, 0, value)


@ti.jit
def scalar_f32(out: ti.Buffer[ti.f32, (1,), ti.WriteOnly], value: ti.f32):
    ti.store(out, 0, value)


@ti.jit
def scalar_f64(out: ti.Buffer[ti.f64, (1,), ti.WriteOnly], value: ti.f64):
    ti.store(out, 0, value)


@pytest.mark.parametrize('kernel,dt,torch_dt,value', [
    (scalar_f16, ti.f16, torch.float16, 1.0007),
    (scalar_bf16, ti.bf16, torch.bfloat16, 1.005),
    (scalar_f32, ti.f32, torch.float32, 2**80 + 2**56 + 1),
    (scalar_f64, ti.f64, torch.float64, 1.0000000000000002)])
def test_rounded_scalar_abi_cpu_gpu(kernel, dt, torch_dt, value):

    import ml_dtypes
    from tila.numeric import scalar_float
    cpu = np.zeros(1, dtype=dt.np_dtype or ml_dtypes.bfloat16)
    gpu = torch.zeros(1, dtype=torch_dt, device='cuda')
    kernel[(1,)](cpu, value)
    kernel[(1,)](gpu, value)
    assert float(cpu[0]) == gpu.item() == scalar_float(value, dt)


@pytest.mark.parametrize('kind', ['numeric', 'boolean'])
def test_c1_mask_cast_shapes_and_values(kind):
    if kind == 'numeric':
        out = torch.full((4, 4), -1, dtype=torch.int32, device='cuda')
        mask_cast_values[(1,)](out)
        expected = np.arange(4)[:, None] < np.arange(4)[None, :]
    else:
        out = torch.zeros(4, dtype=torch.int32, device='cuda')
        source = torch.tensor([5, 6, 7], dtype=torch.int32, device='cuda')
        mask_cast_tail[(1,)](source, out)
        expected = [5, 6, 7, -1]
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


def test_c1_pointer_view_negative_step():
    source = torch.arange(12, dtype=torch.int32, device='cuda')
    backing = torch.full((12,), -1, dtype=torch.int32, device='cuda')
    ptr_view_copy[(1,)](source[2:6], backing[3:7])
    expected = np.full(12, -1, np.int32)
    expected[3:7] = np.arange(2, 6)
    np.testing.assert_array_equal(backing.cpu().numpy(), expected)


@pytest.mark.parametrize('ret,inner,count', [(r, i, c) for r in (False, True)
                                           for i in (False, True) for c in (0, 3)])
def test_c1_return_loop_cpu_gpu(ret, inner, count):
    cpu = np.full(4, -9, np.int32)
    gpu = torch.full((4,), -9, dtype=torch.int32, device='cuda')
    static_return_loop[(1,)](cpu, count, RETURN=ret, INNER=inner)
    static_return_loop[(1,)](gpu, count, RETURN=ret, INNER=inner)
    expected = -9 if ret else count*(1 if inner else 2)
    np.testing.assert_array_equal(cpu, expected)
    np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)


@pytest.mark.parametrize('flag', [False, True])
def test_c1_runtime_return_gate(flag, monkeypatch):
    out = torch.full((4,), -9, dtype=torch.int32, device='cuda')
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported return reached execution backend')
    monkeypatch.setattr(_Launcher, '_execute', forbidden)
    with pytest.raises(TilaError, match='return under runtime if or loop'):
        runtime_return[(1,)](out, flag)
    np.testing.assert_array_equal(out.cpu().numpy(), -9)


@pytest.mark.parametrize('count', [0, 3])
def test_c1_loop_return_gate(count, monkeypatch):
    out = torch.zeros(1, dtype=torch.int32, device='cuda')
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported return reached execution backend')
    monkeypatch.setattr(_Launcher, '_execute', forbidden)
    with pytest.raises(TilaError, match='return under runtime if or loop'):
        loop_return[(1,)](out, count)
    assert out.item() == 0
