"""Correctness closure: no invalid kernel execution; actual CPU/CUDA ABI parity."""
import numpy as np
import pytest
import torch
import tila as ti

from tila.errors import TilaError, TilaLaunchContractError
from tila.runtime import _Launcher
from test_soundness_closure import signed_mask, positive_f32, pointer_roundtrip


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
