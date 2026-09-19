"""Explicit GPU gate: PYTHONPATH=src python tests/gpu_integer_smoke.py.

Not auto-collected by CPU pytest; requires a CUDA runner and fails if absent.
This is a bounded ADR-007 differential, not the M3 support matrix.
"""
import numpy as np
import torch
import triton

from test_integer_semantics import (cast_narrow, divmod_kernel, float_cast,
                                    loop_boundary, scalar_wrap, shifts,
                                    divmod64, const_narrow, wide_cast_index, unsigned_scalar)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runner required for integer differential")
    cases = 0
    x = np.array([-(2**31), -7, -1, 0, 1, 7, 2**31 - 1, 128], dtype=np.int32)
    device_x = torch.from_numpy(x).cuda()
    for divisor in (-3, -1, 1, 3):
        cpu_q, cpu_r = np.zeros_like(x), np.zeros_like(x)
        gpu_q, gpu_r = torch.zeros_like(device_x), torch.zeros_like(device_x)
        divmod_kernel[(1,)](x, cpu_q, cpu_r, divisor)
        divmod_kernel[(1,)](device_x, gpu_q, gpu_r, divisor)
        np.testing.assert_array_equal(cpu_q, gpu_q.cpu().numpy())
        np.testing.assert_array_equal(cpu_r, gpu_r.cpu().numpy())
        cases += 1
    cpu_i8 = np.zeros(1, dtype=np.int8)
    gpu_i8 = torch.zeros(1, dtype=torch.int8, device="cuda")
    scalar_wrap[(1,)](cpu_i8, 127)
    scalar_wrap[(1,)](gpu_i8, 127)
    np.testing.assert_array_equal(cpu_i8, gpu_i8.cpu().numpy())
    cases += 1
    for value, count in ((2**64 - 1, 1), (2**63, 1), (0, 63)):
        cpu_unsigned = np.zeros(1, dtype=np.int64)
        gpu_unsigned = torch.zeros(1, dtype=torch.int64, device="cuda")
        unsigned_scalar[(1,)](cpu_unsigned, value, count)
        unsigned_scalar[(1,)](gpu_unsigned, value, count)
        np.testing.assert_array_equal(cpu_unsigned, gpu_unsigned.cpu().numpy())
        cases += 1
    x64 = np.array([-(2**63), -(2**53 + 1), -7, 0, 7, 2**53 + 1, 2**63 - 1], dtype=np.int64)
    gpu_x64 = torch.from_numpy(x64).cuda()
    for divisor in (-3, -1, 3):
        cpu_q, cpu_r = np.zeros_like(x64), np.zeros_like(x64)
        gpu_q, gpu_r = torch.zeros_like(gpu_x64), torch.zeros_like(gpu_x64)
        divmod64[(1,)](x64, cpu_q, cpu_r, divisor)
        divmod64[(1,)](gpu_x64, gpu_q, gpu_r, divisor)
        np.testing.assert_array_equal(cpu_q, gpu_q.cpu().numpy())
        np.testing.assert_array_equal(cpu_r, gpu_r.cpu().numpy())
        cases += 1
    const_narrow[(1,)](cpu_i8, BIG=2**100 + 255)
    const_narrow[(1,)](gpu_i8, BIG=2**100 + 255)
    np.testing.assert_array_equal(cpu_i8, gpu_i8.cpu().numpy())
    cases += 1
    cpu_wide = np.zeros(8, dtype=np.int32)
    gpu_wide = torch.zeros(8, dtype=torch.int32, device="cuda")
    wide_cast_index[(1,)](cpu_wide)
    wide_cast_index[(1,)](gpu_wide)
    np.testing.assert_array_equal(cpu_wide, gpu_wide.cpu().numpy())
    cases += 1
    for count in (0, 1, 31):
        cpu = np.zeros_like(x)
        gpu = torch.zeros_like(device_x)
        shifts[(1,)](x, cpu, count)
        shifts[(1,)](device_x, gpu, count)
        np.testing.assert_array_equal(cpu, gpu.cpu().numpy())
        cases += 1
    cpu = np.zeros(8, dtype=np.int8)
    gpu = torch.zeros(8, dtype=torch.int8, device="cuda")
    cast_narrow[(1,)](x, cpu)
    cast_narrow[(1,)](device_x, gpu)
    np.testing.assert_array_equal(cpu, gpu.cpu().numpy())
    cases += 1
    for value in (-1.9, 1.9, 16777217.0, -(2**31) - 0.5, 2**31 - 0.5):
        cpu = np.zeros(1, dtype=np.int32)
        gpu = torch.zeros(1, dtype=torch.int32, device="cuda")
        float_cast[(1,)](cpu, value)
        float_cast[(1,)](gpu, value)
        np.testing.assert_array_equal(cpu, gpu.cpu().numpy())
        cases += 1
    cpu = np.zeros(1, dtype=np.int32)
    gpu = torch.zeros(1, dtype=torch.int32, device="cuda")
    loop_boundary[(1,)](cpu, 2**31 - 2, 2**31 - 1)
    loop_boundary[(1,)](gpu, 2**31 - 2, 2**31 - 1)
    np.testing.assert_array_equal(cpu, gpu.cpu().numpy())
    cases += 1
    print(f"{cases} integer GPU differential cases passed")
    print(f"torch={torch.__version__}, triton={triton.__version__}, "
          f"CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
