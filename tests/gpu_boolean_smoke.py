"""ADR-012 GPU gate: PYTHONPATH=src python tests/gpu_boolean_smoke.py."""
import numpy as np
import torch
import triton

from test_boolean_tiles import mixed_mask, broadcast_bool, direct_bool, reductions


def check(kernel, inputs, shape, *scalars):
    cpu = np.full(shape, -77, dtype=np.int32)
    gpu = torch.from_numpy(cpu.copy()).cuda()
    kernel[(1,)](*inputs, cpu, *scalars)
    kernel[(1,)](*(torch.from_numpy(x).cuda() for x in inputs), gpu, *scalars)
    np.testing.assert_array_equal(cpu, gpu.cpu().numpy())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runner required for boolean differential")
    count = 0
    for bits in range(16):
        flags = np.array([bool(bits & (1 << i)) for i in range(4)])
        check(broadcast_bool, [flags], (4, 4))
        check(direct_bool, [flags, np.arange(4, dtype=np.int32)], (4,))
        for toggle in (False, True):
            check(reductions, [flags], (1,), toggle)
        count += 4
    for n in (1, 3, 7, 8):
        check(mixed_mask, [np.arange(n) % 2 == 0, np.arange(n, dtype=np.int32)], (n,))
        count += 1
    print(f"{count} boolean CPU/GPU cases passed; "
          f"{torch.cuda.get_device_name()}; torch={torch.__version__}; "
          f"triton={triton.__version__}; CUDA={torch.version.cuda}")


if __name__ == "__main__":
    main()
