"""Explicit ADR-013 gate: PYTHONPATH=src python tests/gpu_const_bool_smoke.py."""
import numpy as np
import torch
import triton
from test_const_bool import choose, required, short_circuit


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runner required")
    count = 0
    for enabled, other in ((True, False), (False, False), (True, True), (False, True), (True, False)):
        cpu = np.zeros(1, np.int32)
        gpu = torch.zeros(1, dtype=torch.int32, device="cuda")
        choose[(1,)](cpu, ENABLED=enabled, OTHER=other)
        choose[(1,)](gpu, ENABLED=enabled, OTHER=other)
        np.testing.assert_array_equal(cpu, gpu.cpu().numpy())
        count += 1
    for flag in (True, False, True):
        required[(1,)](cpu, FLAG=flag)
        required[(1,)](gpu, FLAG=flag)
        np.testing.assert_array_equal(cpu, gpu.cpu().numpy())
        count += 1
    for flag, divisor in ((True, 0), (False, 1), (False, -1)):
        short_circuit[(1,)](cpu, FLAG=flag, DIV=divisor)
        short_circuit[(1,)](gpu, FLAG=flag, DIV=divisor)
        np.testing.assert_array_equal(cpu, gpu.cpu().numpy())
        count += 1
    print(f"{count} Const bool CPU/GPU cases passed; {torch.cuda.get_device_name()}; "
          f"torch={torch.__version__}; triton={triton.__version__}; CUDA={torch.version.cuda}")


if __name__ == "__main__":
    main()
