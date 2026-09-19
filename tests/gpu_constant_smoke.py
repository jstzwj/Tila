"""ADR-015 explicit GPU gate: PYTHONPATH=src python tests/gpu_constant_smoke.py."""
from pathlib import Path
import math
import tempfile

import numpy as np
import torch
import triton
import ml_dtypes

from test_typed_constants import FORMATS, make_kernel, oracle, rational


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runner required")
    directory = Path(tempfile.mkdtemp(prefix="tila-constant-gpu-"))
    count = 0
    for dtype, eb, fb in FORMATS:
        npdtype = ml_dtypes.bfloat16 if dtype.name == "bf16" else dtype.np_dtype
        tdtype = {"f16": torch.float16, "bf16": torch.bfloat16,
                  "f32": torch.float32, "f64": torch.float64}[dtype.name]
        intdtype = {16: torch.int16, 32: torch.int32, 64: torch.int64}[dtype.bits]
        largest = (((1 << eb) - 2) << fb) | ((1 << fb) - 1)
        values = [0.1, -0.1, 0.0, -0.0, float(rational(1, eb, fb)),
                  float(rational(1, eb, fb)) / 2, -float(rational(1, eb, fb)) / 2,
                  float(rational(1 << fb, eb, fb)), float(rational(largest, eb, fb)),
                  -float(rational(largest, eb, fb)), math.nextafter(1 + 2.0**(-fb - 1), math.inf)]
        if dtype.name in ("f32", "f64"):
            values.append(2**100 + 2**47 + 1)
        for value in values:
            kernel = make_kernel(directory / f"case{count}.py", dtype, value)
            cpu = np.zeros(1, dtype=npdtype)
            gpu = torch.zeros(1, dtype=tdtype, device="cuda")
            kernel[(1,)](cpu)
            kernel[(1,)](gpu)
            expected = oracle(value, eb, fb)
            actual_gpu = gpu.view(intdtype).cpu().numpy().view(f"uint{dtype.bits}")
            assert int(actual_gpu[0]) == expected, (dtype.name, value, actual_gpu, expected)
            assert int(cpu.view(f"uint{dtype.bits}")[0]) == expected
            count += 1
    print(f"{count} typed constant CPU/GPU bit comparisons passed; "
          f"{torch.cuda.get_device_name()}; torch={torch.__version__}; "
          f"triton={triton.__version__}; CUDA={torch.version.cuda}")


if __name__ == "__main__":
    main()
