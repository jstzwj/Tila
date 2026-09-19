"""Target exclusions and intermediate rounding regressions, without a GPU."""
import ml_dtypes
import ast
import json
from pathlib import Path
import numpy as np
import pytest

from m3_capability_support import elementwise
from tila.errors import TilaError
from tila.intrinsics import INTRINSICS, Availability


@pytest.mark.parametrize("dtype", ["f8e4m3fn", "f8e5m2"])
@pytest.mark.parametrize("operation", ["copy", "cast"])
def test_fp8_build_rejected(dtype, operation, tmp_path):
    kernel = elementwise(tmp_path, dtype, operation, "f32" if operation == "cast" else dtype)
    with pytest.raises(TilaError, match="TILA-TARGET-009"):
        kernel.materialize({})


@pytest.mark.parametrize("operation", ["fp8_e4", "fp8_e5"])
def test_fp8_intermediate_build_rejected(operation, tmp_path):
    kernel = elementwise(tmp_path, "f32", operation)
    with pytest.raises(TilaError, match="TILA-TARGET-009"):
        kernel.materialize({})
    with pytest.raises(TilaError, match="TILA-TARGET-009"):
        kernel[(1,)](np.zeros(8, np.float32), np.zeros(8, np.float32))


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("operation", ["exp", "exp2"])
def test_narrow_exponential_rounds_before_widening(dtype, operation, tmp_path):
    ndtype = np.float16 if dtype == "f16" else ml_dtypes.bfloat16
    kernel = elementwise(tmp_path, dtype, operation + "_chain", "f32")
    x = np.array([.125, .25, .5, .75, 1, 1.25, 1.5, 2], dtype=ndtype)
    out = np.zeros(8, np.float32)
    kernel[(1,)](x, out)
    unrounded = getattr(np, operation)(x.astype(np.float32))
    expected = unrounded.astype(ndtype).astype(np.float32)
    assert np.any(expected != unrounded)
    np.testing.assert_array_equal(out, expected)


def test_gpu_evidence_inventory_tracks_registry_and_real_tests():
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads((root / "docs/gpu-capabilities.json").read_text())
    assert set(matrix["operations"]) == {s.name for s in INTRINSICS if s.availability != Availability.DEFERRED}
    for name, row in matrix["operations"].items():
        assert row["verified_dtypes"] and row["shape"] and row["limits"] and row["evidence"], name
        for evidence in row["evidence"]:
            file, function = evidence.split("::")
            tree = ast.parse((root / file).read_text())
            assert any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body), evidence
