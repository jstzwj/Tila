"""ADR-008: explicit dtype policy, wraparound and frontend domain gates."""
import numpy as np
import pytest
import ml_dtypes
import tila as ti
from tila import dtypes as D, tir as T, types as TY
from tila.errors import TilaError
from reduction_support import reduction_kernel


@pytest.mark.parametrize("dt", D.ARITH_DTYPES)
@pytest.mark.parametrize("op", ["sum", "max"])
def test_reduction_dtype_and_cpu(tmp_path, dt, op):
    node = T.TReduce(TY.ScalarT(dt), op, None, 0)
    assert node.input_dtype is node.output_dtype is dt
    assert node.accumulation_dtype is (D.f32 if op == "sum" and dt in (D.f16, D.bf16) else dt)
    kernel = reduction_kernel(tmp_path, dt.name, op)
    dtype = ml_dtypes.bfloat16 if dt is D.bf16 else dt.np_dtype
    value = np.iinfo(dtype).max if dt.is_int else 0.5
    x = np.full((4, 8), value, dtype=dtype)
    out = np.zeros(4, dtype=dtype)
    kernel[(1,)](x, out)
    expected = value
    if op == "sum":
        if dt.is_int:
            expected = (int(value) * 8) % (1 << dt.bits)
            if dt.kind == "int" and expected >= 1 << (dt.bits - 1):
                expected -= 1 << dt.bits
        else:
            expected = 4
    np.testing.assert_array_equal(out, np.full(4, expected, dtype=dtype))


@pytest.mark.parametrize("dtype,code", [("bool", "TILA-TYPE-028"), ("f8e4m3fn", "TILA-TYPE-036"), ("f8e5m2", "TILA-TYPE-036")])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_reduction_rejects_non_arithmetic(tmp_path, dtype, code, op):
    with pytest.raises(TilaError) as exc:
        reduction_kernel(tmp_path, dtype, op)
    assert exc.value.code == code


def test_bool_axis_rejected():
    with pytest.raises(TilaError, match="literal int"):
        @ti.jit
        def kernel(x: ti.Buffer[ti.f32, (8,), ti.ReadOnly]):
            v = ti.load(x, ti.arange(0, 8))
            total = ti.sum(v, False)
