"""ADR-007: executable integer boundaries and proof counterexamples."""
import math

import numpy as np
import pytest

import tila as ti
from tila import dtypes as D, numeric, tir as T, types as TY
from tila.errors import TilaError, TilaLaunchContractError
from tila.interp import Interp
from tila.runtime import _check_index_metadata

N = ti.Dim("N")


@ti.jit
def divmod_kernel(x: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
                  q: ti.Buffer[ti.i32, (N,), ti.WriteOnly],
                  r: ti.Buffer[ti.i32, (N,), ti.WriteOnly], divisor: ti.i32,
                  BLOCK: ti.Const[int, ti.PowerOfTwo] = 8):
    i = ti.arange(0, BLOCK)
    value = ti.load(x, i, mask=i < N)
    ti.store(q, i, value // divisor, mask=i < N)
    ti.store(r, i, value % divisor, mask=i < N)


@ti.jit
def scalar_wrap(out: ti.Buffer[ti.i8, (1,), ti.WriteOnly], value: ti.i8):
    ti.store(out, 0, value + 1)


@ti.jit
def shifts(x: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
           out: ti.Buffer[ti.i32, (N,), ti.WriteOnly], count: ti.i32):
    i = ti.arange(0, 8)
    value = ti.load(x, i, mask=i < N)
    ti.store(out, i, value >> count, mask=i < N)


@ti.jit
def cast_narrow(x: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
                out: ti.Buffer[ti.i8, (N,), ti.WriteOnly]):
    i = ti.arange(0, 8)
    ti.store(out, i, ti.cast[ti.i8](ti.load(x, i, mask=i < N)), mask=i < N)


@ti.jit
def float_cast(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], value: ti.f64):
    ti.store(out, 0, ti.cast[ti.i32](value))


@ti.jit
def middle_overflow(out: ti.Buffer[ti.i32, (N,), ti.WriteOnly],
                    FACTOR: ti.Const[int] = 1073741824):
    lane = ti.arange(0, 8)
    # Cancellation in the final expression must not hide intermediate wrap.
    index = (lane * FACTOR) - (lane * FACTOR) + lane
    ti.store(out, index, ti.zeros((8,), ti.i32), mask=index < N)


@ti.jit
def loop_boundary(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly],
                  start: ti.i32, end: ti.i32):
    acc = ti.cast[ti.i32](0)
    for i in ti.range(start, end, 2):
        acc = acc + 1
    ti.store(out, 0, acc)


@ti.jit
def divmod64(x: ti.Buffer[ti.i64, (N,), ti.ReadOnly],
             q: ti.Buffer[ti.i64, (N,), ti.WriteOnly],
             r: ti.Buffer[ti.i64, (N,), ti.WriteOnly], divisor: ti.i64):
    i = ti.arange(0, 8)
    v = ti.load(x, i, mask=i < N)
    ti.store(q, i, v // divisor, mask=i < N)
    ti.store(r, i, v % divisor, mask=i < N)


@ti.jit
def wide_cast_index(out: ti.Buffer[ti.i32, (N,), ti.WriteOnly]):
    lane = ti.cast[ti.i64](ti.arange(0, 8))
    index = (lane * 1073741824) - (lane * 1073741824) + lane
    ti.store(out, index, ti.zeros((8,), ti.i32), mask=index < N)


@ti.jit
def const_narrow(out: ti.Buffer[ti.i8, (1,), ti.WriteOnly], BIG: ti.Const[int]):
    ti.store(out, 0, ti.cast[ti.i8](BIG))


@ti.jit
def unsigned_scalar(out: ti.Buffer[ti.i64, (1,), ti.WriteOnly], value: ti.u64, count: ti.u64):
    ti.store(out, 0, ti.cast[ti.i64]((value + 1) << count))


def test_unsigned_scalar_abi_and_left_shift():
    out = np.zeros(1, dtype=np.int64)
    for value, count in ((2**64 - 1, 1), (2**63, 1), (0, 63)):
        unsigned_scalar[(1,)](out, value, count)
        assert out[0] == signed_wrap((value + 1) << count, D.i64)


def signed_wrap(value, dt):
    bits = value % (2 ** dt.bits)
    return bits - 2 ** dt.bits if dt.kind == "int" and bits >= 2 ** (dt.bits - 1) else bits


@pytest.mark.parametrize("dt", D.INT_DTYPES + D.UINT_DTYPES)
def test_integer_matrix_matches_python_bitvector_oracle(dt):
    lo, hi = numeric.limits(dt)
    interpreter = Interp(T.TKernel("arithmetic", [], [], [], []), {}, {}, {}, (1,))
    vt = TY.ScalarT(dt)
    cases = [("+", hi, 1), ("-", lo, 1), ("*", hi, hi),
             ("//", hi, 3), ("%", hi, 3), ("<<", 1, dt.bits - 1),
             (">>", lo, dt.bits - 1)]
    if dt.kind == "int":
        cases += [("//", -7, 3), ("//", 7, -3), ("%", -7, 3),
                  ("%", 7, -3), ("//", lo, -1), ("%", lo, -1)]
    import operator
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul,
           "//": operator.floordiv, "%": operator.mod,
           "<<": operator.lshift, ">>": operator.rshift}
    for op, a, b in cases:
        node = T.TBin(vt, op, T.TLit(a, dt), T.TLit(b, dt), operand_dtype=dt)
        actual = interpreter.e(node, {}, (0, 0, 0))
        assert int(actual) == signed_wrap(ops[op](a, b), dt)
        assert np.asarray(actual).dtype == np.dtype(dt.np_dtype)
        block = T.TBin(TY.BlockT(vt, ()), op, T.TName("a"), T.TName("b"), operand_dtype=dt)
        actual_block = interpreter.e(block, {"a": np.array([a], dtype=dt.np_dtype),
                                             "b": np.array([b], dtype=dt.np_dtype)}, (0, 0, 0))
        assert int(actual_block[0]) == int(actual)


@pytest.mark.parametrize("divisor", [-3, -1, 1, 3])
def test_divmod_signed_end_to_end(divisor):
    x = np.array([-(2**31), -7, -1, 0, 1, 7, 2**31 - 1], dtype=np.int32)
    q, r = np.empty_like(x), np.empty_like(x)
    divmod_kernel[(1,)](x, q, r, divisor)
    assert q.tolist() == [signed_wrap(int(v) // divisor, D.i32) for v in x]
    assert r.tolist() == [int(v) % divisor for v in x]


def test_scalar_wrap_and_launch_range():
    out = np.zeros(1, dtype=np.int8)
    scalar_wrap[(1,)](out, 127)
    assert out[0] == -128
    for invalid in (-129, 128):
        with pytest.raises(TilaLaunchContractError, match="does not fit i8"):
            scalar_wrap[(1,)](out, invalid)


def test_invalid_divisor_and_shift_rejected_before_write():
    x = np.arange(8, dtype=np.int32)
    out = np.full(8, 99, dtype=np.int32)
    with pytest.raises(TilaError, match="divisor is nonzero"):
        divmod_kernel[(1,)](x, out, out, 0)
    for count in (-1, 32):
        with pytest.raises(TilaError, match="shift count"):
            shifts[(1,)](x, out, count)
    assert (out == 99).all()


def test_float_cast_domains():
    out = np.zeros(1, dtype=np.int32)
    for value in (1.9, -1.9, 2**31 - 0.5, -(2**31) - 0.5):
        float_cast[(1,)](out, value)
        assert out[0] == math.trunc(value)
    for value in (float("nan"), float("inf"), -(2**31) - 1.0, float(2**31)):
        with pytest.raises(TilaError, match="finite, representable"):
            float_cast[(1,)](out, value)
    @ti.jit
    def bf16_narrow(out: ti.Buffer[ti.i8, (1,), ti.WriteOnly], value: ti.bf16):
        ti.store(out, 0, ti.cast[ti.i8](value))
    with pytest.raises(TilaError, match="finite, representable"):
        # The host float fits after truncation, but its bf16 value is 128.
        bf16_narrow[(1,)](np.zeros(1, dtype=np.int8), 127.9)


def test_middle_overflow_cannot_be_cancelled_or_bypassed_by_warn(monkeypatch):
    out = np.full(8, 42, dtype=np.int32)
    monkeypatch.setenv("TILA_SAFETY", "warn")
    with pytest.raises(TilaError, match="intermediate index"):
        middle_overflow[(1,)](out)
    assert (out == 42).all()
    # Contracts must be checked on every specialization/launch.
    middle_overflow[(1,)](out, FACTOR=2)
    with pytest.raises(TilaError, match="intermediate index"):
        middle_overflow[(1,)](out)


def test_widen_before_multiplication_preserves_safe_index_proof():
    out = np.full(8, 42, dtype=np.int32)
    wide_cast_index[(1,)](out)
    assert (out == 0).all()


def test_compile_time_constant_cast_and_illegal_arithmetic():
    out = np.zeros(1, dtype=np.int8)
    const_narrow[(1,)](out, BIG=2**100 + 255)
    assert out[0] == -1
    with pytest.raises(TilaError, match="intermediate index"):
        middle_overflow.materialize()
    @ti.jit
    def bad(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
        v = ti.cast[ti.i32](1)
        ti.store(out, 0, v // 0)
    with pytest.raises(TilaError, match="divisor"):
        bad.materialize()


def test_i64_float_roundtrip_cannot_hide_cast_overflow():
    @ti.jit
    def bad(out: ti.Buffer[ti.i64, (1,), ti.WriteOnly], value: ti.i64):
        ti.store(out, 0, ti.cast[ti.i64](ti.cast[ti.f64](value)))
    out = np.zeros(1, dtype=np.int64)
    with pytest.raises(TilaError, match="finite, representable"):
        bad[(1,)](out, 2**63 - 1)


def test_nonzero_arange_uses_actual_coordinates_in_proof():
    with pytest.raises(TilaError, match="out-of-bounds"):
        @ti.jit
        def bad(out: ti.Buffer[ti.i32, (8,), ti.WriteOnly]):
            i = ti.arange(8, 16)
            ti.store(out, i, ti.zeros((8,), ti.i32))
        bad[(1,)](np.zeros(8, dtype=np.int32))


def test_grid_and_metadata_narrowing_without_large_allocations():
    out = np.zeros(1, dtype=np.int8)
    for grid in (2**31, -1, 1.5, True):
        with pytest.raises(TilaLaunchContractError, match="grid"):
            scalar_wrap[(grid,)](out, 0)
    for grid in (ti.cdiv(1, 0), ti.cdiv(1.5, 1), ti.cdiv(True, 1)):
        with pytest.raises(TilaLaunchContractError, match="cdiv"):
            scalar_wrap[(grid,)](out, 0)
    with pytest.raises(TilaLaunchContractError, match="three axes"):
        scalar_wrap[(1, 1, 1, 1)](out, 0)
    for shape, strides in [((2**31,), (1,)), ((1,), (2**31,))]:
        with pytest.raises(TilaLaunchContractError, match="i32"):
            _check_index_metadata(shape, strides)


def test_loop_final_increment_is_wide():
    out = np.zeros(1, dtype=np.int32)
    loop_boundary[(1,)](out, 2**31 - 2, 2**31 - 1)
    assert out[0] == 1
    src, _ = loop_boundary.materialize()
    assert "tl.cast(end, tl.int64)" in src


def test_narrow_integer_cast_wraps():
    x = np.array([-257, -129, -128, -1, 0, 127, 128, 256], dtype=np.int32)
    out = np.zeros(8, dtype=np.int8)
    cast_narrow[(1,)](x, out)
    assert out.tolist() == [signed_wrap(int(v), D.i8) for v in x]


def test_untrusted_integer_arithmetic_does_not_inherit_mathematical_facts():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
               out: ti.Buffer[ti.i32, (N,), ti.WriteOnly]):
        i = ti.arange(0, 8)
        v = ti.load(x, i, mask=i < N)
        index = v + 1
        ti.store(out, index, v, mask=(index >= 0) & (index < N))
    x = np.array([2**31 - 1, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    out = np.full(8, 99, dtype=np.int32)
    kernel[(1,)](x, out)
    assert out.tolist() == [-1, 0, 1, 2, 3, 4, 5, 99]
