"""ADR-015: independent rational oracle, exact bit payloads and surface gates."""
from fractions import Fraction
import math
import random

import numpy as np
import pytest
import ml_dtypes

import tila as ti
from tila import dtypes as D, tir as T
from tila.constants import rounded_bits
from tila.constants import float_value
from tila.errors import TilaError
from tila.runtime import _triton_cache_key


FORMATS = [(D.f16, 5, 10), (D.bf16, 8, 7), (D.f32, 8, 23), (D.f64, 11, 52)]


def rational(bits, eb, fb):
    exponent, fraction = bits >> fb, bits & ((1 << fb) - 1)
    bias = (1 << (eb - 1)) - 1
    significand = fraction if exponent == 0 else (1 << fb) + fraction
    power = (1 - bias if exponent == 0 else exponent - bias) - fb
    return Fraction(significand) * Fraction(2) ** power


def oracle(value, eb, fb):
    sign = int(value < 0 or type(value) is float and math.copysign(1, value) < 0)
    source = abs(Fraction(value))
    largest = (((1 << eb) - 2) << fb) | ((1 << fb) - 1)
    # Independent binary search over monotonically ordered finite bit patterns.
    lo, hi = 0, largest
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if rational(mid, eb, fb) <= source:
            lo = mid
        else:
            hi = mid - 1
    below, above = rational(lo, eb, fb), rational(lo + 1, eb, fb)
    bit = lo if source - below < above - source else lo + 1
    if source - below == above - source:
        bit = lo if lo % 2 == 0 else lo + 1
    if bit > largest:
        raise OverflowError
    return (sign << (eb + fb)) | bit


@pytest.mark.parametrize("dtype,eb,fb", FORMATS)
def test_direct_rounding_against_rational_oracle(dtype, eb, fb):
    rng = random.Random(15007)
    bias = (1 << (eb - 1)) - 1
    values = [0.0, -0.0, 0.1, -0.1, 1, -1, 2**100 + 2**47 + 1,
              2**1024 - 2**970, -(2**2000)]
    for _ in range(100):
        value = math.ldexp(rng.uniform(-2, 2), rng.randint(-1073, 1022))
        values.append(value)
    for e in (1 - bias - fb, 1 - bias, 0, bias):
        x = math.ldexp(1.0, e)
        values.extend([x, -x, math.nextafter(x, 0), math.nextafter(x, math.inf)])
    for value in values:
        try:
            expected = oracle(value, eb, fb)
        except OverflowError:
            with pytest.raises(TilaError, match="TILA-NUM-002"):
                rounded_bits(value, dtype)
        else:
            assert rounded_bits(value, dtype) == expected, (dtype.name, value)


@pytest.mark.parametrize("dtype,eb,fb", FORMATS[:3])
def test_midpoint_ties_and_double_rounding(dtype, eb, fb):
    for lower in (0, 1, (1 << fb) - 1, ((1 << (eb - 1)) - 1) << fb):
        midpoint = float((rational(lower, eb, fb) + rational(lower + 1, eb, fb)) / 2)
        for value in (math.nextafter(midpoint, 0), midpoint, math.nextafter(midpoint, math.inf)):
            assert rounded_bits(value, dtype) == oracle(value, eb, fb)
    if dtype in (D.f16, D.bf16):
        midpoint = 1 + 2.0 ** (-fb - 1)
        value = math.nextafter(midpoint, math.inf)
        assert rounded_bits(value, dtype) != rounded_bits(float(np.float32(value)), dtype)


@pytest.mark.parametrize("dtype,eb,fb", FORMATS[:2])
def test_all_finite_16bit_values_are_fixed_points(dtype, eb, fb):
    for bits in range(1 << 16):
        if ((bits >> fb) & ((1 << eb) - 1)) == (1 << eb) - 1:
            continue
        assert rounded_bits(float_value(dtype, bits), dtype) == bits


@pytest.mark.parametrize("dtype,eb,fb", FORMATS)
def test_overflow_rounding_threshold(dtype, eb, fb):
    largest = (((1 << eb) - 2) << fb) | ((1 << fb) - 1)
    midpoint = (rational(largest, eb, fb) + rational(largest + 1, eb, fb)) / 2
    threshold = int(midpoint)  # all supported overflow thresholds are exact integers
    assert rounded_bits(threshold - 1, dtype) == largest
    with pytest.raises(TilaError, match="TILA-NUM-002"):
        rounded_bits(threshold, dtype)
    assert rounded_bits(-threshold + 1, dtype) == (1 << (eb + fb)) | largest


def test_ordinary_literal_rules_remain_strict():
    with pytest.raises(TilaError, match="TILA-TYPE-013"):
        @ti.jit
        def k(out: ti.Buffer[ti.f16, (1,), ti.WriteOnly]):
            ti.store(out, 0, 0.1)


def test_python_numeric_subclasses_rejected(tmp_path):
    class IntSubclass(int):
        pass
    class FloatSubclass(float):
        pass
    for value in (IntSubclass(1), FloatSubclass(0.1)):
        with pytest.raises(TilaError, match="TILA-CONST-011"):
            make_kernel(tmp_path / "bad.py", D.f32, value)


def make_kernel(path, dtype, value, expression="VALUE"):
    source = ("import tila as ti\n@ti.jit\n"
              f"def k(out: ti.Buffer[ti.{dtype.name}, (1,), ti.WriteOnly]):\n"
              f"    value = ti.constant[ti.{dtype.name}]({expression})\n"
              "    ti.store(out, 0, value)\n")
    path.write_text(source)
    namespace = {"VALUE": value, "__name__": "constant_case"}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["k"]


@pytest.mark.parametrize("dtype,eb,fb", FORMATS)
def test_cpu_and_lowered_bit_patterns(tmp_path, dtype, eb, fb):
    npdt = ml_dtypes.bfloat16 if dtype is D.bf16 else dtype.np_dtype
    for value in (0.1, -0.0, 0.0, 2.0 ** (2 - (1 << (eb - 1)) - fb), -1.5):
        k = make_kernel(tmp_path / "kernel.py", dtype, value)
        out = np.zeros(1, dtype=npdt)
        k[(1,)](out)
        expected = oracle(value, eb, fb)
        assert int(out.view(f"uint{dtype.bits}")[0]) == expected
        source, typed = k.materialize()
        assert "bitcast=True" in source
        assert f"tl.uint{dtype.bits}" in source
        assert f"bits=0x{expected:0{dtype.bits // 4}x}" in typed
        assert T.constant_signature(k.tk.body) == ((dtype.name, expected),)
        assert _triton_cache_key(k, {})[-1] == ((dtype.name, expected),)


@pytest.mark.parametrize("value", [True, "0.1", np.float64(0.1), np.int64(1), np.array(1)])
def test_nonexact_sources_rejected(tmp_path, value):
    with pytest.raises(TilaError, match="TILA-CONST-011"):
        make_kernel(tmp_path / "bad.py", D.f32, value)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_rejected(tmp_path, value):
    with pytest.raises(TilaError, match="TILA-NUM-002"):
        make_kernel(tmp_path / "bad.py", D.f32, value)


@pytest.mark.parametrize("expression", ["VALUE + 1", "ti.cast[ti.f32](1)", "out", "True", "'0.1'"])
def test_expression_surface_is_narrow(tmp_path, expression):
    with pytest.raises(TilaError, match="TILA-CONST-011"):
        make_kernel(tmp_path / "bad.py", D.f32, 0.1, expression)


@pytest.mark.parametrize("dtype", [D.i32, D.bool_, D.f8e4m3fn])
def test_unsupported_targets_rejected(tmp_path, dtype):
    with pytest.raises(TilaError, match="TILA-CONST-011"):
        make_kernel(tmp_path / "bad.py", dtype, 1)


def test_const_parameter_is_not_a_literal():
    with pytest.raises(TilaError, match="TILA-CONST-011"):
        @ti.jit
        def k(VALUE: ti.Const[int]):
            value = ti.constant[ti.f32](VALUE)


def test_signed_zero_cache_payload(tmp_path):
    k = make_kernel(tmp_path / "kernel.py", D.f32, 0.0)
    positive = _triton_cache_key(k, {})
    k.tk.body[0].value.bits = 1 << 31
    assert _triton_cache_key(k, {}) != positive


def test_integer_cast_uses_rounded_value(tmp_path):
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
        value = ti.constant[ti.f32](1.9)
        ti.store(out, 0, ti.cast[ti.i32](value))
    out = np.zeros(1, np.int32)
    k[(1,)](out)
    assert out[0] == 1


@ti.jit
def snapshot_constants(a: ti.Buffer[ti.f16, (1,), ti.WriteOnly],
                       b: ti.Buffer[ti.bf16, (1,), ti.WriteOnly],
                       c: ti.Buffer[ti.f32, (1,), ti.WriteOnly],
                       d: ti.Buffer[ti.f64, (1,), ti.WriteOnly]):
    ti.store(a, 0, ti.constant[ti.f16](0.1))
    ti.store(b, 0, ti.constant[ti.bf16](0.1))
    ti.store(c, 0, ti.constant[ti.f32](-0.0))
    ti.store(d, 0, ti.constant[ti.f64](0.1))


def test_typed_constant_goldens():
    from pathlib import Path
    source, typed = snapshot_constants.materialize()
    root = Path(__file__).parent / "golden"
    assert source == (root / "typed_constants.triton.py").read_text()
    assert typed == (root / "typed_constants.tir.txt").read_text()
