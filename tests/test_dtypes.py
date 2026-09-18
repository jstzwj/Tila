"""dtype 域：widening 格、common_type、字面量可表示性（type-system.md §6）。"""

import pytest

from tila import dtypes as D
from tila.dtypes import can_widen, common_type, representable


class TestWidening:
    def test_reflexive(self):
        assert can_widen(D.i32, D.i32)

    def test_lattice(self):
        assert can_widen(D.i8, D.i64)
        assert can_widen(D.u8, D.u64)
        assert can_widen(D.f16, D.f32)
        assert can_widen(D.f16, D.f64)
        assert can_widen(D.bf16, D.f32)
        assert can_widen(D.bf16, D.f64)

    @pytest.mark.parametrize("a,b", [
        (D.i32, D.u32),      # 跨符号：两个方向都不可
        (D.i8, D.u8),
        (D.i64, D.u64),
        (D.i32, D.f32),      # int/float
        (D.u32, D.f32),
        (D.u32, D.i32),
        (D.f16, D.bf16),     # 互不 widening
    ])
    def test_forbidden(self, a, b):
        assert not can_widen(a, b)
        assert not can_widen(b, a)

    @pytest.mark.parametrize("wide,narrow", [
        (D.f32, D.f16),      # narrowing 方向不可；反方向（提升）可以
        (D.i64, D.i32),
        (D.f64, D.f32),
    ])
    def test_narrowing_one_way(self, wide, narrow):
        assert not can_widen(wide, narrow)
        assert can_widen(narrow, wide)


class TestCommonType:
    def test_same(self):
        assert common_type(D.f16, D.f16) is D.f16

    def test_widen(self):
        assert common_type(D.i32, D.i64) is D.i64
        assert common_type(D.i64, D.i32) is D.i64
        assert common_type(D.f16, D.f32) is D.f32

    def test_mixed_signed_unsigned_is_none(self):
        assert common_type(D.i32, D.u32) is None

    def test_int_float_is_none(self):
        assert common_type(D.i32, D.f32) is None

    def test_f16_bf16_is_none(self):
        assert common_type(D.f16, D.bf16) is None


class TestRepresentable:
    def test_int_ranges(self):
        assert representable(255, D.u8)
        assert not representable(256, D.u8)
        assert not representable(-1, D.u8)
        assert representable(-128, D.i8)
        assert not representable(128, D.i8)

    def test_float_wide_always(self):
        assert representable(0.1, D.f32)
        assert representable(0.1, D.f64)

    def test_f16_roundtrip(self):
        assert representable(1.5, D.f16)
        assert representable(0.0, D.f16)
        assert not representable(0.1, D.f16)   # 不允许静默舍入

    def test_int_literal_into_float_context(self):
        assert representable(0, D.f32)
        assert representable(3, D.f16)

    def test_float_literal_into_int_context(self):
        assert not representable(1.0, D.i32)
