"""ADR-014: explicit normalization preserves exact values and use-site gates."""
import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError, TilaLaunchContractError, DiagnosticPhase
from tila.frontend import INTRINSIC_NAMES
from tila.intrinsics import PUBLIC_INTRINSIC_NAMES
from tila.runtime import _triton_cache_key


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64])
def test_numpy_extrema(dtype):
    info = np.iinfo(dtype)
    for value in (info.min, 0, info.max):
        converted = ti.host_int(dtype(value))
        assert type(converted) is int
        assert converted == value


@pytest.mark.parametrize("dtype", [np.int_, np.intp, np.uintp, np.byte, np.ubyte,
                                   np.short, np.ushort])
def test_numpy_type_aliases(dtype):
    assert ti.host_int(dtype(7)) == 7


@pytest.mark.parametrize("value", [0, -1, 2**64 - 1, 2**200, -(2**200)])
def test_python_integer_preserves_value(value):
    assert type(ti.host_int(value)) is int
    assert ti.host_int(value) == value


@pytest.mark.parametrize("value", [True, False, np.bool_(False), 1.0, np.float64(1),
                                   "1", np.array(1), np.array([1]), None])
def test_rejects_non_integer_types(value):
    with pytest.raises(TilaError) as exc:
        ti.host_int(value)
    assert exc.value.code == "TILA-TYPE-037"
    assert exc.value.spec.phases == frozenset({DiagnosticPhase.HOST})
    assert "accepted:" in exc.value.render()
    assert type(value).__name__ in exc.value.render()


def test_no_conversion_or_repr_hooks_for_rejected_types():
    class Hooks:
        def __int__(self):
            raise AssertionError("__int__ executed")
        def __index__(self):
            raise AssertionError("__index__ executed")
        def __repr__(self):
            raise AssertionError("__repr__ executed")
    class PythonSubclass(int, Hooks):
        __int__ = Hooks.__int__
    class NumpySubclass(np.int64, Hooks):
        __int__ = Hooks.__int__
    for value in (Hooks(), PythonSubclass(7), NumpySubclass(7)):
        with pytest.raises(TilaError, match="TILA-TYPE-037"):
            ti.host_int(value)


def test_metaclass_equality_cannot_impersonate_a_supported_type():
    class Impersonator(type):
        def __eq__(self, other):
            raise AssertionError("metaclass equality executed")
        def __hash__(self):
            raise AssertionError("metaclass hash executed")
    class Fake(metaclass=Impersonator):
        def __int__(self):
            raise AssertionError("conversion executed")
    with pytest.raises(TilaError, match="TILA-TYPE-037"):
        ti.host_int(Fake())


def test_torch_tensor_rejected():
    import torch
    with pytest.raises(TilaError, match="TILA-TYPE-037"):
        ti.host_int(torch.tensor(7))


@ti.jit
def write_const(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly],
                VALUE: ti.Const[int, ti.Positive] = 7):
    ti.store(out, 0, VALUE)


def test_normalization_and_cache_equivalence():
    out = np.zeros(1, np.int32)
    for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint64):
        converted = ti.host_int(dtype(7))
        write_const[(1,)](out, VALUE=converted)
        assert out[0] == 7
        assert write_const.materialize({"VALUE": converted}) == write_const.materialize({"VALUE": 7})
        assert _triton_cache_key(write_const, {"VALUE": converted}) == _triton_cache_key(write_const, {"VALUE": 7})


def test_implicit_numpy_const_still_rejected():
    for entry in (write_const.materialize, write_const.explain):
        with pytest.raises(TilaError, match="TILA-CONST-008"):
            entry({"VALUE": np.int64(7)})
    with pytest.raises(TilaLaunchContractError, match="TILA-CONST-008"):
        write_const[(1,)](np.zeros(1, np.int32), VALUE=np.int64(7))


def test_use_site_refinement_and_grid_gates_remain():
    value = ti.host_int(np.int64(-1))
    with pytest.raises(TilaError, match="TILA-CONST-003"):
        write_const.materialize({"VALUE": value})
    large = ti.host_int(np.uint64(2**64 - 1))
    with pytest.raises(TilaLaunchContractError, match="TILA-TYPE-104"):
        write_const[(large,)](np.zeros(1, np.int32))


def test_scalar_range_gate_remains():
    @ti.jit
    def k(out: ti.Buffer[ti.i8, (1,), ti.WriteOnly], value: ti.i8):
        ti.store(out, 0, value)
    with pytest.raises(TilaLaunchContractError):
        k[(1,)](np.zeros(1, np.int8), ti.host_int(np.int64(128)))


@pytest.mark.parametrize("field", ["shape", "stride"])
def test_index_metadata_gates_remain(field):
    from tila.runtime import _check_index_metadata
    large = ti.host_int(np.int64(2**31))
    shape, strides = ((large,), (1,)) if field == "shape" else ((1,), (large,))
    with pytest.raises(TilaLaunchContractError, match="i32"):
        _check_index_metadata(shape, strides)


def test_host_helper_is_not_a_device_intrinsic():
    assert "host_int" in ti.__all__
    assert "host_int" not in INTRINSIC_NAMES
    assert "host_int" not in PUBLIC_INTRINSIC_NAMES
    with pytest.raises(TilaError, match="TILA-SYN-030"):
        @ti.jit
        def k():
            value = ti.host_int(1)
