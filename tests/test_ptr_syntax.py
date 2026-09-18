"""ADR-001：Ptr 公共参数解析、规范化表示与类型打印。"""

import numpy as np
import pytest

import tila as ti
from tila import types as TY
from tila.dims import Cst
from tila.errors import TilaLaunchContractError


N = ti.Dim("N")
SHARED_PTR_SPEC = ti.ReadPtr[ti.f32, N, 16]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (ti.Ptr[ti.f32],
         "Ptr[f32, Global, ReadWrite, UnknownExtent, UnknownAlignment]"),
        (ti.Ptr[ti.f32, ti.ReadOnly],
         "Ptr[f32, Global, ReadOnly, UnknownExtent, UnknownAlignment]"),
        (ti.Ptr[ti.f32, ti.ReadOnly, N],
         "Ptr[f32, Global, ReadOnly, N, UnknownAlignment]"),
        (ti.Ptr[ti.f32, ti.ReadOnly, 64, 16],
         "Ptr[f32, Global, ReadOnly, 64, Aligned[16]]"),
        (ti.Ptr[ti.f32, ti.Global, ti.ReadOnly, N, 16],
         "Ptr[f32, Global, ReadOnly, N, Aligned[16]]"),
    ],
)
def test_ptr_forms_normalize_to_canonical_print(spec, expected):
    assert isinstance(spec, TY.PtrT)
    assert spec.describe() == expected


@pytest.mark.parametrize(
    ("spec", "expected_access"),
    [
        (ti.ReadPtr[ti.f16, N, 8], ti.ReadOnly),
        (ti.WritePtr[ti.f16, N, 8], ti.WriteOnly),
        (ti.RWPtr[ti.f16, N, 8], ti.ReadWrite),
    ],
)
def test_short_aliases_accept_extent_and_alignment(spec, expected_access):
    assert spec.access == expected_access
    assert spec.extent == TY.LinearExtent(N)
    assert spec.alignment == TY.Alignment(8)
    assert spec.address_space == ti.Global


def test_integer_extent_is_normalized_to_dim_expr():
    spec = ti.Ptr[ti.f32, ti.ReadWrite, 32]
    assert spec.extent == TY.LinearExtent(Cst(32))
    assert spec.describe().endswith(", 32, UnknownAlignment]")


def test_internal_ptr_capability_fields_are_controlled_enums():
    spec = ti.Ptr[ti.f32, ti.Global, ti.ReadOnly, N, ti.Aligned[16]]
    assert isinstance(spec.access, TY.Access)
    assert isinstance(spec.address_space, TY.AddressSpace)
    with pytest.raises(TypeError):
        TY.PtrT(ti.f32, address_space="Global", access=ti.ReadOnly)
    with pytest.raises(TypeError):
        TY.PtrT(ti.f32, address_space=ti.Global, access="ReadOnly")


@pytest.mark.parametrize(
    "make",
    [
        lambda: ti.Ptr[ti.f32, ti.Global],
        lambda: ti.Ptr[ti.f32, ti.Shared, ti.ReadWrite, N, 16],
        lambda: ti.Ptr[ti.f32, ti.ReadWrite, -1],
        lambda: ti.Ptr[ti.f32, ti.ReadWrite, True],
        lambda: ti.Ptr[ti.f32, ti.ReadWrite, N, 3],
        lambda: ti.ReadPtr[ti.f32, N, 0],
        lambda: ti.RWPtr[ti.f32, N, 16, "extra"],
    ],
)
def test_invalid_or_ambiguous_ptr_forms_are_rejected(make):
    with pytest.raises(TypeError):
        make()


def test_checker_assigns_internal_region_id_not_public_parameter():
    @ti.jit
    def k(p: ti.Ptr[ti.f32, ti.Global, ti.ReadWrite, N, 16]):
        pass

    spec = k.tk.ptr_params[0].vtype
    assert spec.region_id == TY.ParamRegion(0, "p")
    assert "ParamRegion" not in spec.describe()
    assert spec.describe() == "Ptr[f32, Global, ReadWrite, N, Aligned[16]]"


def test_reused_public_spec_gets_parameter_local_region_ids():
    @ti.jit
    def first(p: SHARED_PTR_SPEC):
        pass

    @ti.jit
    def second(q: SHARED_PTR_SPEC):
        pass

    assert isinstance(SHARED_PTR_SPEC.region_id, TY.UnknownRegion)
    assert first.tk.ptr_params[0].vtype.region_id == TY.ParamRegion(0, "p")
    assert second.tk.ptr_params[0].vtype.region_id == TY.ParamRegion(0, "q")


def test_static_extent_cannot_exceed_available_storage():
    @ti.jit
    def k(p: ti.ReadPtr[ti.f32, 65]):
        pass

    with pytest.raises(TilaLaunchContractError) as exc:
        k[(1,)](np.zeros(64, dtype=np.float32))
    assert exc.value.code == "TILA-TYPE-102"
    assert "extent exceeds available storage" in exc.value.render()


def test_raw_ptr_requires_contiguous_storage():
    @ti.jit
    def k(p: ti.ReadPtr[ti.f32, 6]):
        pass

    noncontiguous = np.zeros((2, 3), dtype=np.float32).T
    with pytest.raises(TilaLaunchContractError) as exc:
        k[(1,)](noncontiguous)
    assert exc.value.code == "TILA-TYPE-102"
    assert "requires contiguous storage" in exc.value.render()


def test_contiguous_multidimensional_argument_uses_flat_ptr_semantics():
    @ti.jit
    def k(p: ti.RWPtr[ti.f32, 6]):
        offsets = ti.arange(0, 6)
        values = ti.load(p + offsets)
        ti.store(p + offsets, values + 1.0)

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    k[(1,)](array)
    assert np.array_equal(array, (np.arange(6) + 1).reshape(2, 3))
