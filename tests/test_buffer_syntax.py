"""ADR-003 / M1-03：Buffer 规范语法、类型打印与 buf.ptr 边界。"""

import numpy as np
import pytest

import tila as ti
from tila import types as TY
from tila.errors import TilaError, TilaLaunchContractError


N = ti.Dim("N")
M = ti.Dim("M")
SHARED_BUFFER_SPEC = ti.Buffer[ti.f32, (N,), ti.ReadOnly, 16]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (ti.Buffer[ti.f32, (N,)],
         "Buffer[f32, (N,), ReadWrite, UnknownAlignment]"),
        (ti.Buffer[ti.f32, (M, N), ti.ReadOnly],
         "Buffer[f32, (M, N), ReadOnly, UnknownAlignment]"),
        (ti.Buffer[ti.f32, (N,), ti.WriteOnly, 16],
         "Buffer[f32, (N,), WriteOnly, Aligned[16]]"),
    ],
)
def test_buffer_forms_normalize_to_four_item_print(spec, expected):
    assert isinstance(spec, TY.BufferT)
    assert spec.describe() == expected
    assert spec.address_space == ti.Global
    assert isinstance(spec.strides, TY.UnboundStrides)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ti.Buffer[ti.f32],
        lambda: ti.Buffer[ti.f32, (N,), ti.ReadOnly, 16, ti.Global],
        lambda: ti.Buffer[ti.f32, (N,), (1,), ti.ReadOnly],
        lambda: ti.Buffer[ti.f32, (N,), ti.Global],
        lambda: ti.Buffer[ti.f32, (N,), ti.ReadOnly, 0],
        lambda: ti.Buffer[ti.f32, (N,), ti.ReadOnly, 3],
        lambda: ti.Buffer[ti.f32, (N,), ti.ReadOnly, True],
    ],
)
def test_invalid_or_explicit_layout_forms_are_rejected(make):
    with pytest.raises(TypeError):
        make()


def test_checker_binds_internal_strides_without_mutating_public_spec():
    @ti.jit
    def first(x: SHARED_BUFFER_SPEC):
        pass

    @ti.jit
    def second(y: SHARED_BUFFER_SPEC):
        pass

    assert isinstance(SHARED_BUFFER_SPEC.strides, TY.UnboundStrides)
    assert tuple(map(str, first.tk.buffers[0].vtype.strides)) == \
        ("x_stride0",)
    assert tuple(map(str, second.tk.buffers[0].vtype.strides)) == \
        ("y_stride0",)


def test_internal_buffer_capability_fields_are_controlled_enums():
    spec = ti.Buffer[ti.f32, (N,), ti.ReadOnly]
    assert isinstance(spec.access, TY.Access)
    assert isinstance(spec.address_space, TY.AddressSpace)
    with pytest.raises(TypeError):
        TY.BufferT(ti.f32, (N,), access="ReadOnly")
    with pytest.raises(TypeError):
        TY.BufferT(ti.f32, (N,), address_space="Global")


def test_rank_two_buffer_ptr_is_rejected_at_stage1():
    with pytest.raises(TilaError) as exc:
        @ti.jit
        def bad(x: ti.Buffer[ti.f32, (M, N), ti.ReadOnly]):
            p = x.ptr

    assert exc.value.code == "TILA-MEM-005"
    assert "requires a rank-1 Buffer" in exc.value.render()
    assert "does not implicitly flatten" in exc.value.render() or \
        "不隐式 flatten" in exc.value.render()


def test_rank_one_nonunit_stride_buffer_ptr_is_rejected_at_launch():
    @ti.jit
    def read(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly]):
        p = x.ptr

    view = np.arange(16, dtype=np.float32)[::2]
    with pytest.raises(TilaLaunchContractError) as exc:
        read[(1,)](view)
    assert exc.value.code == "TILA-MEM-005"
    assert "stride[0] == 1" in exc.value.render()


def test_rank_one_unit_stride_buffer_ptr_inherits_capabilities():
    @ti.jit
    def read(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly, 16]):
        p = x.ptr

    buffer = read.tk.buffers[0]
    ptr = read.tk.types["p"]
    assert ptr.element is ti.f32
    assert ptr.access == ti.ReadOnly
    assert ptr.address_space == ti.Global
    assert ptr.extent == TY.LinearExtent(N)
    assert ptr.alignment == TY.Alignment(16)
    assert ptr.region_id == buffer.region_id
    read[(1,)](np.zeros(8, dtype=np.float32))


def test_noncontiguous_multidimensional_buffer_coordinate_access_still_works():
    @ti.jit
    def copy(x: ti.Buffer[ti.f32, (M, N), ti.ReadOnly],
             out: ti.Buffer[ti.f32, (M, N), ti.WriteOnly]):
        rows = ti.arange(0, 3)[:, None]
        cols = ti.arange(0, 2)[None, :]
        mask = (rows < M) & (cols < N)
        values = ti.load(x, (rows, cols), mask=mask)
        ti.store(out, (rows, cols), values, mask=mask)

    source = np.arange(6, dtype=np.float32).reshape(2, 3).T
    assert not source.flags.c_contiguous
    output = np.zeros((3, 2), dtype=np.float32)
    copy[(1,)](source, output)
    assert np.array_equal(output, source)
