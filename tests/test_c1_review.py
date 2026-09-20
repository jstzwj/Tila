"""C1 independent boundary review: acceptance failures before execution."""
from dataclasses import replace

import numpy as np
import pytest
import tila as ti
from tila import dims as D, tir as T, types as TY
from tila.errors import TilaError, TilaLaunchContractError
from tila.effect_ir import bind_effects
from tila.verifier import verify


def writeonly_load(x: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    value = ti.load(x, 0)


def writeonly_unsafe(x: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    value = ti.unsafe_load(x, 0)


def writeonly_false_mask(x: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    value = ti.load(x, 0, mask=False)


def writeonly_nested(x: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    value = ti.where(True, 0, ti.load(x, 0))


def writeonly_other(x: ti.Buffer[ti.i32, (4,), ti.WriteOnly], y: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
    value = ti.load(y, 0, mask=True, other=ti.load(x, 0))


def writeonly_ptr(x: ti.WritePtr[ti.i32, 4]):
    value = ti.load(x)


def writeonly_ptr_unsafe(x: ti.WritePtr[ti.i32, 4]):
    value = ti.unsafe_load(x)


@pytest.mark.parametrize('fn', [writeonly_load, writeonly_unsafe, writeonly_false_mask,
                               writeonly_nested, writeonly_other, writeonly_ptr, writeonly_ptr_unsafe])
def test_writeonly_never_permits_read_even_under_unsafe_or_false_mask(fn):
    with pytest.raises(TilaError, match='TILA-MEM-001'):
        ti.jit(fn)


@ti.jit
def mask_cast_values(out: ti.Buffer[ti.i32, (4, 4), ti.WriteOnly]):
    i = ti.arange(0, 4)
    mask = i[:, None] < i[None, :]
    boolean_value = ti.where(mask, True, False)
    value = ti.cast[ti.i32](boolean_value)
    ti.store(out, (i[:, None], i[None, :]), value)


@ti.jit
def mask_cast_tail(x: ti.Buffer[ti.i32, (3,), ti.ReadOnly], out: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    i = ti.arange(0, 4)
    boolean_value = ti.where(i < 3, True, False)
    mask = ti.cast[ti.bool](boolean_value) & (i < 3)
    value = ti.load(x, i, mask=mask, other=-1)
    ti.store(out, i, value)


def test_boolean_tile_numeric_cast_retains_rank_and_shape():
    value_type = mask_cast_values.tk.types['value']
    assert isinstance(value_type, TY.BlockT) and value_type.dims == (D.Cst(4), D.Cst(4))
    out = np.full((4, 4), -1, np.int32)
    mask_cast_values[(1,)](out)
    np.testing.assert_array_equal(out, np.arange(4)[:, None] < np.arange(4)[None, :])


def test_boolean_tile_cast_composes_with_explicit_bounds_predicate():
    assert isinstance(mask_cast_tail.tk.types['mask'], TY.MaskT)
    out = np.zeros(4, np.int32)
    mask_cast_tail[(1,)](np.array([5, 6, 7], np.int32), out)
    np.testing.assert_array_equal(out, [5, 6, 7, -1])


def test_cast_mask_cannot_be_used_as_scalar_branch_condition():
    with pytest.raises(TilaError, match='scalar bool|Mask|Block'):
        @ti.jit
        def kernel():
            i = ti.arange(0, 4)
            condition = ti.cast[ti.bool](i < 2)
            if condition:
                result = 1


def test_cast_mask_incompatible_broadcast_rejected():
    with pytest.raises(TilaError, match='cast does not accept Mask'):
        @ti.jit
        def kernel():
            mask = ti.arange(0, 4) < 2
            result = ti.cast[ti.i32](mask) + ti.zeros((8,), ti.i32)


def test_memory_handle_cannot_cast_to_numeric_value():
    with pytest.raises(TilaError):
        @ti.jit
        def kernel(x: ti.ReadPtr[ti.i32, 4]):
            value = ti.cast[ti.i64](x)


@ti.jit
def ptr_view_copy(x: ti.ReadPtr[ti.i32, 4], out: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    i = ti.arange(0, 4)
    forward = ti.cast[ti.i32](3)
    backward = ti.cast[ti.i32](-3)
    advanced = x + forward
    restored = advanced + backward
    ti.store(out, i, ti.load(restored + i))


def test_pointer_view_origin_negative_step_and_sentinels():
    source = np.arange(12, dtype=np.int32)
    backing = np.full(12, -1, np.int32)
    ptr_view_copy[(1,)](source[2:6], backing[3:7])
    expected = np.full(12, -1, np.int32)
    expected[3:7] = source[2:6]
    np.testing.assert_array_equal(backing, expected)


@ti.jit
def ptr_masked_offset(x: ti.ReadPtr[ti.i32, 4], out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], offset: ti.i32):
    value = ti.load(x + offset, mask=(offset >= 0) & (offset < 4), other=-1)
    ti.store(out, 0, value)


@pytest.mark.parametrize('offset', [-3, -1, 0, 3, 4, 7])
def test_pointer_mask_is_relative_to_view_extent(offset):
    source = np.arange(12, dtype=np.int32)[2:6]
    out = np.zeros(1, np.int32)
    ptr_masked_offset[(1,)](source, out, offset)
    assert out[0] == (source[offset] if 0 <= offset < 4 else -1)


def test_pointer_view_cannot_access_backing_before_its_origin():
    with pytest.raises(TilaError, match='TILA-BOUNDS'):
        @ti.jit
        def kernel(x: ti.ReadPtr[ti.i32, 4]):
            before = ti.cast[ti.i32](-1)
            value = ti.load(x + before)


def test_pointer_rejects_negative_stride_binding():
    with pytest.raises((TilaError, TilaLaunchContractError)):
        ptr_view_copy[(1,)](np.arange(4, dtype=np.int32)[::-1], np.zeros(4, np.int32))


@ti.jit
def static_return_loop(out: ti.Buffer[ti.i32, (4,), ti.WriteOnly], count: ti.i32,
                       RETURN: ti.Const[bool], INNER: ti.Const[bool]):
    if RETURN:
        unused = ti.arange(0, 8)
        return
    else:
        unused = ti.arange(0, 4)
    value = ti.zeros((4,), ti.i32)
    for j in ti.range(0, count):
        if INNER:
            value = value + 1
        else:
            value = value + 2
    ti.store(out, unused, value)


@pytest.mark.parametrize('ret,inner,count', [(r, i, c) for r in (False, True)
                                            for i in (False, True) for c in (0, 1, 3)])
def test_static_return_and_zero_trip_composition(ret, inner, count):
    out = np.full(4, -9, np.int32)
    static_return_loop[(1,)](out, count, RETURN=ret, INNER=inner)
    np.testing.assert_array_equal(out, -9 if ret else count*(1 if inner else 2))


def test_loop_cannot_hide_static_shape_variant():
    with pytest.raises(TilaError, match='TILA-TYPE-020|loop'):
        @ti.jit
        def kernel(FLAG: ti.Const[bool]):
            value = ti.zeros((4,), ti.i32)
            for j in ti.range(0, 2):
                if FLAG:
                    value = ti.zeros((4,), ti.i32)
                else:
                    value = ti.zeros((8,), ti.i32)
            result = value + 1


def test_zero_trip_pointer_preserves_initial_binding():
    @ti.jit
    def kernel(x: ti.ReadPtr[ti.i32, 4], out: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
        p = x
        for j in ti.range(0, 0):
            p = p + ti.cast[ti.i32](1)
        i = ti.arange(0, 4)
        ti.store(out, i, ti.load(p + i))

    source, out = np.arange(4, dtype=np.int32), np.zeros(4, np.int32)
    kernel[(1,)](source, out)
    np.testing.assert_array_equal(out, source)


@pytest.mark.parametrize('unsafe', [False, True])
def test_verifier_rejects_forged_writeonly_read_with_fresh_metadata(unsafe):
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        value = ti.load(x, 0)

    tk = kernel.tk
    tk.buffers[0].vtype = replace(tk.buffers[0].vtype, access=TY.WRITE_ONLY)
    tk.body[0].value.unsafe = unsafe
    bind_effects(tk)
    with pytest.raises(TilaError, match='capability'):
        verify(tk)


def test_verifier_rejects_forged_scalar_cast_of_boolean_tile():
    tk = ti.jit(mask_cast_values.fn).tk
    cast = next(s.value for s in tk.body if isinstance(s, T.TAssign) and isinstance(s.value, T.TCast))
    cast.vt = TY.ScalarT(ti.i32)
    bind_effects(tk)
    with pytest.raises(TilaError, match='cast must preserve operand shape'):
        verify(tk)


def test_verifier_rejects_mask_cast_even_when_shape_is_preserved():
    tk = ti.jit(mask_cast_values.fn).tk
    cast = next(s.value for s in tk.body if isinstance(s, T.TAssign) and isinstance(s.value, T.TCast))
    cast.operand = T.TName('mask')
    bind_effects(tk)
    with pytest.raises(TilaError, match='cast does not accept Mask'):
        verify(tk)


def test_return_continuation_lowering_keeps_source_map_and_ir():
    from tila.lowering import Lowering
    tk = static_return_loop.tk
    before = tk.dump()
    lowering = Lowering(tk)
    source = lowering.kernel_source()
    lines = source.splitlines()
    stores = [i for i, line in enumerate(lines, 1) if 'tl.store(' in line]
    assert len(stores) == 1
    assert lines[stores[0]-1].startswith('        tl.store(')
    assert lowering.source_map[stores[0]] == tk.body[-1].line
    assert tk.dump() == before


@ti.jit
def runtime_return(out: ti.Buffer[ti.i32, (4,), ti.WriteOnly], flag: ti.bool):
    if flag:
        i = ti.arange(0, 8)
        return
    else:
        i = ti.arange(0, 4)
    ti.store(out, i, ti.zeros((4,), ti.i32))


@ti.jit
def loop_return(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], count: ti.i32):
    for j in ti.range(0, count):
        ti.store(out, 0, 7)
        return
    ti.store(out, 0, 9)


@pytest.mark.parametrize('flag', [False, True])
def test_runtime_return_shape_matches_live_continuation(flag):
    out = np.full(4, -9, np.int32)
    runtime_return[(1,)](out, flag)
    np.testing.assert_array_equal(out, -9 if flag else 0)


@pytest.mark.parametrize('count', [0, 3])
def test_loop_return_executes_continuation_only_on_empty_loop(count):
    out = np.zeros(1, np.int32)
    loop_return[(1,)](out, count)
    assert out[0] == (7 if count else 9)


def test_bool_cast_rebinding_does_not_reuse_old_mask_fact():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (3,), ti.ReadOnly]):
        i = ti.arange(0, 4)
        active = ti.cast[ti.bool](ti.where(i < 3, True, False))
        active = ti.cast[ti.bool](ti.where(i >= 3, True, False))
        value = ti.load(x, i, mask=active)

    with pytest.raises(TilaError, match='TILA-BOUNDS'):
        kernel[(1,)](np.zeros(3, np.int32))


def test_numeric_cast_does_not_invent_boolean_bounds_fact():
    @ti.jit
    def kernel(flags: ti.Buffer[ti.i32, (4,), ti.ReadOnly], x: ti.Buffer[ti.i32, (3,), ti.ReadOnly]):
        i = ti.arange(0, 4)
        active = ti.cast[ti.bool](ti.load(flags, i))
        value = ti.load(x, i, mask=active)

    with pytest.raises(TilaError, match='TILA-BOUNDS'):
        kernel[(1,)](np.ones(4, np.int32), np.zeros(3, np.int32))
