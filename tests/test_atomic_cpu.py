"""ADR-017 CPU reference, effect integrity, and explicit GPU exclusion."""
from dataclasses import replace
import numpy as np
import pytest
import tila as ti
from tila import tir as T, dtypes as D
from tila.atomic import add, Relaxed, GPU
from tila.effect_ir import bind_effects, verify_effects
from tila.effect_audit import effect_details
from tila.effect_policy import where_warnings
from tila.errors import TilaError, TilaLaunchContractError
from tila.verifier import verify

ORDER_ALIAS = ti.Relaxed
SCOPE_ALIAS = ti.GPU


@ti.jit
def simple(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
    ti.atomic_add(x, 0, 1)


def test_atomic_detail_golden():
    from pathlib import Path
    from tila.effect_audit import render_effect_details
    text = simple.tk.dump() + '\n' + render_effect_details(simple.tk, {}) + '\n'
    assert text == (Path(__file__).with_name('golden') / 'atomic-cpu.txt').read_text()


@ti.jit
def ticket(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite], out: ti.Buffer[ti.i32, (8,), ti.WriteOnly]):
    i = ti.arange(0, 8)
    old = ti.atomic_add(x, i * 0, 1, mask=i < 5, order=ti.Relaxed, scope=ti.GPU)
    ti.store(out, i, old)


def test_duplicate_addresses_mask_and_reference_order():
    x, out = np.zeros(1, np.int32), np.full(8, -1, np.int32)
    ticket[(1,)](x, out)
    np.testing.assert_array_equal(out, [0, 1, 2, 3, 4, 0, 0, 0])
    assert x[0] == 5
    verify(ticket.tk, {}, capability=None)


def test_enum_aliases_derived_pointer_and_zero_loop():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite], COUNT: ti.Const[int] = 0):
        p = x.ptr
        for j in ti.range(0, COUNT):
            ti.atomic_add(p, 1, order=ORDER_ALIAS, scope=SCOPE_ALIAS)
    x = np.zeros(1, np.int32)
    kernel[(1,)](x)
    assert x[0] == 0 and kernel.tk.effect_summary({'COUNT': 0}).effects == ()
    kernel[(1,)](x, COUNT=3)
    assert x[0] == 3


def test_all_false_mask_old_is_positive_zero():
    @ti.jit
    def kernel(x: ti.Buffer[ti.f32, (1,), ti.ReadWrite], out: ti.Buffer[ti.f32, (1,), ti.WriteOnly]):
        old = ti.atomic_add(x, 99, ti.constant[ti.f32](1), mask=False)
        ti.store(out, 0, old)
    x, out = np.array([-3], np.float32), np.array([-1], np.float32)
    kernel[(1,)](x, out)
    assert x[0] == -3 and out.view(np.uint32)[0] == 0


def test_debug_and_runtime_bounds(monkeypatch):
    monkeypatch.setenv('TILA_SAFETY', 'warn')
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite], index: ti.i32):
        v = ti.load(x, 0)
        ti.atomic_add(x, index, 1)
    for debug in ('0', '1'):
        monkeypatch.setenv('TILA_DEBUG', debug)
        with pytest.raises((TilaError, IndexError, AssertionError)):
            kernel[(1,)](np.zeros(1, np.int32), 3)


def test_where_policy_rechecked_for_atomic(monkeypatch):
    monkeypatch.setenv('TILA_EFFECTS', 'warn')
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
        v = ti.where(False, ti.atomic_add(x, 0, 1), 0)
    x = np.zeros(1, np.int32)
    kernel[(1,)](x)
    assert x[0] == 1
    monkeypatch.setenv('TILA_EFFECTS', 'error')
    with pytest.raises(TilaError, match='TILA-EFFECT-007'):
        kernel[(1,)](x)
    assert x[0] == 1


@pytest.mark.parametrize('layout', ['negative', 'zero'])
def test_nonpositive_atomic_strides_rejected(layout):
    x = np.zeros(1, np.int32)
    view = x[::-1] if layout == 'negative' else np.lib.stride_tricks.as_strided(x, (1,), (0,))
    with pytest.raises((TilaError, TilaLaunchContractError)):
        ticket[(0,)](view, np.zeros(8, np.int32))


def test_discard_nested_and_old_value_reuse():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
        old = ti.atomic_add(x, 0, 1)
        used = old + old
        ti.atomic_add(x, 0, ti.atomic_add(x, 0, 1))
    x = np.zeros(1, np.int32)
    kernel[(3,)](x)
    assert x[0] == 21
    assert len(kernel.tk.effects) == 3
    assert isinstance(kernel.tk.body[-1], T.TAtomicStmt)
    assert len(verify_effects(kernel.tk)) == 3


def test_value_and_mask_nested_atoms_execute_in_order():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (3,), ti.ReadWrite]):
        ti.atomic_add(x, 0, ti.atomic_add(x, 1, 1), mask=ti.atomic_add(x, 2, 1) > 0)
    x = np.array([4, 7, 0], np.int32)
    kernel[(1,)](x)
    np.testing.assert_array_equal(x, [4, 8, 1])


def test_pointer_and_alias_parameters_share_storage():
    @ti.jit
    def kernel(a: ti.RWPtr[ti.u32, 1], b: ti.RWPtr[ti.u32, 1]):
        ti.atomic_add(a, 1)
        ti.atomic_add(b, 1)
    x = np.array([2**32 - 1], np.uint32)
    kernel[(1,)](x, x)
    assert x[0] == 1


def test_positive_stride_offset_view_and_multidimensional_buffer():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2, 4), ti.ReadWrite]):
        i = ti.arange(0, 4)
        ti.atomic_add(x, 1, i, 3)
    storage = np.zeros((4, 10), np.int32)
    view = storage[1:3, 1:9:2]
    kernel[(1,)](view)
    np.testing.assert_array_equal(view[1], [3]*4)
    assert storage.sum() == 12


@pytest.mark.parametrize('old,value,expected', [(2**31-1, 1, -2**31), (-2**31, -1, 2**31-1)])
def test_signed_wrap(old, value, expected):
    assert add(old, value, D.i32) == expected


@pytest.mark.parametrize('old,value', [(1., 2**-24), (np.nextafter(np.float32(0), np.float32(1)), 0.),
                                     (-np.finfo(np.float32).tiny/2, -0.), (float('inf'), float('-inf'))])
def test_float_updates_and_unmodified_old_value(old, value):
    @ti.jit
    def kernel(x: ti.Buffer[ti.f32, (1,), ti.ReadWrite], out: ti.Buffer[ti.f32, (1,), ti.WriteOnly], value: ti.f32):
        old = ti.atomic_add(x, 0, value)
        ti.store(out, 0, old)
    x, out = np.array([old], np.float32), np.zeros(1, np.float32)
    original = x.copy()
    kernel[(1,)](x, out, value)
    np.testing.assert_array_equal(out.view(np.uint32), original.view(np.uint32))
    if np.isnan(old + value):
        assert np.isnan(x[0])
    else:
        expected = np.float32(1) if old == 1 else np.float32(np.copysign(0., old))
        assert x[0].view(np.uint32) == expected.view(np.uint32)


def test_effect_metadata_summary_where_and_v2():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite], FLAG: ti.Const[bool] = False):
        if FLAG:
            v = ti.where(True, ti.atomic_add(x, 0, 1), 0)
    node, = verify_effects(kernel.tk)
    assert node.effect.atomic == T.AtomicInfo('Add', Relaxed, GPU)
    assert kernel.tk.effect_summary({'FLAG': False}).effects == ()
    assert len(kernel.tk.effect_summary({'FLAG': True}).effects) == 1
    warning, = where_warnings(kernel.tk)
    assert 'Atomic(Add)' in warning.render()
    data = effect_details(kernel.tk, {'FLAG': True})
    assert data['schema'] == 'tila.effect-details.v2'
    assert data['accesses'][0]['atomic'] == dict(op='Add', order='relaxed', scope='gpu')
    assert 'atomic_add' in kernel.tk.dump()


@pytest.mark.parametrize('corruption', ['effect', 'order', 'scope', 'dtype', 'value', 'mask'])
def test_verifier_rejects_corruption(corruption):
    kernel = ti.jit(ticket.fn).tk
    node = verify_effects(kernel)[0]
    if corruption == 'effect':
        node.effect = replace(node.effect, atomic=None)
    elif corruption in ('order', 'scope'):
        setattr(node, corruption, 'relaxed')
    elif corruption == 'dtype':
        node.vt = ti.types.ScalarT(D.f32)
    elif corruption == 'value':
        node.value = T.TLit(1., D.f32)
    else:
        node.mask = T.TLit(1, D.i32)
    with pytest.raises(TilaError):
        verify(kernel, {}, capability=None)


def test_gpu_materialize_and_lowering_fail_closed():
    from tila.lowering import Lowering
    from tila.target import SUPPORTED
    source, _ = ticket.materialize()
    assert 'sem="relaxed", scope="gpu"' in source
    assert source.count('tl.atomic_add(') == 1
    emitter = Lowering(ticket.tk)
    assert emitter.kernel_source() == source
    assert 4 in emitter.source_map.values()
    with pytest.raises(TilaError, match='TILA-TARGET-012'):
        verify(ticket.tk, {}, capability=replace(SUPPORTED, atomic_add_dtypes=()))


def test_gpu_rejection_diagnostic_golden():
    from pathlib import Path
    from tila.target import SUPPORTED
    with pytest.raises(TilaError) as failure:
        verify(simple.tk, {}, capability=replace(SUPPORTED, atomic_add_dtypes=()))
    assert str(failure.value) + '\n' == (Path(__file__).with_name('golden') / 'atomic-gpu-rejected.txt').read_text()


def test_atomic_triton_source_golden():
    from pathlib import Path
    assert simple.materialize()[0] == (Path(__file__).with_name('golden') / 'atomic.triton.py').read_text()


def test_atomic_helper_name_does_not_shadow_user_kernel():
    @ti.jit
    def _tila_atomic_add(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
        ti.atomic_add(x, 0, 1)
    source, _ = _tila_atomic_add.materialize()
    assert 'def _tila_atomic_add_(' in source
    assert '= _tila_atomic_add_(' in source


@pytest.mark.parametrize('grid', [(0,), (1,)])
def test_unaligned_binding_rejected_even_empty(grid):
    x = np.ndarray((1,), dtype=np.int32, buffer=bytearray(8), offset=1)
    with pytest.raises(TilaLaunchContractError, match='TILA-MEM-008'):
        ticket[grid](x, np.zeros(8, np.int32))
    assert not ticket.tk.runtime_aliases


def test_bounds_and_debug_false_mask():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
        ti.atomic_add(x, 99, 1, mask=False)
    x = np.zeros(1, np.int32)
    kernel[(1,)](x)
    assert x[0] == 0
    with pytest.raises(TilaError, match='TILA-BOUNDS-003'):
        @ti.jit
        def bad(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
            ti.atomic_add(x, 99, 1)


@pytest.mark.parametrize('case', ['dtype', 'access', 'string', 'value', 'rank', 'mask'])
def test_frontend_negative_cases(case):
    # Source-backed definitions keep source locations and reflection realistic.
    with pytest.raises(TilaError):
        if case == 'dtype':
            @ti.jit
            def bad(x: ti.Buffer[ti.f64, (1,), ti.ReadWrite]):
                ti.atomic_add(x, 0, 1.)
        elif case == 'access':
            @ti.jit
            def bad(x: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
                ti.atomic_add(x, 0, 1)
        elif case == 'string':
            @ti.jit
            def bad(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
                ti.atomic_add(x, 0, 1, order='relaxed')
        elif case == 'value':
            @ti.jit
            def bad(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
                ti.atomic_add(x, 0, ti.cast[ti.u32](1))
        elif case == 'rank':
            @ti.jit
            def bad(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite]):
                i = ti.arange(0, 8)
                ti.atomic_add(x, i[:, None], 1)
        else:
            @ti.jit
            def bad(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite]):
                i = ti.arange(0, 8)
                ti.atomic_add(x, i, 1, mask=ti.arange(0, 1) < 1)
