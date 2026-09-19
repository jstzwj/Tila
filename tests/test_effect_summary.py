"""May-effect control flow, independent masks, and conservative loop exits."""
from dataclasses import replace
import numpy as np
import pytest
import tila as ti
from tila import tir as T, dtypes as D, predicates as P
from tila.effect_ir import bind_effects
from tila.effect_summary import summarize_effects, ConditionSource
from tila.errors import TilaError, TilaLaunchContractError


def live(summary):
    return [a for a in summary.accesses if a.may_access]


@ti.jit
def choose(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], FLAG: ti.Const[bool] = True):
    if FLAG:
        ti.store(out, 0, 1)
    else:
        ti.store(out, 0, 2)


def test_symbolic_and_specialized_are_separate_and_read_only():
    assert len(choose.tk.effects) == 2
    first = choose.tk.effect_summary({'FLAG': True})
    second = choose.tk.effect_summary({'FLAG': False})
    assert first.stage == second.stage == 'specialized'
    assert len(first.effects) == len(second.effects) == 1
    assert live(first)[0].effect.site_id != live(second)[0].effect.site_id
    assert choose.tk.effect_summary().stage == 'symbolic'
    assert len(choose.tk.effects) == 2
    with pytest.raises(AttributeError):
        choose.tk.effects.append(T.TEffect('Write', choose.tk.buffers[0].region_id))
    with pytest.raises(AttributeError):
        choose.tk.effects = ()
    with pytest.raises(ValueError):
        choose.tk.effect_summary({'FLAG': 1})
    with pytest.raises(ValueError):
        choose.tk.effect_summary({'runtime_scalar': 1})
    assert len(choose.tk.effects) == 2


def test_return_dominates_later_access_but_condition_load_survives():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.ReadWrite]):
        if ti.load(x, 0) > 0:
            return
        ti.store(x, 1, 3)
    read, write = live(kernel.tk.effect_summary())
    assert read.path is P.TRUE
    assert write.path.op == 'not'
    assert isinstance(write.path.args[0].atom, ConditionSource)
    assert read.mask is write.mask is P.TRUE
    for value in (0, 1):
        x = np.array([value, 0], np.int32)
        kernel[(1,)](x)
        assert x[1] == (3 if value == 0 else 0)


def test_both_returns_remove_continuation_in_constructed_ir():
    branch = T.TIf(T.TName('FLAG'), [T.TReturn()], [T.TReturn()])
    store = choose.tk.body[0].then_body[0]
    kernel = replace(choose.tk, body=[branch, replace(store, effect=None)])
    bind_effects(kernel)
    assert kernel.effects == ()
    assert kernel.effect_summary().accesses[0].path is P.FALSE


def test_mask_false_keeps_nested_other_and_where_reads():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):
        v = ti.load(x, 0, mask=False, other=ti.load(x, 1))
        w = ti.where(False, ti.load(x, 2), ti.load(x, 3))
    summary = kernel.tk.effect_summary()
    assert len(summary.accesses) == 4 and len(summary.effects) == 3
    inner, outer, a, b = summary.accesses
    assert all(s.path is P.TRUE for s in summary.accesses)
    assert inner.mask is a.mask is b.mask is P.TRUE
    assert outer.mask is P.FALSE


def test_path_and_mask_are_not_interchangeable():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], flag: ti.bool):
        if flag:
            i = ti.arange(0, 8)
            a = ti.load(x, i, mask=i < 4)
    access, = live(kernel.tk.effect_summary())
    assert access.path.op == access.mask.op == 'unknown'
    assert access.path is not access.mask
    assert access.path.shape == () and len(access.mask.shape) == 1
    assert access.loops == ()


def test_assume_false_and_unsafe_do_not_erase_effects():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite], n: ti.i32):
        ti.assume((n < 0) & (n >= 0))
        ti.unsafe_store(x, 0, ti.unsafe_load(x, 1))
    summary = kernel.tk.effect_summary()
    assert [e.op for e in summary.effects] == ['Read', 'Write']
    assert all(a.path is P.TRUE for a in summary.accesses)
    # Frontend rejects literal assume(False); a transformed TIR must still not
    # interpret its false predicate as a control-flow branch.
    kernel.tk.body[0].pred = T.TLit(False, D.bool_)
    bind_effects(kernel.tk)
    assert len(kernel.tk.effects) == 2


@pytest.mark.parametrize('count', [0, 1, 3])
def test_zero_loop_and_guaranteed_return(count):
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.ReadWrite], COUNT: ti.Const[int] = 0):
        for j in ti.range(0, COUNT):
            ti.store(x, 0, 1)
            return
        ti.store(x, 1, 2)
    summary = kernel.tk.effect_summary({'COUNT': count})
    inside, after = summary.accesses
    assert bool(inside.may_access) == (count > 0)
    assert bool(after.may_access) == (count == 0)
    assert len(inside.loops) == 1 and after.loops == ()
    assert inside.loops[0].induction.kind == 'induction'
    x = np.zeros(2, np.int32)
    kernel[(1,)](x, COUNT=count)
    np.testing.assert_array_equal(x, [1, 0] if count else [0, 2])


def test_unknown_loop_return_keeps_zero_iteration_exit():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.WriteOnly], n: ti.i32):
        for j in ti.range(0, n):
            ti.store(x, 0, 1)
            return
        ti.store(x, 1, 2)
    a, b = live(kernel.tk.effect_summary())
    assert a.path is a.loops[0].may_enter
    assert b.path.op == 'not' and b.path.args[0] is a.path


def test_loop_carried_boolean_does_not_use_entry_constant_or_leak_exit_guard():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (3,), ti.WriteOnly], flag: ti.bool):
        active = False
        for j in ti.range(0, 2):
            if active:
                ti.store(x, 0, 1)
            active = flag
        if active:
            ti.store(x, 1, 2)
    first, second = live(kernel.tk.effect_summary())
    assert first.path is not P.FALSE and second.path is not P.FALSE
    assert first.path is not second.path
    assert first.path.atom.definition.kind == 'loop'
    assert first.path.atom.loops and second.path.atom.loops == ()


def test_conditional_loop_return_has_unknown_continuation_not_iteration_condition():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (3,), ti.ReadWrite]):
        for j in ti.range(0, 2):
            if ti.load(x, 0) > 0:
                return
        ti.store(x, 2, 3)
    summary = kernel.tk.effect_summary()
    after = live(summary)[-1]
    assert after.path.op == 'unknown'
    assert after.path.atom.site.endswith('/continuation')
    assert after.loops == ()


def test_runtime_integer_overflow_is_not_folded_as_unbounded_const():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
        a = ti.cast[ti.i8](127)
        b = a + 1
        if b < 0:
            ti.store(x, 0, 9)
    assert len(kernel.tk.effects) == 1
    x = np.zeros(1, np.int32)
    kernel[(1,)](x)
    assert x[0] == 9


def test_recompute_after_rebind_no_parallel_list_or_stale_cache():
    kernel = ti.jit(choose.fn).tk
    assert len(kernel.effects) == 2
    kernel.body = []
    bind_effects(kernel)
    assert kernel.effects == ()
    kernel = ti.jit(choose.fn).tk
    kernel.body[0].then_body[0].effect = None
    with pytest.raises(TilaError, match='effect metadata'):
        summarize_effects(kernel)


def test_const_branch_value_merge_and_return():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.WriteOnly], FLAG: ti.Const[bool]):
        active = False
        if FLAG:
            active = True
        if active:
            ti.store(x, 0, 1)
    assert len(kernel.tk.effect_summary({'FLAG': True}).effects) == 1
    assert len(kernel.tk.effect_summary({'FLAG': False}).effects) == 0
    assert len(kernel.tk.effects) == 1


def test_loop_control_reads_are_outside_iteration_context():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.ReadWrite]):
        for j in ti.range(0, ti.load(x, 0)):
            ti.store(x, 1, 2)
    bound, body = live(kernel.tk.effect_summary())
    assert bound.path is P.TRUE and bound.loops == ()
    assert len(body.loops) == 1 and body.loops[0].may_enter.op == 'unknown'


def test_nested_loops_keep_separate_contexts_and_zero_inner_loop():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.WriteOnly], N: ti.Const[int]):
        for i in ti.range(0, 2):
            for j in ti.range(0, N):
                ti.store(x, 0, 1)
            ti.store(x, 1, 2)
    inner, outer = kernel.tk.effect_summary({'N': 0}).accesses
    assert len(inner.loops) == 2 and len(outer.loops) == 1
    assert inner.path is P.FALSE and outer.path is P.TRUE
    assert len(kernel.tk.effect_summary({'N': 1}).effects) == 2


def test_unknown_conditions_remain_distinct_after_rebinding():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.ReadWrite]):
        flag = ti.load(x, 0) > 0
        if flag:
            ti.store(x, 1, 1)
        flag = ti.load(x, 1) > 0
        if flag:
            ti.store(x, 0, 2)
    writes = [a for a in live(kernel.tk.effect_summary()) if a.effect.kind == 'Write']
    assert writes[0].path is not writes[1].path
    assert writes[0].path.atom.definition.site != writes[1].path.atom.definition.site


def test_failed_and_empty_launch_cannot_change_static_summary():
    kernel = ti.jit(choose.fn)
    before = kernel.tk.effects
    kernel[(0,)](np.zeros(1, np.int32), FLAG=False)
    with pytest.raises(TilaLaunchContractError):
        kernel[(1,)](np.zeros(1, np.int32), FLAG=1)
    assert kernel.tk.effects == before
    assert len(kernel.tk.effect_summary({'FLAG': False}).effects) == 1
