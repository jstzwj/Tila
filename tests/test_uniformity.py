"""ADR-019 internal value/control analysis: no synchronization API or launch gate."""
from dataclasses import replace, FrozenInstanceError

import pytest
import tila as ti
from tila import tir as T, dtypes as D, types as TY, predicates as P
from tila.effect_ir import OPERANDS, bind_effects
from tila.errors import TilaError
from tila.uniformity import (Level, UniformityConfig, analyze_uniformity,
                             verify_uniformity, TRANSFERS)

L, G, V, U = Level


def values(summary, name, kind='definition'):
    return [f for f in summary.facts if f.site.startswith('@') and f.definition
            and f.definition.name == name and f.definition.kind == kind]


def value(summary, name):
    found = values(summary, name)
    assert len(found) == 1
    return found[0]


@ti.jit
def basic(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite], n: ti.i32, B: ti.Const[int] = 4):
    c = 3
    scalar = n
    size = B
    programs = ti.num_programs(0)
    pid = ti.program_id(0)
    lanes = ti.arange(0, B)
    offset = pid + lanes
    zeroed = lanes * 0
    loaded = ti.load(x, 0, mask=False, other=0)
    old = ti.atomic_add(x, 0, 1)
    reused = loaded + loaded
    z = ti.zeros((4,), ti.i32)
    a = ti.cast[ti.i8](pid)


def test_layer_sources_and_value_reuse():
    result = analyze_uniformity(basic.tk)
    for name in ('c', 'scalar', 'size', 'programs', 'z'):
        assert value(result, name).value_level == L
    for name in ('pid', 'a'):
        assert value(result, name).value_level == G
    for name in ('lanes', 'offset', 'zeroed'):
        assert value(result, name).value_level == V
    for name in ('loaded', 'old', 'reused'):
        assert value(result, name).value_level == U
    assert len(result.accesses) == 2  # Reusing a load is not another evaluation.
    assert result.accesses[0].mask is P.FALSE
    assert value(result, 'loaded').control.selection_level == L
    assert result.requires_defined_execution and result.stage == 'symbolic'
    assert all(f.source == P.STATIC for f in result.facts)
    verify_uniformity(basic.tk, result)


def test_transfer_coverage_and_immutable_read_only_summary():
    assert set(TRANSFERS) == set(OPERANDS)
    before = basic.tk.dump()
    result = analyze_uniformity(basic.tk)
    with pytest.raises(FrozenInstanceError):
        result.facts[0].reason = 'forged'
    assert basic.tk.dump() == before
    assert all(i in {f.site for f in result.facts} for f in result.facts for i in f.inputs)


def test_branch_value_selector_and_control_reconvergence():
    @ti.jit
    def kernel():
        p = ti.program_id(0)
        if p == 0:
            a = 1
        else:
            a = 2
        out = a
        constant = 3
    result = analyze_uniformity(kernel.tk)
    assert all(f.value_level == L for f in values(result, 'a'))
    assert all(f.control.selection_level == G for f in values(result, 'a'))
    assert values(result, 'a', 'merge')[0].value_level == G
    assert value(result, 'out').value_level == G
    assert value(result, 'constant').control.selection_level == L
    assert result.exit_control.launch_participation == 'Full'


def test_early_return_preserves_remaining_participation():
    @ti.jit
    def kernel():
        if ti.program_id(0) == 0:
            return
        constant = 1
    result = analyze_uniformity(kernel.tk)
    fact = value(result, 'constant')
    assert fact.value_level == L
    assert fact.control.launch_participation == 'Conditional'
    assert fact.control.program_participation == 'Full'
    assert fact.control.returned and fact.control.reachability == 'Unknown'


def test_load_dependent_return_keeps_scalar_control_unknown():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadOnly]):
        if ti.load(x, 0) > 0:
            return
        a = 1
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'a').value_level == L
    assert value(result, 'a').control.program_participation == 'Unknown'


def test_two_returns_and_dead_successor_do_not_fabricate_uniform_control():
    @ti.jit
    def kernel(flag: ti.bool):
        if flag:
            return
        else:
            return
    tk = ti.jit(kernel.fn).tk
    tk.body.append(T.TAssign('dead', T.TLit(1, D.i32)))
    bind_effects(tk)
    result = analyze_uniformity(tk)
    assert value(result, 'dead').control.reachability == 'Unreachable'
    assert result.exit_control.program_participation == 'None'


@ti.jit
def staged(FLAG: ti.Const[bool]):
    if FLAG:
        v = ti.program_id(0)
    else:
        v = 0
    result = v


@pytest.mark.parametrize('binding,expected', [(None, G), ({'FLAG': True}, G), ({'FLAG': False}, L)])
def test_const_branch_selection_and_symbolic_isolation(binding, expected):
    result = analyze_uniformity(staged.tk, binding)
    assert value(result, 'result').value_level == expected
    verify_uniformity(staged.tk, result, binding)
    if binding is not None:
        dead = next(f for f in values(result, 'v')
                    if ('/else/' if binding['FLAG'] else '/then/') in f.site)
        assert dead.control.reachability == 'Unreachable'
    assert value(analyze_uniformity(staged.tk), 'result').value_level == G


@pytest.mark.parametrize('binding', [{'FLAG': 1}, {'FLAG': 0}, {'unknown': True}])
def test_const_domain_is_exact(binding):
    with pytest.raises((ValueError, TilaError)):
        analyze_uniformity(staged.tk, binding)


def test_eager_where_preserves_all_effects_and_selected_value_only():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        a = ti.where(False, ti.load(x, 0), 1)
        b = ti.where(ti.program_id(0) == 0, 1, 2)
        p = ti.program_id(0)
        same = ti.where(ti.load(x, 1) > 0, p, p)
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'a').value_level == L
    assert value(result, 'b').value_level == G
    assert value(result, 'same').value_level == G
    assert len(result.accesses) == 2
    assert all(a.path is P.TRUE for a in result.accesses)


def test_reduction_reshape_and_boolean_reduction():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        lane = ti.arange(0, 4)
        reshaped = ti.reshape(lane, (2, 2))
        partial = ti.sum(reshaped, 0)
        full = ti.sum(lane, 0)
        mask = lane > 0
        any_value = mask.any()
        loaded = ti.load(x, lane)
        unknown = ti.sum(loaded, 0)
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'reshaped').value_level == V
    assert value(result, 'partial').value_level == V
    assert value(result, 'full').value_level == G
    assert value(result, 'any_value').value_level == G
    assert value(result, 'unknown').value_level == U


def test_definition_rebinding_is_not_name_based():
    @ti.jit
    def kernel():
        a = ti.program_id(0)
        before = a
        a = 0
        after = a
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'before').value_level == G
    assert value(result, 'after').value_level == L
    assert {f.value_level for f in values(result, 'a')} == {L, G}


@ti.jit
def loop(N: ti.Const[int]):
    a = 0
    for j in ti.range(0, N):
        a = ti.program_id(0)
        k = j
    result = a


@pytest.mark.parametrize('count,expected', [(0, L), (1, G), (3, G)])
def test_zero_loop_exit_and_fixed_point(count, expected):
    result = analyze_uniformity(loop.tk, {'N': count})
    assert not result.incomplete_reason
    assert value(result, 'result').value_level == expected
    assert values(result, 'j', 'induction')[0].value_level == L
    assert value(result, 'k').control.loops
    if count == 0:
        assert value(result, 'k').control.reachability == 'Unreachable'
    verify_uniformity(loop.tk, result, {'N': count})


def test_runtime_loop_bounds_and_zero_path():
    @ti.jit
    def kernel(n: ti.i32):
        a = 0
        for j in ti.range(0, n):
            a = ti.program_id(0)
        out = a
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'out').value_level == G
    exit_fact = next(f for f in values(result, 'a', 'loop') if '/exit/' in f.site)
    assert len(exit_fact.inputs) >= 3  # Entry, body and trip count dependencies.


def test_program_dependent_loop_induction_and_nested_carries():
    @ti.jit
    def kernel():
        a = 0
        for j in ti.range(0, ti.program_id(0)):
            for k in ti.range(0, 2):
                a = a + j + k
        out = a
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'out').value_level == G
    assert values(result, 'j', 'induction')[0].value_level == G
    assert all(i in {f.site for f in result.facts} for f in result.facts for i in f.inputs)


def test_loop_return_and_memory_bound_do_not_claim_convergence():
    @ti.jit
    def early():
        for j in ti.range(0, 3):
            if j == 0:
                return
        after = 1
    result = analyze_uniformity(early.tk)
    assert value(result, 'after').value_level == L
    assert value(result, 'after').control.program_participation == 'Unknown'
    @ti.jit
    def bound(x: ti.Buffer[ti.i32, (1,), ti.ReadOnly]):
        for j in ti.range(0, ti.load(x, 0)):
            a = j
        after = 1
    result = analyze_uniformity(bound.tk)
    assert values(result, 'j', 'induction')[0].value_level == U
    assert value(result, 'after').control.program_participation == 'Unknown'


def test_assume_does_not_strengthen_or_hide_nested_reads():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadOnly]):
        loaded = ti.load(x, 0)
        ti.assume(ti.program_id(0) == 0)
        p = ti.program_id(0)
    # The source assume syntax deliberately rejects memory predicates. Test
    # nested operand traversal in verified internal TIR without widening it.
    tk = kernel.tk
    tk.body[0] = T.TAssume(T.TBin(TY.ScalarT(D.bool_), '>', tk.body[0].value, T.TLit(0, D.i32)))
    bind_effects(tk)
    result = analyze_uniformity(tk)
    assert value(result, 'p').value_level == G
    assert value(result, 'p').control.selection_level == L
    assert len(result.accesses) == 1


def test_floating_operation_and_dot_are_explicitly_unsupported():
    @ti.jit
    def kernel(x: ti.f32):
        constant = ti.constant[ti.f32](1.0)
        y = x + constant
        z = ti.cast[ti.i32](x)
        a = ti.zeros((16, 16), ti.f16)
        d = ti.dot(a, a)
        partial = ti.sum(a, 0)
        lane = ti.zeros((4,), ti.f32)
        scalar_reduce = ti.sum(lane, 0)
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'constant').value_level == L
    for name in ('y', 'z', 'd', 'partial'):
        assert value(result, name).value_level == U
    assert value(result, 'scalar_reduce').value_level == G
    assert any(f.reason == 'floating-operation-contract-unavailable' for f in result.facts)


@pytest.mark.parametrize('field', ['max_nodes', 'max_depth', 'max_iterations'])
def test_exhaustion_discards_partial_strong_guarantees(field):
    config = replace(UniformityConfig(), **{field: 0})
    result = analyze_uniformity(basic.tk, config=config)
    assert result.incomplete_reason
    assert all(f.value_level in (None, U) for f in result.facts)
    assert result.exit_control.program_participation == 'Unknown'
    verify_uniformity(basic.tk, result, config=config)
    assert value(analyze_uniformity(basic.tk), 'pid').value_level == G


@pytest.mark.parametrize('bad', [-1, True, 1.0])
def test_invalid_budget(bad):
    with pytest.raises(ValueError):
        UniformityConfig(max_nodes=bad)


@pytest.mark.parametrize('mutation', ['missing', 'level', 'edge', 'control', 'consts'])
def test_verifier_rejects_forged_or_stale_summary(mutation):
    result = analyze_uniformity(basic.tk)
    fact = value(result, 'offset')
    if mutation == 'missing':
        bad = replace(result, facts=tuple(f for f in result.facts if f is not fact))
    elif mutation == 'consts':
        bad = replace(result, consts=(('B', 8),))
    else:
        corrupt = (replace(fact, value_level=L) if mutation == 'level' else
                   replace(fact, inputs=()) if mutation == 'edge' else
                   replace(fact, control=replace(fact.control, program_participation='Warp')))
        bad = replace(result, facts=tuple(corrupt if f is fact else f for f in result.facts))
    with pytest.raises(TilaError, match='uniformity summary'):
        verify_uniformity(basic.tk, bad)


def test_invalid_definition_effect_and_unknown_node_are_not_unknown_analysis():
    tk = ti.jit(basic.fn).tk
    name = tk.body[1].value
    name.definition = replace(name.definition, site='wrong')
    with pytest.raises(TilaError, match='definition'):
        analyze_uniformity(tk)
    tk = ti.jit(basic.fn).tk
    tk.body[8].value.effect = None
    with pytest.raises(TilaError, match='effect'):
        analyze_uniformity(tk)
    class NewNode(T.TStmt):
        pass
    tk.body.append(NewNode())
    with pytest.raises(TilaError):
        analyze_uniformity(tk, config=UniformityConfig(max_nodes=0))


def test_pointer_address_and_expand_keep_value_dependencies():
    @ti.jit
    def kernel(p: ti.ReadPtr[ti.i32, 8], x: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):
        base = x.ptr
        offset = p + ti.program_id(0)
        lane = ti.arange(0, 4)
        expanded = lane[:, None]
        loaded = ti.load(offset)
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'base').value_level == L
    assert value(result, 'offset').value_level == G
    assert value(result, 'expanded').value_level == V
    assert value(result, 'loaded').value_level == U


@pytest.mark.parametrize('count', [0, 1, 3])
def test_zero_loop_return_does_not_poison_live_exit(count):
    @ti.jit
    def kernel(N: ti.Const[int]):
        for j in ti.range(0, N):
            if j == 0:
                return
        after = 1
    result = analyze_uniformity(kernel.tk, {'N': count})
    control = value(result, 'after').control
    assert control.program_participation == ('Full' if count == 0 else 'Unknown')
    assert not control.returned if count == 0 else control.returned


def test_unknown_loop_inside_branch_survives_reconvergence():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadOnly], flag: ti.bool):
        if flag:
            for j in ti.range(0, ti.load(x, 0)):
                a = j
        after = 1
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'after').control.program_participation == 'Unknown'
    assert value(result, 'after').control.iteration_sources


def test_fixed_point_propagates_across_mutually_carried_definitions():
    @ti.jit
    def kernel():
        a = 0
        b = 0
        for j in ti.range(0, 3):
            a = b
            b = ti.program_id(0)
        out = a
    result = analyze_uniformity(kernel.tk)
    assert value(result, 'out').value_level == G
    limited = analyze_uniformity(kernel.tk, config=UniformityConfig(max_iterations=2))
    assert limited.incomplete_reason == 'fixed-point-budget'
    assert value(limited, 'out').value_level == U


def test_binding_summary_cannot_be_verified_against_another_const():
    result = analyze_uniformity(staged.tk, {'FLAG': True})
    with pytest.raises(TilaError, match='uniformity summary'):
        verify_uniformity(staged.tk, result, {'FLAG': False})


def test_transfer_registry_drift_fails_closed(monkeypatch):
    monkeypatch.delitem(TRANSFERS, T.TWhere)
    with pytest.raises(TilaError, match='transfer coverage'):
        analyze_uniformity(basic.tk)
