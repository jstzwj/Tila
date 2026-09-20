"""ADR-018 exact inter-program subset: pair proofs and concrete witnesses."""
from dataclasses import replace
import numpy as np
import pytest
import tila as ti
from tila.race import analyze_races, RaceConfig
from tila.solver import ProofConfig
from tila.facts import PROVEN_SAFE, PROVEN_UNSAFE, UNKNOWN


@ti.jit
def partition(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
    i = ti.program_id(0) * 4 + ti.arange(0, 4)
    ti.store(x, i, i, mask=i < 16)


@ti.jit
def collide(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
    ti.store(x, 0, 1)


def run(kernel, **kwargs):
    kwargs.setdefault('grid', (4,))
    kwargs.setdefault('bindings', {'x': np.zeros(16, np.int32)})
    return analyze_races(kernel.tk, **kwargs)


def test_partition_and_read_only_analysis():
    x = np.zeros(16, np.int32)
    result = run(partition, bindings={'x': x})
    assert result.verdict == PROVEN_SAFE
    assert result.unchecked_domains == ('IntraProgram',)
    assert x.tolist() == [0] * 16
    assert result.pairs[0].query and 'check-sat' in result.pairs[0].query


def test_self_pair_confirmed_with_two_programs_and_no_host_addresses():
    x = np.zeros(16, np.int32)
    r = run(collide, bindings={'x': x}).pairs[0]
    assert r.verdict == PROVEN_UNSAFE and r.candidate is None
    assert r.confirmed.pids[0] != r.confirmed.pids[1]
    assert r.confirmed.byte_offsets == (0, 0)
    assert str(x.ctypes.data) not in r.query


@pytest.mark.parametrize('grid', [(0,), (1,), (1, 0), (1, 1, 1)])
def test_zero_or_single_program_has_only_interprogram_guarantee(grid):
    r = run(collide, grid=grid)
    assert r.verdict == PROVEN_SAFE and r.unchecked_domains


def test_three_axis_distinctness():
    assert run(collide, grid=(1, 1, 2)).verdict == PROVEN_UNSAFE


def test_pending_binding_and_grid():
    assert run(collide, grid=None).pairs[0].reason == 'pending-grid'
    assert run(collide, bindings={}).verdict == UNKNOWN


def test_single_writer_and_complementary_program_branches():
    @ti.jit
    def single(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        if p == 0:
            ti.store(x, 0, 1)
    @ti.jit
    def branches(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        if p == 0:
            ti.store(x, 0, 1)
        else:
            ti.store(x, 0, 2)
    assert run(single).verdict == PROVEN_SAFE
    report = run(branches)
    assert report.pairs[1].verdict == PROVEN_UNSAFE


def test_return_and_rebinding_definition_identity():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        if p != 0:
            return
        ti.store(x, p, 1)
        p = 0
        ti.store(x, p, 2)
    assert run(kernel).verdict == PROVEN_SAFE


def test_const_dead_branch_and_zero_loop():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], FLAG: ti.Const[bool], COUNT: ti.Const[int]):
        if FLAG:
            ti.store(x, 0, 1)
        for j in ti.range(0, COUNT):
            ti.store(x, j, 1)
    assert run(kernel, consts={'FLAG': False, 'COUNT': 0}).verdict == PROVEN_SAFE
    result = run(kernel, consts={'FLAG': False, 'COUNT': 2})
    assert result.verdict == PROVEN_UNSAFE
    assert result.pairs[-1].confirmed.iterations
    assert run(kernel, consts={'FLAG': True, 'COUNT': 0}).verdict == PROVEN_UNSAFE


def test_alias_views_and_stride_holes():
    @ti.jit
    def kernel(a: ti.Buffer[ti.i32, (16,), ti.ReadOnly], x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        v = ti.load(a, p)
        ti.store(x, 3 - p, v)
    base = np.zeros(32, np.int32)
    assert run(kernel, bindings={'a': base[::2], 'x': base[1::2]}).verdict == PROVEN_SAFE
    assert run(kernel, bindings={'a': base[:16], 'x': base[:16]}).verdict == PROVEN_UNSAFE
    assert run(kernel, bindings={'a': base[:16], 'x': base[16:]}).verdict == PROVEN_SAFE


def test_different_dtype_partial_byte_overlap():
    @ti.jit
    def kernel(a: ti.Buffer[ti.i8, (16,), ti.WriteOnly], x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        ti.store(a, p, ti.cast[ti.i8](1))
        ti.store(x, p, 1)
    base = np.zeros(16, np.int32)
    report = run(kernel, bindings={'a': base.view(np.int8)[:16], 'x': base})
    assert report.pairs[1].verdict == PROVEN_UNSAFE


def test_pointer_definition_and_view_offset():
    @ti.jit
    def kernel(x: ti.WritePtr[ti.i32, 16]):
        p = x + ti.program_id(0)
        ti.store(p, 1)
    assert run(kernel).verdict == PROVEN_SAFE


def test_compatible_atomic_and_mixed_plain_access():
    @ti.jit
    def atomic(x: ti.Buffer[ti.i32, (16,), ti.ReadWrite]):
        ti.atomic_add(x, 0, 1)
    @ti.jit
    def mixed(x: ti.Buffer[ti.i32, (16,), ti.ReadWrite]):
        ti.atomic_add(x, 0, 1)
        ti.store(x, ti.program_id(0), 1)
    assert run(atomic).pairs[0].reason == 'compatible-atomic-pair'
    assert run(mixed).pairs[1].verdict == PROVEN_UNSAFE


def test_indirect_address_and_opaque_guard_remain_unknown(monkeypatch):
    monkeypatch.setenv('TILA_SAFETY', 'warn')
    @ti.jit
    def indirect(x: ti.Buffer[ti.i32, (16,), ti.ReadWrite]):
        i = ti.load(x, 0)
        ti.store(x, i, 1)
    @ti.jit
    def guarded(x: ti.Buffer[ti.i32, (16,), ti.ReadWrite]):
        v = ti.load(x, 0)
        if v > 0:
            ti.store(x, 0, 1)
    assert run(indirect).verdict == UNKNOWN
    report = run(guarded)
    assert report.verdict == UNKNOWN
    assert any(r.candidate for r in report.pairs)
    assert not any(r.confirmed for r in report.pairs)


def test_unsafe_invalid_prior_access_does_not_confirm_conflict():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.ReadWrite]):
        v = ti.unsafe_load(x, 100)
        ti.store(x, 0, 1)
    r = run(kernel)
    assert r.verdict == UNKNOWN
    assert r.pairs[-1].reason == 'access-validity-not-confirmed'


def test_assume_does_not_hide_conflict():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        ti.assume(p == 0)
        ti.store(x, 0, 1)
    result = run(kernel)
    assert result.verdict == PROVEN_UNSAFE
    assert all(d.kind != 'UserAssumption' for r in result.pairs for d in r.dependencies)


def test_narrowing_and_intermediate_wrap_are_not_mathematical_affine():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        i = ti.cast[ti.u8](p * 256)
        ti.store(x, ti.cast[ti.i32](i), 1)
    assert run(kernel).verdict == PROVEN_UNSAFE


@pytest.mark.parametrize('field', ['timeout_ms', 'rlimit', 'total_ms', 'max_queries', 'max_nodes', 'max_depth', 'max_query_bytes'])
def test_budget_exhaustion_never_claims_safe(field):
    r = run(collide, config=RaceConfig(replace(ProofConfig(), **{field: 0})))
    assert r.verdict == UNKNOWN and r.incomplete_reason and r.unchecked_pairs


def test_pair_budget_includes_unchecked_and_has_no_cache():
    r = run(collide, config=RaceConfig(max_pairs=0))
    assert r.verdict == UNKNOWN and r.unchecked_pairs == 1
    assert run(collide).verdict == PROVEN_UNSAFE


def test_invalid_metadata_and_tir_fail_closed():
    with pytest.raises(ValueError, match='dtype'):
        run(collide, bindings={'x': np.zeros(16, np.float32)})
    with pytest.raises(ValueError, match='grid'):
        run(collide, grid=(True,))
    from copy import deepcopy
    from tila.errors import TilaError
    tk = deepcopy(collide.tk)
    tk.body[0].effect = None
    with pytest.raises(TilaError):
        analyze_races(tk, grid=(2,))


def test_torch_cpu_binding_metadata():
    import torch
    assert run(collide, bindings={'x': torch.zeros(16, dtype=torch.int32)}).verdict == PROVEN_UNSAFE
    with pytest.raises(ValueError, match='real strided'):
        run(collide, bindings={'x': torch.empty(16, dtype=torch.int32, device='meta')})


def test_eager_where_and_outer_false_mask_retain_inner_atomic():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.ReadWrite]):
        v = ti.where(False, ti.atomic_add(x, 0, 1), 0)
        ti.store(x, 0, v)
        ignored = ti.load(x, ti.atomic_add(x, 0, 1), mask=False)
    report = run(kernel)
    assert report.verdict == PROVEN_UNSAFE
    assert any(r.reason == 'unreachable-access' for r in report.pairs)
    assert any(r.confirmed for r in report.pairs)


def test_shared_scalar_binding_is_not_fresh_per_program():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], FLAG: ti.bool):
        if FLAG:
            ti.store(x, 0, 1)
    assert run(kernel, scalars={'FLAG': False}).verdict == PROVEN_SAFE
    assert run(kernel, scalars={'FLAG': True}).verdict == PROVEN_UNSAFE
    assert run(kernel).verdict == UNKNOWN
    @ti.jit
    def converted(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        i = ti.cast[ti.i32](ti.cast[ti.bool](p))
        ti.store(x, i, 1)
    assert run(converted).verdict == PROVEN_UNSAFE


def test_negative_stride_and_refinement_are_not_trusted():
    assert run(collide, bindings={'x': np.zeros(16, np.int32)[::-1]}).verdict == UNKNOWN
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], STEP: ti.Const[int, ti.Positive]):
        ti.store(x, 0, 1)
    with pytest.raises(ValueError, match='refinement'):
        run(kernel, consts={'STEP': 0})


def test_small_affine_domains_against_event_enumeration():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], STEP: ti.Const[int]):
        i = ti.program_id(0) * STEP + ti.arange(0, 4)
        ti.store(x, i, i, mask=(i >= 0) & (i < 16))
    # Fixed seed determines order only; every listed small domain is exhausted.
    cases = [(grid, step) for grid in range(2, 5) for step in range(-2, 7)]
    np.random.default_rng(204018).shuffle(cases)
    for grid, step in cases:
        events = [(p, p * step + lane) for p in range(grid) for lane in range(4)
                  if 0 <= p * step + lane < 16]
        conflict = any(p != q and a == b for p, a in events for q, b in events)
        r = run(kernel, grid=(grid,), consts={'STEP': step})
        assert r.verdict == (PROVEN_UNSAFE if conflict else PROVEN_SAFE), (grid, step, r)
        for pair in r.pairs:
            if pair.confirmed:
                w = pair.confirmed
                assert all((pid[0], offset // 4) in events for pid, offset in zip(w.pids, w.byte_offsets))


def test_replay_query_and_analysis_do_not_mutate_kernel():
    from tila.solver import z3_module
    before = collide.tk.dump(), list(collide.tk.runtime_aliases)
    result = run(collide).pairs[0]
    z = z3_module()
    solver = z.Solver()
    solver.from_string(result.query)
    assert solver.check() == z.sat
    assert (collide.tk.dump(), collide.tk.runtime_aliases) == before


def test_loop_carried_address_and_early_return_are_not_confirmed():
    @ti.jit
    def carried(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        k = 0
        for j in ti.range(0, 2):
            ti.store(x, k, 1)
            k = k + 1
    @ti.jit
    def early(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        for j in ti.range(0, 2):
            if j == 0:
                return
            ti.store(x, 0, 1)
    assert run(carried).verdict == UNKNOWN
    assert run(early).verdict == UNKNOWN
    assert not any(r.confirmed for r in run(early).pairs)


def test_loop_partition_and_distinct_iterations():
    @ti.jit
    def safe(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        for j in ti.range(0, 4):
            ti.store(x, p * 4 + j, 1)
    @ti.jit
    def shifted(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly]):
        p = ti.program_id(0)
        for j in ti.range(0, 4):
            ti.store(x, p + j, 1)
    assert run(safe).verdict == PROVEN_SAFE
    w = run(shifted).pairs[0].confirmed
    assert w and w.iterations[0][0][1] != w.iterations[1][0][1]


def test_i32_intermediate_overflow_before_narrowing():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], n: ti.i32):
        intermediate = n + 1
        i = ti.cast[ti.i32](ti.cast[ti.u8](intermediate))
        ti.store(x, i, 1)
    report = run(kernel, scalars={'n': 2147483647})
    assert report.verdict == PROVEN_UNSAFE
    assert report.pairs[0].confirmed.byte_offsets == (0, 0)
    assert report.pairs[0].confirmed.inputs == (('n', 2147483647),)
    assert report.pairs[0].confirmed.grid == (4, 1, 1)


def test_solver_resource_limit_is_unknown():
    report = run(partition, config=RaceConfig(replace(ProofConfig(), rlimit=1)))
    assert report.verdict == UNKNOWN and report.incomplete_reason == 'solver-unknown'


def test_multi_axis_access_stays_unknown():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4, 4), ti.WriteOnly]):
        ti.store(x, (0, 0), 1)
    report = analyze_races(kernel.tk, grid=(2,), bindings={'x': np.zeros((4, 4), np.int32)})
    assert report.verdict == UNKNOWN
