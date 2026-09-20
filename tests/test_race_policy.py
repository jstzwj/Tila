"""ADR-018 launch gates, same-store lanes, audit and binding/cache isolation."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import tila as ti
from tila import runtime
from tila.errors import TilaError
from tila.facts import PROVEN_SAFE, PROVEN_UNSAFE, UNKNOWN
from tila.race import RaceConfig, analyze_races
from tila.solver import ProofConfig
from tila import race_policy as RP
from race_fixture import collision, duplicate, partition, copy_reverse, matrix


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setenv('TILA_RACE', 'warn')
    RP._CACHE.clear()


def blocked(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('race rejection must happen before _execute')
    monkeypatch.setattr(runtime._Launcher, '_execute', fail)


@pytest.mark.parametrize('policy', ['warn', 'error'])
@pytest.mark.parametrize('safety', ['strict', 'warn'])
@pytest.mark.parametrize('kernel,grid,domain', [(collision, (2,), 'InterProgram'), (duplicate, (1,), 'IntraProgram')])
def test_confirmed_conflicts_never_execute(policy, safety, kernel, grid, domain, monkeypatch):
    monkeypatch.setenv('TILA_RACE', policy)
    monkeypatch.setenv('TILA_SAFETY', safety)
    blocked(monkeypatch)
    x = np.zeros(8, np.int32)
    with pytest.raises(TilaError, match='TILA-RACE-001') as failure:
        kernel[grid](x)
    assert any(p.domain == domain and p.confirmed for p in failure.value.race_report.pairs)
    assert kernel.last_race_report is None and kernel.last_race_details is None
    assert np.all(x == 0)


def test_off_is_audited_and_policy_change_ignores_compiled_cache(monkeypatch):
    x = np.zeros(8, np.int32)
    monkeypatch.setenv('TILA_RACE', 'off')
    collision[(1,)](x)
    assert collision.last_race_report.verdict == UNKNOWN
    assert collision.last_race_report.suppressions
    assert json.loads(collision.last_race_details)['stage'] == 'disabled'
    collision._kern_cache['test-warm'] = object()
    monkeypatch.setenv('TILA_RACE', 'warn')
    blocked(monkeypatch)
    with pytest.raises(TilaError, match='TILA-RACE-001'):
        collision[(2,)](x)
    collision._kern_cache.pop('test-warm')


def test_unknown_warn_and_error_and_no_cache(monkeypatch):
    x = np.zeros((2, 2), np.int32)
    with pytest.warns(RuntimeWarning, match='TILA-RACE-002'):
        matrix[(1,)](x)
    assert x[0, 0] == 4 and matrix.last_race_report.verdict == UNKNOWN
    assert not RP._CACHE
    monkeypatch.setenv('TILA_RACE', 'error')
    blocked(monkeypatch)
    with pytest.raises(TilaError, match='TILA-RACE-002'):
        matrix[(1,)](x)
    assert matrix.last_race_report is None


def test_cache_hit_then_const_and_grid_change(monkeypatch):
    x = np.zeros(8, np.int32)
    monkeypatch.setenv('TILA_RACE', 'error')
    partition[(2,)](x)
    assert partition.last_race_report.cache_status == 'miss'
    partition[(2,)](x)
    assert partition.last_race_report.cache_status == 'hit'
    blocked(monkeypatch)
    with pytest.raises(TilaError, match='TILA-RACE-001'):
        partition[(2,)](x, STEP=0)
    # Even the confirmed negative cache must reapply today's policy.
    with pytest.raises(TilaError) as error:
        partition[(2,)](x, STEP=0)
    assert error.value.race_report.cache_status == 'hit'


def test_alias_view_and_stride_rebinding_do_not_reuse_safe_result(monkeypatch):
    base = np.arange(12, dtype=np.int32)
    a, x = base[:4], base[8:]
    copy_reverse[(4,)](a, x)
    copy_reverse[(4,)](a, x)
    assert copy_reverse.last_race_report.cache_status == 'hit'
    copy_reverse[(4,)](base[::2][:4], base[1::2][:4])
    assert copy_reverse.last_race_report.cache_status == 'miss'
    blocked(monkeypatch)
    with pytest.raises(TilaError, match='TILA-RACE-001'):
        copy_reverse[(4,)](a, a)
    assert copy_reverse.tk.runtime_aliases == []


def test_cache_fingerprint_covers_source_target_options_and_trust():
    x = np.zeros(8, np.int32)
    config = RaceConfig(include_intra=True)
    args = (collision.tk, {'x': x}, {'x_stride0': 1}, {}, (1,), config)
    keys = [RP._key(*args, context) for context in (
        ('source-a', 'contract-a', 'cpu', 4), ('source-b', 'contract-a', 'cpu', 4),
        ('source-a', 'contract-b', 'cpu', 4), ('source-a', 'contract-a', 'cuda', 4),
        ('source-a', 'contract-a', 'cpu', 8))]
    assert len(set(keys)) == len(keys)


def test_runtime_launch_contract_precedes_race_cache(monkeypatch):
    @ti.assume_launch('COUNT == 1')
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.WriteOnly], COUNT: ti.Const[int] = 1):
        ti.store(x, 0, 1)
    x = np.zeros(8, np.int32)
    kernel[(1,)](x)
    kernel[(1,)](x)
    assert kernel.last_race_report.cache_status == 'hit'
    monkeypatch.setattr(RP, 'analyze_launch', lambda *a, **k: pytest.fail('failed contract reached analysis'))
    with pytest.raises(Exception, match='TILA-BOUNDS-010'):
        kernel[(2,)](x, COUNT=2)
    assert kernel.last_race_report is None


def test_empty_and_failed_launch_clear_old_binding_report(monkeypatch):
    x = np.zeros(8, np.int32)
    partition[(2,)](x)
    blocked(monkeypatch)
    partition[(0,)](x)
    assert partition.last_race_report.stage == 'empty-launch'
    assert partition.last_race_report.pairs == ()
    assert partition.tk.runtime_aliases == []
    with pytest.raises(Exception, match='dtype mismatch'):
        partition[(0,)](np.zeros(8, np.float32))
    assert partition.last_race_report is None


def test_off_does_not_disable_alignment_or_verifier(monkeypatch):
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (1,), ti.ReadWrite]):
        ti.atomic_add(x, 0, 1)
    monkeypatch.setenv('TILA_RACE', 'off')
    raw = np.zeros(9, np.uint8)
    x = raw[1:5].view(np.int32)
    with pytest.raises(Exception, match='TILA-MEM-008'):
        kernel[(0,)](x)


@pytest.mark.parametrize('setting', [('TILA_RACE', 'typo'), ('TILA_RACE_MAX_PAIRS', '-1')])
def test_invalid_settings_fail_before_execution(setting, monkeypatch):
    monkeypatch.setenv(*setting)
    blocked(monkeypatch)
    with pytest.raises(TilaError, match='TILA-RACE-003'):
        collision[(1,)](np.zeros(8, np.int32))


def test_budget_unknown_not_reused_at_higher_budget(monkeypatch):
    monkeypatch.setenv('TILA_RACE', 'error')
    monkeypatch.setenv('TILA_RACE_MAX_PAIRS', '0')
    x = np.zeros(8, np.int32)
    with pytest.raises(TilaError, match='TILA-RACE-002'):
        partition[(2,)](x)
    assert not RP._CACHE
    monkeypatch.setenv('TILA_RACE_MAX_PAIRS', '4096')
    partition[(2,)](x)
    assert partition.last_race_report.verdict == PROVEN_SAFE


def test_unmodeled_cross_site_order_unknown_and_scalar_sequence_safe():
    @ti.jit
    def scalar(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite]):
        v = ti.load(x, 0)
        ti.store(x, 0, v + 1)
    scalar[(1,)](np.zeros(8, np.int32))
    assert scalar.last_race_report.verdict == PROVEN_SAFE
    @ti.jit
    def tile(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite]):
        i = ti.arange(0, 4)
        v = ti.load(x, i)
        ti.store(x, i, v)
    with pytest.warns(RuntimeWarning, match='TILA-RACE-002'):
        tile[(1,)](np.zeros(8, np.int32))
    assert any(p.reason == 'intra-program-order-not-modeled' for p in tile.last_race_report.pairs)


def test_loop_duplicate_lane_and_iteration_order_boundary():
    @ti.jit
    def duplicated(x: ti.Buffer[ti.i32, (8,), ti.WriteOnly]):
        i = ti.arange(0, 4)
        for j in ti.range(0, 2):
            ti.store(x, i * 0 + j, i)
    report = analyze_races(duplicated.tk, grid=(1,), bindings={'x': np.zeros(8, np.int32)},
                           config=RaceConfig(include_intra=True))
    pair = next(p for p in report.pairs if p.domain == 'IntraProgram')
    assert pair.verdict == PROVEN_UNSAFE
    assert pair.confirmed.iterations[0] == pair.confirmed.iterations[1]
    assert pair.confirmed.lanes[0] != pair.confirmed.lanes[1]


def test_symbolic_explain_never_reuses_last_binding_and_default_unchanged():
    before = partition.explain()
    partition[(2,)](np.zeros(8, np.int32))
    output = partition.explain(show_races=True)
    assert 'tila.race-details.v1' in output and 'pending-grid' in output
    assert 'confirmed-overlap' not in output
    assert 'race-details:' not in before
    assert 'race-details:' not in partition.explain()


@pytest.mark.parametrize('policy,code', [('off', 0), ('warn', 2), ('error', 2)])
def test_cli_policy_override_and_replay_flags(policy, code, monkeypatch, capsys):
    from tila.cli import main
    fixture = Path(__file__).with_name('race_fixture.py')
    monkeypatch.setenv('TILA_RACE', 'error')
    assert main(['run', str(fixture), '--race', policy, '--show-races', '--show-query', '--show-witness']) == code
    output = capsys.readouterr()
    if code:
        assert 'TILA-RACE-001' in output.err and 'check-sat' in output.err and 'witness' in output.err


def test_cli_pending_error_policy_is_not_a_fake_launch_failure(capsys):
    from tila.cli import main
    fixture = Path(__file__).with_name('race_fixture.py')
    assert main(['explain', str(fixture), '--race', 'error', '--show-races']) == 0
    assert 'pending-grid' in capsys.readouterr().out


def test_render_excludes_solver_variability_unless_requested():
    x = np.zeros(8, np.int32)
    first = RP.analyze_launch(duplicate.tk, {'x': x}, {'x_stride0': 1}, {}, (1,))
    second = RP.analyze_launch(duplicate.tk, {'x': x}, {'x_stride0': 1}, {}, (1,))
    assert second.cache_status == 'hit'
    stable = RP.render(duplicate.tk, first, {})
    assert stable == RP.render(duplicate.tk, second, {})
    assert 'check-sat' not in stable and 'witness' not in stable and str(x.ctypes.data) not in stable
    assert 'check-sat' in RP.render(duplicate.tk, second, {}, show_query=True)
    assert '"cache": "hit"' in RP.render(duplicate.tk, second, {}, show_cache=True)


def test_diagnostic_and_details_goldens():
    report = RP.analyze_launch(duplicate.tk, {'x': np.zeros(8, np.int32)}, {'x_stride0': 1}, {}, (1,))
    root = Path(__file__).with_name('golden')
    assert RP.render(duplicate.tk, report, {}) + '\n' == (root / 'race-details.json').read_text()
    assert RP.diagnostics(duplicate.tk, report, {})[0].render() + '\n' == (root / 'race-error.txt').read_text()


def state_boundaries():
    x = np.zeros(8, np.int32)
    reports = {}
    reports['safe'] = RP.analyze_launch(partition.tk, {'x': x}, {'x_stride0': 1}, {'STEP': 4}, (2,))
    reports['hit'] = RP.analyze_launch(partition.tk, {'x': x}, {'x_stride0': 1}, {'STEP': 4}, (2,))
    reports['confirmed'] = RP.analyze_launch(duplicate.tk, {'x': x}, {'x_stride0': 1}, {}, (1,))
    reports['pending'] = RP.symbolic(collision.tk, {})
    reports['empty'] = RP.analyze_launch(collision.tk, {'x': x}, {'x_stride0': 1}, {}, (0,))
    reports['budget'] = RP.analyze_launch(partition.tk, {'x': x}, {'x_stride0': 1}, {'STEP': 4}, (2,), config=RaceConfig(max_pairs=0))
    reports['disabled'] = replace(reports['empty'], policy='off', stage='disabled', suppressions=('race=off: analysis disabled',))
    @ti.jit
    def opaque(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite]):
        flag = ti.load(x, 0)
        if flag > 0:
            ti.store(x, 0, 1)
    reports['candidate'] = RP.analyze_launch(opaque.tk, {'x': x}, {'x_stride0': 1}, {}, (2,))
    return json.dumps({label: dict(verdict=r.verdict, stage=r.stage, policy=r.policy,
        cache=r.cache_status, suppressions=r.suppressions, incomplete=r.incomplete_reason,
        pairs=[dict(domain=p.domain, verdict=p.verdict, reason=p.reason,
                    counterexample='confirmed' if p.confirmed else 'candidate' if p.candidate else 'none') for p in r.pairs])
        for label, r in reports.items()}, indent=2) + '\n'


def test_state_boundary_golden():
    assert state_boundaries() == (Path(__file__).with_name('golden') / 'race-states.json').read_text()


def test_bounded_cache_and_no_cross_budget_hit():
    config = RaceConfig(proof=replace(ProofConfig(), cache_entries=1), include_intra=True)
    arrays = [np.zeros(8, np.int32), np.zeros(8, np.int32)]
    for x in arrays:
        RP.analyze_launch(collision.tk, {'x': x}, {'x_stride0': 1}, {}, (1,), config=config)
    assert len(RP._CACHE) == 1
    report = RP.analyze_launch(collision.tk, {'x': arrays[0]}, {'x_stride0': 1}, {}, (1,), config=config)
    assert report.cache_status == 'miss'
    larger = replace(config, proof=replace(config.proof, timeout_ms=200))
    report = RP.analyze_launch(collision.tk, {'x': arrays[0]}, {'x_stride0': 1}, {}, (1,), config=larger)
    assert report.cache_status == 'miss'


def test_invalid_policy_in_read_only_explain(monkeypatch):
    monkeypatch.setenv('TILA_RACE', 'typo')
    with pytest.raises(TilaError, match='TILA-RACE-003'):
        collision.explain()
