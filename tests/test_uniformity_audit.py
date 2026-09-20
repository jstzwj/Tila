"""M4-05c opt-in serialization, logical consumer seams and binding isolation."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tila as ti
from tila import runtime
from tila.errors import TilaError, TilaLaunchContractError
from tila.uniformity import Level, UniformityConfig, analyze_uniformity
from tila.uniformity_audit import (uniformity_details, render_uniformity_details,
                                   evaluate_requirement)
from uniformity_audit_fixture import audit


@ti.jit
def states(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly], N: ti.Const[int] = 0):
    c = 1
    p = ti.program_id(0)
    lane = ti.arange(0, 4)
    read = ti.load(x, 0)
    for j in ti.range(0, N):
        if j == 0:
            return
    after = 1


@ti.jit
def views(a: ti.Buffer[ti.i32, (4,), ti.ReadOnly], b: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
    first = ti.load(a, 0)
    second = ti.load(b, 0)


def state_snapshot():
    result = {}
    for label, consts, config in (
        ('symbolic', None, UniformityConfig()),
        ('zero-loop', {'N': 0}, UniformityConfig()),
        ('loop-return', {'N': 2}, UniformityConfig()),
        ('budget', {'N': 0}, UniformityConfig(max_iterations=0)),
        ('truncated', {'N': 0}, UniformityConfig(max_nodes=0)),
    ):
        data = uniformity_details(states.tk, consts, config=config)
        result[label] = {k: data[k] for k in ('stage', 'consts', 'cache', 'complete',
                                            'incomplete_reason', 'consumer_status')}
        result[label]['definitions'] = [
            {k: f[k] for k in ('site', 'value_level', 'reason', 'control')}
            for f in data['facts'] if f['site'].startswith('@definition:')]
        result[label]['exit_control'] = data['exit_control']
        result[label]['predicates'] = data['predicates']
    return result


def test_details_and_default_explain_golden():
    root = Path(__file__).with_name('golden')
    actual = render_uniformity_details(audit.tk, {'FLAG': False}) + '\n'
    assert actual == (root / 'uniformity-details.json').read_text()
    detailed = audit.explain(show_uniformity=True)
    assert detailed + '\n' == (root / 'uniformity-explain.txt').read_text()
    before, block = detailed.split('\nuniformity-details:\n', 1)
    assert before == audit.explain()
    assert json.loads(block) == json.loads(actual)
    assert 'uniformity-details:' not in audit.explain()


def test_state_boundaries_golden():
    expected = Path(__file__).with_name('golden') / 'uniformity-states.json'
    assert state_snapshot() == json.loads(expected.read_text())


def test_schema_graph_closure_and_no_machine_details():
    data = uniformity_details(states.tk, {'N': 2})
    assert data['schema'] == 'tila.uniformity-details.v1'
    assert data['requires_defined_execution'] is True
    assert data['consumers'] == [] and data['consumer_status'] == 'not-installed'
    facts = {f['site'] for f in data['facts']}
    predicates = {p['id'] for p in data['predicates']}
    assert all(i in facts for f in data['facts'] for i in f['inputs'])
    assert all(f['control']['path'] in predicates for f in data['facts'])
    assert all(a['path'] in predicates and a['mask'] in predicates for a in data['accesses'])
    assert all(arg in predicates for p in data['predicates'] for arg in p['args'])
    assert {f['value_level'] for f in data['facts']} >= {
        'LaunchUniform', 'ProgramUniform', 'Varying', 'Unknown'}
    assert all(f['source'] == 'StaticFact' for f in data['facts'])
    text = json.dumps(data)
    assert all(s not in text for s in ('0x', 'timestamp', 'check-sat', 'tensor('))


def test_const_types_and_budget_do_not_reuse_results():
    before = uniformity_details(audit.tk, {'FLAG': False})
    dead = uniformity_details(audit.tk, {'FLAG': True})
    assert dead['exit_control']['reachability'] == 'Unreachable'
    assert before['exit_control']['reachability'] == 'Reachable'
    with pytest.raises(ValueError):
        uniformity_details(audit.tk, {'FLAG': 0})
    limited = uniformity_details(audit.tk, {'FLAG': False}, config=UniformityConfig(max_iterations=0))
    assert not limited['complete'] and limited['incomplete_reason'] == 'fixed-point-budget'
    assert uniformity_details(audit.tk, {'FLAG': False}) == before
    assert uniformity_details(audit.tk)['stage'] == 'symbolic'
    truncated = uniformity_details(states.tk, {'N': 0}, config=UniformityConfig(max_nodes=0))
    assert truncated['exit_control']['reachability'] == 'Unknown'
    returned = uniformity_details(states.tk, {'N': 2})
    assert returned['exit_control']['reachability'] == 'Unknown'


@pytest.mark.parametrize('failure', ['missing', 'dtype', 'grid', 'execute', 'empty'])
def test_launch_failure_empty_and_views_never_bind_explain(failure, monkeypatch):
    kernel = ti.jit(views.fn)
    storage = np.arange(32, dtype=np.int32)
    before = uniformity_details(kernel.tk, {})
    for a, b in ((storage[:4], storage[:4]), (storage[:4], storage[16:20]),
                 (storage[1:9:2], storage[2:10:2])):
        kernel[(1,)](a, b)
        block = kernel.explain(show_uniformity=True).split('uniformity-details:\n')[1]
        assert json.loads(block) == before
        assert str(a.ctypes.data) not in block
    monkeypatch.setattr(kernel, '_kern_cache', {'warm': object()})
    if failure == 'execute':
        def fail(*args, **kwargs):
            raise RuntimeError('injected execution failure')
        monkeypatch.setattr(runtime._Launcher, '_execute', fail)
    if failure == 'empty':
        kernel[(0,)](a, b)
    else:
        with pytest.raises((TilaError, TilaLaunchContractError, RuntimeError)):
            if failure == 'missing':
                kernel[(1,)](a)
            elif failure == 'dtype':
                kernel[(1,)](a, b.astype(np.float32))
            elif failure == 'grid':
                kernel[(-1,)](a, b)
            else:
                kernel[(1,)](a, b)
    assert uniformity_details(kernel.tk, {}) == before
    assert json.loads(kernel.explain(show_uniformity=True).split('uniformity-details:\n')[1]) == before
    assert not hasattr(kernel, 'last_uniformity_report')
    assert list(kernel._kern_cache) == ['warm']


def test_const_failure_and_materialize_leave_no_audit_state():
    before = audit.explain(show_uniformity=True)
    audit.materialize({'FLAG': True})
    with pytest.raises(TilaError):
        audit.explain({'FLAG': 1}, show_uniformity=True)
    assert audit.explain(show_uniformity=True) == before


@pytest.mark.parametrize('command', [('explain',), ('check', '--explain')])
def test_cli_optional_details(command, capsys):
    from tila.cli import main
    fixture = Path(__file__).with_name('uniformity_audit_fixture.py')
    assert main([*command, str(fixture), '--show-uniformity', '--const', 'FLAG=true']) == 0
    output = capsys.readouterr().out
    data = json.loads(output.split('uniformity-details:\n')[1])
    assert data['schema'] == 'tila.uniformity-details.v1'
    assert data['consts'] == [{'name': 'FLAG', 'kind': 'Bool', 'value': True}]


def test_cross_process_hash_seed_stability():
    fixture = Path(__file__).with_name('uniformity_audit_fixture.py')
    outputs = [subprocess.check_output(
        [sys.executable, '-m', 'tila', 'explain', str(fixture), '--show-uniformity'],
        env={**os.environ, 'PYTHONHASHSEED': seed}) for seed in ('1', '73')]
    assert outputs[0] == outputs[1]


def test_flags_compose_without_changing_uniformity():
    plain = audit.explain(show_uniformity=True).split('uniformity-details:\n')[1]
    combined = audit.explain(show_uniformity=True, show_effects=True, show_races=True,
                             show_query=True, show_witness=True, show_cache=True)
    assert combined.split('uniformity-details:\n')[1] == plain


def test_opt_in_is_lazy_and_read_only(monkeypatch):
    import tila.uniformity_audit as ua
    def fail(*args, **kwargs):
        pytest.fail('unexpected compilation, execution or analysis')
    monkeypatch.setattr(audit, 'materialize', fail)
    monkeypatch.setattr(runtime._Launcher, '_execute', fail)
    default = audit.explain()
    assert audit.explain(show_uniformity=True).startswith(default)
    monkeypatch.setattr(ua, 'analyze_uniformity', fail)
    assert audit.explain() == default


def test_changed_tir_does_not_reuse_old_levels():
    from tila import tir as T, types as TY, dtypes as D
    from tila.effect_ir import bind_effects
    @ti.jit
    def kernel():
        v = 1
    before = uniformity_details(kernel.tk)
    kernel.tk.body[0].value = T.TPid(TY.ScalarT(D.i32), 0)
    bind_effects(kernel.tk)
    after = uniformity_details(kernel.tk)
    def level(data):
        return next(f['value_level'] for f in data['facts'] if f['site'].startswith('@definition:'))
    assert level(before) == 'LaunchUniform'
    assert level(after) == 'ProgramUniform'


@pytest.mark.parametrize('name,status', [('c', 'Satisfied'), ('p', 'Satisfied'),
                                        ('lane', 'Unsatisfied'), ('read', 'Unknown')])
def test_logical_consumer_fixture_gates_before_execution(name, status):
    summary = analyze_uniformity(states.tk, {'N': 0})
    site = next(f.site for f in summary.facts if f.site.startswith('@definition:')
                and f.definition.name == name)
    result = evaluate_requirement(states.tk, summary, site, consts={'N': 0},
                                  value_level=Level.ProgramUniform)
    assert result.status == status
    executed = []
    # Fixture only: a future consumer may execute solely on Satisfied. No
    # warning/off setting is passed to the hard requirement evaluator.
    if result.status == 'Satisfied':
        executed.append(site)
    assert bool(executed) == (status == 'Satisfied')
    assert evaluate_requirement(states.tk, summary, site, consts={'N': 0}, scope='CTA').status == 'Unknown'


def test_requirement_control_budget_dead_site_and_forgery():
    @ti.jit
    def conditional():
        if ti.program_id(0) == 0:
            return
        c = 1
    summary = analyze_uniformity(conditional.tk)
    site = next(f.site for f in summary.facts if f.site.startswith('@definition:'))
    assert evaluate_requirement(conditional.tk, summary, site, scope='Launch').status == 'Unsatisfied'
    limited_config = UniformityConfig(max_nodes=0)
    limited = analyze_uniformity(conditional.tk, config=limited_config)
    assert evaluate_requirement(conditional.tk, limited, site, config=limited_config).status == 'Unknown'
    with pytest.raises(TilaError):
        evaluate_requirement(conditional.tk, replace(summary, facts=()), site)
    dead = analyze_uniformity(audit.tk, {'FLAG': True})
    p = next(f.site for f in dead.facts if f.site.startswith('@definition:'))
    assert evaluate_requirement(audit.tk, dead, p, consts={'FLAG': True}).status == 'NotApplicable'
