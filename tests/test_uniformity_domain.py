"""M4-05d: independent finite execution observations, mutations and replay."""
from collections import Counter
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tila as ti
from tila.errors import TilaError
from tila.uniformity import Level, UniformityConfig, analyze_uniformity, verify_uniformity
from tila.uniformity_audit import evaluate_requirement

from uniformity_domain_support import (SEED, GRID, FAMILIES, BINDINGS, Case,
                                       cases, check_guarantees, compile_family, inject, persist,
                                       reference, run_case)


OBSERVATIONS = []


@pytest.fixture(scope='module', autouse=True)
def audit_record(request):
    OBSERVATIONS.clear()
    failures_at_start = request.session.testsfailed
    yield
    path = Path('artifacts/cpu/m4-05d-uniformity-summary.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = len(OBSERVATIONS) == len(FAMILIES) * len(BINDINGS)
    counts = Counter()
    controls = Counter()
    for item in OBSERVATIONS:
        counts.update(item['observed_levels'])
        controls.update(item['observed_controls'])
    status = ('failed' if request.session.testsfailed != failures_at_start else
              'passed' if complete else 'partial')
    path.write_text(json.dumps({'schema': 'tila.uniformity-domain.v1',
        'seed': SEED, 'grid': GRID, 'families': FAMILIES, 'bindings': BINDINGS,
        'complete': complete, 'status': status, 'cases': len(OBSERVATIONS),
        'events': sum(item['events'] for item in OBSERVATIONS),
        'load_evaluations': sum(item['loads'] for item in OBSERVATIONS),
        'observed_levels': dict(counts), 'observed_controls': dict(controls),
        'observations': OBSERVATIONS}, indent=2) + '\n')


@pytest.mark.parametrize('case', cases(), ids=lambda c: f'{c.family}-{c.seed}-{int(c.choose)}-{c.count}-{int(c.flag)}')
def test_concrete_values_and_participation_agree_with_guarantees(case):
    kernel, summary, events, expected, loads, problems = run_case(case)
    if problems:
        record = persist(case, kernel, summary, events, expected, loads, problems)
        pytest.fail(f'uniformity observation failed; replay: {record}; {problems}')
    assert summary.requires_defined_execution and not summary.incomplete_reason
    observed_sites = {event['site'] for event in events}
    counts = Counter(f.value_level.name for f in summary.facts if f.site in observed_sites
                     and f.value_level is not None)
    controls = Counter(f'{scope}:{getattr(f.control, scope.lower() + "_participation")}'
                       for f in summary.facts if f.site in observed_sites
                       for scope in ('Launch', 'Program'))
    OBSERVATIONS.append({'case': asdict(case), 'events': len(events), 'loads': loads,
                         'observed_levels': dict(counts), 'observed_controls': dict(controls)})


def test_domain_is_fixed_and_contains_real_positive_guarantees():
    corpus = cases()
    assert len(corpus) == 68 and len({repr(c) for c in corpus}) == 68
    assert {c.family for c in corpus} == set(FAMILIES)
    assert any(c.count == 0 for c in corpus) and any(c.count > 0 for c in corpus)
    assert {-128, -1, 0, 127} == {c.seed for c in corpus}
    levels = Counter()
    for family in FAMILIES:
        case = next(c for c in corpus if c.family == family)
        summary = analyze_uniformity(compile_family(family).tk, case.consts)
        levels.update(f.value_level for f in summary.facts if f.site.startswith('@definition:'))
    assert all(levels[level] > 0 for level in Level)


@pytest.mark.parametrize('pair', [('branch', 'branch_equiv'), ('where', 'where_equiv')])
def test_boolean_metamorphisms_preserve_concrete_events(pair):
    for seed, choose, count, flag in BINDINGS:
        left = Case(pair[0], seed, choose, count, flag)
        right = Case(pair[1], seed, choose, count, flag)
        assert reference(left) == reference(right)
        a = run_case(left)[2]
        b = run_case(right)[2]
        visible = lambda events: [(e['name'], e['pid'], e['iteration'], e['value'])
                                  for e in events if e['site'].startswith('@definition:')]
        assert visible(a) == visible(b)


def test_dynamic_instances_and_zero_iteration_entry_are_distinct():
    case = Case('loop', -1, False, 3, True)
    _, summary, events, _, _, problems = run_case(case)
    assert not problems
    carried = [e for e in events if e['name'] == 'carried' and e['site'].startswith('@definition:')]
    assert {tuple(e['iteration']) for e in carried} == {(), (0,), (1,), (2,)}
    assert {e['value']['values'][0] for e in carried if e['pid'] == 2} == {0, 2, 5, 9}
    induction = [e for e in events if e['site'].startswith('@induction:')]
    assert len(induction) == 9
    zero = Case('zero_loop', 0, False, 0, False)
    _, result, observed, _, _, problems = run_case(zero)
    assert not problems
    assert not any(e['iteration'] for e in observed)
    after = next(f for f in result.facts if f.site.startswith('@definition:') and f.definition.name == 'after')
    assert after.value_level == Level.ProgramUniform


def test_nested_dynamic_instances_and_load_control_remain_separate():
    case = Case('nested_loop', 0, False, 3, True)
    _, _, events, _, _, problems = run_case(case)
    assert not problems
    updates = [e for e in events if e['site'].startswith('@definition:')
               and e['name'] == 'seen' and e['pid'] == 1]
    assert {tuple(e['iteration']) for e in updates} == {
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}
    loaded = Case('load_branch', 0, False, 0, False)
    kernel, summary, events, _, loads, problems = run_case(loaded)
    assert not problems and loads == GRID[0]
    selected = next(f for f in summary.facts if f.site.startswith('@definition:')
                    and f.definition.name == 'selected' and '/then/' in f.site)
    assert selected.control.launch_participation == 'Unknown'
    assert evaluate_requirement(kernel.tk, summary, selected.site,
                                consts=loaded.consts, scope='Launch').status == 'Unknown'


@pytest.mark.parametrize('mutation,family', [
    ('pid-as-launch', 'branch'), ('lane-as-program', 'linear'),
    ('load-as-program', 'load'), ('matrix-as-program', 'shape'),
    ('partial-as-program', 'shape'), ('where-selector-as-launch', 'where'),
    ('merge-without-selector', 'branch'),
    ('zero-loop-as-launch', 'zero_loop'), ('return-as-full', 'return'),
    ('pid-loop-as-full', 'pid_loop'), ('load-branch-as-full', 'load_branch')])
def test_independent_observer_detects_overstrong_fact(mutation, family):
    case = Case(family, 0, False, 0, False)
    kernel, summary, events, expected, loads, problems = run_case(case, mutation)
    assert problems and all('false ' in item for item in problems)
    with pytest.raises(TilaError, match='uniformity summary'):
        verify_uniformity(kernel.tk, summary, case.consts)


def test_partial_logical_tile_observation_cannot_satisfy_program_full():
    case = Case('linear', 0, False, 0, False)
    _, summary, events, _, _, problems = run_case(case)
    assert not problems
    lane = next(e for e in events if e['name'] == 'lane' and e['pid'] == 0)
    truncated = [replace_event if replace_event is not lane else
                 {**lane, 'participants': lane['participants'][:-1]} for replace_event in events]
    assert any('false Program Full' in problem for problem in check_guarantees(summary, truncated))


@ti.jit
def body_only(N: ti.Const[int]):
    for j in ti.range(0, N):
        inside = j
    after = 1


def test_missing_loop_incoming_value_has_no_full_consumer_guarantee():
    summary = analyze_uniformity(body_only.tk, {'N': 0})
    exit_facts = [f for f in summary.facts if f.site.startswith('@loop:')
                  and '/exit/' in f.site and f.definition.name in ('j', 'inside')]
    assert len(exit_facts) == 2
    for fact in exit_facts:
        assert fact.value_level == Level.Unknown
        assert fact.control.launch_participation == 'Unknown'
        assert fact.control.program_participation == 'Unknown'
        assert evaluate_requirement(body_only.tk, summary, fact.site, consts={'N': 0}).status == 'Unknown'
    assert next(f for f in summary.facts if f.site.startswith('@definition:')
                and f.definition.name == 'after').control.launch_participation == 'Full'


def test_budget_const_identity_and_missing_summary_fail_closed():
    case = Case('const_branch', 0, False, 0, True)
    kernel = compile_family(case.family)
    true = analyze_uniformity(kernel.tk, case.consts)
    false = analyze_uniformity(kernel.tk, {**case.consts, 'FLAG': False})
    merged = lambda s: next(f for f in s.facts if f.site.startswith('@definition:')
                            and f.definition.name == 'merged')
    assert merged(true).value_level == Level.ProgramUniform
    assert merged(false).value_level == Level.LaunchUniform
    with pytest.raises(TilaError):
        verify_uniformity(kernel.tk, true, {**case.consts, 'FLAG': False})
    with pytest.raises((ValueError, TilaError)):
        analyze_uniformity(kernel.tk, {**case.consts, 'FLAG': 1})
    with pytest.raises(TilaError):
        verify_uniformity(kernel.tk, replace(true, facts=()), case.consts)
    limited = UniformityConfig(max_nodes=0)
    incomplete = analyze_uniformity(kernel.tk, case.consts, config=limited)
    assert incomplete.incomplete_reason and all(f.value_level in (None, Level.Unknown)
                                               for f in incomplete.facts)
    assert evaluate_requirement(kernel.tk, incomplete, merged(true).site,
                                consts=case.consts, config=limited).status == 'Unknown'
    assert evaluate_requirement(kernel.tk, true, merged(true).site,
                                consts=case.consts, scope='CTA').status == 'Unknown'


def test_single_case_failure_record_replays_same_counterexample(tmp_path):
    case = Case('branch', 0, False, 0, False)
    kernel, summary, events, expected, loads, problems = run_case(case, 'merge-without-selector')
    assert problems
    path = persist(case, kernel, summary, events, expected, loads, problems,
                   root=tmp_path, mutation='merge-without-selector')
    record = json.loads(path.read_text())
    assert record['seed'] == SEED and record['case'] == asdict(case)
    assert record['bindings'] == {'consts': case.consts, 'scalars': case.scalars}
    assert record['config'] == asdict(summary.config) and record['minimized'] is False
    assert record['events'] and record['facts'] and record['problems'] == problems
    assert path.with_name('kernel.py').read_text() and path.with_name('kernel.tir.txt').read_text()
    proc = subprocess.run([sys.executable, 'tests/uniformity_domain_support.py', str(path)],
                          env={**os.environ, 'PYTHONPATH': 'src:tests'}, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)['problems'] == problems
