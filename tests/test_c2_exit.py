"""C2 bounded cross-layer combinations; no arbitrary-language completeness claim."""
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import c2_audit_support as support
from c2_audit_support import Case, SEED, CONFIG, FAMILIES, MASKS, audit, cases, check_case, oracle
from tila import tir as T
from tila.facts import PROVEN_SAFE, PROVEN_UNSAFE
from tila.solver import ProofConfig


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.setenv('TILA_SAFETY', 'strict')
    monkeypatch.setenv('TILA_DEBUG', '1')
    # This corpus has a single program and disjoint output lanes. Race analysis
    # is separately audited; bounds/typing/numeric gates remain enabled.
    monkeypatch.setenv('TILA_RACE', 'off')
    for key, value in asdict(CONFIG).items():
        monkeypatch.setenv('TILA_PROOF_' + key.upper(), str(value))


@pytest.fixture(scope='module', autouse=True)
def audit_record(request):
    support.OBSERVATIONS.clear()
    failures = request.session.testsfailed
    yield
    observations = support.OBSERVATIONS
    programs = len(FAMILIES) * 2 * 2 * len(MASKS)
    complete = len(observations) == programs * 8
    path = Path('artifacts/cpu/c2-summary.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema='tila.c2-audit.v1', seed=SEED,
        status=('passed' if complete else 'partial') if request.session.testsfailed == failures else 'failed',
        complete=complete, config=asdict(CONFIG), programs=programs, cases=len(observations),
        verdicts=dict(Counter(o['verdict'] for o in observations)),
        launches=sum(o['launched'] for o in observations),
        oracle_unsafe=sum(bool(o['outside']) for o in observations),
        checked_negative_gates=sum(o.get('gate_checked', False) for o in observations),
        observations=observations), indent=2) + '\n')


@pytest.mark.parametrize('family,dtype,op,mask', product(FAMILIES, ('i8', 'u8'), ('+', '*'), MASKS))
def test_generated_accesses_values_and_proof_conclusions(family, dtype, op, mask):
    observations = [audit(case) for case in cases(family, dtype, op, mask)]
    # Reject-all is not an implementation of the accepted, freshly masked
    # straight-line subset. Other conservative rejections stay visible in JSON.
    if family in ('plain', 'broadcast', 'ptr') and mask in ('fresh', 'false'):
        assert all(o['launched'] for o in observations), observations


def test_generator_has_both_safety_outcomes_and_boolean_metamorphisms():
    defaults = asdict(ProofConfig())
    assert all(v >= defaults[k] for k, v in asdict(CONFIG).items() if k != 'cache_entries')
    unsafe, safe = 0, 0
    for family, dtype, op in product(FAMILIES, ('i8', 'u8'), ('+', '*')):
        for case in cases(family, dtype, op, 'fresh'):
            events, values = oracle(case)
            transformed, output = oracle(replace(case, mask='equivalent'))
            assert transformed == events and np.array_equal(values, output)
        for mask in MASKS:
            for case in cases(family, dtype, op, mask):
                events, _ = oracle(case)
                invalid = any(active and not 0 <= index < 8 for _, index, active in events)
                unsafe += invalid
                safe += not invalid
    assert unsafe > 50 and safe > 500


@pytest.mark.parametrize('verdict', (PROVEN_SAFE, PROVEN_UNSAFE))
def test_oracle_detects_forged_proof_conclusions(verdict, monkeypatch):
    case = Case('plain', 'i8', '+', 'stale', 0, 8 if verdict == PROVEN_SAFE else 0, False, 0)
    real = support.prove
    def corrupt(kernel, case):
        return [(ob, replace(result, verdict=verdict) if ob.kind == 'load' else result)
                for ob, result in real(kernel, case)]
    monkeypatch.setattr(support, 'prove', corrupt)
    assert any('false ' in p for p in check_case(case)[3])


def test_oracle_detects_cast_shape_loss(monkeypatch):
    original = support.AccessTrace.e
    def corrupt(self, expr, env, pids):
        value = original(self, expr, env, pids)
        if isinstance(expr, T.TCast) and expr.dtype.name == 'i32':
            return np.asarray(value).flat[0]
        return value
    monkeypatch.setattr(support.AccessTrace, 'e', corrupt)
    case = Case('broadcast', 'i8', '+', 'fresh', 0, 0, False, 0)
    assert 'CPU access trace differs from independent oracle' in check_case(case)[3]


def test_oracle_detects_pointer_origin_shift(monkeypatch):
    original = support.AccessTrace._ptr_value
    def corrupt(self, expr, env, pids):
        name, offset = original(self, expr, env, pids)
        return name, offset + 1
    monkeypatch.setattr(support.AccessTrace, '_ptr_value', corrupt)
    case = Case('ptr', 'u8', '+', 'fresh', 0, 0, False, 0)
    assert 'CPU access trace differs from independent oracle' in check_case(case)[3]


def test_failure_record_replays_one_binding_and_preserves_source(tmp_path, monkeypatch):
    monkeypatch.setenv('TILA_C2_FAILURE_DIR', str(tmp_path))
    case = Case('plain', 'i8', '+', 'stale', 0, 8, False, 0)
    kernel, results, observation, problems, events = check_case(case)
    assert not problems and observation['outside']
    with pytest.raises(AssertionError, match='replay:'):
        support.persist(case, kernel, results, observation, ['injected failure'], events)
    path, = tmp_path.glob('failure-*/case.json')
    record = json.loads(path.read_text())
    assert record['case'] == asdict(case) and record['seed'] == SEED
    assert record['config'] == asdict(CONFIG) and record['first_invalid_event']
    assert record['policy'] == dict(TILA_SAFETY='strict', TILA_RACE='off', TILA_DEBUG='1')
    assert any(p['query'] for p in record['proofs'])
    assert path.with_name('kernel.py').read_text() == record['source']
    replay = subprocess.run([sys.executable, 'tests/c2_audit_support.py', str(path)],
        env={**os.environ, 'PYTHONPATH': 'src:tests'}, capture_output=True, text=True)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert json.loads(replay.stdout)['problems'] == []
