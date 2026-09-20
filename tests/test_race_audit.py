"""M4-04d exhaustive bounded events, metamorphisms and reproducibility."""
from itertools import product
from dataclasses import replace
from pathlib import Path
import json
import os
import random
import subprocess
import sys

import pytest
import race_audit_support as support
from tila.facts import PROVEN_SAFE
from race_audit_support import SEED, audit, check_case, persist


@pytest.fixture(scope='module', autouse=True)
def audit_record(request):
    support.OBSERVATIONS.clear()
    failures = request.session.testsfailed
    yield
    path = Path('artifacts/cpu/m4-04d-race-summary.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema': 'tila.race-enumeration.v1', 'seed': SEED,
                    'status': 'passed' if request.session.testsfailed == failures else 'failed',
                    'analysis_budget': support.CONFIG.proof.__dict__,
                    'observations': support.OBSERVATIONS}, indent=2) + '\n')


@pytest.mark.parametrize('grid', range(5))
def test_affine_and_boolean_lane_permutation(grid):
    cases = list(product((-2, 0, 1, 4, 7), (0, 1, 2), (-1, 0, 14)))
    random.Random(SEED + grid).shuffle(cases)
    for step, scale, bias in cases:
        case = {'family': 'tile', 'grid': grid,
                'consts': {'STEP': step, 'SCALE': scale, 'BIAS': bias}}
        original = audit(case)
        transformed = audit({**case, 'family': 'equivalent'})
        assert original.verdict == transformed.verdict


@pytest.mark.parametrize('grid', (1, 2, 4))
def test_narrow_cast_and_intermediate_wrap(grid):
    for step, scale, bias in product((0, 1, 255, 256), (0, 1),
                                    (-2147483648, -1, 0, 255, 256, 2147483647)):
        audit({'family': 'wrapped', 'grid': grid,
               'consts': {'STEP': step, 'SCALE': scale, 'BIAS': bias}})


@pytest.mark.parametrize('grid', (0, 1, 3))
def test_early_return_zero_loop_and_iteration_pairs(grid):
    for step, end, limit in product((0, 1, 4), (0, 1, 3), (0, 1, 3)):
        audit({'family': 'control', 'grid': grid,
               'consts': {'STEP': step, 'END': end, 'LIMIT': limit}})


@pytest.mark.parametrize('grid', (1, 2, 4))
def test_alias_partial_bytes_stride_and_translation(grid):
    for offset, stride in product((0, 15, 16, 17, 20, 32), (1, 2, 4, 8)):
        # setup allocates fresh storage every time: absolute address relocation
        # must not affect the result or relative witness validity.
        case = {'family': 'aliases', 'grid': grid, 'offset': offset, 'stride': stride}
        assert audit(case).verdict == audit(case).verdict


def test_failure_capture_and_single_case_replay(tmp_path, monkeypatch):
    monkeypatch.setenv('TILA_RACE_AUDIT_FAILURE_DIR', str(tmp_path))
    case = {'family': 'tile', 'grid': 2,
            'consts': {'STEP': 0, 'SCALE': 1, 'BIAS': 0}}
    report, problems, events = check_case(case)
    assert not problems and events
    with pytest.raises(AssertionError, match='replay:'):
        persist(case, report, ['injected audit failure'], events)
    path, = tmp_path.glob('failure-*/case.json')
    record = json.loads(path.read_text())
    assert record['seed'] == SEED and record['case'] == case
    assert len(record['minimal_event_pair'][0]) == 2
    assert any(p['query'] for p in record['report']['pairs'])
    result = subprocess.run([sys.executable, 'tests/race_audit_support.py', str(path)],
                            env={**os.environ, 'PYTHONPATH': 'src:tests'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['problems'] == []


@pytest.mark.parametrize('mutation', ('false-safe', 'invalid-byte', 'same-program'))
def test_oracle_rejects_corrupted_solver_conclusions(mutation, monkeypatch):
    case = {'family': 'tile', 'grid': 2,
            'consts': {'STEP': 0, 'SCALE': 1, 'BIAS': 0}}
    report, problems, _ = check_case(case)
    assert not problems
    pair = next(p for p in report.pairs if p.confirmed)
    if mutation == 'false-safe':
        bad = replace(pair, verdict=PROVEN_SAFE, confirmed=None)
    elif mutation == 'invalid-byte':
        bad = replace(pair, confirmed=replace(pair.confirmed, byte_offsets=(999, 999)))
    else:
        bad = replace(pair, confirmed=replace(pair.confirmed,
                      pids=(pair.confirmed.pids[0], pair.confirmed.pids[0])))
    corrupt = replace(report, pairs=tuple(bad if p is pair else p for p in report.pairs))
    monkeypatch.setattr(support, 'analyze_races', lambda *a, **kw: corrupt)
    assert check_case(case)[1]
