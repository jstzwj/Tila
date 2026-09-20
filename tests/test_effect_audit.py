"""M4-01d: stable details and specialization/launch isolation."""
import json
from pathlib import Path

import numpy as np
import pytest
import tila as ti
from tila.effect_audit import effect_details, render_effect_details
from tila.errors import TilaError, TilaLaunchContractError
from tila import runtime


@ti.jit
def select(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], FLAG: ti.Const[bool] = True):
    if FLAG:
        ti.store(out, 0, 1)
    else:
        ti.store(out, 0, 2)


@ti.jit
def pair(a: ti.Buffer[ti.i32, (4,), ti.ReadOnly, 16],
         b: ti.Buffer[ti.i32, (4,), ti.ReadOnly, 16]):
    x = ti.load(a, 0)
    y = ti.load(b, 0)


def test_effect_golden_and_default_compatibility():
    actual = render_effect_details(select.tk, {'FLAG': True}) + '\n'
    expected = Path(__file__).with_name('golden') / 'effect-details.json'
    assert actual == expected.read_text()
    assert 'effect-details:' not in select.explain()
    detailed = select.explain(show_effects=True)
    block = detailed.split('effect-details:\n', 1)[1].split('\naliases:', 1)[0]
    assert json.loads(block) == json.loads(actual)


def test_const_and_compilation_cache_isolation(monkeypatch):
    before = effect_details(select.tk, {'FLAG': True})
    # Neither a cold nor a warm backend cache can be an effect fact source.
    monkeypatch.setattr(select, '_kern_cache', {})
    select.materialize({'FLAG': False})
    select._kern_cache[('unrelated',)] = object()
    other = effect_details(select.tk, {'FLAG': False})
    assert [a['status'] for a in before['accesses']] == ['may-access', 'excluded']
    assert [a['status'] for a in other['accesses']] == ['excluded', 'may-access']
    assert effect_details(select.tk, {'FLAG': True}) == before
    assert len(effect_details(select.tk)['accesses']) == 2
    with pytest.raises(ValueError):
        effect_details(select.tk, {'FLAG': 1})
    assert len(select._kern_cache) == 1


@pytest.mark.parametrize('failure', ['missing', 'dtype', 'grid', 'execute', 'empty'])
def test_failed_or_empty_binding_clears_overlays(failure, monkeypatch):
    kernel = ti.jit(pair.fn)
    x, y = np.zeros(4, np.int32), np.zeros(4, np.int32)
    before = effect_details(kernel.tk, {})
    kernel[(1,)](x, x)
    assert kernel.tk.runtime_aliases
    assert kernel.tk.runtime_alignments
    kernel[(1,)](x, y)
    assert effect_details(kernel.tk, {}) == before
    if failure == 'empty':
        kernel[(0,)](x, y)
    else:
        if failure == 'execute':
            def fail(*args, **kwargs):
                raise RuntimeError('execution failed')
            monkeypatch.setattr(runtime._Launcher, '_execute', fail)
        with pytest.raises((TilaError, TilaLaunchContractError, RuntimeError)):
            if failure == 'missing':
                kernel[(1,)](x)
            elif failure == 'dtype':
                kernel[(1,)](x, np.zeros(4, np.float32))
            elif failure == 'grid':
                kernel[(-1,)](x, y)
            else:
                kernel[(1,)](x, y)
    assert not kernel.tk.runtime_aliases
    assert not kernel.tk.runtime_alignments
    assert not kernel.last_alignment_facts
    assert kernel.last_backend_resources is None
    assert effect_details(kernel.tk, {}) == before
    assert 'runtime launch binding' not in kernel.explain(show_effects=True)


def test_loops_masks_unknown_and_assume_sources():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly], flag: ti.bool, n: ti.i32, N: ti.Const[int] = 0):
        ti.assume(n > 0)
        for i in ti.range(0, N):
            if flag:
                v = ti.unsafe_load(x, 0, mask=False, other=ti.unsafe_load(x, 1))
    data = effect_details(kernel.tk, {'N': 0})
    assert all(a['status'] == 'excluded' for a in data['accesses'])
    data = effect_details(kernel.tk, {'N': 2})
    assert [a['status'] for a in data['accesses']] == ['may-access', 'excluded']
    assert all(len(a['loops']) == 1 for a in data['accesses'])
    assert any(p['op'] == 'unknown' and p['source']['definition']['name'] == 'flag'
               for p in data['predicates'])
    assert data == effect_details(kernel.tk, {'N': 2})


@pytest.mark.parametrize('command', [('explain',), ('check', '--explain')])
def test_cli_effect_details(command, capsys):
    from tila.cli import main
    fixture = Path(__file__).with_name('effect_audit_fixture.py')
    assert main([*command, str(fixture), '--show-effects']) == 0
    output = capsys.readouterr().out
    block = output.split('effect-details:\n', 1)[1].split('\naliases:', 1)[0]
    assert json.loads(block)['schema'] == 'tila.effect-details.v2'


def test_hash_seed_independence():
    import os
    import subprocess
    import sys
    fixture = Path(__file__).with_name('effect_audit_fixture.py')
    outputs = [subprocess.check_output(
        [sys.executable, '-m', 'tila', 'explain', str(fixture), '--show-effects'],
        env=dict(os.environ, PYTHONHASHSEED=seed)) for seed in ('1', '73')]
    assert outputs[0] == outputs[1]


def test_view_binding_and_readonly_explain():
    @ti.jit
    def kernel(a: ti.Buffer[ti.i32, (4,), ti.ReadOnly],
               b: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        x = ti.load(a, 0)
        y = ti.load(b, 0)
    storage = np.arange(32, dtype=np.int32)
    baseline = effect_details(kernel.tk, {})
    aliases = []
    for a, b in [(storage[:4], storage[:4]),
                 (storage[:4], storage[16:20]),
                 (storage[1:9:2], storage[2:10:2])]:
        kernel[(1,)](a, b)
        aliases.append(tuple(kernel.tk.runtime_aliases))
        state = (kernel.last_report, kernel.last_alignment_facts,
                 kernel.last_backend_resources, dict(kernel._kern_cache),
                 dict(kernel.tk.runtime_alignments), tuple(kernel.tk.runtime_aliases))
        kernel.explain(show_effects=True)
        assert state == (kernel.last_report, kernel.last_alignment_facts,
                         kernel.last_backend_resources, dict(kernel._kern_cache),
                         dict(kernel.tk.runtime_alignments), tuple(kernel.tk.runtime_aliases))
        assert effect_details(kernel.tk, {}) == baseline
    assert aliases[0] != aliases[1]
