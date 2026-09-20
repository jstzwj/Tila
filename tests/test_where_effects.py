"""ADR-010: evaluation sites, definition reuse and independent policy gates."""
from pathlib import Path
import numpy as np
import pytest
import tila as ti
from tila.effect_policy import where_warnings, diagnostics
from tila.errors import TilaError


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv('TILA_EFFECTS', 'warn')


def eager_kernel():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        v = ti.where(True, ti.load(x, 0), ti.load(x, 1))
    return kernel


def test_nested_load_golden():
    kernel = eager_kernel()
    text = '\n\n'.join(w.render() for w in where_warnings(kernel.tk)) + '\n'
    assert text == (Path(__file__).with_name('golden') / 'where-effects.txt').read_text()
    assert text.split('\n\n')[0] in kernel.report().replace('\n  ', '\n')


def test_value_reuse_rebinding_merges_and_loop_carried():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly], flag: ti.bool):
        a = ti.load(x, 0)
        b = ti.where(flag, a + 1, a)
        a = ti.load(x, 1)
        if flag:
            a = ti.load(x, 2)
        c = ti.where(flag, a, b)
        for i in ti.range(0, 2):
            c = ti.where(flag, a, c)
            a = ti.load(x, 3)
    assert where_warnings(kernel.tk) == []
    assert not any(w.code == 'TILA-EFFECT-007' for w in kernel.tk.warnings)


def test_condition_read_is_not_value_branch():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        v = ti.where(ti.load(x, 0) > 0, 1, 2)
    assert where_warnings(kernel.tk) == []


def test_nested_where_ownership_and_false_mask_other():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        v = ti.where(True, ti.where(ti.load(x, 0) > 0,
                                   ti.unsafe_load(x, 1, mask=False, other=ti.load(x, 2)), 0),
                     ti.load(x, 3) + 1)
    warnings = where_warnings(kernel.tk)
    assert len(warnings) == 3
    sites = [line for w in warnings for line in w.details if line.startswith('Read site=')]
    assert len(sites) == len(set(sites)) == 4
    assert any(len([d for d in w.details if d.startswith('Read site=')]) == 2 for w in warnings)
    assert any('/cond/' in d for d in sites)


def test_load_coordinates_are_traversed():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
        v = ti.where(True, ti.unsafe_load(x, ti.load(x, 0)), 0)
    warning, = where_warnings(kernel.tk)
    assert len([d for d in warning.details if d.startswith('Read site=')]) == 2


@pytest.mark.parametrize('policy', ['off', 'warn', 'error'])
def test_policy_switch_rechecks_existing_kernel_and_cache(policy, monkeypatch):
    kernel = eager_kernel()
    kernel.materialize()
    kernel._kern_cache[('already-compiled',)] = object()
    monkeypatch.setenv('TILA_EFFECTS', policy)
    if policy == 'error':
        for action in (lambda: kernel.materialize(),
                       lambda: kernel[(1,)](np.zeros(4, np.int32)),
                       lambda: kernel[(0,)](np.zeros(4, np.int32))):
            with pytest.raises(TilaError, match='TILA-EFFECT-007'):
                action()
        assert 'error[TILA-EFFECT-007]' in kernel.explain()
        with pytest.raises(TilaError, match='TILA-EFFECT-007'):
            eager_kernel()
    else:
        kernel.materialize()
        kernel[(1,)](np.zeros(4, np.int32))
        assert ('warning[TILA-EFFECT-007]' in kernel.explain()) == (policy == 'warn')
    assert len(kernel._kern_cache) == 1


@pytest.mark.parametrize('safety', ['strict', 'warn'])
def test_error_independent_of_bounds_policy(safety, monkeypatch):
    monkeypatch.setenv('TILA_SAFETY', safety)
    monkeypatch.setenv('TILA_EFFECTS', 'error')
    with pytest.raises(TilaError, match='TILA-EFFECT-007'):
        eager_kernel()


def test_off_does_not_disable_bounds(monkeypatch):
    monkeypatch.setenv('TILA_EFFECTS', 'off')
    with pytest.raises(TilaError, match='TILA-BOUNDS-003'):
        @ti.jit
        def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
            v = ti.where(True, ti.load(x, 99), 0)


def test_invalid_policy_and_corrupt_effect_are_rejected(monkeypatch):
    kernel = eager_kernel()
    monkeypatch.setenv('TILA_EFFECTS', 'typo')
    with pytest.raises(TilaError, match='TILA-EFFECT-008'):
        kernel.materialize()
    monkeypatch.setenv('TILA_EFFECTS', 'off')
    kernel.tk.body[0].value.a.effect = None
    with pytest.raises(TilaError, match='effect metadata'):
        diagnostics(kernel.tk, enforce=True)


@pytest.mark.parametrize('policy,code', [('off', 0), ('warn', 0), ('error', 2)])
def test_cli_policy(policy, code, monkeypatch, capsys):
    from tila.cli import main
    monkeypatch.setenv('TILA_EFFECTS', 'error')
    fixture = Path(__file__).with_name('where_effect_fixture.py')
    assert main(['check', str(fixture), '--effects', policy]) == code
    result = capsys.readouterr()
    output = result.out + result.err
    assert ('TILA-EFFECT-007' in output) == (policy != 'off')


def test_cli_inherits_environment(monkeypatch, capsys):
    from tila.cli import main
    monkeypatch.setenv('TILA_EFFECTS', 'error')
    fixture = Path(__file__).with_name('where_effect_fixture.py')
    assert main(['check', str(fixture)]) == 2
    assert 'TILA-EFFECT-007' in capsys.readouterr().err


def test_error_golden_and_no_execution(monkeypatch):
    from tila import runtime
    kernel = eager_kernel()
    def unexpected(*args, **kwargs):
        pytest.fail('effect error must precede execution')
    monkeypatch.setattr(runtime._Launcher, '_execute', unexpected)
    monkeypatch.setenv('TILA_EFFECTS', 'error')
    with pytest.raises(TilaError) as failure:
        kernel[(1,)](np.zeros(4, np.int32))
    golden = (Path(__file__).with_name('golden') / 'where-effects.txt').read_text()
    assert str(failure.value) == golden.split('\n\n')[0].replace('warning[', 'error[', 1)


def test_static_diagnostic_retains_dead_and_masked_reads():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly], FLAG: ti.Const[bool] = False):
        if FLAG:
            v = ti.where(True, ti.load(x, 0, mask=False, other=0), 1)
    warning, = where_warnings(kernel.tk)
    assert 'then' in warning.title
    assert len(kernel.tk.effect_summary({'FLAG': False}).effects) == 0
