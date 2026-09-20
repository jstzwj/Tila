"""M4-04c actual CUDA bindings and prelaunch race gates; never run racy kernels."""
import numpy as np
import pytest
import torch
import tila as ti
from tila.runtime import _Launcher
from tila.errors import TilaError
from race_fixture import collision, duplicate, copy_reverse, matrix


@pytest.mark.parametrize('warps', [4, 8])
def test_cuda_duplicate_lanes_never_launch(warps, monkeypatch):
    monkeypatch.setenv('TILA_RACE', 'warn')
    x = torch.zeros(8, dtype=torch.int32, device='cuda')
    monkeypatch.setattr(_Launcher, '_execute', lambda *a: pytest.fail('racy CUDA kernel launched'))
    with pytest.raises(TilaError, match='TILA-RACE-001') as error:
        duplicate[(1,)].with_options(num_warps=warps)(x)
    assert any(p.domain == 'IntraProgram' and p.confirmed for p in error.value.race_report.pairs)
    assert torch.count_nonzero(x).item() == 0


def test_cuda_alias_rebinding_on_warm_cache(monkeypatch):
    monkeypatch.setenv('TILA_RACE', 'error')
    kernel = ti.jit(copy_reverse.fn)
    a = torch.arange(4, dtype=torch.int32, device='cuda')
    out = torch.empty_like(a)
    kernel[(4,)](a, out)
    kernel[(4,)](a, out)
    assert kernel.last_race_report.cache_status == 'hit'
    np.testing.assert_array_equal(out.cpu().numpy(), [3, 2, 1, 0])
    assert len(kernel._kern_cache) == 1
    monkeypatch.setattr(_Launcher, '_execute', lambda *a: pytest.fail('alias conflict reached CUDA execution'))
    with pytest.raises(TilaError, match='TILA-RACE-001'):
        kernel[(4,)](a, a)
    np.testing.assert_array_equal(a.cpu().numpy(), [0, 1, 2, 3])
    assert kernel.last_race_report is None and kernel.tk.runtime_aliases == []


@pytest.mark.parametrize('warps', [4, 8])
def test_cuda_off_to_warn_checks_precompiled_kernel(warps, monkeypatch):
    kernel = ti.jit(collision.fn)
    x = torch.zeros(8, dtype=torch.int32, device='cuda')
    monkeypatch.setenv('TILA_RACE', 'off')
    kernel[(1,)].with_options(num_warps=warps)(x)  # One program: no conflicting writes.
    assert len(kernel._kern_cache) == 1 and kernel.last_race_report.suppressions
    monkeypatch.setenv('TILA_RACE', 'warn')
    monkeypatch.setattr(_Launcher, '_execute', lambda *a: pytest.fail('policy switch reused an unchecked kernel'))
    with pytest.raises(TilaError, match='TILA-RACE-001'):
        kernel[(2,)].with_options(num_warps=warps)(x)
    assert x[0].item() == 1


@pytest.mark.parametrize('policy', ['warn', 'error'])
def test_cuda_unknown_policy(policy, monkeypatch):
    monkeypatch.setenv('TILA_RACE', policy)
    x = torch.zeros((2, 2), dtype=torch.int32, device='cuda')
    if policy == 'error':
        monkeypatch.setattr(_Launcher, '_execute', lambda *a: pytest.fail('Unknown error policy launched CUDA kernel'))
        with pytest.raises(TilaError, match='TILA-RACE-002'):
            matrix[(1,)](x)
        assert torch.count_nonzero(x).item() == 0
    else:
        with pytest.warns(RuntimeWarning, match='TILA-RACE-002'):
            matrix[(1,)](x)
        assert x[0, 0].item() == 4
