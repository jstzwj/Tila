"""M3-02 real-device launch/caching guards; explicit GPU gate only."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import tila as ti
from tila import tir as T
from tila.errors import TilaError, TilaLaunchContractError
from tila.runtime import _Launcher
from test_m3_launch import copy, aligned


@pytest.mark.parametrize("grid", [(0,), (0, 65535, 65535)])
def test_cuda_zero_grid_never_executes(grid, monkeypatch):
    kernel = ti.jit(copy.fn)
    monkeypatch.setattr(_Launcher, "_execute", lambda *a: pytest.fail("empty launch executed"))
    kernel[grid](torch.empty(0, dtype=torch.int32, device="cuda"))
    assert not kernel._kern_cache and "no-op" in kernel.last_report


@pytest.mark.parametrize("grid", [(0, 65536), (0, 1, 65536)])
def test_cuda_grid_axis_rejected(grid, monkeypatch):
    monkeypatch.setattr(_Launcher, "_execute", lambda *a: pytest.fail("invalid grid executed"))
    with pytest.raises(TilaLaunchContractError) as exc:
        copy[grid](torch.empty(0, dtype=torch.int32, device="cuda"))
    assert exc.value.code == "TILA-TYPE-104"


@pytest.mark.parametrize("grid", [(0,), (1,)])
def test_cuda_alignment_failure_never_executes(grid, monkeypatch):
    monkeypatch.setattr(_Launcher, "_execute", lambda *a: pytest.fail("invalid alignment executed"))
    x = torch.zeros(16, dtype=torch.int32, device="cuda")[1:]
    with pytest.raises(TilaLaunchContractError) as exc:
        aligned[grid](x)
    assert exc.value.code == "TILA-MEM-003"


def test_cuda_unknown_tir_never_executes(monkeypatch):
    kernel = ti.jit(copy.fn)
    kernel.tk.body = [object()]
    monkeypatch.setattr(_Launcher, "_execute", lambda *a: pytest.fail("invalid TIR executed"))
    with pytest.raises(TilaError) as exc:
        kernel[(1,)](torch.zeros(8, dtype=torch.int32, device="cuda"))
    assert exc.value.code == "TILA-TARGET-009"


def test_cuda_cache_hit_and_source_view_options_isolation():
    kernel = ti.jit(copy.fn)
    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    kernel[(1,)](x)
    kernel[(1,)](x)
    assert len(kernel._kern_cache) == 1
    np.testing.assert_array_equal(x.cpu().numpy(), np.full(8, 2))
    backing = torch.zeros(16, dtype=torch.int32, device="cuda")
    kernel[(1,)](backing[::2])
    assert len(kernel._kern_cache) == 2
    np.testing.assert_array_equal(backing.cpu().numpy(), np.tile([1, 0], 8))
    kernel[(1,)].with_options(num_warps=8)(x)
    assert len(kernel._kern_cache) == 3
    # Same JIT object and ABI, new semantic source: must not reuse +1 kernel.
    store = kernel.tk.body[-1]
    assert isinstance(store, T.TStore) and isinstance(store.value, T.TBin)
    kernel.tk.body[-1] = replace(store, value=replace(store.value, op="-"))
    kernel[(1,)](x)
    assert len(kernel._kern_cache) == 4
    np.testing.assert_array_equal(x.cpu().numpy(), np.full(8, 2))
