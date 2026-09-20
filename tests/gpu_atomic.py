"""ADR-017 real SM86 atomics: exact invariants and bounded FP histories."""
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import tila as ti
from atomic_support import kernel
from tila.atomic import add
from tila import dtypes as D
from tila.errors import TilaError, TilaLaunchContractError

NP = dict(i32=np.int32, u32=np.uint32, f32=np.float32)


def make(tmp_path, dtype='i32', kind='buffer'):
    return kernel(Path(os.environ.get('TILA_GPU_KERNELS', tmp_path)), dtype, kind)


@pytest.mark.parametrize('dtype', ['i32', 'u32', 'f32'])
@pytest.mark.parametrize('kind', ['buffer', 'ptr'])
@pytest.mark.parametrize('warps', [4, 8])
@pytest.mark.parametrize('count', [0, 5, 61, 64])
def test_atomic_tickets_and_tail(dtype, kind, warps, count, tmp_path):
    k = make(tmp_path, dtype, kind)
    values = np.ones(64, NP[dtype])
    cpu, old_cpu = np.zeros(64, NP[dtype]), np.zeros(64, NP[dtype])
    k[(2,)](cpu, values, old_cpu, COUNT=count)
    x, old = torch.from_numpy(cpu*0).cuda(), torch.from_numpy(old_cpu*0).cuda()
    launch = k[(2,)].with_options(num_warps=warps)
    for _ in range(3):
        x.zero_()
        launch(x, torch.from_numpy(values).cuda(), old, COUNT=count)
        actual, tickets = x.cpu().numpy(), old.cpu().numpy()
        np.testing.assert_array_equal(actual, cpu)
        np.testing.assert_array_equal(np.sort(tickets[:count]), np.arange(count, dtype=NP[dtype]))
        np.testing.assert_array_equal(tickets[count:], 0)
    assert len(k._kern_cache) == 1


@pytest.mark.parametrize('dtype', ['i32', 'u32', 'f32'])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('warps', [4, 8])
def test_atomic_noncolliding_views(dtype, strided, warps, tmp_path):
    k = make(tmp_path, dtype)
    initial = np.arange(130).astype(NP[dtype])
    if dtype == 'i32':
        initial[1] = 2**31-1
    if dtype == 'u32':
        initial[1] = 2**32-1
    cpu = initial.copy()
    gpu = torch.from_numpy(initial).cuda()
    cv = cpu[1:129:2] if strided else cpu[1:65]
    gv = gpu[1:129:2] if strided else gpu[1:65]
    values, old = np.ones(64, NP[dtype]), np.zeros(64, NP[dtype])
    go = torch.from_numpy(old).cuda()
    launch = k[(2,)].with_options(num_warps=warps)
    launch(cv, values, old, COLLIDE=False, COUNT=61)
    launch(gv, torch.from_numpy(values).cuda(), go, COLLIDE=False, COUNT=61)
    np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)
    np.testing.assert_array_equal(go.cpu().numpy(), old)


@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('warps', [4, 8])
def test_atomic_float_legal_histories(keep, warps, tmp_path):
    k = make(tmp_path, 'f32')
    # Cancellation and half-ULP ties distinguish sequential updates from
    # reassociated contributions. Validate old values AND final value jointly.
    contributions = np.array([2**24, 1, -2**24, 2**-24], np.float32)
    histories = set()
    for order in itertools.permutations(range(4)):
        state = np.float32(1)
        old = np.zeros(4, np.float32)
        for lane in order:
            old[lane], state = state, add(state, contributions[lane], D.f32)
        histories.add((state.tobytes(), old.tobytes()))
    values = np.zeros(64, np.float32)
    values[:4] = contributions
    x = torch.zeros(64, device='cuda')
    old = torch.zeros_like(x)
    for _ in range(8):
        x.zero_()
        x[0] = 1
        k[(2,)].with_options(num_warps=warps)(x, torch.from_numpy(values).cuda(), old, COUNT=4, KEEP=keep)
        final = x.cpu().numpy()[0].tobytes()
        observed = old.cpu().numpy()[:4].tobytes()
        assert ((final, observed) in histories if keep else any(final == h[0] for h in histories))


@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('warps', [4, 8])
def test_atomic_float_special_values(keep, warps, tmp_path):
    k = make(tmp_path, 'f32')
    tiny = np.nextafter(np.float32(0), np.float32(1))
    initial = np.resize(np.array([tiny, -tiny, -0., 0., np.inf, -np.inf, np.nan, 1.], np.float32), 64)
    values = np.zeros(64, np.float32)
    values[7::8] = np.float32(2**-24)
    cpu, out = initial.copy(), np.zeros(64, np.float32)
    k[(2,)](cpu, values, out, KEEP=keep, COLLIDE=False)
    x, old = torch.from_numpy(initial).cuda(), torch.zeros(64, device='cuda')
    k[(2,)].with_options(num_warps=warps)(x, torch.from_numpy(values).cuda(), old, KEEP=keep, COLLIDE=False)
    actual = x.cpu().numpy()
    np.testing.assert_equal(np.isnan(actual), np.isnan(cpu))
    finite = ~np.isnan(cpu)
    np.testing.assert_array_equal(actual[finite].view(np.uint32), cpu[finite].view(np.uint32))
    if keep:
        np.testing.assert_array_equal(old.cpu().numpy()[finite].view(np.uint32), initial[finite].view(np.uint32))


@pytest.mark.parametrize('warps', [4, 8])
def test_atomic_random_float_error_bound(warps, tmp_path):
    k = make(tmp_path, 'f32')
    rng = np.random.default_rng(208009)
    for _ in range(4):
        values = rng.uniform(-1, 1, 64).astype(np.float32)
        x, old = torch.zeros(64, device='cuda'), torch.zeros(64, device='cuda')
        k[(2,)].with_options(num_warps=warps)(x, torch.from_numpy(values).cuda(), old)
        total = sum(map(float, values))
        u = 2**-24
        bound = 64*u/(1-64*u)*sum(map(lambda v: abs(float(v)), values))
        assert abs(x[0].item()-total) <= bound


def test_atomic_nested_scalar_alias_and_unused():
    @ti.jit
    def nested(a: ti.RWPtr[ti.i32, 1], b: ti.RWPtr[ti.i32, 1], out: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
        old = ti.atomic_add(a, 1)
        ti.atomic_add(b, ti.atomic_add(a, 1))
        ti.store(out, 0, old + old)
    cpu, co = np.zeros(1, np.int32), np.zeros(1, np.int32)
    gpu = torch.zeros(1, device='cuda', dtype=torch.int32)
    go = torch.zeros_like(gpu)
    nested[(1,)](cpu, cpu, co)
    nested[(1,)](gpu, gpu, go)
    np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)
    np.testing.assert_array_equal(go.cpu().numpy(), co)


@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('value', [0., -0., 1.])
def test_atomic_scalar_float_zero_is_rmw(keep, value):
    @ti.jit
    def scalar(x: ti.Buffer[ti.f32, (1,), ti.ReadWrite], out: ti.Buffer[ti.f32, (1,), ti.WriteOnly],
               value: ti.f32, KEEP: ti.Const[bool]):
        if KEEP:
            old = ti.atomic_add(x, 0, value)
            ti.store(out, 0, old)
        else:
            ti.atomic_add(x, 0, value)
    initial = np.array([np.nextafter(np.float32(0), np.float32(1))], np.float32)
    cpu, co = initial.copy(), np.zeros(1, np.float32)
    gpu, go = torch.from_numpy(initial).cuda(), torch.zeros(1, device='cuda')
    scalar[(1,)](cpu, co, value, KEEP=keep)
    scalar[(1,)](gpu, go, value, KEEP=keep)
    np.testing.assert_array_equal(gpu.cpu().numpy().view(np.uint32), cpu.view(np.uint32))
    if keep:
        np.testing.assert_array_equal(go.cpu().numpy().view(np.uint32), co.view(np.uint32))


def test_atomic_nested_mask_eager_and_const_cache():
    @ti.jit
    def nested(x: ti.Buffer[ti.i32, (3,), ti.ReadWrite], FLAG: ti.Const[bool]):
        if FLAG:
            ti.atomic_add(x, 0, ti.atomic_add(x, 1, 1), mask=ti.atomic_add(x, 2, 1) > 0)
    for flag in (False, True, True):
        cpu = np.array([4, 7, 0], np.int32)
        gpu = torch.from_numpy(cpu.copy()).cuda()
        nested[(1,)](cpu, FLAG=flag)
        nested[(1,)](gpu, FLAG=flag)
        np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)
    assert len(nested._kern_cache) == 2


@pytest.mark.parametrize('keep', [False, True])
def test_atomic_literal_zero_preserves_rmw(keep):
    @ti.jit
    def kernel(x: ti.Buffer[ti.f32, (1,), ti.ReadWrite], out: ti.Buffer[ti.f32, (1,), ti.WriteOnly], KEEP: ti.Const[bool]):
        if KEEP:
            old = ti.atomic_add(x, 0, 0.0)
            ti.store(out, 0, old)
        else:
            ti.atomic_add(x, 0, 0.0)
    initial = np.array([np.nextafter(np.float32(0), np.float32(1))], np.float32)
    x, out = torch.from_numpy(initial).cuda(), torch.zeros(1, device='cuda')
    kernel[(1,)](x, out, KEEP=keep)
    assert x.item() == 0
    if keep:
        np.testing.assert_array_equal(out.cpu().numpy().view(np.uint32), initial.view(np.uint32))


@pytest.mark.parametrize('count', [0, 3])
def test_atomic_loop_and_return(count):
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.ReadWrite], COUNT: ti.Const[int]):
        for j in ti.range(0, COUNT):
            ti.atomic_add(x, 0, 1)
        if COUNT == 0:
            return
        ti.atomic_add(x, 1, 1)
    x = torch.zeros(2, device='cuda', dtype=torch.int32)
    kernel[(2,)](x, COUNT=count)
    np.testing.assert_array_equal(x.cpu().numpy(), [count*2, 2 if count else 0])


def test_atomic_false_mask_invalid_address():
    @ti.jit
    def kernel(x: ti.Buffer[ti.f32, (1,), ti.ReadWrite], out: ti.Buffer[ti.f32, (1,), ti.WriteOnly]):
        old = ti.atomic_add(x, 99, 1.0, mask=False)
        ti.store(out, 0, old)
    x = torch.ones(1, device='cuda')
    out = torch.full_like(x, -1)
    kernel[(1,)](x, out)
    assert x.item() == 1 and out.cpu().numpy().view(np.uint32)[0] == 0


@pytest.mark.parametrize('flag', [False, True])
def test_atomic_runtime_short_circuit(flag):
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (2,), ti.ReadWrite], flag: ti.bool):
        a = flag and ti.atomic_add(x, 0, 1) > 0
        b = flag or ti.atomic_add(x, 1, 1) > 0
    cpu = np.zeros(2, np.int32)
    gpu = torch.zeros(2, device='cuda', dtype=torch.int32)
    kernel[(1,)](cpu, flag)
    kernel[(1,)](gpu, flag)
    np.testing.assert_array_equal(gpu.cpu().numpy(), cpu)
    np.testing.assert_array_equal(cpu, [int(flag), int(not flag)])


def test_atomic_cache_const_layout_and_invalid_launch(tmp_path, monkeypatch):
    k = make(tmp_path)
    x = torch.zeros(130, device='cuda', dtype=torch.int32)
    v = torch.ones(64, device='cuda', dtype=torch.int32)
    old = torch.zeros_like(v)
    k[(2,)](x[:64], v, old, COUNT=5)
    k[(2,)](x[1:129:2], v, old, COUNT=5)
    k[(2,)].with_options(num_warps=8)(x[:64], v, old, COUNT=64, COLLIDE=False)
    assert len(k._kern_cache) == 3
    from tila.runtime import _Launcher
    monkeypatch.setattr(_Launcher, '_execute', lambda *args: pytest.fail('invalid launch executed'))
    for grid in ((0,), (2,)):
        with pytest.raises(TilaLaunchContractError):
            k[grid](x[:1].expand(64), v, old)
        with pytest.raises(TilaLaunchContractError):
            k[grid](x[:64].to(torch.float32), v, old)


@pytest.mark.parametrize('grid', [(0,), (2,)])
def test_atomic_target_gate_on_warm_cache(grid, tmp_path, monkeypatch):
    from dataclasses import replace
    from tila.runtime import _Launcher
    k = make(tmp_path)
    x = torch.zeros(64, device='cuda', dtype=torch.int32)
    value = torch.ones_like(x)
    old = torch.zeros_like(x)
    k[(2,)](x, value, old)
    resolve = _Launcher._resolve_target
    def unavailable(self, tensors):
        target = resolve(self, tensors)
        return replace(target, capability=replace(target.capability, atomic_add_dtypes=()))
    monkeypatch.setattr(_Launcher, '_resolve_target', unavailable)
    monkeypatch.setattr(_Launcher, '_execute', lambda *args: pytest.fail('unsupported target executed'))
    monkeypatch.setenv('TILA_EFFECTS', 'off')
    with pytest.raises(TilaError, match='TILA-TARGET-012'):
        k[grid](x, value, old)
    assert x[0].item() == 64 and len(k._kern_cache) == 1
