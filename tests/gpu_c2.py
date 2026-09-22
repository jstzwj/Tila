"""C2 selected generated programs: oracle/CPU/CUDA parity and no-launch negatives."""
from dataclasses import asdict
from itertools import product

import numpy as np
import pytest
import torch

from c2_audit_support import (Case, CONFIG, FAMILIES, bindings,
                              check_case, oracle, persist)
from tila.errors import TilaError
from tila.runtime import _Launcher


@pytest.fixture(autouse=True)
def audit_policy(monkeypatch):
    monkeypatch.setenv('TILA_RACE', 'off')
    for key, value in asdict(CONFIG).items():
        monkeypatch.setenv('TILA_PROOF_' + key.upper(), str(value))


def gpu_bindings(case):
    arrays, source, output = bindings(case)
    xbase = torch.tensor(source, device='cuda')
    outbase = torch.tensor(output, device='cuda')
    x = xbase[2:2 + 8 * case.stride:case.stride]
    out = outbase[2:-2:2].reshape(arrays['out'].shape)
    return x, out, xbase, outbase, source, output


POSITIVE = [Case(family, dtype, op, 'fresh', seed, 1, seed == 255, 0 if seed == 0 else 3,
                 1 if family == 'ptr' else 2)
            for family, dtype, op, seed in product(FAMILIES, ('i8', 'u8'), ('+', '*'), (0, 255))]


@pytest.mark.parametrize('case', POSITIVE)
def test_c2_generated_cpu_gpu_parity(case):
    kernel, proofs, observation, problems, events = check_case(case)
    assert not problems and not observation['outside'] and observation['launched'], observation
    x, out, xbase, outbase, source, expected_backing = gpu_bindings(case)
    _, expected = oracle(case)
    try:
        kernel[(1,)](x, out, **case.kwargs)
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        expected_backing[2:-2:2] = expected.ravel()
        np.testing.assert_array_equal(outbase.cpu().numpy(), expected_backing)
        np.testing.assert_array_equal(xbase.cpu().numpy(), source)
    except Exception as exc:
        observation['backend'] = 'cuda'
        observation['actual_output'] = out.cpu().tolist()
        persist(case, kernel, proofs, observation, [f'{type(exc).__name__}: {exc}'], events)


NEGATIVE = [Case(family, dtype, '+', mask, 0, 8 if mask == 'stale' else -8,
                 family in ('branch', 'runtime'), 1)
            for family, dtype, mask in product(FAMILIES, ('i8', 'u8'), ('stale', 'signed'))]


@pytest.mark.parametrize('case', NEGATIVE)
def test_c2_invalid_generated_access_never_reaches_gpu(case, monkeypatch):
    kernel, proofs, observation, problems, events = check_case(case)
    assert not problems and observation['outside'], observation
    x, out, _, outbase, _, expected_backing = gpu_bindings(case)
    def forbidden(*args, **kwargs):
        raise AssertionError('oracle-invalid access reached GPU execution backend')
    monkeypatch.setattr(_Launcher, '_execute', forbidden)
    try:
        with pytest.raises(TilaError) as failure:
            kernel[(1,)](x, out, **case.kwargs)
        assert failure.value.code in ('TILA-BOUNDS-001', 'TILA-BOUNDS-002', 'TILA-BOUNDS-003')
        np.testing.assert_array_equal(outbase.cpu().numpy(), expected_backing)
    except Exception as exc:
        observation['backend'] = 'cuda-gate'
        persist(case, kernel, proofs, observation, [f'{type(exc).__name__}: {exc}'], events)
