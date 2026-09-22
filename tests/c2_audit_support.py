"""Bounded C2 source generator, independent oracle and append-only replay.

Replay: PYTHONPATH=src:tests python tests/c2_audit_support.py CASE.json
The corpus contains one input load and one disjoint output store per program.
It makes no concurrency claim and never executes an oracle-invalid GPU access.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import linecache
import os
from pathlib import Path
import random
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import tila as ti
from tila import numeric
from tila.errors import TilaError
from tila.facts import Facts, PROVEN_SAFE, PROVEN_UNSAFE, evaluate_obligation
from tila.interp import Interp
from tila.runtime import _Launcher
from tila.solver import ProofConfig, ProofSession

SEED = 20260921
# At least production work limits, with more wall time on shared runners.
# Unknown remains conservative; the audit must not hide conclusions merely by
# assigning less deterministic solver work than a normal launch.
CONFIG = ProofConfig(timeout_ms=2000, total_ms=10_000,
                     rlimit=200_000, cache_entries=0)
FAMILIES = ('plain', 'branch', 'runtime', 'loop', 'return', 'broadcast', 'ptr')
MASKS = ('fresh', 'equivalent', 'stale', 'signed', 'false')
OBSERVATIONS = []
SENTINEL = -999


@dataclass(frozen=True)
class Case:
    family: str
    dtype: str
    op: str
    mask: str
    seed: int
    shift: int
    flag: bool
    end: int
    stride: int = 1

    @property
    def consts(self):
        return dict(SEED=self.seed, SHIFT=self.shift, FLAG=self.flag, END=self.end)

    @property
    def kwargs(self):
        return {**self.consts, **({'choose': self.flag} if self.family == 'runtime' else {})}


def cases(family, dtype, op, mask):
    # Stable edge coverage, shuffled only to exercise binding/cache history.
    values = [Case(family, dtype, op, mask, seed, shift, flag, end,
                   1 if family == 'ptr' else (1 if i % 2 else 2))
              for i, (seed, shift, flag, end) in enumerate(zip(
                  (-129, -128, -1, 0, 6, 127, 255, 256),
                  (-1, 0, 8, 0, 1, -1, 0, 8),
                  (False, True, False, True, False, True, False, True),
                  (0, 1, 3, 0, 1, 3, 0, 3)))]
    random.Random(SEED).shuffle(values)
    return values


def source_for(case):
    assert case.family in FAMILIES and case.dtype in ('i8', 'u8')
    assert case.op in ('+', '*') and case.mask in MASKS
    shape = '(4, 4)' if case.family == 'broadcast' else '(4,)'
    source_type = 'ti.ReadPtr[ti.i32, 8]' if case.family == 'ptr' else 'ti.Buffer[ti.i32, (8,), ti.ReadOnly]'
    lines = ['import tila as ti',
             f'def kernel(x: {source_type}, out: ti.Buffer[ti.i32, {shape}, ti.WriteOnly],',
             '           SEED: ti.Const[int], SHIFT: ti.Const[int], FLAG: ti.Const[bool], END: ti.Const[int]):',
             '    lane = ti.arange(0, 4)']
    if case.family == 'runtime':
        lines[2] = lines[2].replace('SEED:', 'choose: ti.bool, SEED:')
    lane = 'lane[:, None] + lane[None, :]' if case.family == 'broadcast' else 'lane'
    lines += [f'    value = ti.cast[ti.{case.dtype}]({lane} + SEED)',
              f'    value = value {case.op} ti.cast[ti.{case.dtype}](3)',
              '    index = ti.cast[ti.i32](value)',
              '    active = (index >= 0) & (index < 8)']
    if case.family in ('branch', 'runtime'):
        condition = 'choose' if case.family == 'runtime' else 'FLAG'
        lines += [f'    if {condition}:', '        index = index + SHIFT',
                  '    else:', '        index = index - SHIFT']
    elif case.family == 'loop':
        lines += ['    for step in ti.range(0, END):', '        index = index + SHIFT']
    else:
        if case.family == 'return':
            lines += ['    if FLAG:', '        return']
        lines += ['    index = index + SHIFT']
    mask = {'fresh': '(index >= 0) & (index < 8)',
            'equivalent': '~((index < 0) | (index >= 8))',
            'stale': 'active', 'signed': '(-index >= 0) & (-index < 8)',
            'false': 'False'}[case.mask]
    lines += [f'    active = {mask}']
    if case.family == 'ptr':
        lines += ['    forward = x + ti.cast[ti.i32](3)',
                  '    restored = forward + ti.cast[ti.i32](-3)',
                  '    result = ti.load(restored + index, mask=active, other=-99)']
    else:
        lines += ['    result = ti.load(x, index, mask=active, other=-99)']
    coords = '(lane[:, None], lane[None, :])' if case.family == 'broadcast' else 'lane'
    lines += [f'    ti.store(out, {coords}, result)']
    return '\n'.join(lines) + '\n'


@lru_cache(maxsize=128)
def compile_source(source):
    # inspect.getsource sees stable generated source without a mutable temp file.
    name = '<tila-c2-' + sha256(source.encode()).hexdigest()[:16] + '>'
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    namespace = {}
    exec(compile(source, name, 'exec'), namespace)
    return ti.jit(namespace['kernel'])


def wrap8(value, dtype):
    value %= 256
    return value - 256 if dtype == 'i8' and value >= 128 else value


def oracle(case):
    """Only Python integers/booleans; no Tila expressions, casts or interpreter."""
    shape = (4, 4) if case.family == 'broadcast' else (4,)
    output = np.full(shape, SENTINEL, np.int32)
    events = []
    if case.family == 'return' and case.flag:
        return events, output
    for loc in product(*(range(n) for n in shape)):
        index = wrap8(sum(loc) + case.seed, case.dtype)
        index = wrap8(index + 3 if case.op == '+' else index * 3, case.dtype)
        old_active = 0 <= index < 8
        if case.family in ('branch', 'runtime'):
            index += case.shift if case.flag else -case.shift
        elif case.family == 'loop':
            index += case.end * case.shift
        else:
            index += case.shift
        active = {'fresh': 0 <= index < 8, 'equivalent': not (index < 0 or index >= 8),
                  'stale': old_active, 'signed': 0 <= -index < 8, 'false': False}[case.mask]
        events.append((loc, index, active))
        # Invalid reads are never dereferenced, even in the trace model.
        output[loc] = 100 + index if active and 0 <= index < 8 else -99
    return events, output


def bindings(case):
    backing = np.full(24, -777, np.int32)
    x = backing[2:2 + 8 * case.stride:case.stride]
    x[:] = np.arange(8, dtype=np.int32) + 100
    shape = (4, 4) if case.family == 'broadcast' else (4,)
    output_backing = np.full(2 + 2 * int(np.prod(shape)) + 2, SENTINEL, np.int32)
    out = output_backing[2:-2:2].reshape(shape)
    return {'x': x, 'out': out}, backing, output_backing


class AccessTrace(Interp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def _load(self, arr, strides, coords, mask, other):
        assert len(coords) == 1, 'C2 trace accepts one-dimensional input loads only'
        indexes, enabled = np.broadcast_arrays(coords[0], True if mask is None else mask)
        value = np.full(indexes.shape, -99, np.int32)
        for loc in np.ndindex(indexes.shape):
            index, active = int(indexes[loc]), bool(enabled[loc])
            self.events.append((loc, index, active))
            if active and 0 <= index < len(arr):
                value[loc] = arr[index]
        return value


def prove(kernel, case):
    facts = Facts(num={k: v for k, v in case.consts.items() if type(v) is int},
                  bools={'FLAG': case.flag}, sym_lo=dict(kernel.tk.sym_lo),
                  sym_hi=dict(kernel.tk.sym_hi), grid_checked=True,
                  launch_bindings=tuple(sorted(case.consts.items())) + (('grid', (1,)),))
    session = ProofSession(CONFIG)
    return [(ob, evaluate_obligation(ob, facts, kernel.tk.nonneg_syms, session))
            for ob in kernel.tk.obligations]


def check_case(case):
    kernel = compile_source(source_for(case))
    numeric.validate(kernel.tk, case.consts, {'choose': case.flag}, (1, 1, 1))
    expected_events, expected = oracle(case)
    arrays, input_backing, output_backing = bindings(case)
    scalars = {f'{name}_stride{i}': stride // arr.itemsize
               for name, arr in arrays.items() for i, stride in enumerate(arr.strides)}
    scalars['choose'] = case.flag
    trace = AccessTrace(kernel.tk, arrays, scalars, case.consts, (1,), debug=True)
    trace.run()
    problems = []
    if trace.events != expected_events:
        problems.append('CPU access trace differs from independent oracle')
    if not np.array_equal(arrays['out'], expected):
        problems.append('CPU values/shape differ from independent oracle')
    outside = [event for event in expected_events if event[2] and not 0 <= event[1] < 8]
    # A runtime bool is not specialized by the proof. Its claim quantifies both
    # values, while the concrete launch/trace above uses this binding only.
    proof_outside = outside
    if case.family == 'runtime':
        from dataclasses import replace
        alternative, _ = oracle(replace(case, flag=not case.flag))
        proof_outside = outside + [e for e in alternative if e[2] and not 0 <= e[1] < 8]
    results = prove(kernel, case)
    read = [(ob, result) for ob, result in results if ob.kind == 'load']
    assert len(read) == 1, 'generator must retain exactly one input load obligation'
    verdict = read[0][1].verdict
    if verdict == PROVEN_SAFE and proof_outside:
        problems.append('false Safe: independent oracle contains an active out-of-bounds read')
    if verdict == PROVEN_UNSAFE and not proof_outside:
        problems.append('false confirmed Unsafe: no concrete read is out of bounds')
    # Run the normal CPU launch only for independently safe cases. Unknown
    # remains a recorded rejection; it never substitutes for positive coverage.
    launched, rejection, gate_checked = False, None, False
    if outside and not problems:
        try:
            with patch.object(_Launcher, '_execute', side_effect=AssertionError('invalid read reached backend')):
                kernel[(1,)](arrays['x'], arrays['out'], **case.kwargs)
            problems.append('oracle-invalid CPU launch was accepted')
        except TilaError as exc:
            rejection = exc.code
            gate_checked = exc.code in ('TILA-BOUNDS-001', 'TILA-BOUNDS-002', 'TILA-BOUNDS-003')
            if not gate_checked:
                problems.append(f'unexpected invalid-launch rejection: {exc.code}')
        except AssertionError as exc:
            problems.append(str(exc))
    if not outside and not problems:
        arrays['out'][:] = SENTINEL
        before = input_backing.copy()
        try:
            kernel[(1,)](arrays['x'], arrays['out'], **case.kwargs)
            launched = True
        except TilaError as exc:
            if exc.code not in ('TILA-BOUNDS-001', 'TILA-BOUNDS-002') and not (
                    exc.code == 'TILA-BOUNDS-003' and proof_outside):
                problems.append(f'unexpected launch rejection: {exc.code}: {exc}')
            rejection = exc.code
        if launched and not np.array_equal(arrays['out'], expected):
            problems.append('public CPU launch differs from independent oracle')
        if not np.array_equal(before, input_backing):
            problems.append('read-only input or backing sentinel changed')
        unused = np.ones(len(output_backing), dtype=bool)
        unused[2:-2:2] = False
        if np.any(output_backing[unused] != SENTINEL):
            problems.append('output view backing sentinel changed')
    observation = dict(case=asdict(case), verdict=verdict, reason=read[0][1].reason, outside=outside,
                       proof_outside=proof_outside, launched=launched,
                       rejection=rejection, gate_checked=gate_checked)
    return kernel, results, observation, problems, expected_events


def persist(case, kernel, results, observation, problems, events):
    root = Path(os.environ.get('TILA_C2_FAILURE_DIR', 'artifacts/c2-audit'))
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='failure-', dir=root))
    arrays, _, _ = bindings(case)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True).stdout.strip()
    payload = dict(schema='tila.c2-case.v1', seed=SEED, case=asdict(case),
                   policy=dict(TILA_SAFETY='strict', TILA_RACE='off',
                               TILA_DEBUG=os.environ.get('TILA_DEBUG', '0')),
                   revision=revision, node=os.environ.get('PYTEST_CURRENT_TEST', '').rsplit(' (', 1)[0],
                   bindings={name: dict(shape=arr.shape, byte_strides=arr.strides,
                                        values=arr.tolist(), backing_offset_elements=2)
                             for name, arr in arrays.items()},
                   launch_kwargs=case.kwargs,
                   source=source_for(case), config=asdict(CONFIG), problems=problems,
                   observation=observation, expected_events=events,
                   # One concrete binding; first bad event, not a global AST minimizer.
                   first_invalid_event=next(iter(observation['outside']), None),
                   proofs=[dict(obligation=ob.describe(), verdict=r.verdict,
                                reason=r.reason, query=r.query) for ob, r in results])
    (folder / 'case.json').write_text(json.dumps(payload, indent=2) + '\n')
    (folder / 'kernel.py').write_text(payload['source'])
    if kernel is not None:
        (folder / 'kernel.tir').write_text(kernel.tk.dump())
    raise AssertionError(f'{problems}; replay: PYTHONPATH=src:tests python tests/c2_audit_support.py {folder / "case.json"}')


def audit(case):
    try:
        result = check_case(case)
    except Exception as exc:
        observation = dict(case=asdict(case), verdict='error', outside=[],
                           launched=False, rejection=type(exc).__name__)
        OBSERVATIONS.append(observation)
        persist(case, None, [], observation, [f'{type(exc).__name__}: {exc}'], oracle(case)[0])
    kernel, proofs, observation, problems, events = result
    OBSERVATIONS.append(observation)
    if problems:
        persist(case, kernel, proofs, observation, problems, events)
    return observation


if __name__ == '__main__':
    import sys
    record = json.loads(Path(sys.argv[1]).read_text())
    case = Case(**record['case'])
    if record['source'] != source_for(case):
        raise SystemExit('generator/source changed; use the revision recorded with the audit')
    if record['config'] != asdict(CONFIG):
        raise SystemExit('audit budget changed; use the preserved audit source')
    os.environ.update(record['policy'])
    os.environ.update({'TILA_PROOF_' + k.upper(): str(v) for k, v in record['config'].items()})
    if record['observation'].get('backend', '').startswith('cuda'):
        node = record['node']
        if not node.startswith('tests/gpu_c2.py::test_c2_'):
            raise SystemExit('missing GPU test node; use tools/replay_gpu_audit.py')
        raise SystemExit(subprocess.run([sys.executable, '-m', 'pytest', '-q', node]).returncode)
    _, _, observation, problems, _ = check_case(case)
    print(json.dumps(dict(observation=observation, problems=problems)))
    raise SystemExit(bool(problems))
