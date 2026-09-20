"""M4-04d independent finite event oracle and single-case failure replay.

PYTHONPATH=src:tests python tests/race_audit_support.py CASE.json
"""
from dataclasses import asdict
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import tila as ti
from tila import tir as T
from tila.facts import PROVEN_SAFE, PROVEN_UNSAFE, UNKNOWN
from tila.effect_summary import summarize_effects
from tila.interp import Interp
from tila.race import RaceConfig, analyze_races
from tila.solver import ProofConfig

SEED = 204_0426
OBSERVATIONS = []
CONFIG = RaceConfig(ProofConfig(timeout_ms=2000, total_ms=10000,
                               rlimit=2000000, cache_entries=0), include_intra=True)


@ti.jit
def tile(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], STEP: ti.Const[int],
         SCALE: ti.Const[int], BIAS: ti.Const[int]):
    i = ti.program_id(0) * STEP + ti.arange(0, 4) * SCALE + BIAS
    ti.store(x, i, i * 0 + 1, mask=(i >= 0) & (i < 16))


@ti.jit
def equivalent(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], STEP: ti.Const[int],
               SCALE: ti.Const[int], BIAS: ti.Const[int]):
    i = BIAS + SCALE * (3 - ti.arange(0, 4)) + STEP * ti.program_id(0)
    ti.store(x, i, i * 0 + 1, mask=~((i < 0) | (i >= 16)))


@ti.jit
def wrapped(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], STEP: ti.Const[int],
            SCALE: ti.Const[int], BIAS: ti.Const[int]):
    i = ti.cast[ti.i32](ti.cast[ti.u8](ti.program_id(0) * STEP + ti.arange(0, 4) * SCALE + BIAS))
    ti.store(x, i, i * 0 + 1, mask=i < 16)


@ti.jit
def control(x: ti.Buffer[ti.i32, (16,), ti.WriteOnly], STEP: ti.Const[int],
            END: ti.Const[int], LIMIT: ti.Const[int]):
    p = ti.program_id(0)
    if p >= LIMIT:
        return
    for j in ti.range(0, END):
        i = p * STEP + j
        ti.store(x, i, 1, mask=(i >= 0) & (i < 16))


@ti.jit
def aliases(a: ti.Buffer[ti.i8, (4,), ti.WriteOnly],
            x: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    p = ti.program_id(0)
    ti.store(a, p, ti.cast[ti.i8](1))
    ti.store(x, p, 1)


class StoreTrace(Interp):
    """Use CPU expression/control evaluation, suppress all writes.

    This harness only accepts these store-only fixtures. It does not pretend to
    model arbitrary memory-dependent kernels or concurrent execution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []
        self.current = None

    def stmt(self, node, env, pids):
        old = self.current
        if isinstance(node, T.TStore):
            self.current = (node.buffer, pids)
        try:
            super().stmt(node, env, pids)
        finally:
            self.current = old

    def _store(self, arr, strides, coords, value, mask):
        name, pids = self.current
        indices, active, _ = np.broadcast_arrays(coords[0], True if mask is None else mask, value)
        for lane, (index, enabled) in enumerate(zip(indices.flat, active.flat)):
            if enabled:
                assert 0 <= int(index) < len(arr)
                self.events.append((name, pids[0], lane, int(index) * arr.strides[0]))


def setup(case):
    family = case['family']
    grid = case['grid']
    if family == 'aliases':
        # Exact same backing allocation; normalize its base away in all output.
        base = np.zeros(128, np.uint8)
        a = np.ndarray((4,), dtype=np.int8, buffer=base,
                       offset=case['offset'], strides=(case['stride'],))
        x = np.ndarray((4,), dtype=np.int32, buffer=base, offset=16, strides=(4,))
        bindings, consts = {'a': a, 'x': x}, {}
        events = [(name, p, 0, p * arr.strides[0])
                  for name, arr in bindings.items() for p in range(grid)]
    else:
        bindings = {'x': np.zeros(16, np.int32)}
        consts = case['consts']
        events = []
        for p in range(grid):
            if family == 'control':
                if p >= consts['LIMIT']:
                    continue
                terms = [(0, j) for j in range(consts['END'])]
            else:
                terms = [(lane, (3 - lane if family == 'equivalent' else lane)
                          * consts['SCALE'] + consts['BIAS']) for lane in range(4)]
            for lane, term in terms:
                # Arbitrary precision Python oracle, independent of Tila wrap.
                index = p * consts['STEP'] + term
                if family == 'wrapped':
                    index %= 256
                if 0 <= index < 16:
                    events.append(('x', p, lane, index * 4))
    return globals()[family], bindings, consts, events


def check_case(case):
    kernel, bindings, consts, events = setup(case)
    report = analyze_races(kernel.tk, bindings=bindings, consts=consts,
                           grid=(case['grid'],), config=CONFIG)
    strides = {f'{name}_stride0': arr.strides[0] // arr.itemsize
               for name, arr in bindings.items()}
    trace = StoreTrace(kernel.tk, bindings, strides, consts, (case['grid'],), debug=False)
    trace.run()
    problems = []
    if sorted(trace.events) != sorted(events):
        problems.append('CPU events disagree with independent Python events')
    origin = min(a.ctypes.data for a in bindings.values())
    bases = {k: a.ctypes.data - origin for k, a in bindings.items()}

    def overlaps(a, b):
        left, right = bases[a[0]] + a[3], bases[b[0]] + b[3]
        return left < right + bindings[b[0]].itemsize and right < left + bindings[a[0]].itemsize

    # Scalar accesses within a program are ordered; one tile store's lanes are
    # unordered. These fixtures contain no cross-site vector ordering claim.
    conflicts = [(a, b) for a, b in combinations(events, 2)
                 if (a[1] != b[1] or (case['family'] not in ('control', 'aliases')
                                     and a[2] != b[2])) and overlaps(a, b)]
    expected = PROVEN_UNSAFE if conflicts else PROVEN_SAFE
    def allowed_unknown(pair):
        return (case['family'] == 'wrapped' and pair.verdict == UNKNOWN
                and pair.reason == 'numeric-gate-not-confirmed')

    if report.verdict != expected and not (
            report.verdict == UNKNOWN and all(p.verdict != UNKNOWN or allowed_unknown(p)
                                             for p in report.pairs)):
        problems.append(f'expected {expected}, got {report.verdict}')
    if report.unchecked_domains or report.incomplete_reason:
        problems.append('exact fixture was not fully checked')
    sites = {a.effect.site_id: a.node.buffer
             for a in summarize_effects(kernel.tk, consts).accesses}
    for pair in report.pairs:
        names = tuple(sites[s] for s in pair.sites)
        pair_conflicts = [(a, b) for a, b in conflicts
                         if ((a[0], b[0]) == names or (b[0], a[0]) == names)
                         and ((a[1] != b[1]) == (pair.domain == 'InterProgram'))]
        if pair.verdict != (PROVEN_UNSAFE if pair_conflicts else PROVEN_SAFE) and not allowed_unknown(pair):
            problems.append(f'pair {pair.sites}/{pair.domain} disagrees with event oracle')
        w = pair.confirmed
        if w is None:
            continue
        selected = [(name, pid[0], lane, offset)
                    for name, pid, lane, offset in zip(w.parameters, w.pids, w.lanes, w.byte_offsets)]
        if len(selected) != 2 or any(e not in events for e in selected):
            problems.append('confirmed witness is not an active concrete event pair')
        elif not overlaps(*selected):
            problems.append('confirmed witness bytes do not overlap')
        elif pair.domain == 'InterProgram' and selected[0][1] == selected[1][1]:
            problems.append('inter-program witness reuses one program')
        elif pair.domain == 'IntraProgram' and (selected[0][1] != selected[1][1]
                                               or selected[0][2] == selected[1][2]):
            problems.append('intra-program witness is not two distinct lanes')
        if case['family'] == 'control':
            for event, iterations in zip(selected, w.iterations):
                if (len(iterations) != 1 or not 0 <= iterations[0][1] < consts['END']
                        or event[3] != 4 * (event[1] * consts['STEP'] + iterations[0][1])):
                    problems.append('loop witness iteration does not reproduce its address')
    return report, problems, conflicts[:1]


def audit(case):
    report, problems, minimal_events = check_case(case)
    OBSERVATIONS.append({'case': case, 'verdict': report.verdict,
                         'pairs': [{'domain': p.domain, 'verdict': p.verdict, 'reason': p.reason}
                                   for p in report.pairs]})
    if problems:
        persist(case, report, problems, minimal_events)
    return report


def persist(case, report, problems, minimal_events):
    root = Path(os.environ.get('TILA_RACE_AUDIT_FAILURE_DIR', 'artifacts/race-audit'))
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='failure-', dir=root))
    record = {'schema': 'tila.race-audit.v1', 'seed': SEED, 'case': case,
              'problems': problems, 'minimal_event_pair': minimal_events,
              'report': asdict(report)}
    (directory / 'case.json').write_text(json.dumps(record, indent=2) + '\n')
    raise AssertionError(f'{problems}; replay: PYTHONPATH=src:tests python '
                         f'tests/race_audit_support.py {directory / "case.json"}')


if __name__ == '__main__':
    import sys
    record = json.loads(Path(sys.argv[1]).read_text())
    report, problems, _ = check_case(record['case'])
    print(json.dumps({'verdict': report.verdict, 'problems': problems}))
    raise SystemExit(bool(problems))
