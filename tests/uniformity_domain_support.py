"""M4-05d finite-domain observer and independent Python reference.

Replay: PYTHONPATH=src:tests python tests/uniformity_domain_support.py CASE.json
Only logical TIR values/control on defined CPU executions are audited. This is
not a physical CTA/warp or synchronization model.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
import linecache
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import tila as ti
from tila import tir as T
from tila.interp import Interp
from tila.uniformity import Level, UniformityConfig, analyze_uniformity


SEED = 20260923
GRID = (3,)
FAMILIES = ('linear', 'branch', 'branch_equiv', 'host_branch', 'const_branch',
            'loop', 'pid_loop', 'nested_loop', 'zero_loop', 'return', 'shape',
            'rebind', 'where', 'where_equiv', 'load', 'eager_where', 'load_branch')
BINDINGS = ((-128, False, 0, False), (-1, True, 1, True),
            (0, False, 3, True), (127, True, 0, False))


@dataclass(frozen=True)
class Case:
    family: str
    seed: int
    choose: bool
    count: int
    flag: bool

    @property
    def consts(self):
        return {'N': self.count, 'FLAG': self.flag}

    @property
    def scalars(self):
        return {'seed': self.seed, 'choose': self.choose}


def cases():
    result = [Case(family, *binding) for family in FAMILIES for binding in BINDINGS]
    random.Random(SEED).shuffle(result)
    return result


def source_for(family):
    if family not in FAMILIES:
        raise ValueError(family)
    body = {
        'linear': ['value = ti.cast[ti.i8](lane + seed)',
                   'copy = value', 'zeroed = value * ti.cast[ti.i8](0)'],
        'branch': ['if pid == 0:', '    selected = 1', 'else:', '    selected = 2',
                   'merged = selected', 'after = 5'],
        'branch_equiv': ['if pid != 0:', '    selected = 2', 'else:', '    selected = 1',
                         'merged = selected', 'after = 5'],
        'host_branch': ['if choose:', '    selected = 1', 'else:', '    selected = 2',
                        'merged = selected', 'after = 5'],
        'const_branch': ['if FLAG:', '    selected = pid', 'else:', '    selected = 0',
                         'merged = selected', 'after = 5'],
        'loop': ['carried = 0', 'iteration = 0', 'for j in ti.range(0, N):',
                 '    carried = carried + pid + j', '    iteration = carried',
                 'after = carried'],
        'pid_loop': ['carried = 0', 'iteration = 0', 'for j in ti.range(0, pid):',
                     '    carried = carried + j', '    iteration = carried',
                     'after = carried'],
        'nested_loop': ['carried = 0', 'for j in ti.range(0, N):',
                        '    for k in ti.range(0, 2):',
                        '        carried = carried + pid + j + k',
                        '        seen = carried', 'after = carried'],
        'zero_loop': ['carried = pid', 'for j in ti.range(0, N):',
                      '    carried = 0', 'after = carried'],
        'return': ['if pid == 0:', '    return', 'after = 1'],
        'shape': ['matrix = lane[:, None] + lane[None, :]',
                  'reshaped = ti.reshape(lane, (2, 2))',
                  'partial = ti.sum(reshaped, 0)',
                  'full = ti.sum(lane, 0)'],
        'rebind': ['selected = pid', 'before = selected',
                   'selected = 0', 'after = selected'],
        'where': ['picked = ti.where(lane > 1, lane, 1)',
                  'same = ti.where(pid == 0, pid, pid)',
                  'different = ti.where(pid == 0, 1, 2)'],
        'where_equiv': ['picked = ti.where(~(lane <= 1), lane, 1)',
                        'same = ti.where(pid != 0, pid, pid)',
                        'different = ti.where(pid != 0, 2, 1)'],
        'load': ['loaded = ti.load(x, lane)'],
        'eager_where': ['picked = ti.where(False, ti.load(x, lane), lane)'],
        'load_branch': ['if ti.load(x, pid) > 15:', '    selected = 1',
                        'else:', '    selected = 2', 'after = selected'],
    }[family]
    buffer = ('x: ti.Buffer[ti.i32, (4,), ti.ReadOnly], '
              if family in ('load', 'eager_where', 'load_branch') else '')
    lines = ['import tila as ti',
             'def kernel(' + buffer + 'seed: ti.i8, choose: ti.bool, N: ti.Const[int], FLAG: ti.Const[bool]):',
             '    pid = ti.program_id(0)', '    lane = ti.arange(0, 4)']
    lines.extend('    ' + line for line in body)
    return '\n'.join(lines) + '\n'


@lru_cache(maxsize=len(FAMILIES))
def compile_family(family):
    source = source_for(family)
    name = '<tila-uniformity-' + sha256(source.encode()).hexdigest()[:16] + '>'
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    namespace = {}
    exec(compile(source, name, 'exec'), namespace)
    return ti.jit(namespace['kernel'])


def wrap_i8(value):
    return (value + 128) % 256 - 128


def freeze(value):
    array = np.asarray(value)
    if array.dtype.kind not in 'biu':
        raise AssertionError('finite corpus must contain only integer and bool values')
    return {'shape': list(array.shape), 'values': [int(x) for x in array.flat]}


def reference(case):
    """Source-level Python oracle. No TIR traversal, Interp or transfer rules."""
    result = []
    def put(name, pid, value, iteration=()):
        result.append({'name': name, 'pid': pid, 'iteration': list(iteration),
                       'value': freeze(value)})
    lane = list(range(4))
    for pid in range(GRID[0]):
        put('pid', pid, pid)
        put('lane', pid, lane)
        if case.family == 'linear':
            values = [wrap_i8(x + case.seed) for x in lane]
            put('value', pid, values)
            put('copy', pid, values)
            put('zeroed', pid, [0] * 4)
        elif case.family in ('branch', 'branch_equiv', 'host_branch', 'const_branch'):
            value = ((1 if pid == 0 else 2) if case.family in ('branch', 'branch_equiv') else
                     (1 if case.choose else 2) if case.family == 'host_branch' else
                     (pid if case.flag else 0))
            put('selected', pid, value)
            put('merged', pid, value)
            put('after', pid, 5)
        elif case.family in ('loop', 'pid_loop'):
            carried = 0
            put('carried', pid, carried)
            put('iteration', pid, 0)
            count = case.count if case.family == 'loop' else pid
            for j in range(count):
                carried += pid + j if case.family == 'loop' else j
                put('carried', pid, carried, (j,))
                put('iteration', pid, carried, (j,))
            put('after', pid, carried)
        elif case.family == 'nested_loop':
            carried = 0
            put('carried', pid, carried)
            for j in range(case.count):
                for k in range(2):
                    carried += pid + j + k
                    put('carried', pid, carried, (j, k))
                    put('seen', pid, carried, (j, k))
            put('after', pid, carried)
        elif case.family == 'zero_loop':
            carried = pid
            put('carried', pid, carried)
            for j in range(case.count):
                carried = 0
                put('carried', pid, carried, (j,))
            put('after', pid, carried)
        elif case.family == 'return':
            if pid != 0:
                put('after', pid, 1)
        elif case.family == 'shape':
            put('matrix', pid, [[a + b for b in lane] for a in lane])
            put('reshaped', pid, [[0, 1], [2, 3]])
            put('partial', pid, [2, 4])
            put('full', pid, 6)
        elif case.family == 'rebind':
            put('selected', pid, pid)
            put('before', pid, pid)
            put('selected', pid, 0)
            put('after', pid, 0)
        elif case.family in ('where', 'where_equiv'):
            put('picked', pid, [1, 1, 2, 3])
            put('same', pid, pid)
            put('different', pid, 1 if pid == 0 else 2)
        elif case.family == 'load':
            put('loaded', pid, [10, 20, 30, 40])
        elif case.family == 'eager_where':
            put('picked', pid, lane)
        elif case.family == 'load_branch':
            value = 1 if (10 + 10 * pid) > 15 else 2
            put('selected', pid, value)
            put('after', pid, value)
    return result


class Observer(Interp):
    """Observe real interpreter execution without implementing its operators."""
    def __init__(self, kernel, case):
        buffers = {'x': np.asarray([10, 20, 30, 40], dtype=np.int32)} if kernel.buffers else {}
        scalars = {**case.scalars, **({'x_stride0': 1} if buffers else {})}
        super().__init__(kernel, buffers, scalars, case.consts, GRID)
        self.block_paths = {}
        self.loop_bodies = {}
        self.site_by_stmt = {}
        self.iteration = []
        self.events = []
        self.loads = 0
        self._index(kernel.body, 'body')

    def _load(self, arr, strides, coords, mask, other):
        self.loads += 1
        return super()._load(arr, strides, coords, mask, other)

    def _index(self, block, path):
        self.block_paths[id(block)] = path
        for index, node in enumerate(block):
            site = f'{path}/{index}'
            self.site_by_stmt[id(node)] = site
            if isinstance(node, (T.TIf, T.TStaticIf)):
                self._index(node.then_body, site + '/then')
                self._index(node.else_body, site + '/else')
            elif isinstance(node, T.TFor):
                self._index(node.body, site + '/body')
                self.loop_bodies[id(node.body)] = (site, node.var)

    def _record(self, site, name, value, pids):
        frozen = freeze(value)
        # The currently accepted scalar control executes a whole logical
        # value. Keep element identities explicit so a partial observation
        # cannot accidentally satisfy a Program Full claim.
        participants = [list(index) for index in np.ndindex(tuple(frozen['shape']))]
        self.events.append({'site': site, 'name': name, 'pid': pids[0],
                            'iteration': [index for _, index in self.iteration],
                            'value': frozen, 'participants': participants})

    def stmts(self, stmts, env, pids):
        loop = self.loop_bodies.get(id(stmts))
        if loop:
            site, var = loop
            index = int(env[var])
            self.iteration.append((site, index))
            self._record(f'@induction:{site}:{var}', var, env[var], pids)
            for name in self._pseudo_names('loop', site + '/carried/'):
                if name in env:
                    self._record(f'@loop:{site}/carried/{name}:{name}', name, env[name], pids)
        try:
            return super().stmts(stmts, env, pids)
        finally:
            if loop:
                self.iteration.pop()

    def _pseudo_names(self, kind, prefix):
        # Populated by the checked summary in observe(); identity is taken
        # from the TIR site, never from a current same-named variable alone.
        return self.pseudo.get((kind, prefix), ())

    def stmt(self, stmt, env, pids):
        site = self.site_by_stmt[id(stmt)]
        super().stmt(stmt, env, pids)
        if isinstance(stmt, T.TAssign):
            self._record(f'@definition:{site}:{stmt.name}', stmt.name, env[stmt.name], pids)
        elif isinstance(stmt, (T.TIf, T.TStaticIf)):
            for name in self._pseudo_names('merge', site + '/'):
                if name in env:
                    self._record(f'@merge:{site}/{name}:{name}', name, env[name], pids)
        elif isinstance(stmt, T.TFor):
            for name in self._pseudo_names('loop', site + '/exit/'):
                if name in env:
                    self._record(f'@loop:{site}/exit/{name}:{name}', name, env[name], pids)


def observe(kernel, case, summary):
    observer = Observer(kernel, case)
    pseudo = defaultdict(list)
    for fact in summary.facts:
        ref = fact.definition
        if ref and ref.kind in ('merge', 'loop', 'induction'):
            prefix = ref.site.rsplit('/', 1)[0] + '/'
            if ref.kind == 'merge':
                pseudo[('merge', prefix)].append(ref.name)
            elif ref.kind == 'loop':
                pseudo[('loop', prefix)].append(ref.name)
    observer.pseudo = pseudo
    observer.run()
    return observer.events, observer.loads


def check_guarantees(summary, events):
    """Check only positive guarantees against concrete events, not V/U claims."""
    problems = []
    grouped = defaultdict(list)
    for event in events:
        grouped[(event['site'], tuple(event['iteration']))].append(event)
    for fact in summary.facts:
        if fact.definition is None:
            continue
        groups = [(iteration, group) for (site, iteration), group in grouped.items()
                  if site == fact.site]
        if fact.control.reachability == 'Unreachable' and groups:
            problems.append(f'{fact.site}: claimed Unreachable but executed')
        for iteration, group in groups:
            if fact.value_level == Level.LaunchUniform:
                values = [v for e in group for v in e['value']['values']]
                if len(set(values)) > 1:
                    problems.append(f'{fact.site} {iteration}: false LaunchUniform')
            elif fact.value_level == Level.ProgramUniform:
                for pid in range(GRID[0]):
                    values = [v for e in group if e['pid'] == pid for v in e['value']['values']]
                    if len(set(values)) > 1:
                        problems.append(f'{fact.site} {iteration}: false ProgramUniform in {pid}')
            if fact.control.launch_participation == 'Full':
                observed = {e['pid'] for e in group}
                if observed != set(range(GRID[0])):
                    problems.append(f'{fact.site} {iteration}: false Launch Full: {sorted(observed)}')
            if fact.control.program_participation == 'Full':
                # Scalar if/for/return is the only accepted control shape.
                # Compare observed logical elements with the complete shape.
                for event in group:
                    complete = [list(index) for index in np.ndindex(tuple(event['value']['shape']))]
                    if event['participants'] != complete:
                        problems.append(f'{fact.site} {iteration}: false Program Full in {event["pid"]}')
    return problems


def inject(summary, mutation):
    targets = {
        'pid-as-launch': ('pid', 'definition', Level.LaunchUniform, None),
        'lane-as-program': ('lane', 'definition', Level.ProgramUniform, None),
        'load-as-program': ('loaded', 'definition', Level.ProgramUniform, None),
        'matrix-as-program': ('matrix', 'definition', Level.ProgramUniform, None),
        'partial-as-program': ('partial', 'definition', Level.ProgramUniform, None),
        'where-selector-as-launch': ('different', 'definition', Level.LaunchUniform, None),
        'merge-without-selector': ('merged', 'definition', Level.LaunchUniform, None),
        'zero-loop-as-launch': ('after', 'definition', Level.LaunchUniform, None),
        'return-as-full': ('after', 'definition', None, 'Full'),
        'pid-loop-as-full': ('carried', 'definition', None, 'Full'),
        'load-branch-as-full': ('selected', 'definition', None, 'Full'),
    }
    name, kind, level, participation = targets[mutation]
    matching = [f for f in summary.facts if f.site.startswith('@definition:') and f.definition is not None
                and f.definition.name == name and f.definition.kind == kind]
    if mutation == 'pid-loop-as-full':
        matching = [f for f in matching if '/body/' in f.site]
    if mutation == 'load-branch-as-full':
        matching = [f for f in matching if '/then/' in f.site]
    if len(matching) != 1:
        raise AssertionError(f'injection {mutation} needs exactly one definition: {matching}')
    target = matching[0]
    corrupt = replace(target, value_level=level) if level is not None else replace(
        target, control=replace(target.control, launch_participation=participation))
    return replace(summary, facts=tuple(corrupt if f is target else f for f in summary.facts))


def run_case(case, mutation=None, config=UniformityConfig()):
    kernel = compile_family(case.family)
    summary = analyze_uniformity(kernel.tk, case.consts, config=config)
    if mutation is not None:
        summary = inject(summary, mutation)
    events, loads = observe(kernel.tk, case, summary)
    expected = reference(case)
    actual = [{key: event[key] for key in ('name', 'pid', 'iteration', 'value')} for event in events
              if event['site'].startswith('@definition:')]
    problems = []
    if actual != expected:
        problems.append('interpreter execution differs from independent reference')
    if loads != (GRID[0] if case.family in ('load', 'eager_where', 'load_branch') else 0):
        problems.append(f'eager read count differs from reference: {loads}')
    problems.extend(check_guarantees(summary, events))
    return kernel, summary, events, expected, loads, problems


def persist(case, kernel, summary, events, expected, loads, problems, root=None, mutation=None):
    root = Path(root or os.environ.get('TILA_UNIFORMITY_FAILURE_DIR', 'artifacts/uniformity-audit'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = root / ('failure-' + stamp + '-' + sha256(repr(case).encode()).hexdigest()[:8])
    directory.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'tila.uniformity-case.v1', 'seed': SEED, 'case': asdict(case),
              'grid': GRID, 'source_sha256': sha256(source_for(case.family).encode()).hexdigest(),
              'bindings': {'consts': case.consts, 'scalars': case.scalars},
              'config': asdict(summary.config), 'minimized': False,
              'mutation': mutation, 'loads': loads,
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'replay': 'PYTHONPATH=src:tests python tests/uniformity_domain_support.py ' + str(directory / 'case.json'),
              'problems': problems, 'expected': expected, 'events': events,
              'facts': [{'site': f.site, 'value_level': f.value_level.name if f.value_level is not None else None,
                         'launch_participation': f.control.launch_participation,
                         'program_participation': f.control.program_participation,
                         'reachability': f.control.reachability}
                        for f in summary.facts if f.definition is not None]}
    (directory / 'case.json').write_text(json.dumps(record, indent=2) + '\n')
    (directory / 'kernel.py').write_text(source_for(case.family))
    (directory / 'kernel.tir.txt').write_text(kernel.tk.dump())
    return directory / 'case.json'


def replay(path):
    record = json.loads(Path(path).read_text())
    if record['schema'] != 'tila.uniformity-case.v1' or record['seed'] != SEED:
        raise ValueError('replay schema or seed drift')
    case = Case(**record['case'])
    if record['bindings'] != {'consts': case.consts, 'scalars': case.scalars}:
        raise ValueError('recorded bindings disagree with case')
    source = source_for(case.family)
    if sha256(source.encode()).hexdigest() != record['source_sha256'] or source != Path(path).with_name('kernel.py').read_text():
        raise ValueError('generated source drift; use recorded source and revision')
    kernel, summary, events, expected, loads, problems = run_case(
        case, record['mutation'], UniformityConfig(**record['config']))
    if expected != record['expected'] or events != record['events'] or loads != record['loads']:
        raise AssertionError('replayed execution differs from recorded event trace')
    if problems != record['problems']:
        raise AssertionError('replayed failure differs from recorded problem list')
    return {'case': asdict(case), 'problems': problems,
            'recorded_problems': record['problems'], 'events': len(events)}


if __name__ == '__main__':
    print(json.dumps(replay(sys.argv[1]), indent=2))
