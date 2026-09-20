"""ADR-018 bounded memory conflict analysis; policy/cache live in race_policy.

Consumes verified Effect IR; never interprets the serial CPU result as race safety.
Unknown dataflow and unmodeled intra-program ordering stay unknown.
"""
from dataclasses import dataclass
from time import monotonic

from . import tir as T, types as TY, dtypes as D, predicates as P
from .dims import eval_num
from .effect_ir import OPERANDS
from .effect_summary import summarize_effects
from .facts import PROVEN_SAFE, PROVEN_UNSAFE, UNKNOWN
from .solver import ProofConfig, Limit, Unsupported, z3_module, _LOCK
from .verifier import verify


@dataclass(frozen=True)
class RaceConfig:
    proof: ProofConfig = ProofConfig()
    max_pairs: int = 4096
    include_intra: bool = False

    def __post_init__(self):
        if type(self.max_pairs) is not int or self.max_pairs < 0:
            raise ValueError('max_pairs must be a nonnegative integer')
        if type(self.include_intra) is not bool:
            raise ValueError('include_intra must be bool')


@dataclass(frozen=True)
class RaceWitness:
    pids: tuple
    lanes: tuple
    # Relative to each named view; no process addresses in public results.
    byte_offsets: tuple
    parameters: tuple
    iterations: tuple = ()  # per side: ((loop site, induction value), ...)
    inputs: tuple = ()  # shared scalar/Const values; no tensor contents
    grid: tuple = ()


@dataclass(frozen=True)
class RacePairResult:
    sites: tuple
    verdict: str
    reason: str
    dependencies: tuple = ()
    candidate: RaceWitness | None = None
    confirmed: RaceWitness | None = None
    query: str | None = None
    domain: str = 'InterProgram'


@dataclass(frozen=True)
class RaceReport:
    pairs: tuple
    unchecked_domains: tuple = ('IntraProgram',)
    incomplete_reason: str | None = None
    unchecked_pairs: int = 0
    stage: str = 'bound'  # symbolic/pending if grid or memory bindings missing
    policy: str | None = None
    suppressions: tuple = ()
    cache_status: str = 'disabled'

    @property
    def verdict(self):
        """Covered-pair verdict; callers must also inspect unchecked_domains."""
        if any(p.verdict == PROVEN_UNSAFE for p in self.pairs):
            return PROVEN_UNSAFE
        if self.incomplete_reason or self.suppressions or any(p.verdict == UNKNOWN for p in self.pairs):
            return UNKNOWN
        return PROVEN_SAFE


@dataclass(frozen=True)
class _View:
    base: int
    size: int
    stride: int  # bytes, includes actual view layout
    width: int
    device: str


class _Session:
    def __init__(self, config):
        self.config = config.proof
        self.deadline = monotonic() + self.config.total_ms / 1000
        self.nodes = self.queries = 0
        self.z = z3_module()

    def tick(self, depth=0):
        self.nodes += 1
        if monotonic() >= self.deadline:
            raise Limit('total-budget')
        if self.nodes > self.config.max_nodes or depth > self.config.max_depth:
            raise Limit('construction-budget')

    def integer(self, n):
        if type(n) is not int or n.bit_length() > self.config.max_integer_bits:
            raise Limit('integer-budget')
        return n

    def check(self, formula):
        self.tick()
        c, z = self.config, self.z
        if not c.timeout_ms or not c.rlimit or self.queries >= c.max_queries:
            raise Limit('query-budget')
        self.queries += 1
        solver = z.Solver()
        solver.set(timeout=max(1, min(c.timeout_ms, int((self.deadline - monotonic()) * 1000))),
                   rlimit=c.rlimit)
        solver.add(formula)
        query = solver.to_smt2()
        if len(query.encode()) > c.max_query_bytes:
            raise Limit('query-bytes-budget')
        result = solver.check()
        self.tick()
        if result == z.unknown:
            raise Limit('solver-unknown')
        return result, solver.model() if result == z.sat else None, query


def _index(kernel, session):
    nodes, definitions = {}, {}

    def walk(node, site, depth):
        session.tick(depth)
        nodes[site] = node
        if isinstance(node, T.TAssign):
            definitions[site] = node.value
        for field in OPERANDS[type(node)]:
            value = getattr(node, field)
            children = enumerate(value) if isinstance(value, list) else [(None, value)]
            for i, child in children:
                if child is not None:
                    walk(child, site + '/' + field + ('' if i is None else f'/{i}'), depth + 1)
        if isinstance(node, (T.TIf, T.TStaticIf)):
            block(node.then_body, site + '/then', depth + 1)
            block(node.else_body, site + '/else', depth + 1)
        elif isinstance(node, T.TFor):
            block(node.body, site + '/body', depth + 1)

    def block(body, site, depth):
        for i, node in enumerate(body):
            walk(node, f'{site}/{i}', depth)
    block(kernel.body, 'body', 0)
    return nodes, definitions


def _views(kernel, bindings, values, session):
    """Validate metadata only. No tensor values read and no device work launched."""
    import numpy as np
    result = {}
    for p in kernel.buffers + kernel.ptr_params:
        if p.name not in bindings:
            continue
        a, dt = bindings[p.name], p.vtype.elem
        if isinstance(a, np.ndarray):
            if dt is D.bf16 and str(a.dtype) == 'bfloat16':
                raise Unsupported('bfloat16-numpy-binding')
            if dt.np_dtype is None or a.dtype != np.dtype(dt.np_dtype):
                raise ValueError('race binding dtype mismatch: ' + p.name)
            shape, strides = a.shape, a.strides
            base, width, device = int(a.ctypes.data), a.itemsize, 'cpu'
        else:
            # Torch metadata, including CUDA, without importing/copying storage.
            from .runtime import torch, _TORCH_DT
            if torch is None or not isinstance(a, torch.Tensor) or _TORCH_DT.get(str(a.dtype).removeprefix('torch.')) is not dt:
                raise ValueError('unsupported race binding: ' + p.name)
            if a.device.type not in ('cpu', 'cuda') or a.layout != torch.strided:
                raise ValueError('race binding requires real strided CPU/CUDA storage')
            shape, width = tuple(a.shape), a.element_size()
            strides = tuple(s * width for s in a.stride())
            base, device = a.data_ptr(), str(a.device)
        if len(shape) != 1 or strides[0] <= 0 or strides[0] % width:
            raise Unsupported('binding-layout')
        if isinstance(p, T.TPtrParam) and strides[0] != width:
            raise Unsupported('pointer-layout')
        if p.name in kernel.buffer_ptr_views and strides[0] != width:
            raise Unsupported('buffer-pointer-layout')
        if p.vtype.aligned is not None and base % p.vtype.aligned:
            raise ValueError('race binding alignment mismatch: ' + p.name)
        if shape[0] > 2147483647 or strides[0] // width > 2147483647:
            raise ValueError('race binding index metadata exceeds i32')
        if isinstance(p, T.TBufferParam) and len(p.vtype.shape) != 1:
            raise Unsupported('binding-rank')
        if isinstance(p, T.TPtrParam) and not isinstance(p.vtype.extent, TY.LinearExtent):
            raise Unsupported('unknown-extent')
        expected = (eval_num(p.vtype.shape[0], values) if isinstance(p, T.TBufferParam)
                    else eval_num(p.vtype.extent.expr, values))
        if (shape[0] != expected if isinstance(p, T.TBufferParam) else shape[0] < expected):
            raise ValueError('race binding extent mismatch: ' + p.name)
        for n in (base, int(shape[0]), strides[0]):
            session.integer(n)
        result[p.name] = _View(base, expected, strides[0], width, device)
    return result


class _Encoder:
    def __init__(self, session, nodes, definitions, values, grid, views, side, concrete=None):
        self.s, self.z = session, session.z
        self.nodes, self.definitions, self.values = nodes, definitions, values
        self.grid, self.views, self.side = grid, views, side
        self.concrete = concrete
        self.pid = tuple(self.z.Int(f'{side}_pid_{i}') for i in range(3)) if concrete is None else concrete[0]
        self.lane = self.z.Int(f'{side}_lane') if concrete is None else concrete[1]
        self.constraints, self.exact = [], True
        self.induction = {}
        self.has_lane = False

    def boolean(self, op, *args):
        if self.concrete is not None:
            return (all(args) if op == 'and' else any(args) if op == 'or' else not args[0])
        return {'and': self.z.And, 'or': self.z.Or, 'not': self.z.Not}[op](*args)

    def wrap(self, value, dt):
        if dt is D.bool_:
            if type(value) is bool or self.z.is_bool(value):
                return value
            return value != 0
        if dt is None or not dt.is_int:
            raise Unsupported('noninteger-expression')
        if type(value) is bool:
            value = int(value)
        elif self.z.is_bool(value):
            value = self.z.If(value, 1, 0)
        if self.concrete is not None or type(value) is int:
            value %= 1 << dt.bits
            return value - (1 << dt.bits) if dt.kind == 'int' and value >= 1 << (dt.bits - 1) else value
        # Exact fixed-width normalization in integer arithmetic. This is not an
        # unbounded affine rewrite: every operation retains its modulo step.
        bias = (1 << (dt.bits - 1)) if dt.kind == 'int' else 0
        return (value + bias) % (1 << dt.bits) - bias

    def expr(self, node, depth=0):
        self.s.tick(depth)
        e = lambda n: self.expr(n, depth + 1)
        if isinstance(node, T.TLit):
            if type(node.value) not in (int, bool):
                raise Unsupported('noninteger-literal')
            return node.value if type(node.value) is bool else self.s.integer(node.value)
        if isinstance(node, T.TName):
            ref = node.definition
            if ref.kind == 'parameter' and ref.name in self.values:
                return self.values[ref.name]
            if ref.kind == 'definition':
                return e(self.definitions[ref.site])
            if ref.kind == 'induction' and ref.site in self.induction:
                return self.induction[ref.site]
            raise Unsupported('unbound-or-merged-definition')
        if isinstance(node, T.TPid):
            return self.pid[node.axis]
        if isinstance(node, T.TNumPrograms):
            return self.grid[node.axis]
        if isinstance(node, T.TArange):
            self.has_lane = True
            end = e(node.end)
            if type(end) is not int or type(node.start) is not int:
                raise Unsupported('symbolic-lane-domain')
            self.constraints.extend((self.lane >= 0, self.lane < end - node.start))
            return self.lane + node.start
        if isinstance(node, T.TCast):
            return self.wrap(e(node.operand), node.dtype)
        if isinstance(node, T.TUna):
            value = e(node.operand)
            boolean_type = (getattr(node.vt, 'dtype', None) is D.bool_
                            or isinstance(node.vt, TY.MaskT))
            if node.op == 'not' or node.op == '~' and boolean_type:
                return self.boolean('not', value)
            if node.op in ('-', '+'):
                dt = getattr(getattr(node.vt, 'elem', node.vt), 'dtype', None)
                result = -value if node.op == '-' else value
                return result if node.staged else self.wrap(result, dt)
        if isinstance(node, T.TBin):
            a, b = e(node.left), e(node.right)
            op = node.op
            if op in ('and', 'or'):
                return self.boolean(op, a, b)
            if op in ('==', '!=', '<', '<=', '>', '>='):
                return {'==': lambda: a == b, '!=': lambda: a != b, '<': lambda: a < b,
                        '<=': lambda: a <= b, '>': lambda: a > b, '>=': lambda: a >= b}[op]()
            if op in ('&', '|') and (getattr(node.vt, 'dtype', None) is D.bool_ or isinstance(node.vt, TY.MaskT)):
                return self.boolean('and' if op == '&' else 'or', a, b)
            if op not in ('+', '-', '*') or op == '*' and not (type(a) is int or type(b) is int):
                raise Unsupported('nonaffine-or-unsupported-operation')
            value = a + b if op == '+' else a - b if op == '-' else a * b
            dt = getattr(getattr(node.vt, 'elem', node.vt), 'dtype', None)
            return value if node.staged else self.wrap(value, dt)
        raise Unsupported('unsupported-expression:' + type(node).__name__)

    def guard(self, predicate, depth=0):
        self.s.tick(depth)
        if predicate.op in ('true', 'false'):
            return predicate.op == 'true'
        if predicate.op in ('and', 'or', 'not'):
            return self.boolean(predicate.op, *(self.guard(p, depth + 1) for p in predicate.args))
        source = predicate.atom
        try:
            ref = source.definition
            node = T.TName(ref.name, ref) if ref else self.nodes[source.site]
            return self.expr(node, depth + 1)
        except (Unsupported, KeyError, AttributeError):
            self.exact = False
            if self.concrete is not None:
                raise Unsupported('opaque-guard')
            # Independent overapproximation, not shared across loop iterations or sides.
            return self.z.FreshBool(self.side + '_opaque')

    def pointer(self, node, depth=0):
        self.s.tick(depth)
        if isinstance(node, T.TName):
            ref = node.definition
            if ref.kind == 'parameter':
                return ref.name, 0
            if ref.kind == 'definition':
                return self.pointer(self.definitions[ref.site], depth + 1)
        if isinstance(node, T.TBufPtr):
            return node.buffer, 0
        if isinstance(node, T.TPAdd):
            name, offset = self.pointer(node.ptr, depth + 1)
            return name, offset + self.expr(node.offset, depth + 1)
        raise Unsupported('unknown-pointer')

    def access(self, context):
        if not context.may_access:
            return None
        for loop in context.loops:
            start = 0 if loop.start is None else self.expr(loop.start)
            end, step = self.expr(loop.end), self.expr(loop.step)
            if not all(type(v) is int for v in (start, end, step)) or step <= 0:
                raise Unsupported('nonconstant-loop-domain')
            index = (self.z.Int(f'{self.side}_iteration_{len(self.induction)}') if self.concrete is None
                     else dict(self.concrete[2])[loop.site_id])
            self.induction[loop.site_id] = index
            self.constraints.extend((index >= start, index < end, (index - start) % step == 0))
            if any(isinstance(node, T.TReturn) and site.startswith(loop.site_id + '/body/')
                   for site, node in self.nodes.items()):
                self.exact = False
        n = context.node
        if n.buffer is not None:
            if len(n.coords) != 1:
                raise Unsupported('access-rank')
            name, offset = n.buffer, self.expr(n.coords[0])
        else:
            name, offset = self.pointer(n.ptr)
        view = self.views.get(name)
        if view is None:
            raise Unsupported('pending-memory-binding')
        active = self.boolean('and', self.guard(context.path), self.guard(context.mask))
        # A scalar has one logical lane. All tile expressions use this side's lane.
        if not self.has_lane:
            self.constraints.append(self.lane == 0)
        domain = self.boolean('and', *self.constraints,
                              *(self.boolean('and', p >= 0, p < g) for p, g in zip(self.pid, self.grid)))
        return name, offset * view.stride, active, domain, self.boolean('and', offset >= 0, offset < view.size)


def analyze_races(kernel, *, grid=None, bindings=None, consts=None, scalars=None, config=None):
    """Internal analysis only; does not enforce policy, execute, cache, or mutate TIR.

    Explicit scalar/Const bindings describe a single specialization. Missing facts
    yield Unknown, not assumptions. Memory bindings are actual NumPy/Torch views.
    """
    config = RaceConfig() if config is None else config
    with _LOCK:
        return _analyze(kernel, grid, bindings or {}, consts or {}, scalars or {}, config)


def _analyze(kernel, grid, bindings, consts, scalars, config):
    verify(kernel, capability=None)
    declarations = {p.name: p for p in kernel.consts}
    if consts.keys() - declarations.keys() or scalars.keys() - {p.name for p in kernel.scalars}:
        raise ValueError('unknown race binding')
    for name, value in consts.items():
        if type(value) is not (bool if declarations[name].value_kind == 'Bool' else int):
            raise ValueError('invalid Const binding')
        if any(not r.check(value) for r in declarations[name].refinements):
            raise ValueError('Const refinement violated')
    for p in kernel.scalars:
        if p.name in scalars:
            value = scalars[p.name]
            if p.dtype is D.bool_:
                valid = type(value) is bool
            else:
                valid = p.dtype.is_int and type(value) is int and D._INT_RANGE[p.dtype][0] <= value <= D._INT_RANGE[p.dtype][1]
            if not valid:
                raise ValueError('invalid scalar binding: ' + p.name)
            if any(not r.check(value) for r in p.refined):
                raise ValueError('scalar refinement violated: ' + p.name)
    if declarations.keys() <= consts.keys():
        verify(kernel, consts, capability=None)
    if grid is not None:
        if not 1 <= len(grid) <= 3 or any(type(g) is not int or g < 0 or g > (2147483647 if i == 0 else 65535) for i, g in enumerate(grid)):
            raise ValueError('invalid race grid')
        grid = tuple(grid) + (1,) * (3 - len(grid))
    summary = summarize_effects(kernel, consts)
    accesses = summary.accesses
    total = len(accesses) * (len(accesses) + 1) // 2 * (2 if config.include_intra else 1)
    unchecked = () if config.include_intra else ('IntraProgram',)
    session, results = _Session(config), []
    values = dict(scalars, **consts)
    try:
        for value in values.values():
            if type(value) is int:
                session.integer(value)
        nodes, definitions = _index(kernel, session)
        views = _views(kernel, bindings, values, session)
        if len({v.device for v in views.values()}) > 1:
            raise Unsupported('mixed-device-bindings')
        atomic_regions = {a.effect.region_id for a in accesses if a.effect.atomic}
        for p in kernel.buffers + kernel.ptr_params:
            region = p.region_id if isinstance(p, T.TBufferParam) else p.vtype.region_id
            if region in atomic_regions and p.name in views and views[p.name].base % 4:
                raise ValueError('atomic binding requires natural alignment')
    except (Limit, Unsupported, KeyError) as exc:
        return RaceReport((), unchecked_domains=unchecked, incomplete_reason=str(exc), unchecked_pairs=total,
                          stage='pending' if grid is None else 'bound')
    z = session.z
    stage = 'pending' if grid is None or len(views) != len(kernel.buffers + kernel.ptr_params) else 'bound'
    origins = (P.Origin(P.STATIC, detail='verified Effect IR'),
               P.Origin(P.CHECKED, detail='validated analysis binding metadata'))
    numeric_valid = False
    if grid is not None:
        from .numeric import validate
        from .errors import TilaError
        try:
            validate(kernel, consts, scalars, grid)
            numeric_valid = True
        except TilaError:
            # A pending/failed arithmetic gate cannot support a confirmed race.
            pass

    def encoder(side, concrete=None):
        return _Encoder(session, nodes, definitions, values, grid, views, side, concrete)

    def scalar_access(a):
        operands = a.node.coords if a.node.buffer is not None else [a.node.ptr]
        return all(not isinstance(getattr(n, 'vt', getattr(getattr(n, 'definition', None), 'vtype', None)),
                                  (TY.BlockT, TY.MaskT)) for n in operands)

    def pair(a, b, domain):
        sites = (a.effect.site_id, b.effect.site_id)
        def result(verdict, reason, **kw):
            return RacePairResult(sites, verdict, reason, origins, domain=domain, **kw)
        if not a.may_access or not b.may_access:
            return result(PROVEN_SAFE, 'unreachable-access')
        if a.effect.kind == b.effect.kind == 'Read':
            return result(PROVEN_SAFE, 'read-only-pair')
        if domain == 'IntraProgram' and scalar_access(a) and scalar_access(b):
            return result(PROVEN_SAFE, 'ordered-scalar-accesses')
        if grid is None:
            return result(UNKNOWN, 'pending-grid')
        if 0 in grid or domain == 'InterProgram' and grid == (1, 1, 1):
            return result(PROVEN_SAFE, 'no-distinct-programs')
        ea, eb = encoder('a'), encoder('b')
        aa, bb = ea.access(a), eb.access(b)
        na, oa, pa, da, va = aa
        nb, ob, pb, db, vb = bb
        av, bv = views[na], views[nb]
        if av.device != bv.device:
            raise Unsupported('mixed-device-bindings')
        # Normalize relative bases so replay queries do not expose host pointers.
        delta = bv.base - av.base
        if av.size == 0 or bv.size == 0:
            return result(UNKNOWN, 'empty-memory-extent')
        overlap = z.And(oa < delta + ob + bv.width, delta + ob < oa + av.width)
        instances = (z.Or(*(p != q for p, q in zip(ea.pid, eb.pid))) if domain == 'InterProgram'
                     else z.And(*(p == q for p, q in zip(ea.pid, eb.pid)),
                                ea.lane != eb.lane if a is b else True))
        conflict = z.And(da, db, pa, pb, instances, overlap)
        # Compatible atomic pairs are excluded only after ruling out partial overlap.
        atomic = (a.effect.atomic is not None and a.effect.atomic == b.effect.atomic
                  and a.effect.element_dtype is b.effect.element_dtype and av.width == bv.width)
        if atomic:
            partial, _, _ = session.check(z.And(conflict, oa != delta + ob))
            if partial == z.unsat:
                return result(PROVEN_SAFE, 'compatible-atomic-pair')
        answer, model, query = session.check(conflict)
        if answer == z.unsat:
            return result(PROVEN_SAFE, 'disjoint-or-unreachable', query=query)
        if domain == 'IntraProgram' and a is b and a.effect.kind == 'Write' and a.loops:
            # Confirm only duplicate lanes in the SAME dynamic store. Different
            # iterations remain an ordering question even if a SAT model exists.
            same_iteration = z.And(*(ea.induction[site] == eb.induction[site] for site in ea.induction))
            answer_same, same_model, same_query = session.check(z.And(conflict, same_iteration))
            if answer_same == z.sat:
                model, query = same_model, same_query
            else:
                return result(UNKNOWN, 'intra-program-iteration-order-not-modeled', query=query)
        def number(value):
            return value if type(value) is int else model.eval(value, model_completion=True).as_long()
        witness = RaceWitness(tuple(tuple(number(p) for p in e.pid) for e in (ea, eb)),
                              (number(ea.lane), number(eb.lane)), (number(oa), number(ob)), (na, nb),
                              tuple(tuple((site, number(v)) for site, v in e.induction.items()) for e in (ea, eb)),
                              tuple(sorted(values.items())), grid)
        if domain == 'IntraProgram' and (a is not b or a.effect.kind != 'Write'):
            return result(UNKNOWN, 'intra-program-order-not-modeled', candidate=witness, query=query)
        if not ea.exact or not eb.exact:
            return result(UNKNOWN, 'candidate-path-not-confirmed', candidate=witness, query=query)
        if not numeric_valid:
            return result(UNKNOWN, 'numeric-gate-not-confirmed', candidate=witness, query=query)
        # Definedness of arbitrary value computations and short-circuit side
        # effects is not yet modeled by the may-effect summary.
        if any(isinstance(n, T.TBin) and n.op in ('/', '//', '%', '<<', '>>', 'and', 'or')
               or isinstance(n, T.TCast) and not n.dtype.is_int and n.dtype is not D.bool_
               for n in nodes.values()):
            return result(UNKNOWN, 'execution-definedness-not-confirmed', candidate=witness, query=query)
        # Prior invalid/opaque memory events can invalidate reachability. Prove all
        # active accesses valid independently, without assuming bounds/unsafe facts.
        for context in accesses:
            if not context.may_access:
                continue
            ec = encoder('valid')
            try:
                _, _, active, valid_domain, valid = ec.access(context)
            except Unsupported:
                return result(UNKNOWN, 'kernel-reachability-not-confirmed', candidate=witness, query=query)
            if not ec.exact:
                return result(UNKNOWN, 'kernel-reachability-not-confirmed', candidate=witness, query=query)
            invalid, _, _ = session.check(z.And(valid_domain, active, z.Not(valid)))
            if invalid != z.unsat:
                return result(UNKNOWN, 'access-validity-not-confirmed', candidate=witness, query=query)
        # Independent Python integer evaluation of both concrete events, including
        # finite-width arithmetic and guards; no execution of the racy kernel.
        concrete = [encoder(side, (pid, lane, iterations)).access(context)
                    for side, pid, lane, iterations, context in
                    zip(('a', 'b'), witness.pids, witness.lanes, witness.iterations, (a, b))]
        ca, cb = concrete
        if not (ca[2] and ca[3] and ca[4] and cb[2] and cb[3] and cb[4]
                and ca[1] < delta + cb[1] + bv.width and delta + cb[1] < ca[1] + av.width):
            return result(UNKNOWN, 'witness-evaluation-failed', candidate=witness, query=query)
        return result(PROVEN_UNSAFE, 'confirmed-overlap', confirmed=witness, query=query)

    for domain in (('InterProgram', 'IntraProgram') if config.include_intra else ('InterProgram',)):
        for i, a in enumerate(accesses):
            for j in range(i, len(accesses)):
                b = accesses[j]
                if len(results) >= config.max_pairs:
                    return RaceReport(tuple(results), unchecked_domains=unchecked, incomplete_reason='pair-budget',
                                      unchecked_pairs=total - len(results), stage=stage)
                try:
                    session.nodes = 0
                    session.tick()
                    results.append(pair(a, b, domain))
                except Limit as exc:
                    return RaceReport(tuple(results), unchecked_domains=unchecked, incomplete_reason=str(exc),
                                      unchecked_pairs=total - len(results), stage=stage)
                except (Unsupported, KeyError) as exc:
                    results.append(RacePairResult((a.effect.site_id, b.effect.site_id), UNKNOWN, str(exc), origins, domain=domain))
    return RaceReport(tuple(results), unchecked_domains=unchecked, stage=stage)
