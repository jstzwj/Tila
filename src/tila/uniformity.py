"""ADR-019 internal logical uniformity. No launch policy or synchronization API.

Facts are conditional on defined execution, not bounds/numeric/race proofs.
The structural definition graph shares ValueRef and Effect IR identities.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import IntEnum
import operator

from . import tir as T, types as TY, dtypes as D, predicates as P
from .effect_ir import OPERANDS
from .effect_summary import ConditionSource, LoopContext, summarize_effects
from .verifier import verify, fail


class Level(IntEnum):
    LaunchUniform = 0
    ProgramUniform = 1
    Varying = 2
    Unknown = 3


L, G, V, U = Level


@dataclass(frozen=True)
class UniformityConfig:
    max_nodes: int = 10000
    max_depth: int = 128
    max_iterations: int = 128

    def __post_init__(self):
        for f in fields(self):
            if type(getattr(self, f.name)) is not int or getattr(self, f.name) < 0:
                raise ValueError(f'{f.name} must be a nonnegative integer')


@dataclass(frozen=True)
class Control:
    selection_level: Level
    launch_participation: str
    program_participation: str
    reachability: str
    path: P.Predicate
    selectors: tuple[str, ...] = ()
    loops: tuple[LoopContext, ...] = ()
    returned: tuple[str, ...] = ()
    reason: str = 'structured-control'
    iteration_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class UniformityFact:
    site: str
    value_level: Level | None
    definition: T.ValueRef | None
    inputs: tuple[str, ...]
    rule: str
    reason: str
    control: Control
    line: int = 0
    source: str = P.STATIC


@dataclass(frozen=True)
class UniformitySummary:
    facts: tuple[UniformityFact, ...]
    accesses: tuple  # Original AccessContext: path/mask/loops remain independent.
    exit_control: Control
    stage: str
    consts: tuple
    config: UniformityConfig
    incomplete_reason: str | None = None
    requires_defined_execution: bool = True
    version: int = 1


# Deliberately explicit: a new semantic node requires a reviewed transfer.
TRANSFERS = {
    T.TName: 'reference', T.TLit: 'literal', T.TConstant: 'constant',
    T.TPid: 'program-id', T.TNumPrograms: 'num-programs', T.TArange: 'lane-index',
    T.TZeros: 'zeros', T.TBufPtr: 'buffer-base', T.TPAdd: 'pointer-add',
    T.TBin: 'elementwise', T.TUna: 'unary', T.TCast: 'cast',
    T.TReshape: 'reshape', T.TExpand: 'expand', T.TWhere: 'eager-select',
    T.TReduce: 'reduction', T.TDot: 'unsupported-dot',
    T.TLoad: 'memory-result', T.TAtomicAdd: 'atomic-result',
    T.TAssign: 'definition', T.TStore: 'store', T.TAtomicStmt: 'atomic-statement',
    T.TAssume: 'assume-no-strengthening', T.TIf: 'branch', T.TStaticIf: 'const-branch',
    T.TFor: 'loop', T.TReturn: 'return',
}


def _key(ref):
    return f'@{ref.kind}:{ref.site}:{ref.name}'


def _and(a, b):
    return P.FALSE if a is P.FALSE or b is P.FALSE else P.conjunction(a, b)


def _or(a, b):
    return b if a is P.FALSE else a if b is P.FALSE else P.disjunction(a, b)


@dataclass(frozen=True)
class _Flow:
    path: P.Predicate = P.TRUE
    selectors: tuple = ()
    loops: tuple = ()
    returned: tuple = ()
    uncertain: bool = False
    iteration_sources: tuple = ()


@dataclass
class _Equation:
    inputs: tuple
    rule: str
    flow: _Flow
    definition: T.ValueRef | None = None
    line: int = 0
    fixed: Level | None = None
    no_value: bool = False


class _Limit(Exception):
    pass


class _Analysis:
    def __init__(self, kernel, consts, config):
        self.kernel, self.consts, self.config = kernel, consts, config
        self.equations = {}
        self.constants = {}

    def put(self, site, inputs, rule, flow, ref=None, line=0, fixed=None, no_value=False):
        if site not in self.equations and len(self.equations) >= self.config.max_nodes:
            raise _Limit('construction-budget')
        self.equations[site] = _Equation(tuple(inputs), rule, flow, ref, line, fixed, no_value)
        return site

    def depth(self, depth):
        if depth > self.config.max_depth:
            raise _Limit('depth-budget')

    def expr(self, node, site, flow, depth=0, line=0):
        self.depth(depth)
        line = getattr(node, 'line', 0) or line
        children = []
        for field in OPERANDS[type(node)]:
            value = getattr(node, field)
            for index, child in (enumerate(value) if isinstance(value, list) else [(None, value)]):
                if child is not None:
                    suffix = field if index is None else f'{field}/{index}'
                    children.append(self.expr(child, site + '/' + suffix, flow, depth + 1, line))
        rule, fixed, ref = TRANSFERS[type(node)], None, None
        if type(node) is T.TName:
            ref = node.definition
            children = [_key(ref)]
        elif type(node) in (T.TLit, T.TConstant, T.TNumPrograms, T.TZeros, T.TBufPtr):
            fixed = L
        elif type(node) is T.TPid:
            fixed = G
        elif type(node) is T.TArange:
            fixed = V
        elif type(node) in (T.TLoad, T.TAtomicAdd, T.TDot):
            fixed = U
        elif type(node) is T.TWhere:
            known = self.constant(node.cond)
            if type(known) is bool:
                children = [site + ('/a' if known else '/b')]
                rule = 'constant-select-value-only'
            elif isinstance(node.a, T.TName) and isinstance(node.b, T.TName) and node.a.definition == node.b.definition:
                children = [site + '/a']
                rule = 'same-definition-select'
        elif type(node) in (T.TBin, T.TUna, T.TCast):
            types = [getattr(node, 'vt', None)]
            for field in OPERANDS[type(node)]:
                child = getattr(node, field)
                types.append(self.vtype(child))
            if any(getattr(getattr(t, 'elem', t), 'dtype', None) in D.FLOAT_DTYPES for t in types):
                fixed, rule = U, 'floating-operation-contract-unavailable'
            if type(node) is T.TUna and node.op in ('any', 'all'):
                rule = 'full-reduction'
        elif type(node) is T.TReduce:
            rule = 'full-reduction' if isinstance(node.vt, TY.ScalarT) else 'partial-reduction'
            if rule == 'partial-reduction' and node.input_dtype.is_float:
                fixed, rule = U, 'floating-reduction-contract-unavailable'
        self.put(site, children, rule, flow, ref, line, fixed)
        if type(node) is T.TLit and type(node.value) in (bool, int):
            self.constants[site] = node.value
        return site

    @staticmethod
    def vtype(node):
        if isinstance(node, T.TName):
            return node.definition.vtype
        if isinstance(node, T.TLit) and node.dtype is not None:
            return TY.ScalarT(node.dtype)
        return getattr(node, 'vt', None)

    def constant(self, node, depth=0):
        # Small staged-only evaluator; never samples scalar bindings or memory.
        if depth > min(self.config.max_depth, 64):
            return None
        if type(node) is T.TLit:
            return node.value if type(node.value) in (int, bool) else None
        if type(node) is T.TName:
            return self.constants.get(_key(node.definition))
        if type(node) not in (T.TBin, T.TUna):
            return None
        if type(node) is T.TUna:
            a = self.constant(node.operand, depth + 1)
            if node.op == 'not' and type(a) is bool:
                return not a
            if node.staged and type(a) is int and node.op in ('+', '-'):
                return a if node.op == '+' else -a
            return None
        a, b = self.constant(node.left, depth + 1), self.constant(node.right, depth + 1)
        if a is None or b is None:
            return None
        comparisons = {'==': operator.eq, '!=': operator.ne, '<': operator.lt,
                       '<=': operator.le, '>': operator.gt, '>=': operator.ge}
        if type(a) is type(b) and node.op in comparisons:
            return comparisons[node.op](a, b)
        if type(a) is type(b) is bool and node.op in ('and', 'or'):
            return a and b if node.op == 'and' else a or b
        if node.staged and type(a) is type(b) is int and max(a.bit_length(), b.bit_length()) <= 256:
            op = {'+': operator.add, '-': operator.sub, '*': operator.mul,
                  '//': operator.floordiv, '%': operator.mod}.get(node.op)
            if op:
                try:
                    value = op(a, b)
                    return value if value.bit_length() <= 256 else None
                except ArithmeticError:
                    pass
        return None

    def guard(self, node, site, flow):
        value = self.constant(node)
        if type(value) is bool:
            return P.TRUE if value else P.FALSE
        ref = node.definition if isinstance(node, T.TName) else None
        return P.Predicate('unknown', atom=ConditionSource(site, ref, tuple(l.site_id for l in flow.loops)))

    @staticmethod
    def assigned(body):
        names = set()
        for node in body:
            if type(node) is T.TAssign:
                names.add(node.name)
            elif type(node) in (T.TIf, T.TStaticIf):
                names |= _Analysis.assigned(node.then_body) | _Analysis.assigned(node.else_body)
            elif type(node) is T.TFor:
                names |= _Analysis.assigned(node.body) | {node.var}
        return names

    def joined(self, left, right, path, flow, selector, kind='merge', selected=None, line=0):
        result = {}
        for name in sorted(left.keys() | right.keys()):
            a, b = left.get(name), right.get(name)
            if a == b:
                result[name] = a
                continue
            types = [r.vtype for r in (a, b) if r is not None]
            vt = types[0] if types and all(t == types[0] for t in types) else None
            ref = T.ValueRef(kind, path + '/' + name, name, vt)
            chosen = (a,) if selected == 0 else (b,) if selected == 1 else (a, b)
            inputs = tuple(_key(r) for r in chosen if r is not None)
            # Missing incoming definitions are never a uniformity fact.
            fixed = U if any(r is None for r in chosen) else None
            # A synthetic merge has no dynamic value on a predecessor without
            # a definition. Its control cannot certify Full participation for
            # a consumer at this ValueRef, even if the block exit reconverges.
            fact_flow = replace(flow, uncertain=True) if fixed == U else flow
            self.put(_key(ref), inputs + (() if selected is not None else tuple(selector)),
                     'selected-definition' if selected is not None else kind + '-join', fact_flow, ref, line, fixed=fixed)
            if selected is not None and chosen[0] is not None:
                self.constants[_key(ref)] = self.constants.get(_key(chosen[0]))
            result[name] = ref
        return result

    def block(self, body, env, flow, path='body', depth=0):
        self.depth(depth)
        env = dict(env)
        falls = True  # Lexical fallthrough, matching effect_ir's ValueRef ownership.
        for index, node in enumerate(body):
            site = f'{path}/{index}'
            line = getattr(node, 'line', 0)
            kind = type(node)
            inputs = []
            for field in OPERANDS[kind]:
                value = getattr(node, field)
                for i, child in (enumerate(value) if isinstance(value, list) else [(None, value)]):
                    if child is not None:
                        suffix = field if i is None else f'{field}/{i}'
                        inputs.append(self.expr(child, site + '/' + suffix, flow, depth + 1, line))
            self.put(site, inputs, TRANSFERS[kind], flow, line=line, no_value=True)
            if kind is T.TAssign:
                ref = T.ValueRef('definition', site, node.name, self.vtype(node.value))
                self.put(_key(ref), (site + '/value',), 'definition', flow, ref, line)
                self.constants[_key(ref)] = self.constant(node.value)
                env[node.name] = ref
            elif kind in (T.TIf, T.TStaticIf):
                predicate = self.guard(node.cond, site + '/cond', flow)
                value = self.constant(node.cond)
                selector = site + '/cond'
                then_flow = replace(flow, path=_and(flow.path, predicate), selectors=flow.selectors + (selector,))
                negated = P.FALSE if predicate is P.TRUE else P.TRUE if predicate is P.FALSE else P.negate(predicate)
                else_flow = replace(flow, path=_and(flow.path, negated), selectors=flow.selectors + (selector,))
                a, af, ac = self.block(node.then_body, env, then_flow, site + '/then', depth + 1)
                b, bf, bc = self.block(node.else_body, env, else_flow, site + '/else', depth + 1)
                selected = 0 if value is True else 1 if value is False else None
                env = self.joined(a, b, site, flow, (selector,), selected=selected, line=line) if af == bf else (a if af else b)
                falls = falls and (af or bf)
                if selected is not None:
                    flow = ac if selected == 0 else bc
                elif (ac.returned == bc.returned == flow.returned and not ac.uncertain and not bc.uncertain
                      and ac.iteration_sources == bc.iteration_sources == flow.iteration_sources):
                    pass  # Both structured arms reconverge without an exit.
                else:
                    flow = replace(flow, path=_or(ac.path, bc.path),
                                   selectors=tuple(dict.fromkeys(ac.selectors + bc.selectors)),
                                   returned=tuple(dict.fromkeys(ac.returned + bc.returned)),
                                   uncertain=ac.uncertain or bc.uncertain,
                                   iteration_sources=tuple(dict.fromkeys(ac.iteration_sources + bc.iteration_sources)))
            elif kind is T.TReturn:
                if flow.path is not P.FALSE:
                    flow = replace(flow, path=P.FALSE, returned=flow.returned + (site,))
                falls = False
            elif kind is T.TFor:
                before = flow
                start = 0 if node.start is None else self.constant(node.start)
                end, step = self.constant(node.end), self.constant(node.step)
                nonempty = None
                if all(type(v) is int for v in (start, end, step)) and step != 0:
                    nonempty = start < end if step > 0 else start > end
                entry = P.FALSE if nonempty is False else P.TRUE if nonempty is True else P.Predicate(
                    'unknown', atom=ConditionSource(site + '/entry', None, tuple(l.site_id for l in flow.loops)))
                induction = T.ValueRef('induction', site, node.var, TY.ScalarT(D.i32))
                loop = LoopContext(site, induction, node.start, node.end, node.step, entry)
                loop_flow = replace(flow, path=_and(flow.path, entry), loops=flow.loops + (loop,),
                                    selectors=flow.selectors + tuple(inputs))
                self.put(_key(induction), inputs, 'induction', loop_flow, induction, line)
                inside = dict(env)
                carried = []
                for name in sorted(self.assigned(node.body) & env.keys()):
                    ref = T.ValueRef('loop', site + '/carried/' + name, name, env[name].vtype)
                    carried.append((name, ref))
                    inside[name] = ref
                inside[node.var] = induction
                after, _, body_flow = self.block(node.body, inside, loop_flow, site + '/body', depth + 1)
                returned = body_flow.returned != before.returned and nonempty is not False
                for name, ref in carried:
                    self.put(_key(ref), (_key(env[name]), _key(after[name])) + tuple(inputs),
                             'loop-carried', loop_flow, ref, line, U if returned else None)
                env = self.joined(env, after, site + '/exit', before, inputs, 'loop',
                                  selected=0 if nonempty is False else 1 if nonempty is True else None, line=line)
                if returned:
                    # No iteration quantification: refuse to infer convergence.
                    flow = replace(before, uncertain=True, selectors=before.selectors + tuple(inputs),
                                   returned=body_flow.returned)
                    for key, equation in self.equations.items():
                        if any(l.site_id == site for l in equation.flow.loops):
                            equation.flow = replace(equation.flow, uncertain=True)
                        if equation.definition and equation.definition.kind == 'loop' and equation.definition.site.startswith(site + '/exit/'):
                            equation.fixed = U
                else:
                    flow = before if nonempty is False else replace(
                        before, iteration_sources=before.iteration_sources + tuple(inputs))
        return env, falls, flow

    def build(self):
        env = {}
        for param in self.kernel.buffers + self.kernel.ptr_params + self.kernel.scalars + self.kernel.consts:
            vt = TY.ScalarT(param.dtype) if hasattr(param, 'dtype') else getattr(param, 'vtype', None)
            ref = T.ValueRef('parameter', 'parameter/' + param.name, param.name, vt)
            env[param.name] = ref
            self.put(_key(ref), (), 'parameter', _Flow(), ref, fixed=L)
            if param.name in self.consts:
                self.constants[_key(ref)] = self.consts[param.name]
        _, _, flow = self.block(self.kernel.body, env, _Flow())
        return flow

    def solve(self):
        # Least fixed point in weakening order. Nothing is published until the
        # entire graph has converged; all cycles have explicit loop entry edges.
        levels = {key: L for key in self.equations}
        for _ in range(self.config.max_iterations):
            updated = {}
            for key, eq in self.equations.items():
                value = max((levels.get(i, U) for i in eq.inputs), default=L)
                if eq.rule == 'full-reduction' and value != U:
                    value = G
                if eq.rule == 'induction' and value >= V:
                    value = U
                if eq.fixed is not None:
                    value = eq.fixed
                updated[key] = max(levels[key], value)
            if updated == levels:
                return levels
            levels = updated
        raise _Limit('fixed-point-budget')


def _control(flow, levels, incomplete=None):
    level = max((levels.get(s, U) for s in flow.selectors), default=L)
    unknown_iterations = any(levels.get(s, U) >= V for s in flow.iteration_sources)
    if flow.uncertain or incomplete or unknown_iterations:
        level = U
    reachability = ('Unknown' if incomplete else 'Unreachable' if flow.path is P.FALSE
                    else 'Unknown' if flow.uncertain or unknown_iterations
                    else 'Reachable' if flow.path is P.TRUE else 'Unknown')
    def participation(required):
        if reachability == 'Unreachable':
            return 'None'
        if level == U:
            return 'Unknown'
        return 'Full' if level <= required else 'Conditional'
    return Control(level, participation(L), participation(G), reachability, flow.path,
                   flow.selectors, flow.loops, flow.returned,
                   incomplete or ('loop-exit-not-modeled' if flow.uncertain else
                                  'loop-participation-unknown' if unknown_iterations else 'structured-control'),
                   flow.iteration_sources)


def analyze_uniformity(kernel, consts=None, *, config=UniformityConfig()):
    """Read-only, uncached logical analysis. Accepts Const facts only.

    Verification precedes analysis budgets. Exhaustion publishes no partial
    strong guarantees; missing facts must be treated as Unknown by consumers.
    """
    verify(kernel, consts, capability=None)
    if set(TRANSFERS) != set(OPERANDS):
        fail('uniformity transfer coverage disagrees with Effect IR')
    effects = summarize_effects(kernel, consts)  # Also validates exact Const types.
    bindings = {} if consts is None else dict(consts)
    declarations = {p.name: p for p in kernel.consts}
    for name, value in bindings.items():
        if any(not refinement.check(value) for refinement in declarations[name].refinements):
            raise ValueError('invalid uniformity Const refinement: ' + name)
    analysis = _Analysis(kernel, bindings, config)
    incomplete, flow = None, _Flow(uncertain=True)
    try:
        flow = analysis.build()
        levels = analysis.solve()
    except (_Limit, RecursionError) as exc:
        incomplete = str(exc) if isinstance(exc, _Limit) else 'depth-budget'
        levels = {key: U for key in analysis.equations}
    facts = []
    for key, eq in sorted(analysis.equations.items()):
        level = None if eq.no_value else levels[key]
        reason = incomplete or (eq.rule if eq.fixed == U else 'unknown-dependency' if level == U else eq.rule)
        facts.append(UniformityFact(key, level, eq.definition, eq.inputs, eq.rule, reason,
                                   _control(eq.flow, levels, incomplete), eq.line))
    return UniformitySummary(tuple(facts), effects.accesses, _control(flow, levels, incomplete),
                             effects.stage, tuple(sorted(bindings.items())), config, incomplete)


def _equivalent(left, right):
    """Structural comparison of identity-shared DAGs, without exponential expansion."""
    todo, seen = [(left, right)], set()
    while todo:
        a, b = todo.pop()
        key = (id(a), id(b))
        if key in seen:
            continue
        seen.add(key)
        if type(a) is not type(b):
            return False
        if is_dataclass(a):
            todo.extend((getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
        elif isinstance(a, (tuple, list)):
            if len(a) != len(b):
                return False
            todo.extend(zip(a, b))
        elif a != b:
            return False
    return True


def verify_uniformity(kernel, summary, consts=None, *, config=UniformityConfig()):
    """Reject missing, stale or forged derived facts rather than repairing them."""
    expected = analyze_uniformity(kernel, consts, config=config)
    if type(summary) is not UniformitySummary or not _equivalent(summary, expected):
        fail('invalid or stale uniformity summary')
