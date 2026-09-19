"""Conservative may-access contexts derived from verified TIR (ADR-016).

No solver, memory-content inference, launch facts, or user-assumption pruning.
Each call owns its constant environment and opaque predicates; no summary cache.
"""
from dataclasses import dataclass
import operator

from . import tir as T, predicates as P, types as TY, dtypes as D
from .effect_ir import OPERANDS, verify_effects


@dataclass(frozen=True)
class ConditionSource:
    site: str
    definition: T.ValueRef | None
    loops: tuple[str, ...]


@dataclass(frozen=True)
class LoopContext:
    site_id: str
    induction: T.ValueRef
    start: object
    end: object
    step: object
    may_enter: P.Predicate


@dataclass(frozen=True)
class AccessContext:
    effect: T.MemoryEffect
    node: object
    path: P.Predicate
    mask: P.Predicate
    loops: tuple[LoopContext, ...]

    @property
    def may_access(self):
        return self.path is not P.FALSE and self.mask is not P.FALSE


@dataclass(frozen=True)
class EffectSummary:
    stage: str
    accesses: tuple[AccessContext, ...]  # Includes explicitly dead sites for audit.

    @property
    def effects(self):
        return tuple(T.TEffect(a.effect.kind, a.effect.region_id)
                     for a in self.accesses if a.may_access)


def _and(a, b):
    return P.FALSE if a is P.FALSE or b is P.FALSE else P.conjunction(a, b)


def _or(a, b):
    if a is P.FALSE:
        return b
    if b is P.FALSE:
        return a
    return P.disjunction(a, b)


def _not(a):
    return P.FALSE if a is P.TRUE else P.TRUE if a is P.FALSE else P.negate(a)


UNKNOWN = object()


def summarize_effects(kernel, consts=None):
    # Verification never repairs stale metadata. Mutation requires explicit
    # rebinding by the TIR producer before requesting a new summary.
    verify_effects(kernel)
    supplied = {} if consts is None else dict(consts)
    declarations = {p.name: p for p in kernel.consts}
    if supplied.keys() - declarations.keys():
        raise ValueError('effect specialization accepts Const bindings only')
    for name, value in supplied.items():
        expected = bool if declarations[name].value_kind == 'Bool' else int
        if type(value) is not expected:
            raise ValueError(f'invalid effect Const binding: {name}')
    env = {name: supplied.get(name, UNKNOWN) for name in declarations}
    accesses, opaque = [], {}

    def unknown(site, node, loops):
        definition = node.definition if isinstance(node, T.TName) else None
        # A loop predicate represents an arbitrary current iteration, not one
        # value shared across all iterations. Loop exit gets a fresh predicate.
        source = ConditionSource(site if definition is None else definition.site,
                                 definition, tuple(loop.site_id for loop in loops))
        key = (source.site, source.loops,
               None if definition is None else (definition.kind, definition.name))
        if key not in opaque:
            vt = getattr(node, 'vt', None)
            if definition is not None:
                vt = definition.vtype
            opaque[key] = P.Predicate('unknown', atom=source,
                                        shape=getattr(vt, 'dims', ()))
        return opaque[key]

    def evaluate(node, values, budget=None):
        # Only literal/Const values and staged arithmetic are folded. Never
        # apply Python unbounded arithmetic to runtime fixed-width operations.
        budget = [256] if budget is None else budget
        budget[0] -= 1
        if budget[0] < 0:
            return UNKNOWN
        if isinstance(node, T.TLit):
            return node.value if type(node.value) in (bool, int) else UNKNOWN
        if isinstance(node, T.TName):
            return values.get(node.name, UNKNOWN)
        if isinstance(node, T.TUna):
            a = evaluate(node.operand, values, budget)
            if a is UNKNOWN:
                return UNKNOWN
            if node.op == 'not' and type(a) is bool:
                return not a
            if node.staged and node.op == '-':
                return -a
            return UNKNOWN
        if isinstance(node, T.TBin):
            a = evaluate(node.left, values, budget)
            if node.op == 'and' and a is False:
                return False
            if node.op == 'or' and a is True:
                return True
            b = evaluate(node.right, values, budget)
            if a is UNKNOWN or b is UNKNOWN:
                return UNKNOWN
            comparisons = {'==': operator.eq, '!=': operator.ne, '<': operator.lt,
                           '<=': operator.le, '>': operator.gt, '>=': operator.ge}
            if node.op in comparisons:
                return comparisons[node.op](a, b)
            if node.op in ('and', 'or') and type(a) is type(b) is bool:
                return (a and b) if node.op == 'and' else (a or b)
            if node.staged and type(a) is type(b) is int and max(a.bit_length(), b.bit_length()) <= 2048:
                operation = {'+': operator.add, '-': operator.sub, '*': operator.mul,
                             '//': operator.floordiv, '%': operator.mod}.get(node.op)
                if operation:
                    try:
                        return operation(a, b)
                    except ArithmeticError:
                        pass
        return UNKNOWN

    def guard(node, values, site, loops):
        if node is None:
            return P.TRUE
        value = evaluate(node, values)
        if type(value) is bool:
            return P.TRUE if value else P.FALSE
        if isinstance(node, T.TUna) and node.op == 'not':
            return _not(guard(node.operand, values, site + '/operand', loops))
        if isinstance(node, T.TBin) and node.op in ('and', 'or'):
            a = guard(node.left, values, site + '/left', loops)
            b = guard(node.right, values, site + '/right', loops)
            return _and(a, b) if node.op == 'and' else _or(a, b)
        return unknown(site, node, loops)

    def expr(node, values, site, path, loops):
        # Operand evaluation happens even when the outer memory mask is false;
        # where selection never becomes a guard for either value operand.
        for field in OPERANDS[type(node)]:
            value = getattr(node, field)
            children = enumerate(value) if isinstance(value, list) else [(None, value)]
            for index, child in children:
                if child is not None:
                    suffix = field if index is None else f'{field}/{index}'
                    expr(child, values, site + '/' + suffix, path, loops)
        if isinstance(node, (T.TLoad, T.TStore)):
            accesses.append(AccessContext(node.effect, node, path,
                                          guard(node.mask, values, site + '/mask', loops), loops))

    def changed(body):
        names = set()
        for node in body:
            if isinstance(node, T.TAssign):
                names.add(node.name)
            elif isinstance(node, (T.TIf, T.TStaticIf)):
                names.update(changed(node.then_body) | changed(node.else_body))
            elif isinstance(node, T.TFor):
                names.add(node.var)
                names.update(changed(node.body))
        return names

    def join(a, b):
        return {name: a.get(name, UNKNOWN) if type(a.get(name, UNKNOWN)) is type(b.get(name, UNKNOWN))
                and a.get(name, UNKNOWN) == b.get(name, UNKNOWN) else UNKNOWN
                for name in a.keys() | b.keys()}

    def block(body, values, site, path, loops):
        values = dict(values)
        for index, node in enumerate(body):
            current = f'{site}/{index}'
            expr(node, values, current, path, loops)
            if isinstance(node, T.TAssign):
                values[node.name] = evaluate(node.value, values)
            elif isinstance(node, (T.TIf, T.TStaticIf)):
                condition = guard(node.cond, values, current + '/cond', loops)
                a, af = block(node.then_body, values, current + '/then', _and(path, condition), loops)
                b, bf = block(node.else_body, values, current + '/else', _and(path, _not(condition)), loops)
                values = b if af is P.FALSE else a if bf is P.FALSE else join(a, b)
                path = _or(af, bf)
            elif isinstance(node, T.TReturn):
                path = P.FALSE
            elif isinstance(node, T.TFor):
                start = 0 if node.start is None else evaluate(node.start, values)
                end, step = evaluate(node.end, values), evaluate(node.step, values)
                if all(type(v) is int for v in (start, end, step)) and step != 0:
                    nonempty = start < end if step > 0 else start > end
                    entry = P.TRUE if nonempty else P.FALSE
                else:
                    entry = unknown(current + '/entry', None, loops)
                loop = LoopContext(current, T.ValueRef('induction', current, node.var, TY.ScalarT(D.i32)),
                                   node.start, node.end, node.step, entry)
                inside = dict(values)
                for name in changed(node.body) | {node.var}:
                    inside[name] = UNKNOWN
                body_path = _and(path, entry)
                _, fallthrough = block(node.body, inside, current + '/body', body_path, loops + (loop,))
                if entry is not P.FALSE:
                    for name in changed(node.body) | {node.var}:
                        values[name] = UNKNOWN
                    if fallthrough is P.FALSE:
                        path = _and(path, _not(entry))
                    elif fallthrough is not body_path:
                        # Do not leak a current-iteration condition to the
                        # exit or claim a precise quantification of past returns.
                        path = _and(path, unknown(current + '/continuation', None, loops))
        return values, path

    block(kernel.body, env, 'body', P.TRUE, ())
    stage = 'symbolic' if consts is None else 'partial' if supplied.keys() < declarations.keys() else 'specialized'
    return EffectSummary(stage, tuple(accesses))
