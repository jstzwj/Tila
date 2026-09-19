"""Instruction-local effects and lexical definition references (ADR-016).

Binding runs once on newly checked TIR. Verification recomputes expectations
without edits. Control-flow summaries live in effect_summary; no race analysis.
"""
from . import tir as T, types as TY, dtypes as D

# Exhaustive semantic operands; metadata is not an executable child.
OPERANDS = {
    T.TName: (), T.TLit: (), T.TAssign: ('value',),
    T.TStore: ('ptr', 'coords', 'value', 'mask'), T.TAssume: ('pred',),
    T.TIf: ('cond',), T.TStaticIf: ('cond',),
    T.TFor: ('start', 'end', 'step'), T.TReturn: (),
    T.TBin: ('left', 'right'), T.TUna: ('operand',), T.TCast: ('operand',),
    T.TConstant: (), T.TArange: ('end',), T.TPid: (), T.TNumPrograms: (),
    T.TZeros: ('shape',), T.TReshape: ('operand', 'shape'),
    T.TExpand: ('operand',), T.TWhere: ('cond', 'a', 'b'),
    T.TDot: ('a', 'b', 'acc'), T.TReduce: ('operand',),
    T.TLoad: ('ptr', 'coords', 'mask', 'other'), T.TBufPtr: (),
    T.TPAdd: ('ptr', 'offset'),
}


def bind_effects(kernel):
    return _analyze(kernel, bind=True)


def verify_effects(kernel):
    return _analyze(kernel, bind=False)


def _analyze(kernel, *, bind):
    from .verifier import fail
    buffers = {p.name: p for p in kernel.buffers}
    env = {}
    for p in kernel.buffers + kernel.ptr_params + kernel.scalars + kernel.consts:
        vt = getattr(p, 'vtype', None)
        if hasattr(p, 'dtype'):
            vt = TY.ScalarT(p.dtype)
        env[p.name] = T.ValueRef('parameter', f'parameter/{p.name}', p.name, vt)
    accesses, active, owners = [], set(), {}

    def put(node, field, expected, line):
        if bind:
            setattr(node, field, expected)
        elif getattr(node, field) != expected:
            fail(f'invalid or missing {field} metadata', line)

    def vt(node, scope):
        if isinstance(node, T.TExpr):
            return node.vt
        if isinstance(node, T.TName):
            return scope[node.name].vtype
        if isinstance(node, T.TLit) and node.dtype is not None:
            return TY.ScalarT(node.dtype)
        return None

    def assigned(body):
        names = set()
        for node in body:
            if type(node) is T.TAssign:
                names.add(node.name)
            elif type(node) in (T.TIf, T.TStaticIf):
                names.update(assigned(node.then_body) | assigned(node.else_body))
            elif type(node) is T.TFor:
                names.add(node.var)
                names.update(assigned(node.body))
        return names

    def joined(left, right, path, kind='merge'):
        result = {}
        for name in sorted(left.keys() | right.keys()):
            a, b = left.get(name), right.get(name)
            if a == b:
                result[name] = a
            else:
                types = [r.vtype for r in (a, b) if r is not None]
                value_type = types[0] if types and all(t == types[0] for t in types) else None
                result[name] = T.ValueRef(kind, path + '/' + name, name, value_type)
        return result

    def visit(node, scope, path, line):
        if type(node) not in OPERANDS:
            fail(f'unsupported effect TIR node {type(node).__name__}', line)
        if id(node) in active:
            fail('cyclic effect operand graph', line)
        active.add(id(node))
        line = getattr(node, 'line', 0) or line
        if type(node) is T.TName:
            if node.name not in scope:
                fail(f'unbound effect definition {node.name}', line)
            previous = owners.get(id(node))
            if previous is not None and previous != scope[node.name]:
                fail('shared name crosses definition scopes', line)
            owners[id(node)] = scope[node.name]
            put(node, 'definition', scope[node.name], line)
        for field in OPERANDS[type(node)]:
            value = getattr(node, field)
            children = enumerate(value) if isinstance(value, list) else [(None, value)]
            for index, child in children:
                if child is not None:
                    suffix = field if index is None else f'{field}/{index}'
                    visit(child, scope, path + '/' + suffix, line)
        if type(node) in (T.TLoad, T.TStore):
            if id(node) in owners:
                fail('shared memory node has multiple evaluation sites', line)
            owners[id(node)] = path
            if node.buffer is not None:
                param = buffers[node.buffer]
                region, elem, space = param.region_id, param.vtype.elem, param.vtype.space
            else:
                pointer = vt(node.ptr, scope)
                if isinstance(pointer, TY.BlockT):
                    pointer = pointer.elem
                if not isinstance(pointer, TY.PtrT):
                    fail('memory pointer lacks a definition-point pointer type', line)
                region, elem, space = pointer.region_id, pointer.elem, pointer.space
            if type(region) not in (TY.BufferRegion, TY.ParamRegion, TY.InternalRegion, TY.UnknownRegion):
                fail('invalid memory RegionId', line)
            value_type = node.vt if type(node) is T.TLoad else vt(node.value, scope)
            if isinstance(value_type, TY.BlockT):
                value_type = value_type.elem
            value_dtype = D.bool_ if isinstance(value_type, TY.MaskT) else getattr(value_type, 'dtype', None)
            if value_dtype is not None and value_dtype is not elem:
                fail('memory value dtype disagrees with access dtype', line)
            effect = T.MemoryEffect(path, 'Read' if type(node) is T.TLoad else 'Write',
                                    region, space, elem, T.EffectLocation(line))
            put(node, 'effect', effect, line)
            accesses.append(node)
        active.remove(id(node))

    def block(body, scope, path):
        scope = dict(scope)
        falls = True
        for index, node in enumerate(body):
            site = f'{path}/{index}'
            visit(node, scope, site, getattr(node, 'line', 0))
            if type(node) is T.TAssign:
                scope[node.name] = T.ValueRef('definition', site, node.name, vt(node.value, scope))
            elif type(node) in (T.TIf, T.TStaticIf):
                a, af = block(node.then_body, scope, site + '/then')
                b, bf = block(node.else_body, scope, site + '/else')
                scope = joined(a, b, site) if af == bf else (a if af else b)
                falls = falls and (af or bf)
            elif type(node) is T.TFor:
                inside = dict(scope)
                for name in assigned(node.body) & scope.keys():
                    inside[name] = T.ValueRef('loop', site + '/carried/' + name, name, scope[name].vtype)
                inside[node.var] = T.ValueRef('induction', site, node.var, TY.ScalarT(D.i32))
                after, _ = block(node.body, inside, site + '/body')
                scope = joined(scope, after, site + '/exit', 'loop')
            elif type(node) is T.TReturn:
                falls = False
        return scope, falls

    block(kernel.body, env, 'body')
    return tuple(accesses)
