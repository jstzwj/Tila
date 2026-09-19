"""Immutable, identity-shared Boolean DAGs. No distributive expansion.

Identity equality/hashing is intentional: revisiting a shared expression must
not recursively hash its children (which can itself be exponential).
"""
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Origin:
    kind: str
    line: int = 0
    detail: str = ""


STATIC = "StaticFact"
CHECKED = "CheckedLaunchContract"
USER = "UserAssumption"


@dataclass(frozen=True, eq=False)
class Predicate:
    op: str
    args: tuple = ()
    atom: object = None
    line: int = 0
    shape: tuple = ()
    # Expand/broadcast nodes retain the coordinate mapping for the SMT encoder.
    mapping: tuple = ()


TRUE = Predicate("true")
FALSE = Predicate("false")


def unknown(line=0, shape=()):
    return Predicate("unknown", line=line, shape=shape)


def atom(pred, line=0, shape=()):
    return Predicate("atom", atom=pred, line=line, shape=shape)


def conjunction(a, b):
    if a is TRUE:
        return b
    if b is TRUE or a is b:
        return a
    return Predicate("and", (a, b))


def disjunction(a, b):
    if a is b:
        return a
    if a is TRUE or b is TRUE:
        return TRUE
    if (a.op == "not" and a.args[0] is b or
            b.op == "not" and b.args[0] is a):
        return TRUE
    return Predicate("or", (a, b))


def negate(a):
    return a.args[0] if a.op == "not" else Predicate("not", (a,))


def select(condition, a, b):
    if a is b:
        return a
    return disjunction(conjunction(condition, a), conjunction(negate(condition), b))


def mapped(a, shape, mapping):
    return Predicate("map", (a,), shape=shape, mapping=mapping)


def nodes(root):
    """Iterative unique-node walk, also suitable for construction budgeting."""
    seen, todo = set(), [root]
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        yield node
        todo.extend(node.args)


def atoms(root):
    return tuple(n.atom for n in nodes(root) if n.op == "atom")


def guaranteed(root):
    """Small fast path: atoms entailed structurally, without DNF or SAT.

    Negation stays in the DAG for the general solver. In particular an unknown
    Boolean is never replaced by True, including beneath Not/Or.
    """
    memo, todo = {}, [(root, False)]
    while todo:
        node, ready = todo.pop()
        if node in memo:
            continue
        if not ready:
            todo.append((node, True))
            todo.extend((c, False) for c in node.args if c not in memo)
            continue
        if node.op == "atom":
            out = {node.atom.key(): node.atom}
        elif node.op == "map":
            out = dict(memo[node.args[0]])
        elif node.op in ("and", "or"):
            a, b = (memo[c] for c in node.args)
            keys = a.keys() | b.keys() if node.op == "and" else a.keys() & b.keys()
            out = {}
            for key in keys:
                p = a.get(key, b.get(key))
                if key in a and key in b:
                    from dataclasses import replace
                    p = replace(p, origins=a[key].origins | b[key].origins)
                out[key] = p
        else:
            out = {}
        memo[node] = out
    return tuple(memo[root].values())


def is_conjunction(root):
    return all(n.op in ("and", "atom", "map") for n in nodes(root))


def map_atoms(root, transform, shape, mapping):
    """Apply an explicit lane mapping once per shared node."""
    from dataclasses import replace
    memo, todo = {}, [(root, False)]
    while todo:
        node, ready = todo.pop()
        if node in memo:
            continue
        if not ready:
            todo.append((node, True))
            todo.extend((c, False) for c in node.args if c not in memo)
            continue
        pred = node.atom
        if node.op == "atom":
            pred = replace(pred, left=transform(pred.left), right=transform(pred.right))
        children = tuple(memo[c] for c in node.args)
        memo[node] = (node if pred is node.atom and children == node.args else
                      Predicate(node.op, children, pred, node.line, node.shape, node.mapping))
    return mapped(memo[root], shape, mapping)
