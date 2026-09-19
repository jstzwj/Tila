"""Versioned, solver-independent serialization of verified may-access effects."""
import json

from . import tir as T
from .effect_summary import ConditionSource


def effect_details(kernel, consts=None):
    summary = kernel.effect_summary(consts)
    predicates, identifiers = [], {}

    def reference(value):
        if value is None:
            return None
        return dict(kind=value.kind, name=value.name, site=value.site)

    def predicate(root):
        # Identity sharing avoids recursively expanding a Boolean DAG. Traversal
        # order is structural, never hash iteration or solver formatting.
        todo = [root]
        while todo:
            node = todo.pop()
            if node in identifiers:
                continue
            identifiers[node] = f"p{len(identifiers)}"
            predicates.append(node)
            todo.extend(reversed(node.args))
        return identifiers[root]

    accesses = []
    printer = T._Printer(kernel)
    for access in summary.accesses:
        effect = access.effect
        accesses.append(dict(
            site=effect.site_id, kind=effect.kind,
            region=str(effect.region_id), dtype=effect.element_dtype.name,
            address_space=effect.address_space.value, line=effect.location.line,
            status="may-access" if access.may_access else "excluded",
            path=predicate(access.path), mask=predicate(access.mask),
            loops=[dict(site=loop.site_id, induction=reference(loop.induction),
                        start=printer.o(loop.start), end=printer.o(loop.end),
                        step=printer.o(loop.step), may_enter=predicate(loop.may_enter))
                   for loop in access.loops]))
    nodes = []
    for node in predicates:
        if node.op not in ('true', 'false', 'unknown', 'and', 'or', 'not'):
            raise ValueError(f"unsupported effect predicate: {node.op}")
        source = node.atom
        if node.op == 'unknown' and not isinstance(source, ConditionSource):
            raise ValueError("effect predicate lacks a condition source")
        nodes.append(dict(id=identifiers[node], op=node.op,
                          args=[identifiers[a] for a in node.args],
                          shape=[str(d) for d in node.shape],
                          source=None if source is None else dict(
                              site=source.site, definition=reference(source.definition),
                              loops=list(source.loops))))
    return dict(schema="tila.effect-details.v1", stage=summary.stage,
                consts=[dict(name=name, kind="Bool" if type(value) is bool else "Int", value=value)
                        for name, value in sorted((consts or {}).items())],
                provenance="verified TIR and Const only",
                assumptions="not used to remove accesses",
                launch_bindings="excluded", cache="recomputed",
                predicates=nodes, accesses=accesses)


def render_effect_details(kernel, consts=None):
    return json.dumps(effect_details(kernel, consts), ensure_ascii=False, indent=2)
