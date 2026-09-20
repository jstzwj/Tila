"""Stable logical-uniformity audit; always recomputed from TIR and Const only."""
from dataclasses import asdict, dataclass
import json

from . import tir as T
from .effect_summary import ConditionSource
from .uniformity import (Level, UniformityConfig, analyze_uniformity,
                         verify_uniformity)


def _reference(ref):
    return None if ref is None else dict(kind=ref.kind, site=ref.site, name=ref.name)


class _Predicates:
    def __init__(self):
        self.nodes, self.ids = [], {}

    def add(self, root):
        todo = [root]
        while todo:
            node = todo.pop()
            if node in self.ids:
                continue
            self.ids[node] = f'p{len(self.nodes)}'
            self.nodes.append(node)
            todo.extend(reversed(node.args))
        return self.ids[root]

    def details(self):
        result = []
        for node in self.nodes:
            if node.op not in ('true', 'false', 'unknown', 'and', 'or', 'not'):
                raise ValueError('unsupported uniformity predicate: ' + node.op)
            source = node.atom
            if source is not None and not isinstance(source, ConditionSource):
                raise ValueError('uniformity predicate lacks a condition source')
            result.append(dict(id=self.ids[node], op=node.op,
                               args=[self.ids[a] for a in node.args],
                               shape=[str(d) for d in node.shape],
                               source=None if source is None else dict(
                                   site=source.site, definition=_reference(source.definition),
                                   loops=list(source.loops))))
        return result


def uniformity_details(kernel, consts=None, *, config=UniformityConfig()):
    summary = analyze_uniformity(kernel, consts, config=config)
    predicates = _Predicates()
    printer = T._Printer(kernel)

    def loops(context):
        return [dict(site=l.site_id, induction=_reference(l.induction),
                     start='0' if l.start is None else printer.o(l.start),
                     end=printer.o(l.end), step=printer.o(l.step),
                     may_enter=predicates.add(l.may_enter)) for l in context]

    def control(c):
        return dict(selection_level=c.selection_level.name,
                    participation=dict(Launch=c.launch_participation, Program=c.program_participation),
                    reachability=c.reachability, path=predicates.add(c.path),
                    selectors=list(c.selectors), loops=loops(c.loops),
                    returned=list(c.returned), iteration_sources=list(c.iteration_sources),
                    reason=c.reason)

    facts = [dict(site=f.site, definition=_reference(f.definition), line=f.line,
                  value_level=None if f.value_level is None else f.value_level.name,
                  inputs=list(f.inputs), rule=f.rule, reason=f.reason, source=f.source,
                  control=control(f.control)) for f in summary.facts]
    accesses = [dict(site=a.effect.site_id, kind=a.effect.kind, region=str(a.effect.region_id),
                     dtype=a.effect.element_dtype.name, line=a.effect.location.line,
                     path=predicates.add(a.path), mask=predicates.add(a.mask), loops=loops(a.loops))
                for a in summary.accesses]
    exit_control = control(summary.exit_control)
    parameters = []
    for kind, params in (('Buffer', kernel.buffers), ('Ptr', kernel.ptr_params),
                         ('Scalar', kernel.scalars), ('Const', kernel.consts)):
        for p in params:
            dtype = p.vtype.elem if kind in ('Buffer', 'Ptr') else getattr(p, 'dtype', None)
            parameters.append(dict(name=p.name, kind=kind,
                                   dtype=dtype.name if dtype is not None else p.value_kind,
                                   definition=f'@parameter:parameter/{p.name}:{p.name}'))
    return dict(schema='tila.uniformity-details.v1', kernel=kernel.name, stage=summary.stage,
                parameters=parameters,
                consts=[dict(name=k, kind='Bool' if type(v) is bool else 'Int', value=v)
                        for k, v in summary.consts],
                scope='logical Launch/Program; no CTA/Warp mapping',
                requires_defined_execution=summary.requires_defined_execution,
                provenance='verified TIR and Const only',
                assumptions='not used to strengthen uniformity',
                launch_bindings='excluded', cache='recomputed',
                budget=asdict(config), complete=summary.incomplete_reason is None,
                incomplete_reason=summary.incomplete_reason,
                consumers=[], consumer_status='not-installed',
                facts=facts, accesses=accesses, exit_control=exit_control,
                predicates=predicates.details())


def render_uniformity_details(kernel, consts=None, *, config=UniformityConfig()):
    return json.dumps(uniformity_details(kernel, consts, config=config), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class RequirementResult:
    status: str
    reason: str


def evaluate_requirement(kernel, summary, site, *, scope='Program', value_level=None,
                         consts=None, config=UniformityConfig()):
    """Internal consumer contract test seam, NOT a synchronization API.

    Only Satisfied permits a fixture consumer. No policy can upgrade Unknown or
    Unsatisfied. A physical consumer still requires a separate target contract.
    """
    verify_uniformity(kernel, summary, consts, config=config)
    if scope not in ('Launch', 'Program'):
        return RequirementResult('Unknown', 'physical-scope-not-modeled')
    if value_level is not None and (type(value_level) is not Level or value_level > Level.ProgramUniform):
        raise ValueError('required value level must be LaunchUniform or ProgramUniform')
    if summary.incomplete_reason:
        return RequirementResult('Unknown', summary.incomplete_reason)
    fact = next((f for f in summary.facts if f.site == site), None)
    if fact is None:
        return RequirementResult('Unknown', 'missing-site')
    if fact.control.reachability == 'Unreachable':
        return RequirementResult('NotApplicable', 'unreachable-site')
    participation = (fact.control.program_participation if scope == 'Program'
                     else fact.control.launch_participation)
    if participation == 'Unknown' or value_level is not None and fact.value_level in (None, Level.Unknown):
        return RequirementResult('Unknown', 'required-guarantee-unknown')
    if participation != 'Full' or value_level is not None and fact.value_level > value_level:
        return RequirementResult('Unsatisfied', 'required-guarantee-not-established')
    return RequirementResult('Satisfied', 'logical-guarantee-established')
