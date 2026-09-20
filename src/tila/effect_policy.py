"""ADR-010: structural eager-read diagnostics, independent of bounds policy."""
import os

from . import tir as T
from .effect_ir import OPERANDS, verify_effects
from .errors import Loc, TilaError, Warning_


def mode():
    value = os.environ.get('TILA_EFFECTS', 'warn')
    if value not in ('off', 'warn', 'error'):
        raise TilaError('TILA-EFFECT-008', 'invalid effects policy',
                        details=[f'effects={value}'],
                        fixes=['Use TILA_EFFECTS=off|warn|error or --effects off|warn|error'])
    return value


def where_warnings(kernel):
    verify_effects(kernel)
    groups = {}

    def visit(node, site, line=0, owner=None):
        line = getattr(node, 'line', 0) or line
        if type(node) is T.TWhere:
            visit(node.cond, site + '/cond', line, owner)
            for side, field in (('then', 'a'), ('else', 'b')):
                key = (site, side, line)
                visit(getattr(node, field), site + '/' + field, line, key)
            return
        # Verified TName is a definition-point reference, not re-evaluation.
        for field in OPERANDS[type(node)]:
            value = getattr(node, field)
            children = enumerate(value) if isinstance(value, list) else [(None, value)]
            for index, child in children:
                if child is not None:
                    suffix = field if index is None else f'{field}/{index}'
                    visit(child, site + '/' + suffix, line, owner)
        if type(node) is T.TLoad and owner is not None:
            groups.setdefault(owner, []).append(node.effect)
        if type(node) in (T.TIf, T.TStaticIf):
            block(node.then_body, site + '/then')
            block(node.else_body, site + '/else')
        elif type(node) is T.TFor:
            block(node.body, site + '/body')

    def block(body, site):
        for index, node in enumerate(body):
            visit(node, f'{site}/{index}')

    block(kernel.body, 'body')
    return [Warning_(
        'TILA-EFFECT-007',
        f"eager read in where '{side}' operand", Loc(line),
        [f'where site: {site}',
         'Both value operands are evaluated; each load retains its own mask.',
         *[f'Read site={e.site_id} region={e.region_id} line={e.location.line}' for e in effects]],
        ['Use load(a, mask=cond, other=pure_value) and load(b, mask=~cond, other=pure_value);',
         'for scalar bool use not cond; combine with existing bounds masks when shapes permit.',
         'A load nested in other= is still evaluated.'])
        for (site, side, line), effects in groups.items()]


def diagnostics(kernel, *, enforce=False):
    policy = mode()
    candidates = where_warnings(kernel)
    if policy == 'off':
        return []
    if policy == 'error':
        errors = [TilaError(w.code, w.title, w.loc, w.details, w.fixes) for w in candidates]
        if enforce and errors:
            raise errors[0]
        return errors
    return candidates


def report_diagnostics(kernel):
    return [w for w in kernel.warnings if w.code != 'TILA-EFFECT-007'] + diagnostics(kernel)
