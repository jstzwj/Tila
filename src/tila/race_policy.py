"""ADR-018 launch policy, provenance-sensitive bounded cache, stable audit view."""
from collections import OrderedDict
from dataclasses import asdict, fields, is_dataclass, replace
from enum import Enum
import json
import os
import warnings

from . import dtypes as D
from .effect_audit import effect_details
from .errors import Loc, TilaError, Warning_
from .facts import PROVEN_SAFE, PROVEN_UNSAFE, UNKNOWN
from .race import RaceConfig, RaceReport, analyze_races
from .solver import ProofConfig, _LOCK, z3_module
from .verifier import verify

ANALYSIS_VERSION = 3
_CACHE = OrderedDict()


def mode():
    value = os.environ.get('TILA_RACE', 'warn')
    if value not in ('off', 'warn', 'error'):
        raise TilaError('TILA-RACE-003', 'invalid race policy', details=[f'race={value}'])
    return value


def configuration():
    try:
        return RaceConfig(ProofConfig.environment(), int(os.environ.get('TILA_RACE_MAX_PAIRS', '4096')), True)
    except ValueError as exc:
        raise TilaError('TILA-RACE-003', 'invalid race analysis budget') from exc


def _freeze(value):
    """Semantic data, including definitions/effects; never object identity/repr."""
    if isinstance(value, D.DType):
        return ('dtype', value.name)
    if isinstance(value, Enum):
        return (type(value).__name__, value.value)
    if isinstance(value, str):
        return ('str', str(value))  # e.g. torch.torch_version.TorchVersion
    if value is None or type(value) in (int, bool, str):
        return (type(value).__name__, value)
    if type(value) is float:
        return ('float', value.hex())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(v) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if is_dataclass(value):
        return (type(value).__name__, tuple((f.name, _freeze(getattr(value, f.name))) for f in fields(value)))
    raise TypeError('unsupported fingerprint component')


def _key(kernel, tensors, scalars, consts, grid, config, context):
    from .runtime import _tensor_info
    from .intrinsics import INTRINSIC_REGISTRY_SEMANTIC_REVISION
    bindings = []
    for name, tensor in sorted(tensors.items()):
        dtype, shape, strides, pointer = _tensor_info(tensor)
        # Exact addresses are private cache keys only. Address reuse with identical
        # layout/type/device preserves all alias relations; no tensor is retained.
        bindings.append((name, dtype.name, shape, strides, int(pointer), str(getattr(tensor, 'device', 'cpu'))))
    return _freeze((ANALYSIS_VERSION, INTRINSIC_REGISTRY_SEMANTIC_REVISION, z3_module().get_version_string(),
                    kernel.name, kernel.body, kernel.buffers, kernel.ptr_params, kernel.scalars,
                    kernel.consts, kernel.buffer_ptr_views, bindings, scalars, consts, grid, config, context))


def analyze_launch(kernel, tensors, scalars, consts, grid, *, context=(), config=None):
    """Call only AFTER current launch contracts/target/bounds have been checked."""
    policy = mode()
    if policy == 'off':
        return RaceReport((), stage='disabled', policy=policy, suppressions=('race=off: analysis disabled',))
    config = configuration() if config is None else replace(config, include_intra=True)
    with _LOCK:
        verify(kernel, consts, capability=None)
        if 0 in grid:
            return RaceReport((), unchecked_domains=(), stage='empty-launch', policy=policy)
        key = None
        if config.proof.cache_entries:
            try:
                key = _key(kernel, tensors, scalars, consts, grid, config, context)
            except (TypeError, RecursionError):
                pass  # Unfingerprintable bindings are analyzed without caching.
        if key is not None and key in _CACHE:
            result = _CACHE.pop(key)
            _CACHE[key] = result
            return replace(result, policy=policy, cache_status='hit')
        # Float scalar values can be valid launch arguments, but are outside the
        # exact integer encoder. Omit their facts rather than rejecting the launch.
        integers = {p.name: scalars[p.name] for p in kernel.scalars
                    if p.name in scalars and (p.dtype.is_int or p.dtype is D.bool_)}
        result = analyze_races(kernel, grid=grid, bindings=tensors, consts=consts,
                               scalars=integers, config=config)
        complete = (result.stage == 'bound' and not result.unchecked_domains and not result.incomplete_reason
                    and all(p.verdict != UNKNOWN for p in result.pairs))
        if key is not None and complete:
            _CACHE[key] = result
            while len(_CACHE) > config.proof.cache_entries:
                _CACHE.popitem(last=False)
        return replace(result, policy=policy, cache_status='miss' if key is not None else 'disabled')


def symbolic(kernel, consts):
    policy = mode()
    if policy == 'off':
        return RaceReport((), stage='disabled', policy=policy, suppressions=('race=off: analysis disabled',))
    return replace(analyze_races(kernel, consts=consts, config=configuration()), policy=policy)


def details(kernel, report, consts, *, show_witness=False, show_query=False, show_cache=False):
    effects = effect_details(kernel, consts)
    access = {a['site']: a for a in effects['accesses']}
    pairs = []
    for pair in report.pairs:
        entry = dict(sites=list(pair.sites), domain=pair.domain, verdict=pair.verdict, reason=pair.reason,
                     accesses=[access[s] for s in pair.sites],
                     dependencies=[asdict(d) for d in pair.dependencies],
                     counterexample='confirmed' if pair.confirmed else 'candidate' if pair.candidate else 'none')
        if show_witness:
            entry['witness'] = asdict(pair.confirmed or pair.candidate) if pair.confirmed or pair.candidate else None
        if show_query:
            entry['query'] = pair.query
        pairs.append(entry)
    result = dict(schema='tila.race-details.v1', kernel=kernel.name, stage=report.stage,
                  policy=report.policy, verdict=report.verdict, unchecked_domains=list(report.unchecked_domains),
                  incomplete_reason=report.incomplete_reason, unchecked_pairs=report.unchecked_pairs,
                  suppressions=list(report.suppressions), assumptions='not used to remove accesses',
                  predicates=effects['predicates'], pairs=pairs)
    if show_cache:
        result['cache'] = report.cache_status
    return result


def render(kernel, report, consts, **options):
    return json.dumps(details(kernel, report, consts, **options), ensure_ascii=False, indent=2)


def diagnostics(kernel, report, consts):
    if report.policy == 'off':
        return []
    unsafe = next((p for p in report.pairs if p.verdict == PROVEN_UNSAFE), None)
    unknown = next((p for p in report.pairs if p.verdict == UNKNOWN), None)
    pair = unsafe or unknown
    if pair is None and not report.incomplete_reason and not report.unchecked_domains:
        return []
    code = 'TILA-RACE-001' if unsafe else 'TILA-RACE-002'
    title = 'confirmed unordered memory conflict' if unsafe else 'memory conflict analysis is incomplete'
    access = {a['site']: a for a in effect_details(kernel, consts)['accesses']}
    lines = [f'race policy: {report.policy}', f'analysis stage: {report.stage}']
    line = 0
    if pair:
        lines += [f'domain: {pair.domain}', f'result: {pair.verdict}; reason: {pair.reason}',
                  'counterexample: ' + ('confirmed' if pair.confirmed else 'candidate (not confirmed)' if pair.candidate else 'none')]
        for site in pair.sites:
            a = access[site]
            line = line or a['line']
            lines += [f"{a['kind']} site={site} region={a['region']} dtype={a['dtype']} line={a['line']}",
                      f"  path={a['path']} mask={a['mask']} loops=" + ','.join(l['site'] for l in a['loops'])]
        lines += [f'dependency: {d.kind}: {d.detail}' for d in pair.dependencies]
    if report.incomplete_reason:
        lines += [f'incomplete: {report.incomplete_reason}; unchecked pairs: {report.unchecked_pairs}']
        if pair is None and access:
            first = next(iter(access.values()))
            line = first['line']
            lines += [f"first unexamined access: {first['kind']} site={first['site']} region={first['region']} line={line}",
                      f"  path={first['path']} mask={first['mask']}"]
    if report.unchecked_domains:
        lines += ['unchecked domains: ' + ', '.join(report.unchecked_domains)]
    fixes = ['Partition addresses by program/lane or remove duplicate writes.',
             'Use atomic_add only for additive updates with its supported dtype/order/scope.',
             'Inspect --show-races; model-specific witnesses and queries are opt-in.']
    cls = TilaError if unsafe or report.policy == 'error' else Warning_
    diagnostic = cls(code, title, Loc(line), lines, fixes)
    diagnostic.race_report = report
    diagnostic.race_details = details(kernel, report, consts, show_query=True, show_witness=True)
    return [diagnostic]


def enforce(kernel, report, consts):
    for diagnostic in diagnostics(kernel, report, consts):
        if isinstance(diagnostic, TilaError):
            raise diagnostic
        warnings.warn(diagnostic.render(), RuntimeWarning, stacklevel=3)
