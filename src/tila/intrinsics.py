"""Tila 表面 intrinsic 的唯一机器可读 catalog（ADR-006）。

Registry 管理名字、表面形式、调用形态、阶段、effect、bounds、可达 TIR、
backend 覆盖、target 要求和文档状态关联。复杂类型/shape/proof 规则仍由
checker 的专用 handler 实现。

本模块刻意不导入 checker、TIR 或 backend，避免 catalog 产生循环依赖。
handler 与 TIR op 均使用稳定符号 ID，由各消费方显式解析并在测试中校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping


INTRINSIC_REGISTRY_SCHEMA_VERSION = 1
INTRINSIC_REGISTRY_SEMANTIC_REVISION = 10  # ADR-016 derived control-flow effect summaries


class SurfaceForm(str, Enum):
    PUBLIC_CALL = "public-call"
    SUBSCRIPT_CALL = "subscript-call"
    METHOD_CALL = "method-call"
    LOOP_FORM = "loop-form"


class Availability(str, Enum):
    IMPLEMENTED = "Implemented"
    PARTIAL = "Partial"
    DESIGNED = "Designed"
    DEFERRED = "Deferred"


class Backend(str, Enum):
    CHECKER = "Check"
    INTERPRETER = "CPU"
    TRITON = "Triton"
    LAUNCH = "Launch"


class StageRule(str, Enum):
    RUNTIME = "runtime"
    CONST_INT_INPUT = "const-int-input"
    LOOP_CONTROL = "loop-control"
    EXPLICIT_CAST = "explicit-cast"
    TYPED_CONSTANT = "typed-constant"
    FACT_INJECTION = "fact-injection"
    STATIC_ASSERT = "static-assert"
    METHOD_REDUCTION = "method-reduction"


class EffectRule(str, Enum):
    PURE = "pure"
    READ_MEMORY = "read-memory"
    WRITE_MEMORY = "write-memory"
    INJECT_FACT = "inject-fact"
    COMPILE_TIME_ASSERT = "compile-time-assert"
    UNAVAILABLE = "unavailable"


class BoundsRule(str, Enum):
    BUFFER_OR_PTR_ACCESS = "buffer-or-ptr-access"


class TargetRequirement(str, Enum):
    TARGET_NEUTRAL = "target-neutral"
    FLOAT_ELEMENTWISE = "float-elementwise"
    DOT_V0 = "dot-v0"
    GLOBAL_MEMORY_V0 = "global-memory-v0"


@dataclass(frozen=True)
class Arity:
    min_positional: int
    max_positional: int

    def __post_init__(self):
        if self.min_positional < 0 or self.max_positional < self.min_positional:
            raise ValueError(f"invalid intrinsic arity: {self}")

    def describe(self) -> str:
        if self.min_positional == self.max_positional:
            n = self.min_positional
            return f"exactly {n} positional argument{'s' if n != 1 else ''}"
        return (f"{self.min_positional}..{self.max_positional} positional "
                "arguments")


@dataclass(frozen=True)
class KeywordSpec:
    name: str
    required: bool = False


@dataclass(frozen=True)
class IntrinsicSpec:
    name: str
    surface_forms: frozenset[SurfaceForm]
    public_export: bool
    availability: Availability
    status_key: str
    arity: Arity
    keywords: tuple[KeywordSpec, ...]
    stage_rule: StageRule
    checker_handler: str
    effect_rule: EffectRule
    bounds_rule: BoundsRule | None
    tir_ops: tuple[str, ...]
    backend_expectation: frozenset[Backend]
    target_requirement: TargetRequirement
    docs_anchor: str

    @property
    def keyword_names(self) -> frozenset[str]:
        return frozenset(keyword.name for keyword in self.keywords)


_ALL_BACKENDS = frozenset(
    {Backend.CHECKER, Backend.INTERPRETER, Backend.TRITON}
)
_CHECK_ONLY = frozenset({Backend.CHECKER})


def _spec(
    name: str,
    *,
    forms: Iterable[SurfaceForm] = (SurfaceForm.PUBLIC_CALL,),
    public: bool = True,
    availability: Availability = Availability.IMPLEMENTED,
    arity: tuple[int, int],
    keywords: Iterable[str] = (),
    stage: StageRule = StageRule.RUNTIME,
    handler: str | None = None,
    effect: EffectRule = EffectRule.PURE,
    bounds: BoundsRule | None = None,
    tir: Iterable[str] = (),
    backends: frozenset[Backend] = _ALL_BACKENDS,
    target: TargetRequirement = TargetRequirement.TARGET_NEUTRAL,
    anchor: str,
) -> IntrinsicSpec:
    return IntrinsicSpec(
        name=name,
        surface_forms=frozenset(forms),
        public_export=public,
        availability=availability,
        status_key=f"intrinsic:{name}",
        arity=Arity(*arity),
        keywords=tuple(KeywordSpec(keyword) for keyword in keywords),
        stage_rule=stage,
        checker_handler=handler or name,
        effect_rule=effect,
        bounds_rule=bounds,
        tir_ops=tuple(tir),
        backend_expectation=backends,
        target_requirement=target,
        docs_anchor=anchor,
    )


# Catalog 顺序是公共文档/导出稳定顺序；不是分派优先级。
INTRINSICS: tuple[IntrinsicSpec, ...] = (
    _spec("program_id", arity=(1, 1), stage=StageRule.CONST_INT_INPUT,
          tir=("TPid",), anchor="2.1"),
    _spec("num_programs", arity=(1, 1), stage=StageRule.CONST_INT_INPUT,
          tir=("TNumPrograms",), anchor="2.1"),
    _spec("arange", arity=(2, 2), stage=StageRule.CONST_INT_INPUT,
          tir=("TArange",), anchor="2.1"),
    _spec("range", forms=(SurfaceForm.LOOP_FORM,), arity=(2, 3),
          stage=StageRule.LOOP_CONTROL, tir=("TFor",), anchor="2.1"),
    _spec("load", arity=(1, 2), keywords=("mask", "other"),
          availability=Availability.PARTIAL, effect=EffectRule.READ_MEMORY,
          bounds=BoundsRule.BUFFER_OR_PTR_ACCESS, tir=("TLoad",),
          target=TargetRequirement.GLOBAL_MEMORY_V0, anchor="2.3"),
    _spec("store", arity=(2, 3), keywords=("mask",),
          availability=Availability.PARTIAL, effect=EffectRule.WRITE_MEMORY,
          bounds=BoundsRule.BUFFER_OR_PTR_ACCESS, tir=("TStore",),
          target=TargetRequirement.GLOBAL_MEMORY_V0, anchor="2.3"),
    _spec("unsafe_load", arity=(1, 2), keywords=("mask", "other"),
          effect=EffectRule.READ_MEMORY,
          bounds=BoundsRule.BUFFER_OR_PTR_ACCESS, tir=("TLoad",),
          target=TargetRequirement.GLOBAL_MEMORY_V0, anchor="2.3"),
    _spec("unsafe_store", arity=(2, 3), keywords=("mask",),
          effect=EffectRule.WRITE_MEMORY,
          bounds=BoundsRule.BUFFER_OR_PTR_ACCESS, tir=("TStore",),
          target=TargetRequirement.GLOBAL_MEMORY_V0, anchor="2.3"),
    _spec("cast", forms=(SurfaceForm.SUBSCRIPT_CALL,), arity=(1, 1),
          stage=StageRule.EXPLICIT_CAST, tir=("TCast",), anchor="2.4"),
    _spec("constant", forms=(SurfaceForm.SUBSCRIPT_CALL,), arity=(1, 1),
          stage=StageRule.TYPED_CONSTANT, tir=("TConstant",), anchor="2.4"),
    _spec("where", arity=(3, 3), tir=("TWhere",), anchor="2.5"),
    _spec("dot", arity=(2, 2), keywords=("acc",), tir=("TDot",),
          target=TargetRequirement.DOT_V0, anchor="2.6"),
    _spec("zeros", arity=(2, 2), stage=StageRule.CONST_INT_INPUT,
          tir=("TZeros",), anchor="2.2"),
    _spec("sum", arity=(2, 2), stage=StageRule.CONST_INT_INPUT,
          tir=("TReduce",), anchor="2.7"),
    _spec("max", arity=(2, 2), stage=StageRule.CONST_INT_INPUT,
          tir=("TReduce",), anchor="2.7"),
    _spec("exp", arity=(1, 1), tir=("TUna",),
          target=TargetRequirement.FLOAT_ELEMENTWISE, anchor="2.7"),
    _spec("exp2", arity=(1, 1), tir=("TUna",),
          target=TargetRequirement.FLOAT_ELEMENTWISE, anchor="2.7"),
    _spec("assume", arity=(1, 1), stage=StageRule.FACT_INJECTION,
          effect=EffectRule.INJECT_FACT, tir=("TAssume",), anchor="2.10"),
    _spec("static_assert", availability=Availability.IMPLEMENTED, arity=(1, 1),
          stage=StageRule.STATIC_ASSERT,
          effect=EffectRule.COMPILE_TIME_ASSERT, backends=_CHECK_ONLY,
          anchor="2.10"),
    _spec("byte_offset", availability=Availability.DEFERRED, arity=(2, 2),
          effect=EffectRule.UNAVAILABLE, backends=_CHECK_ONLY, anchor="2.3"),
    _spec("reshape", arity=(2, 2), stage=StageRule.CONST_INT_INPUT,
          tir=("TReshape",), anchor="2.8"),
    _spec("any", forms=(SurfaceForm.METHOD_CALL,), public=False,
          arity=(1, 1), stage=StageRule.METHOD_REDUCTION, tir=("TUna",),
          anchor="2.5"),
    _spec("all", forms=(SurfaceForm.METHOD_CALL,), public=False,
          arity=(1, 1), stage=StageRule.METHOD_REDUCTION, tir=("TUna",),
          anchor="2.5"),
)


def validate_registry(specs: Iterable[IntrinsicSpec]) -> tuple[IntrinsicSpec, ...]:
    """Validate structural ADR-006 invariants without importing consumers."""
    items = tuple(specs)
    if not items:
        raise ValueError("intrinsic registry must not be empty")
    for label, values in (
        ("name", [spec.name for spec in items]),
        ("status_key", [spec.status_key for spec in items]),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate intrinsic {label}: {', '.join(duplicates)}")

    for spec in items:
        if not spec.name.isidentifier():
            raise ValueError(f"invalid intrinsic name: {spec.name!r}")
        if not spec.surface_forms:
            raise ValueError(f"{spec.name}: surface_forms must not be empty")
        if Backend.CHECKER not in spec.backend_expectation:
            raise ValueError(f"{spec.name}: every surface intrinsic needs checker coverage")
        if len(spec.keyword_names) != len(spec.keywords):
            raise ValueError(f"{spec.name}: duplicate keyword metadata")
        if any(keyword.required for keyword in spec.keywords):
            raise ValueError(f"{spec.name}: required keyword args are not supported in v0")
        if spec.public_export and spec.surface_forms == {SurfaceForm.METHOD_CALL}:
            raise ValueError(f"{spec.name}: method-only intrinsic cannot be public")
        if not spec.public_export and SurfaceForm.METHOD_CALL not in spec.surface_forms:
            raise ValueError(f"{spec.name}: non-public intrinsic must have a method form")
        needs_tir = bool(spec.backend_expectation &
                         {Backend.INTERPRETER, Backend.TRITON})
        if needs_tir and not spec.tir_ops:
            raise ValueError(f"{spec.name}: backend coverage requires declared TIR ops")
        memory_effect = spec.effect_rule in {
            EffectRule.READ_MEMORY, EffectRule.WRITE_MEMORY
        }
        if memory_effect != (spec.bounds_rule is not None):
            raise ValueError(
                f"{spec.name}: memory effect and bounds rule must be declared separately"
            )
        if not spec.docs_anchor or not spec.status_key:
            raise ValueError(f"{spec.name}: docs/status linkage is required")
    return items


INTRINSICS = validate_registry(INTRINSICS)
INTRINSIC_REGISTRY: Mapping[str, IntrinsicSpec] = MappingProxyType(
    {spec.name: spec for spec in INTRINSICS}
)


def get_intrinsic(name: str) -> IntrinsicSpec | None:
    return INTRINSIC_REGISTRY.get(name)


def intrinsic_names_for_surface(surface: SurfaceForm) -> tuple[str, ...]:
    return tuple(spec.name for spec in INTRINSICS
                 if surface in spec.surface_forms)


def call_shape_problem(
    spec: IntrinsicSpec,
    positional_count: int,
    keyword_names: Iterable[str],
) -> str | None:
    """Return a stable call-shape diagnostic, or ``None`` when valid."""
    names = tuple(keyword_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return f"{spec.name} got duplicate keyword(s): {', '.join(duplicates)}"
    unknown = sorted(set(names) - spec.keyword_names)
    if unknown:
        allowed = ", ".join(sorted(spec.keyword_names)) or "none"
        return (f"{spec.name} got unknown keyword(s): {', '.join(unknown)}; "
                f"allowed: {allowed}")
    if not (spec.arity.min_positional <= positional_count <=
            spec.arity.max_positional):
        return f"{spec.name} expects {spec.arity.describe()}"
    return None


# Backward-compatible, derived read-only views. No parallel hand-maintained list.
PUBLIC_INTRINSIC_NAMES = tuple(
    spec.name for spec in INTRINSICS if spec.public_export
)
METHOD_INTRINSIC_NAMES = intrinsic_names_for_surface(SurfaceForm.METHOD_CALL)
CHECKER_INTRINSIC_NAMES = tuple(spec.name for spec in INTRINSICS)
