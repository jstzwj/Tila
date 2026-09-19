"""Tila 诊断：结构化错误与警告（docs/surface-language.md §8）。

每个 Tila 诊断统一携带：码（TILA-<FAMILY>-<NNN>）、一句话标题、源码位置、
涉及的类型/事实细节、至少一条修复建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping


class DiagnosticPhase(str, Enum):
    HOST = "host"
    FRONTEND = "frontend"
    CHECK = "check"
    SPECIALIZE = "specialize"
    LAUNCH = "launch"
    TARGET = "target"
    INTERNAL = "internal"


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ERROR_OR_WARNING = "error-or-warning"


@dataclass(frozen=True)
class DiagnosticSpec:
    code: str
    phases: frozenset[DiagnosticPhase]
    summary: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    default_fix: str = "参考对应语言规则并修正输入"

    @property
    def family(self) -> str:
        return self.code.split("-")[1]


def _specs(family: str, phases: Iterable[DiagnosticPhase], entries: dict,
           *, severity=DiagnosticSeverity.ERROR, default_fix: str):
    phase_set = frozenset(phases)
    return {
        f"TILA-{family}-{number}": DiagnosticSpec(
            f"TILA-{family}-{number}", phase_set, summary, severity,
            default_fix,
        )
        for number, summary in entries.items()
    }


_FRONTEND_FIX = "只使用 docs/surface-language.md 列出的 Python 子集与表面语法"
_TYPE_FIX = "对齐操作数类型/shape，必要时使用显式 ti.cast"
_CONST_FIX = "按声明使用精确 Python int/bool，或只依赖 Const 的表达式"
_MEM_FIX = "检查 Buffer/Ptr capability、extent、offset 与 alignment 声明"
_BOUNDS_FIX = "补充逐轴 mask/契约，或显式使用 unsafe 访问并接受审计"


_DIAGNOSTICS = {
    **_specs("PROOF", (DiagnosticPhase.CHECK, DiagnosticPhase.SPECIALIZE,
                        DiagnosticPhase.LAUNCH), {
        "001": "proof solver dependency or configuration is invalid",
    }, default_fix="安装项目锁定的 Z3 依赖，检查 TILA_PROOF_* 预算参数"),
    **_specs("NUM", (DiagnosticPhase.CHECK, DiagnosticPhase.SPECIALIZE,
                      DiagnosticPhase.LAUNCH), {
        "001": "integer arithmetic domain or index overflow could not be validated",
    }, default_fix="检查整数范围、除数、移位和转换；必要时显式扩大索引位宽"),
    **_specs("NUM", (DiagnosticPhase.CHECK,), {
        "002": "typed constant is nonfinite or overflows its target dtype",
    }, default_fix="使用有限源值；缩小常量数值或选择更大范围的浮点 dtype"),
    **_specs("SYN", (DiagnosticPhase.FRONTEND,), {
        "000": "JIT entry or source retrieval failed",
        "001": "invalid Python syntax",
        "002": "syntax node is outside the Tila subset",
        "003": "invalid loop form",
        "004": "unsupported kernel parameter form",
        "010": "annotation evaluation failed",
        "011": "parameter default is not a literal",
        "012": "kernel parameter annotation is missing",
        "014": "a non-Const parameter has a default",
        "015": "parameter annotation is unsupported",
        "020": "assignment target is unsupported",
        "021": "kernel value return is unsupported",
        "022": "local variable annotation is unsupported",
        "023": "loop target is unsupported",
        "024": "range loop call shape is invalid",
        "026": "Unit value is used as data",
        "027": "unsupported keyword or call form",
        "028": "range is used outside a loop header",
        "030": "unknown or unbound intrinsic",
        "031": "literal kind is unsupported",
        "032": "operator is unsupported",
        "033": "comparison form is unsupported",
        "034": "attribute form is unsupported",
        "035": "subscripted intrinsic form is invalid",
        "036": "intrinsic call shape is invalid",
        "037": "dynamic kwargs are unsupported",
        "038": "subscript form is unsupported",
        "039": "cast target is not a dtype",
        "040": "name is undefined",
        "041": "captured Python object is unsupported",
        "050": "reserved syntax is not available in v0",
        "060": "kernel launch syntax is invalid",
        "061": "assume_launch decorator form is invalid",
        "062": "assume_launch predicate is invalid",
        "063": "assume_launch expression is unsupported",
    }, default_fix=_FRONTEND_FIX),
    **_specs("TYPE", (DiagnosticPhase.CHECK,), {
        "012": "implicit dtype conversion is unsafe",
        "013": "literal cannot be represented by the required dtype",
        "014": "operator is outside the dtype capability",
        "015": "where branch dtypes differ",
        "016": "memory value dtype differs from element dtype",
        "017": "Unit value cannot be assigned",
        "018": "expression cannot be typed or materialized",
        "019": "condition is not a scalar bool",
        "020": "branch result types differ",
        "021": "loop-carried type changed",
        "022": "block predicate is used as a scalar bool",
        "023": "name is unavailable on this control-flow path",
        "024": "loop bound is not an integer scalar",
        "025": "loop step is not a positive Const int",
        "026": "pointer offset is not an integer",
        "027": "numeric operands are required",
        "028": "signed and unsigned operands are mixed",
        "029": "unary operand domain is invalid",
        "030": "operation requires exact element dtype",
        "031": "dot dtype capability is unsupported",
        "032": "where predicate type is invalid",
        "033": "elementwise exponential needs Float",
        "034": "assume predicate type is invalid",
        "035": "memory coordinate is not an integer index",
        "036": "storage-only dtype is used for arithmetic",
    }, default_fix=_TYPE_FIX),
    **_specs("CONST", (DiagnosticPhase.FRONTEND, DiagnosticPhase.CHECK), {
        "011": "typed constant source or target is unsupported",
    }, default_fix="使用 ti.constant[ti.f16/bf16/f32/f64](精确 Python int/float 字面量或模块常量)"),
    **_specs("TYPE", (DiagnosticPhase.HOST,), {
        "037": "host integer normalization requires a supported exact type",
    }, default_fix="使用原生 Python int 或明确支持的 NumPy 整数 scalar"),
    **_specs("TYPE", (DiagnosticPhase.LAUNCH,), {
        "101": "launch argument type or count mismatch",
        "102": "runtime buffer shape/rank contract mismatch",
        "103": "runtime scalar refinement contract failed",
        "104": "launch grid is invalid",
        "105": "automatic launch grid cannot be derived",
    }, default_fix="检查 kernel 签名、实际参数、grid 与 launch contract"),
    **_specs("SHAPE", (DiagnosticPhase.CHECK, DiagnosticPhase.SPECIALIZE), {
        "003": "shapes are not broadcast-compatible",
        "004": "symbolic shape equality failed",
        "005": "reshape element count differs",
        "006": "expand-dims operand is invalid",
        "008": "arange extent is invalid",
        "009": "reduction shape or axis is invalid",
        "010": "mask shape differs from access shape",
        "011": "coordinate rank differs from Buffer rank",
        "012": "store value shape differs from access shape",
    }, default_fix="对齐 rank/shape，并将 specialization 维写成 Const[int]"),
    **_specs("CONST", (DiagnosticPhase.CHECK, DiagnosticPhase.SPECIALIZE,
                       DiagnosticPhase.LAUNCH), {
        "001": "runtime value is used in a Const int context",
        "002": "literal axis is invalid",
        "003": "Const refinement contract failed",
        "004": "assume predicate cannot become a sound fact",
        "005": "static assertion evaluated to false",
        "006": "static assertion is not a staged bool",
        "007": "Const parameter has no specialization value",
        "008": "Const value does not match its exact Python int/bool domain",
        "009": "Const expression evaluation failed",
        "010": "unknown Const override was supplied",
    }, default_fix=_CONST_FIX),
    **_specs("MEM", (DiagnosticPhase.CHECK,), {
        "001": "memory access violates Read/Write capability",
        "002": "pointer offset unit or category is invalid",
        "004": "Buffer pointer linearization is invalid",
        "005": "memory operand is not a Buffer or Ptr",
        "006": "Buffer access does not name a direct parameter",
    }, default_fix=_MEM_FIX),
    **_specs("MEM", (DiagnosticPhase.LAUNCH,), {
        "003": "runtime alignment or contiguous-storage contract failed",
    }, default_fix=_MEM_FIX),
    **_specs("BOUNDS", (DiagnosticPhase.CHECK, DiagnosticPhase.SPECIALIZE,
                        DiagnosticPhase.LAUNCH), {
        "001": "memory safety could not be proved",
        "002": "mask constrains the wrong dimension",
        "003": "out-of-bounds access is provable",
    }, severity=DiagnosticSeverity.ERROR_OR_WARNING,
       default_fix=_BOUNDS_FIX),
    **_specs("BOUNDS", (DiagnosticPhase.LAUNCH,), {
        "010": "assume_launch contract failed",
    }, default_fix=_BOUNDS_FIX),
    **_specs("EFFECT", (DiagnosticPhase.CHECK,), {
        "007": "where eagerly evaluates a memory effect",
    }, severity=DiagnosticSeverity.WARNING,
       default_fix="改用具有相同谓词的 masked load/store"),
    **_specs("TARGET", (DiagnosticPhase.TARGET,), {
        "004": "required backend package is unavailable",
        "005": "backend compilation or execution failed",
        "006": "unsupported launch option",
        "007": "unvalidated CUDA target or backend version",
        "008": "tensor arguments use different devices",
        "009": "TIR is not supported by the selected backend target",
        "010": "backend compilation or loading failed",
        "011": "compiled kernel exceeds target resources",
    }, default_fix="检查 target、设备和 Triton/CUDA 支持矩阵"),
    **_specs("INTERNAL", (DiagnosticPhase.INTERNAL,), {
        "001": "unexpected internal failure at a user-facing boundary",
    }, default_fix="使用 TILA_DEBUG=1 获取 traceback，并提交最小复现"),
}

DIAGNOSTIC_REGISTRY: Mapping[str, DiagnosticSpec] = MappingProxyType(
    _DIAGNOSTICS
)


def diagnostic_spec(code: str) -> DiagnosticSpec:
    try:
        return DIAGNOSTIC_REGISTRY[code]
    except KeyError as exc:
        raise ValueError(f"unregistered Tila diagnostic code: {code}") from exc


@dataclass
class Loc:
    line: int = 0
    col: int = 0
    src_line: str = ""

    def render(self) -> str:
        if not self.src_line:
            return f"line {self.line}" if self.line else "<unknown>"
        caret = " " * max(self.col - 1, 0) + "^"
        return f"line {self.line}: {self.src_line.strip()}\n    {caret}"


class TilaError(Exception):
    """语言层 / 特化期静态错误（E 码族）。"""

    def __init__(self, code: str, title: str, loc: Loc | None = None,
                 details: list[str] | None = None, fixes: list[str] | None = None,
                 *, proof_result=None):
        self.spec = diagnostic_spec(code)
        self.code = code
        self.title = title
        self.loc = loc or Loc()
        self.details = details or []
        self.fixes = fixes or [self.spec.default_fix]
        self.proof_result = proof_result
        super().__init__(self.render())

    def render(self) -> str:
        head = f"error[{self.code}]: {self.title}"
        phases = ", ".join(sorted(phase.value for phase in self.spec.phases))
        body = [head, f"    at: {self.loc.render()}", f"    phase: {phases}"]
        body += [f"    {d}" for d in self.details]
        if self.fixes:
            body.append("    fix:")
            body += [f"      {f}" for f in self.fixes]
        return "\n".join(body)

    __str__ = render


@dataclass
class Warning_:
    code: str
    title: str
    loc: Loc
    details: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)

    def __post_init__(self):
        spec = diagnostic_spec(self.code)
        if not self.fixes:
            self.fixes.append(spec.default_fix)

    def render(self) -> str:
        spec = diagnostic_spec(self.code)
        phases = ", ".join(sorted(phase.value for phase in spec.phases))
        body = [f"warning[{self.code}]: {self.title}",
                f"    at: {self.loc.render()}", f"    phase: {phases}"]
        body += [f"    {d}" for d in self.details]
        body.append("    fix:")
        body += [f"      {fix}" for fix in self.fixes]
        return "\n".join(body)


class TilaLaunchContractError(Exception):
    """launch 契约失败（标量精化 / assume_launch / 对齐声明 / 参数类型）。

    与 TilaError（编译期）分开：契约是每次启动都重新检查的运行期条件。
    """

    def __init__(self, code: str, title: str, details: list[str] | None = None,
                 fixes: list[str] | None = None):
        self.spec = diagnostic_spec(code)
        self.code = code
        self.title = title
        self.details = details or []
        self.fixes = fixes or [self.spec.default_fix]
        super().__init__(self.render())

    def render(self) -> str:
        body = [f"launch contract error[{self.code}]: {self.title}"]
        body.append("    at: <launch boundary>")
        body.append("    phase: launch")
        body += [f"    {d}" for d in self.details]
        if self.fixes:
            body.append("fix:")
            body += [f"    {f}" for f in self.fixes]
        return "\n".join(body)

    __str__ = render
