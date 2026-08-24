"""Tila 诊断系统：TilaError（E 码，语言错误）与 BackendError（B 码，后端兼容性）。

E01–E17 由 checker / frontend 抛出（编译期拒绝，不产出任何 Triton 代码）；
B01–Bxx 由 backend/triton/validator.py 报告（"backend validation"，不走 Tila 诊断渲染）。
两套诊断严格分开（docs/semantic-model.md §8）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Loc:
    """源码位置（1-based）。"""

    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.line}:{self.col}"


@dataclass(frozen=True)
class Note:
    """诊断附注：操作数类型回显、定义位置、修复建议。一行一条。"""

    text: str


# ---------------------------------------------------------------------------
# E 码目录（docs/type-checker.md §7，2026-08 修订版）
# ---------------------------------------------------------------------------

ERROR_NAMES = {
    "E01": "UnboundName",
    "E02": "TypeMismatch",
    "E03": "ShapeMismatch",
    "E04": "RankMismatch",
    "E05": "LayoutMismatch",
    "E06": "ArangeInvalid",
    "E07": "InvalidOperand",
    "E08": "UnitMisuse",
    "E09": "BadCastForm",
    "E10": "NonConstantExpr",
    "E11": "NotInSubset",
    "E12": "InvalidSignature",
    "E13": "BadIntrinsicForm",
    "E14": "Reassignment",
    "E15": "BadConstexprDefault",
    "E16": "CapabilityError",
    "E17": "LaunchPlanUninferable",
    "E18": "DotConstraints",
    "E19": "AddressForm",
    "E20": "LoopForm",
    "E21": "ReduceConstraints",
}


class TilaError(Exception):
    """唯一的 Tila 编译期错误类型；绝不以 Python 原生异常泄漏。

    subcode（可选）：同码内的细分场景（如 E19 的 FlatAddressing /
    CoordinateArity / CoordinatePosition / CoordinateKind）——渲染仍显示
    主码（用户视角不变），测试与内部分派用 subcode 组织（docs/type-checker.md §7）。
    """

    def __init__(
        self,
        loc: Optional[Loc],
        code: str,
        message: str,
        notes: Tuple[Note, ...] = (),
        subcode: Optional[str] = None,
    ):
        self.loc = loc
        self.code = code
        self.message = message
        self.notes = tuple(notes)
        self.subcode = subcode
        name = ERROR_NAMES.get(code, "Error")
        super().__init__(f"{code} {name}: {message}")

    def render(self) -> str:
        name = ERROR_NAMES.get(self.code, "Error")
        head = f"{self.code} {name}"
        if self.loc is not None:
            head += f" at {self.loc}"
        lines = [head, self.message]
        lines.extend(f"  {n.text}" for n in self.notes)
        return "\n".join(lines)


def err(loc, code: str, message: str, *notes: str, subcode: str = None) -> TilaError:
    """构造 TilaError 的简写；notes 每条一行。"""
    return TilaError(loc, code, message, tuple(Note(t) for t in notes), subcode=subcode)


# ---------------------------------------------------------------------------
# B 码（后端兼容性，docs/triton-lowering.md §8）
# ---------------------------------------------------------------------------

BACKEND_ERROR_NAMES = {
    "B01": "UnsupportedDTypeOnTarget",
    "B02": "TileNumelExceeded",
    "B03": "UnrealizableTileConfig",
}


@dataclass
class BackendError:
    code: str
    message: str
    notes: Tuple[Note, ...] = field(default=())

    def render(self) -> str:
        name = BACKEND_ERROR_NAMES.get(self.code, "BackendError")
        lines = [f"{self.code} {name}", self.message]
        lines.extend(f"  {n.text}" for n in self.notes)
        return "\n".join(lines)
