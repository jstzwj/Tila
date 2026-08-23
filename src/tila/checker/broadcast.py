"""⊗ 与 shape 规则的检查入口（docs/type-checker.md §4.1，E03）。"""

from __future__ import annotations

from ..diagnostics import Loc, err
from ..types import Shape, broadcast, shape_str


def require_broadcast(s1: Shape, s2: Shape, loc: Loc, what: str) -> Shape:
    """Σ₁ ⊗ Σ₂ 良式；否则 E03（同轴均 >1 且不等）。"""
    if len(s1) != len(s2):
        raise err(loc, "E04", f"rank mismatch in {what}: {shape_str(s1)} vs {shape_str(s2)}")
    out = broadcast(s1, s2)
    if out is None:
        dim_notes = []
        for i, (a, b) in enumerate(zip(s1, s2)):
            av, bv = getattr(a, "value", None), getattr(b, "value", None)
            if av is not None and bv is not None and av != bv and av != 1 and bv != 1:
                dim_notes.append(f"dimension {i}: {av} vs {bv}")
        notes = [f"lhs: {shape_str(s1)}", f"rhs: {shape_str(s2)}"]
        notes.extend(dim_notes)
        notes.append("only equal or size-1 dimensions are broadcastable")
        raise err(loc, "E03", f"cannot broadcast shapes in {what}", *notes)
    return out


def require_same_shape(s1: Shape, s2: Shape, loc: Loc, what: str) -> None:
    """v0.1 load/store 的严格口径：Σ' ≡ Σ。"""
    if len(s1) != len(s2):
        raise err(loc, "E04", f"rank mismatch in {what}: {shape_str(s1)} vs {shape_str(s2)}")
    if s1 != s2:
        raise err(
            loc, "E03", f"shape mismatch in {what}",
            f"lhs: {shape_str(s1)}",
            f"rhs: {shape_str(s2)}",
            "v0.1 requires the mask/value shape to match the address shape exactly",
        )
