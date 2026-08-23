"""Shape 迷你系统（docs/type-system.md §2）。

Shape ::= () | (Dim, …)    Dim ::= Const(n) | Symbol(name) | Product(Dim, Dim)

- 相等 Σ₁ ≡ Σ₂：逐维结构比较，无子类型、无松弛。
- 广播 Σ₁ ⊗ Σ₂：右对齐逐轴，相等取之、一方为 1 取另一方，否则 None（E03）。
- v0.1 的 Tile shape 必须全静态；符号维只出现在 Buffer 注解 shape 里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class Product:
    """constexpr 积（如 2*BLOCK）。v0.1 代数上预留、不产生。"""

    lhs: object
    rhs: object


Dim = object  # Const | Symbol | Product
Shape = Tuple[Dim, ...]


def const_shape(*values: int) -> Shape:
    return tuple(Const(v) for v in values)


def shape_eq(s1: Shape, s2: Shape) -> bool:
    return s1 == s2


def dim_str(d: Dim) -> str:
    if isinstance(d, Const):
        return str(d.value)
    if isinstance(d, Symbol):
        return d.name
    if isinstance(d, Product):
        return f"{dim_str(d.lhs)}*{dim_str(d.rhs)}"
    raise AssertionError(f"unknown dim {d!r}")


def shape_str(s: Shape) -> str:
    if len(s) == 1:
        return f"({dim_str(s[0])},)"
    if len(s) == 0:
        return "()"
    return "(" + ",".join(dim_str(d) for d in s) + ")"


def dim_value(d: Dim) -> Optional[int]:
    """静态可求值的维度取值；符号维返回 None。"""
    if isinstance(d, Const):
        return d.value
    if isinstance(d, Product):
        lv, rv = dim_value(d.lhs), dim_value(d.rhs)
        if lv is not None and rv is not None:
            return lv * rv
    return None


def numel(s: Shape) -> Optional[int]:
    n = 1
    for d in s:
        v = dim_value(d)
        if v is None:
            return None
        n *= v
    return n


def is_static(s: Shape) -> bool:
    return numel(s) is not None


def _dim1(d: Dim) -> bool:
    v = dim_value(d)
    return v == 1


def broadcast(s1: Shape, s2: Shape) -> Optional[Shape]:
    """Σ₁ ⊗ Σ₂：右对齐逐轴。返回结果 shape；不可广播返回 None。

    v0.1 的实际使用是 标量→任意 shape（R4，由运算层处理）与严格相等；
    size-1 轴规则按 type-system §2 的定义实现（v0.2 预览 R14 放宽后直接可用）。
    """
    if len(s1) != len(s2):
        # v0.1 rank 必须一致；⊗ 定义在右对齐上，但不同 rank 的组合
        # （非标量）在 v0.1 一律交给调用方报 E03/E04。
        return None
    out = []
    for a, b in zip(s1, s2):
        av, bv = dim_value(a), dim_value(b)
        if av is not None and av == bv:
            out.append(a)
        elif _dim1(a):
            out.append(b)
        elif _dim1(b):
            out.append(a)
        else:
            return None
    return tuple(out)
