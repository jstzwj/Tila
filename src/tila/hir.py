"""Tila HIR：frontend 从 Python AST 脱糖出的高层 IR（docs/surface-language.md）。

表达式节点全部携带 Loc；语句为直线 + If/For 两类控制流。
intrinsic 在 frontend 已解析为名字（'load'、'store'…），不再保留 Python
属性链。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import Loc
from .types import RefinedScalar, ScalarT, ConstT, BufferT, PtrT


@dataclass
class Param:
    name: str
    spec: object            # BufferT | PtrT | RefinedScalar | ConstT | ScalarT
    default: object | None = None   # Const 参数可带默认值


@dataclass
class Kernel:
    name: str
    params: list
    body: list              # list[Stmt]
    source: str
    lines: list = field(default_factory=list)   # 源码行（诊断用）


# ---------------------------------------------------------------------------
# 表达式
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    loc: Loc


@dataclass
class Lit(Expr):
    value: object           # int | float | bool


@dataclass
class Name(Expr):
    id: str
    kind: str = "local"     # local | dim | const_int | const_float | const_bool


@dataclass
class BinOp(Expr):
    op: str                 # + - * // / % & | ^ << >>
    left: Expr
    right: Expr


@dataclass
class UnaOp(Expr):
    op: str                 # - ~ not
    operand: Expr


@dataclass
class Cmp(Expr):
    op: str                 # < <= > >= == !=
    left: Expr
    right: Expr


@dataclass
class BoolOp(Expr):
    op: str                 # and | or（仅 bool 标量）
    values: list


@dataclass
class Call(Expr):
    func: str               # intrinsic 名（'load'、'store'、'cast'…）
    args: list
    kwargs: dict
    cast_dtype: object = None   # cast[U](x) 的 U（DType）


@dataclass
class Expand(Expr):
    """x[:, None] ⇒ expand_dims(x, 1)（intrinsics.md §2.8 的下标语法糖）。"""
    value: Expr
    axis: int


@dataclass
class BufPtr(Expr):
    """buf.ptr：Buffer → Ptr（能力继承）。"""
    buf: str


@dataclass
class HTuple(Expr):
    """仅 load/store 坐标位置的元组：(rows, cols)。"""
    items: list


@dataclass
class HDtype(Expr):
    """表达式位置的 dtype 引用（zeros 的 dtype 参数等）。"""
    dtype: object


# ---------------------------------------------------------------------------
# 语句
# ---------------------------------------------------------------------------

@dataclass
class Stmt:
    loc: Loc


@dataclass
class Assign(Stmt):
    target: str
    value: Expr


@dataclass
class ExprStmt(Stmt):
    value: Expr


@dataclass
class If(Stmt):
    cond: Expr
    then_body: list
    else_body: list


@dataclass
class For(Stmt):
    var: str
    start: Expr
    end: Expr
    step: Expr
    body: list


@dataclass
class Return(Stmt):
    """裸 return（无值，docs/surface-language.md §2.1）：提前退出当前
    program instance。带值的 return 在 frontend 即被 TILA-SYN-021 拒绝，
    故 HIR 层不存在有值返回。"""
