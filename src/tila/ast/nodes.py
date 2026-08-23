"""tila_ast 节点定义（docs/ast.md §2）。

表面子集的干净树：frozen dataclass，均携带 loc；不含任何 Python 特有节点、
不含任何类型/语义信息。intrinsic resolution 已在转换期完成
（Call.intrinsic 是 INTRINSICS 表解析后的名字）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ..diagnostics import Loc


@dataclass(frozen=True)
class Node:
    loc: Loc


# ---------------------------------------------------------------------------
# 类型注解（只出现在参数签名位置）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ann(Node):
    pass


@dataclass(frozen=True)
class TensorAnn(Ann):
    dtype: str                        # 内部短名（"f32" …，tila.float32 解析后）
    dims: Tuple[object, ...]          # (StaticDim | SymDim, …)，rank ≥ 1


@dataclass(frozen=True)
class ConstexprAnn(Ann):
    pass


@dataclass(frozen=True)
class StaticDim(Node):
    value: int


@dataclass(frozen=True)
class SymDim(Node):
    name: str                         # 自动绑定为运行时符号参数


# ---------------------------------------------------------------------------
# 顶层与语句
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelDef(Node):
    name: str
    params: Tuple["Param", ...]
    body: Tuple["Stmt", ...]          # ≥ 1 条（frontend 保证）


@dataclass(frozen=True)
class Param(Node):
    name: str
    ann: Optional[Ann]                # None → Scalar(i32)
    default: Optional[int]            # 仅 ConstexprAnn 允许非 None


@dataclass(frozen=True)
class Assign(Node):
    target: str                       # 单一名字（文法保证）
    value: "Expr"


@dataclass(frozen=True)
class ExprStmt(Node):
    call: "Call"                      # checker 强制 intrinsic == "store"（E08）


Stmt = Union[Assign, ExprStmt]


# ---------------------------------------------------------------------------
# 表达式
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntLit(Node):
    value: int                        # 可为负：-1 由转换期脱糖为负字面量


@dataclass(frozen=True)
class FloatLit(Node):
    value: float                      # 缺省 f32；类别匹配上下文按该 dtype 解释（R12）


@dataclass(frozen=True)
class NameRef(Node):
    name: str


@dataclass(frozen=True)
class BinOp(Node):
    op: str                           # + - * / < <= > >= == != & | （含 const-expr 的 //）
    lhs: "Expr"
    rhs: "Expr"


@dataclass(frozen=True)
class DTypeRef(Node):
    name: str                         # 内部短名；只允许出现在 Cast 第二实参与 TensorAnn


@dataclass(frozen=True)
class Call(Node):
    intrinsic: str                    # program_id | arange | load | store | cast
    args: Tuple["Expr", ...]
    kwargs: Tuple[Tuple[str, "Expr"], ...]  # 键 ∈ {mask, other}；tuple 保持可哈希


@dataclass(frozen=True)
class Cast(Node):
    dtype: str                        # tila.cast(x, tila.float32) 脱糖而来
    operand: "Expr"


Expr = Union[IntLit, FloatLit, NameRef, BinOp, Call, Cast, DTypeRef]
