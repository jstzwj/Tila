"""TIR：typed straight-line SSA（docs/ast.md §4）。

kernel 体的扁平指令序列：每个值一个 %id，先定义后使用，单赋值；每条指令携带
推导出的完整类型（含 layout term）与来源信息。TIR 不做类型推断——类型在
checker 已全部 resolve（不变量 6，lowering 能 total 的前提）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from ..diagnostics import Loc
from ..types.layout import LayoutTerm
from ..types.type import TilaType

# 编译期常量表达式：字面量 | constexpr 名 | 整数运算（发射按表达式渲染，不折叠——模型 B）
ConstExpr = Union[int, str, "CBinOp"]


@dataclass(frozen=True)
class CBinOp:
    op: str  # + - * //
    lhs: ConstExpr
    rhs: ConstExpr


def constexpr_str(ce: ConstExpr) -> str:
    if isinstance(ce, int):
        return str(ce)
    if isinstance(ce, str):
        return ce
    return f"{constexpr_str(ce.lhs)} {ce.op} {constexpr_str(ce.rhs)}"


@dataclass(frozen=True)
class TOp:
    """所有指令的基类。id 为 None 表示语句级指令（store/return，不产生值）。"""

    id: Optional[str]
    tila_type: Optional[TilaType]
    src_name: Optional[str]                    # 源码赋值目标名；匿名指令为 None
    origin_layout: Optional[LayoutTerm]        # 规范化前的原始 term（仅诊断用）


@dataclass(frozen=True)
class TConstInt(TOp):
    value: int                              # opcode: const


@dataclass(frozen=True)
class TConstFloat(TOp):
    value: float                            # opcode: const


@dataclass(frozen=True)
class TSymRef(TOp):
    name: str                               # opcode: sym_ref    N → Scalar(i32)


@dataclass(frozen=True)
class TConstParamRef(TOp):
    name: str                               # opcode: constexpr_ref  检查用值、发射用名


@dataclass(frozen=True)
class TProgramId(TOp):
    axis: int                               # opcode: program_id


@dataclass(frozen=True)
class TArange(TOp):
    start: int                              # 恒 0
    end: ConstExpr                          # 发射按表达式渲染；typing 用 build 期特化值


@dataclass(frozen=True)
class TAddPtr(TOp):
    base: str                               # opcode: addptr  R7：Buffer + Tile[i32] → Address
    offs: str


@dataclass(frozen=True)
class TArith(TOp):
    op: str                                 # opcode: add/sub/mul/div（R3/R4/R5）
    lhs: str
    rhs: str


@dataclass(frozen=True)
class TCmp(TOp):
    op: str                                 # opcode: lt/le/gt/ge/eq/ne（R6）
    lhs: str
    rhs: str


@dataclass(frozen=True)
class TLogic(TOp):
    op: str                                 # opcode: and/or（R10，Tile[bool]）
    lhs: str
    rhs: str


@dataclass(frozen=True)
class TLoad(TOp):
    ptr: str                                # opcode: load
    mask: Optional[str] = None
    other: Optional[str] = None


@dataclass(frozen=True)
class TStore(TOp):
    ptr: str                                # opcode: store（语句级，无值 id）
    value: str = ""
    mask: Optional[str] = None


@dataclass(frozen=True)
class TCast(TOp):
    dtype: str                              # opcode: cast（R11）
    operand: str = ""


@dataclass(frozen=True)
class TReturn(TOp):                         # opcode: return（语句级）
    pass


@dataclass(frozen=True)
class TExpandDim(TOp):
    """opcode: expand_dim（R13，v0.2 预览）；axis 按结果张量轴编号。"""

    tile: str = ""
    axis: int = 0


@dataclass(frozen=True)
class TDot(TOp):
    """opcode: dot（R16，v0.2 matmul fragment）：MMA 布局的构造点。

    操作数布局无前提（dot 即布局家族边界：blocked 进，MMA 出）；
    结果 Tile[f32, (M,N), Mma(M,N,K)]。
    """

    lhs: str = ""
    rhs: str = ""


# v0.2 预览：TExpandDim 已实现（docs/v0.2-preview-2d.md）；size-1 广播隐含于
# TArith/TCmp/TLogic 操作数的 shape，不设独立指令。


@dataclass(frozen=True)
class GridAxis:
    """一维 grid 轴：维（符号维名 | 静态值）、tiling constexpr（名 | 字面量）。"""

    dim: Union[str, int]
    tiling: Union[str, int]


@dataclass(frozen=True)
class TParam:
    name: str
    kind: str                               # "buffer" | "sym" | "constexpr" | "scalar"
    tila_type: TilaType
    default: Optional[int] = None           # constexpr 用


@dataclass(frozen=True)
class LaunchPlan:
    """launch analysis phase 的推导结果；lowering 只发射不推导（E17 在 checker 消化）。"""

    axes: Tuple[GridAxis, ...]
    rank_asserts: Tuple[Tuple[str, int], ...] = ()          # (buffer 名, 注解 rank)
    static_dim_asserts: Tuple[Tuple[str, int, int], ...] = ()  # (buffer 名, 维下标, 静态值)
    sym_dim_asserts: Tuple[Tuple[str, int, str], ...] = ()  # (buffer 名, 维下标, 符号维名)


@dataclass(frozen=True)
class TKernel:
    name: str
    params: Tuple[TParam, ...]              # lowering 生成签名的唯一依据
    ops: Tuple[TOp, ...]
    launch_plan: LaunchPlan
    loc: Loc = field(default=Loc(1, 1))
