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
    """opcode: addptr  R7/R7'：Buffer + 索引 → Address。

    coords 是逐轴坐标 tile 的引用元组（长度 = buffer rank）。线性化不进 TIR：
    lowering 与 interpreter 从 base 参数的 MemoryLayout 完成（v0.3-strides §3）。
    """

    base: str
    coords: Tuple[str, ...] = ()


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


@dataclass(frozen=True)
class TZeros(TOp):
    """opcode: zeros（R17，v0.4-kloop §2.1）：常量分布种子。

    shape 每项是 ConstExpr（发射按表达式渲染，typing 用特化值）——与
    TArange.end 同款模型 B。累加器的唯一合法播种方式。
    """

    shape: Tuple[ConstExpr, ...] = ()
    dtype: str = "f32"


@dataclass(frozen=True)
class TFor(TOp):
    """opcode: for（R18，v0.4-kloop §3）：一层可嵌套的循环容器。

    id 即归纳变量（Scalar(i32)）；start 隐含字面量 0（规范化）；end 是
    运行期标量操作数；step 是 ConstExpr。body 是嵌套指令序列——TIR 从
    扁平变为"一层可嵌套"的唯一位置（嵌套 TFor 由 checker E20 拦截）。
    """

    end: str = ""
    step: ConstExpr = 1
    body: Tuple[TOp, ...] = ()


@dataclass(frozen=True)
class TPhi(TOp):
    """opcode: phi（R19，v0.4-kloop §3）：structured loop-carried φ。

    位于 body 顶部；back 指向同 body 内后文定义的最后一个 += 结果——
    **TIR 唯一允许前向引用的指令**（SSA φ 的标准形态）。自带 tila_type
    （= back 操作数的类型；pre 在 v0.4 恒为 zeros 种子——join 单位元，L7，
    printer 无需按 id 回查）。当前 IR 无一般 CFG（控制流倾向 select），
    φ 在 TIR 中无歧义；若将来引入 CFG φ，那是另行设计的课题。
    """

    pre: str = ""
    back: str = ""


@dataclass(frozen=True)
class TReduce(TOp):
    """opcode: sum / max（R20，v0.5-reduce §2.1）：轴归约——分布的边缘化。

    op/axis 已在 checker 特化（axis ∈ {0,1}，ConstExpr 不进 TIR）；
    结果 dtype = 输入 dtype（无隐式提升——Triton 侧的默认提升策略由
    lowering 显式抵消，v0.5-reduce §4）。
    """

    op: str = "sum"
    tile: str = ""
    axis: int = 1


@dataclass(frozen=True)
class TElem(TOp):
    """opcode: exp / exp2 / sqrt / abs（R21，v0.5-reduce §2.2）：逐元素一元。

    律 L8 是记法不是 term——结果 layout = 操作数 layout，IR 不物化 ElemL。
    """

    op: str = "exp"
    operand: str = ""


@dataclass(frozen=True)
class TWhere(TOp):
    """opcode: where（R22，v0.5-reduce §2.3）：值选择（select 形态的条件值）。

    语义只规定选中值（result_i = c_i ? a_i : b_i）；未选分支是否求值不可
    观察（非严格——backend 可自由优化）。字面量分支以 const 指令物化。
    """

    cond: str = ""
    a: str = ""
    b: str = ""


@dataclass(frozen=True)
class TNumPrograms(TOp):
    """opcode: num_programs（R23，v0.5-reduce §2.4）：program-context query。

    只读观测当前 grid 的该维大小——不参与 launch 推导（observational，
    E17 的推导输入只有 program_id）。
    """

    axis: int = 0


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
    stride_bindings: Tuple[Tuple[str, int, Union[int, str]], ...] = ()  # v0.3：(buffer, 轴, 静态值|符号名)


@dataclass(frozen=True)
class TKernel:
    name: str
    params: Tuple[TParam, ...]              # lowering 生成签名的唯一依据
    ops: Tuple[TOp, ...]
    launch_plan: LaunchPlan
    loc: Loc = field(default=Loc(1, 1))
