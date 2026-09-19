"""Tila TIR：typed SSA 风格中间表示（straight-line + If/For 区块）。

checker 单遍产出；特化期把 Const 符号代入（zeros 形状、range 步长、
arange 上界等），lowering 与 interpreter 共同消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import types as TY


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

@dataclass
class TBufferParam:
    name: str
    vtype: TY.BufferT
    region_id: TY.RegionId


@dataclass
class TPtrParam:
    """裸指针参数：公共签名含 extent/alignment，RegionId 由 checker 生成。"""
    name: str
    vtype: TY.PtrT


@dataclass(frozen=True)
class TEffect:
    op: str
    region_id: TY.RegionId

    def describe(self):
        return f"{self.op}[{self.region_id}]"


@dataclass
class TScalarParam:
    name: str
    dtype: object            # D.DType（i32 维/步长/标量；或声明 dtype）
    refined: tuple = ()      # 标量精化（launch 契约）


@dataclass
class TConstParam:
    name: str
    refinements: tuple
    default: int | None


# ---------------------------------------------------------------------------
# 操作数与表达式
# ---------------------------------------------------------------------------

@dataclass
class TOperand:
    pass


@dataclass
class TName(TOperand):
    name: str


@dataclass
class TLit(TOperand):
    value: object
    dtype: object


@dataclass
class TExpr(TOperand):
    """所有值产生节点；vt 为完整 Tila 类型。"""
    vt: object


@dataclass
class TBin(TExpr):
    op: str
    left: TOperand
    right: TOperand
    staged: bool = False
    checked_index: bool = False
    operand_dtype: object = None


@dataclass
class TUna(TExpr):
    op: str
    operand: TOperand
    staged: bool = False
    checked_index: bool = False


@dataclass
class TCast(TExpr):
    dtype: object
    operand: TOperand


@dataclass
class TArange(TExpr):
    start: object            # int（恒 0 语法限制之外仍保留）
    end: TOperand            # TLit（Const 已代入后）或 TName（Const 参数）


@dataclass
class TPid(TExpr):
    axis: int


@dataclass
class TNumPrograms(TExpr):
    axis: int


@dataclass
class TZeros(TExpr):
    shape: list              # TOperand 列表（Const 已知后为 TLit）


@dataclass
class TReshape(TExpr):
    """reshape（intrinsics.md §2.8）：numel 保持的形状重排。

    shape 与 TZeros 相同——Const 语境 operand（TLit / TName）；
    TName 为 Const 参数名（Triton 路径上是 tl.constexpr，interp 路径
    由 consts 环境取值）。checker 已保证 numel(S1) = numel(S2)。
    """
    operand: TOperand
    shape: list              # TOperand 列表


@dataclass
class TExpand(TExpr):
    operand: TOperand
    axis: int


@dataclass
class TWhere(TExpr):
    cond: TOperand
    a: TOperand
    b: TOperand


@dataclass
class TDot(TExpr):
    a: TOperand
    b: TOperand
    acc: TOperand | None


@dataclass
class TReduce(TExpr):
    op: str                  # sum | max
    operand: TOperand
    axis: int


@dataclass
class TLoad(TExpr):
    buffer: str | None       # Buffer 形式
    coords: list             # Buffer 形式的坐标 operands
    ptr: TOperand | None     # Ptr 形式的指针值
    mask: TOperand | None
    other: TOperand | None
    unsafe: bool = False


@dataclass
class TBufPtr(TExpr):
    buffer: str


@dataclass
class TPAdd(TExpr):
    ptr: TOperand
    offset: TOperand


@dataclass
class TAssumeExpr(TExpr):
    """assume 的谓词表达式（bool/Mask）。"""
    operand: TOperand


# ---------------------------------------------------------------------------
# 语句（line 放末尾，构造按语义顺序传参）
# ---------------------------------------------------------------------------

class TStmt:
    pass


@dataclass
class TAssign(TStmt):
    name: str
    value: TOperand
    line: int = 0


@dataclass
class TStore(TStmt):
    buffer: str | None
    coords: list
    ptr: TOperand | None
    value: TOperand
    mask: TOperand | None
    unsafe: bool = False
    line: int = 0


@dataclass
class TAssume(TStmt):
    pred: TOperand
    line: int = 0


@dataclass
class TIf(TStmt):
    cond: TOperand
    then_body: list
    else_body: list
    line: int = 0


@dataclass
class TStaticIf(TStmt):
    """StaticIf（docs/type-system.md §5.2）：条件由 Const 参数构成的 if。

    与 TIf 严格分离：装饰期条件不可定值（Const 数值到特化期才代入），
    两分支均进入 IR；实际执行的分支由每次特化决定——Triton 路径上
    条件引用 tl.constexpr 实参（trace 期消除），interp 路径上由
    self.consts 求值后走 python 分支。
    """
    cond: TOperand
    then_body: list
    else_body: list
    line: int = 0


@dataclass
class TFor(TStmt):
    var: str
    end: TOperand
    step: TOperand           # Const（特化后 TLit）
    body: list
    line: int = 0
    # 运行期起点（intrinsics.md §2.1）：None = 起点 0（既有形态，
    # lowering 保持 range(0, end, step)）。
    start: TOperand | None = None


@dataclass
class TReturn(TStmt):
    """裸 return：提前退出当前 program instance（early exit）。
    无值——带值返回在 frontend 即被拒绝（TILA-SYN-021）。"""
    line: int = 0


@dataclass
class TKernel:
    name: str
    buffers: list            # TBufferParam（保序）
    scalars: list            # TScalarParam（显式 + 隐式维/步长，保序）
    consts: list             # TConstParam
    body: list
    ptr_params: list = field(default_factory=list)   # TPtrParam（裸指针参数，保序）
    # 声明序参数表 (kind, name)：kind ∈ buffer | ptr | scalar | const。
    # runtime 位置实参映射的唯一依据（buffer/ptr/显式标量可位置传参）。
    param_order: list = field(default_factory=list)
    types: dict = field(default_factory=dict)   # name -> Type（最后定义点）
    obligations: list = field(default_factory=list)
    effects: list = field(default_factory=list)     # list[TEffect]
    aliases: list = field(default_factory=list)     # list[TY.AliasFact]
    runtime_aliases: list = field(default_factory=list)  # 上次 launch 快照
    runtime_alignments: dict = field(default_factory=dict)  # RegionId -> bytes
    buffer_ptr_views: set = field(default_factory=set)  # 使用过 `.ptr` 的 Buffer 名
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)   # static-if 解析等说明
    hints: list = field(default_factory=list)   # (var, span) max_contiguous 发射点
    hint_origins: dict = field(default_factory=dict)  # name -> set[Origin]; no assume-derived hints yet
    deferred: list = field(default_factory=list)  # 特化期复查的 shape 等价约束
    refinements: dict = field(default_factory=dict)  # pid sym -> (bound_expr, step_expr) 由 launch 登记
    nonneg_syms: set = field(default_factory=set)   # 结构性非负符号（证明用）
    implicit_names: set = field(default_factory=set)  # 隐式标量（维/步长符号）名
    explicit_scalars: set = field(default_factory=set)  # 用户显式声明的标量参数名
    sym_hi: dict = field(default_factory=dict)   # 符号区间上界快照（证明用）
    sym_lo: dict = field(default_factory=dict)   # 符号区间下界快照

    def dump(self) -> str:
        return _Printer(self).dump()


# ---------------------------------------------------------------------------
# canonical dump
# ---------------------------------------------------------------------------

def shape_tuple_text(shape: list, render) -> str:
    """shape operand 列表 → "d0, d1" 文本（单项补尾逗号保持 tuple 语义：
    (512,) 是 tuple，(512) 是 int——Triton reshape 的形状实参须为 tuple）。"""
    inner = ", ".join(render(d) for d in shape)
    return inner + "," if len(shape) == 1 else inner


class _Printer:
    def __init__(self, k: TKernel):
        self.k = k

    def dump(self) -> str:
        out = [f"kernel @{self.k.name}"]
        for b in self.k.buffers:
            out.append(f"  buffer {b.name} : {b.vtype.describe()}"
                       f"  region={b.region_id}")
        for p in self.k.ptr_params:
            out.append(f"  ptr    {p.name} : {p.vtype.describe()}"
                       f"  region={p.vtype.region_id}")
        for s in self.k.scalars:
            r = "".join(f" | {x.text}" for x in s.refined)
            out.append(f"  scalar {s.name} : {s.dtype.name}{r}")
        for c in self.k.consts:
            d = f" = {c.default}" if c.default is not None else ""
            out.append(f"  const  {c.name} : Const[int]{d}")
        out.append("  {")
        for st in self.k.body:
            out.extend(self.stmt(st, 2))
        out.append("  }")
        for ob in self.k.obligations:
            out.append(f"  obligation {ob.describe()}")
        for effect in self.k.effects:
            out.append(f"  effect {effect.describe()}")
        for alias in self.k.aliases:
            out.append(f"  alias {alias.describe()}")
        return "\n".join(out) + "\n"

    def stmt(self, s: TStmt, ind: int) -> list[str]:
        pad = " " * ind
        if isinstance(s, TAssign):
            return [f"{pad}{s.name} = {self.e(s.value)}"]
        if isinstance(s, TStore):
            dst = self.dst(s.buffer, s.coords, s.ptr)
            m = f", mask={self.o(s.mask)}" if s.mask is not None else ""
            u = "unsafe_" if s.unsafe else ""
            return [f"{pad}{u}store {dst} <- {self.o(s.value)}{m}"]
        if isinstance(s, TAssume):
            return [f"{pad}assume {self.o(s.pred)}"]
        if isinstance(s, TIf):
            out = [f"{pad}if {self.o(s.cond)}:"]
            for x in s.then_body:
                out.extend(self.stmt(x, ind + 2))
            if s.else_body:
                out.append(f"{pad}else:")
                for x in s.else_body:
                    out.extend(self.stmt(x, ind + 2))
            return out
        if isinstance(s, TStaticIf):
            out = [f"{pad}static_if {self.o(s.cond)}:"
                   f"  # both branches present pre-specialization"]
            for x in s.then_body:
                out.extend(self.stmt(x, ind + 2))
            if s.else_body:
                out.append(f"{pad}else:")
                for x in s.else_body:
                    out.extend(self.stmt(x, ind + 2))
            return out
        if isinstance(s, TFor):
            start = f"{self.o(s.start)}, " if s.start is not None else "0, "
            out = [f"{pad}for {s.var} in range({start}{self.o(s.end)}, {self.o(s.step)}):"]
            for x in s.body:
                out.extend(self.stmt(x, ind + 2))
            return out
        if isinstance(s, TReturn):
            return [f"{pad}return"]
        return [f"{pad}?{s!r}"]

    def dst(self, buffer, coords, ptr):
        if buffer is not None:
            cs = ", ".join(self.o(c) for c in coords)
            return f"{buffer}[{cs}]"
        return f"*{self.o(ptr)}"

    def o(self, x: TOperand) -> str:
        if isinstance(x, TName):
            return x.name
        if isinstance(x, TLit):
            v = x.value
            if isinstance(v, float) and v == int(v):
                return f"{int(v)}.0"
            return str(v)
        return self.e(x)

    def e(self, x: TExpr) -> str:
        # 赋值值可为纯操作数形态（名字拷贝 x = y）——TName/TLit 走 o()。
        if isinstance(x, (TName, TLit)):
            return self.o(x)
        if isinstance(x, TBin):
            return f"({self.o(x.left)} {x.op} {self.o(x.right)})"
        if isinstance(x, TUna):
            return f"({x.op}{self.o(x.operand)})"
        if isinstance(x, TCast):
            return f"cast[{x.dtype.name}]({self.o(x.operand)})"
        if isinstance(x, TArange):
            return f"arange({x.start}, {self.o(x.end)})"
        if isinstance(x, TPid):
            return f"program_id({x.axis})"
        if isinstance(x, TNumPrograms):
            return f"num_programs({x.axis})"
        if isinstance(x, TZeros):
            return "zeros((" + ", ".join(self.o(d) for d in x.shape) + f"), {x.vt.elem.dtype.name})"
        if isinstance(x, TReshape):
            return (f"reshape({self.o(x.operand)}, "
                    "(" + shape_tuple_text(x.shape, self.o) + "))")
        if isinstance(x, TExpand):
            return f"expand({self.o(x.operand)}, axis={x.axis})"
        if isinstance(x, TWhere):
            return f"where({self.o(x.cond)}, {self.o(x.a)}, {self.o(x.b)})"
        if isinstance(x, TDot):
            acc = f", acc={self.o(x.acc)}" if x.acc is not None else ""
            return f"dot({self.o(x.a)}, {self.o(x.b)}{acc})"
        if isinstance(x, TReduce):
            return f"{x.op}({self.o(x.operand)}, axis={x.axis})"
        if isinstance(x, TLoad):
            dst = self.dst(x.buffer, x.coords, x.ptr)
            m = f", mask={self.o(x.mask)}" if x.mask is not None else ""
            o = f", other={self.o(x.other)}" if x.other is not None else ""
            u = "unsafe_" if x.unsafe else ""
            return f"{u}load({dst}{m}{o})"
        if isinstance(x, TBufPtr):
            return f"{x.buffer}.ptr"
        if isinstance(x, TPAdd):
            return f"({self.o(x.ptr)} + {self.o(x.offset)})"
        if isinstance(x, TAssumeExpr):
            return f"assume({self.o(x.operand)})"
        return f"?{x!r}"
