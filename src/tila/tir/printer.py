"""TIR canonical dump（docs/ast.md §5）。输出是黄金测试的比对物，格式必须确定。

规则：token 间以单个空格分隔；指令形如 `%id = opcode operand* : type [#src_name]`；
语句级指令（store/return）无 id 无类型；dist 打印正规形式，相同正规形式按首次
出现顺序命名 L0, L1, …，dump 首部列印 `L0 = identity(128)` 定义行。
"""

from __future__ import annotations

from ..fmt import fmt_float
from ..types import AddressType, TileType, type_str
from ..types.dist import Identity, Lift, Mma, Product, Seed, Slice, dist_str
from . import ops

_ARITH_OPCODE = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
_CMP_OPCODE = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_LOGIC_OPCODE = {"&": "and", "|": "or"}


def _fmt_value(v) -> str:
    if isinstance(v, int):
        return str(v)
    return fmt_float(v)


class DistNamer:
    """正规形式 dist → L0/L1/…（首次出现顺序；子项先于复合项命名）。"""

    def __init__(self):
        self._names = {}

    def register_type(self, t) -> None:
        if isinstance(t, (TileType, AddressType)):
            self._register(t.dist)

    def _register(self, term) -> None:
        if term in self._names:
            return
        if isinstance(term, Product):
            for c in term.components:
                self._register(c)
        elif isinstance(term, (Lift, Slice)):
            self._register(term.of if isinstance(term, Lift) else term.parent)
        self._names.setdefault(term, f"L{len(self._names)}")

    def name_of(self, term):
        return self._names.get(term)

    def definitions(self):
        """按命名顺序返回 `L0 = identity(128)` 定义行（顶层项打印本体，子项打印名字）。"""
        lines = []
        for term, name in self._names.items():
            sub = lambda t: None if t is term else self.name_of(t)  # noqa: E731
            lines.append(f"{name} = {dist_str(term, sub)}")
        return lines


def _params_str(kernel: ops.TKernel) -> str:
    parts = []
    for p in kernel.params:
        if p.kind == "constexpr":
            parts.append(f"{p.name}: Constexpr(i32)={p.default}")
        else:
            parts.append(f"{p.name}: {type_str(p.tila_type)}")
    return ", ".join(parts)


def _op_line(op: ops.TOp, namer: DistNamer) -> str:
    toks = []
    if op.id is not None:
        toks.append(f"%{op.id} =")
    if isinstance(op, ops.TConstInt):
        toks += ["const", str(op.value)]
    elif isinstance(op, ops.TConstFloat):
        toks += ["const", fmt_float(op.value)]
    elif isinstance(op, ops.TSymRef):
        toks += ["sym_ref", op.name]
    elif isinstance(op, ops.TConstParamRef):
        toks += ["constexpr_ref", op.name]
    elif isinstance(op, ops.TProgramId):
        toks += ["program_id", str(op.axis)]
    elif isinstance(op, ops.TArange):
        toks += ["arange", str(op.start), ops.constexpr_str(op.end)]
    elif isinstance(op, ops.TAddPtr):
        # 单坐标与 v0.2 逐字节一致；多坐标用方括号逗号表（v0.3-strides §3）
        if len(op.coords) == 1:
            toks += ["addptr", f"%{op.base}", f"%{op.coords[0]}"]
        else:
            inner = ", ".join(f"%{c}" for c in op.coords)
            toks += ["addptr", f"%{op.base}", f"[{inner}]"]
    elif isinstance(op, ops.TArith):
        toks += [_ARITH_OPCODE[op.op], f"%{op.lhs}", f"%{op.rhs}"]
    elif isinstance(op, ops.TCmp):
        toks += [_CMP_OPCODE[op.op], f"%{op.lhs}", f"%{op.rhs}"]
    elif isinstance(op, ops.TLogic):
        toks += [_LOGIC_OPCODE[op.op], f"%{op.lhs}", f"%{op.rhs}"]
    elif isinstance(op, ops.TLoad):
        toks += ["load", f"%{op.ptr}"]
        if op.mask is not None:
            toks.append(f"mask=%{op.mask}")
        if op.other is not None:
            toks.append(f"other=%{op.other}")
    elif isinstance(op, ops.TCast):
        toks += ["cast", f"%{op.operand}", op.dtype]
    elif isinstance(op, ops.TExpandDim):
        toks += ["expand_dim", f"%{op.tile}", str(op.axis)]
    elif isinstance(op, ops.TDot):
        toks += ["dot", f"%{op.lhs}", f"%{op.rhs}"]
    elif isinstance(op, ops.TZeros):
        shape = "(" + ", ".join(ops.constexpr_str(s) for s in op.shape) + ")"
        toks += ["zeros", shape, op.dtype]
    elif isinstance(op, ops.TFull):
        shape = "(" + ", ".join(ops.constexpr_str(s) for s in op.shape) + ")"
        toks += ["full", shape, _fmt_value(op.value), op.dtype]
    elif isinstance(op, ops.TMaximum):
        toks += ["maximum", f"%{op.lhs}", f"%{op.rhs}"]
    elif isinstance(op, ops.TLaunchAssert):
        toks += ["launch_assert", op.cond]
    elif isinstance(op, ops.TReduce):
        toks += [op.op, f"%{op.tile}", str(op.axis)]
    elif isinstance(op, ops.TElem):
        toks += [op.op, f"%{op.operand}"]
    elif isinstance(op, ops.TWhere):
        toks += ["where", f"%{op.cond}", f"%{op.a}", f"%{op.b}"]
    elif isinstance(op, ops.TNumPrograms):
        toks += ["num_programs", str(op.axis)]
    elif isinstance(op, ops.TPhi):
        toks += ["phi", f"%{op.pre}", f"%{op.back}"]
    elif isinstance(op, ops.TStore):
        toks += ["store", f"%{op.ptr}", f"%{op.value}"]
        if op.mask is not None:
            toks.append(f"mask=%{op.mask}")
    elif isinstance(op, ops.TReturn):
        toks.append("return")
    else:  # pragma: no cover
        raise AssertionError(f"unknown op {op!r}")

    if op.id is not None and op.tila_type is not None:
        toks += [":", type_str(op.tila_type, namer.name_of)]
    if op.src_name is not None:
        toks.append(f"[#{op.src_name}]")
    return " ".join(toks)


def _for_header(op: ops.TFor, namer: DistNamer) -> str:
    """`%k0 = for 0 %K BK : Scalar(i32) [#k0]`（start 隐含字面量 0）。"""
    toks = [f"%{op.id} =", "for", "0", f"%{op.end}", ops.constexpr_str(op.step)]
    if op.tila_type is not None:
        toks += [":", type_str(op.tila_type, namer.name_of)]
    if op.src_name is not None:
        toks.append(f"[#{op.src_name}]")
    return " ".join(toks)


def _walk(ops_seq, namer: DistNamer, depth: int, out: list) -> None:
    """平铺指令行；TFor 的 body 递归缩进 2 空格（v0.4-kloop §3）。"""
    pad = "  " * depth
    for op in ops_seq:
        if isinstance(op, ops.TFor):
            out.append(pad + _for_header(op, namer))
            _walk(op.body, namer, depth + 1, out)
        else:
            out.append(pad + _op_line(op, namer))


def _register_ops(ops_seq, namer: DistNamer) -> None:
    for op in ops_seq:
        if op.tila_type is not None:
            namer.register_type(op.tila_type)
        if isinstance(op, ops.TFor):
            _register_ops(op.body, namer)


def dump(kernel: ops.TKernel) -> str:
    """canonical dump（单空格分隔、LF、恰一个尾换行）。"""
    namer = DistNamer()
    _register_ops(kernel.ops, namer)

    lines = [f"func @{kernel.name}({_params_str(kernel)})"]
    lines.extend(namer.definitions())
    lines.append("")
    out: list = []
    _walk(kernel.ops, namer, 0, out)
    lines.extend(out)
    return "\n".join(lines) + "\n"