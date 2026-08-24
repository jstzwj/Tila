"""Triton 源码渲染：表达式、优先级括号、匿名内联（docs/triton-lowering.md §4–§5）。"""

from __future__ import annotations

import math
from typing import Dict

from ... import tir
from ...fmt import fmt_float
from ...types import TRITON_DTYPE
from ...types.layout import strides_of
from ...types.shape import dim_str

# Python 运算符优先级（比较 = 6，| = 8，& = 9，+ - = 11，* / = 12；原子最高）
_PREC = {"<": 6, "<=": 6, ">": 6, ">=": 6, "==": 6, "!=": 6,
         "|": 8, "&": 9,
         "+": 11, "-": 11, "*": 12, "/": 12}
_ATOM = 100
_ADDPTR_PREC = 11


class OpRenderer:
    def __init__(self, kernel: tir.TKernel):
        self.kernel = kernel
        self.by_id: Dict[str, tir.TOp] = {}
        self.use_count: Dict[str, int] = {}
        self._materialized: set = set()
        self._walk(kernel.ops)

    def _walk(self, ops_seq) -> None:
        """by_id / use_count 递归统计（v0.4 起含循环体；匿名编号空间跨层唯一）。"""
        for op in ops_seq:
            if op.id:
                self.by_id[op.id] = op
            for oid in self._operands(op):
                self.use_count[oid] = self.use_count.get(oid, 0) + 1
            if isinstance(op, tir.TFor):
                self._walk(op.body)

    @staticmethod
    def _operands(op: tir.TOp):
        if isinstance(op, (tir.TArith, tir.TCmp, tir.TLogic)):
            return (op.lhs, op.rhs)
        if isinstance(op, tir.TAddPtr):
            return tuple(op.coords)  # base 是参数名，不是值 id
        if isinstance(op, tir.TLoad):
            return tuple(x for x in (op.ptr, op.mask, op.other) if x is not None)
        if isinstance(op, tir.TStore):
            return tuple(x for x in (op.ptr, op.value, op.mask) if x is not None)
        if isinstance(op, tir.TCast):
            return (op.operand,)
        if isinstance(op, tir.TExpandDim):
            return (op.tile,)
        if isinstance(op, tir.TDot):
            return (op.lhs, op.rhs)
        if isinstance(op, tir.TPhi):
            return (op.pre, op.back)
        if isinstance(op, tir.TFor):
            return (op.end,)
        if isinstance(op, tir.TReduce):
            return (op.tile,)
        if isinstance(op, tir.TElem):
            return (op.operand,)
        if isinstance(op, tir.TWhere):
            return (op.cond, op.a, op.b)
        return ()

    # ------------------------------------------------------------------

    def emission_name(self, op: tir.TOp) -> str:
        """值 id 的发射名：具名 → 源码名；合成引用 → 参数名；匿名 → 内联。"""
        if op.src_name is not None:
            return op.src_name
        if isinstance(op, (tir.TSymRef, tir.TConstParamRef)):
            return op.name
        raise ValueError(f"anonymous op {op.id} has no emission name")

    def render_operand(self, oid: str, min_prec: int = 0) -> str:
        """渲染操作数引用：具名 → 源码名；合成引用/参数名 → 原名；匿名 → 内联其表达式。

        min_prec 仅约束内联的二元宇宙表达式（嵌套二元运算的最小括号规则）；
        具名/合成引用是标识符（原子），调用实参上下文传 0（无需括号）。
        """
        op = self.by_id.get(oid)
        if op is None:
            return oid  # 未注解 scalar 参数引用（不是 op id）
        if op.src_name is not None or isinstance(op, (tir.TSymRef, tir.TConstParamRef)):
            return self.emission_name(op)
        expr = self.render_expr(op)
        prec = self._expr_prec(op)
        if prec < min_prec:
            return f"({expr})"
        return expr

    @staticmethod
    def _expr_prec(op: tir.TOp) -> int:
        if isinstance(op, (tir.TArith, tir.TCmp, tir.TLogic)):
            return _PREC[op.op]
        if isinstance(op, tir.TAddPtr):
            return _ADDPTR_PREC
        return _ATOM

    def _binop(self, op_str: str, lhs: str, rhs: str, prec: int) -> str:
        l = self.render_operand(lhs, prec)        # 左结合：左侧同优先级不加括号
        r = self.render_operand(rhs, prec + 1)    # 右侧同优先级必须加括号
        return f"{l} {op_str} {r}"

    def render_expr(self, op: tir.TOp) -> str:
        if isinstance(op, tir.TConstInt):
            return str(op.value)
        if isinstance(op, tir.TConstFloat):
            if math.isinf(op.value):
                # tila.neg_inf 等：非有限字面量在 Python 源码里只能这样拼写
                return f'float("{op.value}")'
            return fmt_float(op.value)
        if isinstance(op, (tir.TSymRef, tir.TConstParamRef)):
            return op.name
        if isinstance(op, tir.TProgramId):
            return f"tl.program_id({op.axis})"
        if isinstance(op, tir.TArange):
            return f"tl.arange({op.start}, {tir.constexpr_str(op.end)})"
        if isinstance(op, tir.TArith):
            return self._binop(op.op, op.lhs, op.rhs, _PREC[op.op])
        if isinstance(op, tir.TCmp):
            return self._binop(op.op, op.lhs, op.rhs, _PREC[op.op])
        if isinstance(op, tir.TLogic):
            return self._binop(op.op, op.lhs, op.rhs, _PREC[op.op])
        if isinstance(op, tir.TAddPtr):
            return self._addptr(op)
        if isinstance(op, tir.TLoad):
            args = [self.render_operand(op.ptr)]
            if op.mask is not None:
                args.append(f"mask={self.render_operand(op.mask)}")
            if op.other is not None:
                args.append(f"other={self.render_operand(op.other)}")
            return f"tl.load({', '.join(args)})"
        if isinstance(op, tir.TCast):
            return f"{self.render_operand(op.operand, _ATOM)}.to({TRITON_DTYPE[op.dtype]})"
        if isinstance(op, tir.TExpandDim):
            return f"tl.expand_dims({self.render_operand(op.tile)}, {op.axis})"
        if isinstance(op, tir.TDot):
            return (f"tl.dot({self.render_operand(op.lhs)}, "
                    f"{self.render_operand(op.rhs)})")
        if isinstance(op, tir.TZeros):
            inner = ", ".join(tir.constexpr_str(s) for s in op.shape)
            shape = f"({inner},)" if len(op.shape) == 1 else f"({inner})"
            return f"tl.zeros({shape}, dtype={TRITON_DTYPE[op.dtype]})"
        if isinstance(op, tir.TReduce):
            x = self.render_operand(op.tile, _ATOM)
            d = op.tila_type.dtype
            if op.op == "sum":
                # dtype 恒显式：Tila 语义钉死结果/累加 dtype——不继承 Triton
                # 的默认提升策略（int<32 → i32/u32，triton 3.7.1 实测）
                return f"tl.sum({x}, {op.axis}, dtype={TRITON_DTYPE[d]})"
            # tl.max 无 dtype 形参，且对 <32 位 dtype 内部提升 f32/i32 并以
            # 提升后 dtype 返回（3.7.1 实测）——cast 恢复 Tila dtype（max
            # 无舍入，恢复无损）
            if d in ("f16", "bf16", "i8", "i16", "u8", "u16"):
                return f"tl.cast(tl.max({x}, {op.axis}), {TRITON_DTYPE[d]})"
            return f"tl.max({x}, {op.axis})"
        if isinstance(op, tir.TElem):
            return f"tl.{op.op}({self.render_operand(op.operand, _ATOM)})"
        if isinstance(op, tir.TWhere):
            return (f"tl.where({self.render_operand(op.cond, _ATOM)}, "
                    f"{self.render_operand(op.a, _ATOM)}, "
                    f"{self.render_operand(op.b, _ATOM)})")
        if isinstance(op, tir.TNumPrograms):
            return f"tl.num_programs({op.axis})"
        if isinstance(op, (tir.TFor, tir.TPhi)):
            # 循环头/φ 以 src_name 在使用点渲染（k0 / acc），从不作为表达式内联
            return self.emission_name(op)
        raise AssertionError(f"cannot render {op!r}")  # pragma: no cover

    def _stride_terms(self, op: tir.TAddPtr):
        """逐轴 stride 渲染文本（v0.3-strides §4.1）；系数来自唯一的
        `types.layout.strides_of`（Address Function，评审 §22 采纳的事实源共享）。"""
        bparam = next((p for p in self.kernel.params
                       if p.kind == "buffer" and p.name == op.base), None)
        if bparam is None:  # pragma: no cover - typing 保证 base 是 buffer 参数
            raise AssertionError(f"addptr base {op.base!r} is not a buffer param")
        bty = bparam.tila_type
        return [dim_str(d) for d in strides_of(bty.mem, bty.shape)]

    def _addptr(self, op: tir.TAddPtr) -> str:
        """`base + (c₀*s₀ + c₁*s₁ + …)`；单坐标且步长 1 时退化为 `base + c`
        （rank-1 RowMajor 与 v0.2 逐字节一致）。"""
        terms = []
        for c, s in zip(op.coords, self._stride_terms(op)):
            coord = self.render_operand(c, _ATOM)
            terms.append(coord if s == "1" else f"{coord} * {s}")
        if len(terms) == 1:
            return f"{op.base} + {terms[0]}"
        return f"{op.base} + ({' + '.join(terms)})"

    # ------------------------------------------------------------------

    def body_lines(self) -> list:
        """kernel 体：每条源码赋值一行 + store 一行；匿名值按 use_count 决策。
        v0.4 起 TFor 递归渲染，每层 +4 空格（TPhi 与合成引用一样不发射）。"""
        taken = {op.src_name for op in self.kernel.ops if op.src_name}
        taken |= {p.name for p in self.kernel.params}

        def collect_names(ops_seq) -> None:
            for op in ops_seq:
                if op.src_name:
                    taken.add(op.src_name)
                if isinstance(op, tir.TFor):
                    collect_names(op.body)

        collect_names(self.kernel.ops)
        lines = []
        self._emit_ops(self.kernel.ops, 0, lines, taken)
        return lines

    def _emit_ops(self, ops_seq, depth: int, lines: list, taken: set) -> None:
        pad = "    " * depth
        for op in ops_seq:
            if isinstance(op, (tir.TReturn, tir.TSymRef, tir.TConstParamRef,
                               tir.TPhi)):
                continue  # 语句终结 / 合成引用 / φ：均不发射
            if isinstance(op, tir.TFor):
                end = self.render_operand(op.end)
                step = tir.constexpr_str(op.step)
                lines.append(f"{pad}for {op.src_name} in range(0, {end}, {step}):")
                self._emit_ops(op.body, depth + 1, lines, taken)
                continue
            if isinstance(op, tir.TStore):
                args = [self.render_operand(op.ptr), self.render_operand(op.value)]
                if op.mask is not None:
                    args.append(f"mask={self.render_operand(op.mask)}")
                lines.append(f"{pad}tl.store({', '.join(args)})")
                continue
            if op.src_name is not None:
                lines.append(f"{pad}{op.src_name} = {self.render_expr(op)}")
                continue
            uses = self.use_count.get(op.id, 0)
            if uses == 1:
                continue  # 匿名且恰被使用一次：在使用点内联（发射策略依据）
            if uses == 0:
                continue  # 死匿名值：不发射
            # 多次使用的匿名值：物化为变量（将来 CSE 产物走此路径）
            name = op.id
            while name in taken:
                name = "_" + name
            taken.add(name)
            lines.append(f"{pad}{name} = {self.render_expr(op)}")
