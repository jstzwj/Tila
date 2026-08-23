"""Triton 源码渲染：表达式、优先级括号、匿名内联（docs/triton-lowering.md §4–§5）。"""

from __future__ import annotations

from typing import Dict

from ... import tir
from ...fmt import fmt_float
from ...types import TRITON_DTYPE

# Python 运算符优先级（比较 = 6，| = 8，& = 9，+ - = 11，* / = 12；原子最高）
_PREC = {"<": 6, "<=": 6, ">": 6, ">=": 6, "==": 6, "!=": 6,
         "|": 8, "&": 9,
         "+": 11, "-": 11, "*": 12, "/": 12}
_ATOM = 100
_ADDPTR_PREC = 11


class OpRenderer:
    def __init__(self, kernel: tir.TKernel):
        self.kernel = kernel
        self.by_id: Dict[str, tir.TOp] = {op.id: op for op in kernel.ops if op.id}
        self.use_count: Dict[str, int] = {}
        for op in kernel.ops:
            for oid in self._operands(op):
                self.use_count[oid] = self.use_count.get(oid, 0) + 1
        self._materialized: set = set()

    @staticmethod
    def _operands(op: tir.TOp):
        if isinstance(op, (tir.TArith, tir.TCmp, tir.TLogic)):
            return (op.lhs, op.rhs)
        if isinstance(op, tir.TAddPtr):
            return (op.offs,)  # base 是参数名，不是值 id
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
            return f"{op.base} + {self.render_operand(op.offs, _ADDPTR_PREC + 1)}"
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
        raise AssertionError(f"cannot render {op!r}")  # pragma: no cover

    # ------------------------------------------------------------------

    def body_lines(self) -> list:
        """kernel 体：每条源码赋值一行 + store 一行；匿名值按 use_count 决策。"""
        taken = {op.src_name for op in self.kernel.ops if op.src_name}
        taken |= {p.name for p in self.kernel.params}
        lines = []
        for op in self.kernel.ops:
            if isinstance(op, tir.TReturn):
                continue
            if isinstance(op, (tir.TSymRef, tir.TConstParamRef)):
                continue  # 合成引用：按参数名在使用点直接渲染，永不物化
            if isinstance(op, tir.TStore):
                args = [self.render_operand(op.ptr), self.render_operand(op.value)]
                if op.mask is not None:
                    args.append(f"mask={self.render_operand(op.mask)}")
                lines.append(f"tl.store({', '.join(args)})")
                continue
            if op.src_name is not None:
                lines.append(f"{op.src_name} = {self.render_expr(op)}")
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
            lines.append(f"{name} = {self.render_expr(op)}")
        return lines
