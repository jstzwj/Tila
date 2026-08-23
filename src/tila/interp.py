"""Reference interpreter（docs/development-plan.md §8）：TIR → NumPy（CPU）。

作为独立 oracle：同一 TIR 走 interpreter 与 Triton lowering 双路径，结果应一致。
masked load 在 other 缺省时 masked-out 通道取 0（Triton 语义为未定义—— sanctioned
用法下这些通道永远不会被写回，差分测试因此不受影响）。
"""

from __future__ import annotations

import itertools
from typing import Dict, Optional

import numpy as np

from . import tir
from .fmt import f32_round

_NP: Dict[str, object] = {
    "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
    "f16": np.float16, "f32": np.float32, "f64": np.float64,
    "bool": np.bool_,
}
try:  # bf16 / fp8 需要 ml_dtypes；缺失时相关 dtype 明确报不支持
    import ml_dtypes

    _NP["bf16"] = ml_dtypes.bfloat16
    _NP["fp8e4m3"] = ml_dtypes.float8_e4m3
    _NP["fp8e5m2"] = ml_dtypes.float8_e5m2
    _NP["fp8e4m3fn"] = ml_dtypes.float8_e4m3fn
except ImportError:  # pragma: no cover - 环境差异
    pass


def np_dtype(dt_name: str):
    if dt_name not in _NP:
        raise RuntimeError(
            f"interpreter: dtype '{dt_name}' needs the optional 'ml_dtypes' package "
            f"(pip install ml_dtypes)")
    return _NP[dt_name]


class UnsupportedInterpreterFeature(RuntimeError):
    pass


def _eval_constexpr(ce, constexpr_values: Dict[str, int]) -> int:
    if isinstance(ce, int):
        return ce
    if isinstance(ce, str):
        return constexpr_values[ce]
    lhs = _eval_constexpr(ce.lhs, constexpr_values)
    rhs = _eval_constexpr(ce.rhs, constexpr_values)
    if ce.op == "+":
        return lhs + rhs
    if ce.op == "-":
        return lhs - rhs
    if ce.op == "*":
        return lhs * rhs
    if ce.op == "//":
        return lhs // rhs
    raise AssertionError(f"unknown constexpr op {ce.op}")  # pragma: no cover


class Interpreter:
    def __init__(self, kernel: tir.TKernel):
        self.kernel = kernel

    def run(self, buffers: Dict[str, "np.ndarray"],
            scalars: Optional[Dict[str, int]] = None,
            constexpr: Optional[Dict[str, int]] = None) -> None:
        scalars = dict(scalars or {})
        constexpr_values = {
            p.name: p.default for p in self.kernel.params
            if p.kind == "constexpr" and p.default is not None
        }
        constexpr_values.update(constexpr or {})

        # 符号维从首个绑定 buffer 的 shape 读取（launcher 同款逻辑）
        first_binding = {}
        for b, i, s in self.kernel.launch_plan.sym_dim_asserts:
            first_binding.setdefault(s, (b, i))
        for s, (b, i) in first_binding.items():
            scalars.setdefault(s, int(buffers[b].shape[i]))

        plan = self.kernel.launch_plan
        grid_dims = []
        for axis in plan.axes:
            dim = scalars[axis.dim] if isinstance(axis.dim, str) else int(axis.dim)
            tiling = constexpr_values[axis.tiling] if isinstance(axis.tiling, str) \
                else int(axis.tiling)
            grid_dims.append(max(1, -(-dim // tiling)))
        if not grid_dims:
            grid_dims = [1]

        for pids in itertools.product(*(range(g) for g in grid_dims)):
            self._run_program(pids, buffers, scalars, constexpr_values)

    # ------------------------------------------------------------------

    def _run_program(self, pids, buffers, scalars, constexpr_values) -> None:
        vals: Dict[str, object] = {
            p.name: scalars[p.name]
            for p in self.kernel.params if p.kind == "scalar"
        }

        def v(oid: str):
            return vals[oid]

        for op in self.kernel.ops:
            if isinstance(op, tir.TProgramId):
                vals[op.id] = pids[op.axis]
            elif isinstance(op, (tir.TSymRef, tir.TConstParamRef)):
                vals[op.id] = scalars[op.name] if isinstance(op, tir.TSymRef) \
                    else constexpr_values[op.name]
            elif isinstance(op, tir.TConstInt):
                vals[op.id] = op.value
            elif isinstance(op, tir.TConstFloat):
                vals[op.id] = f32_round(op.value)
            elif isinstance(op, tir.TArange):
                end = _eval_constexpr(op.end, constexpr_values)
                vals[op.id] = np.arange(op.start, end, dtype=np.int32)
            elif isinstance(op, tir.TAddPtr):
                vals[op.id] = (op.base, v(op.offs))
            elif isinstance(op, (tir.TArith, tir.TCmp, tir.TLogic)):
                vals[op.id] = self._binop(op, v(op.lhs), v(op.rhs))
            elif isinstance(op, tir.TLoad):
                name, offs = v(op.ptr)
                flat = buffers[name].reshape(-1)  # idx 是行主序展平偏移（R7 语义）
                if op.mask is not None:
                    mask = np.asarray(v(op.mask), dtype=np.bool_)
                    safe = np.where(mask, offs, 0)
                    gathered = flat[safe]
                    other = v(op.other) if op.other is not None else 0
                    out = np.where(mask, gathered, other)
                else:
                    out = flat[np.asarray(offs)]
                vals[op.id] = np.asarray(out).astype(np_dtype(op.tila_type.dtype))
            elif isinstance(op, tir.TCast):
                x = v(op.operand)
                if isinstance(x, np.ndarray):
                    vals[op.id] = x.astype(np_dtype(op.dtype))
                else:
                    vals[op.id] = np_dtype(op.dtype)(x)
            elif isinstance(op, tir.TExpandDim):
                vals[op.id] = np.expand_dims(v(op.tile), op.axis)
            elif isinstance(op, tir.TDot):
                x = np.asarray(v(op.lhs)).astype(np.float32)
                y = np.asarray(v(op.rhs)).astype(np.float32)
                vals[op.id] = x @ y  # fp16 乘积在 f32 中精确；累加序差异由容差覆盖
            elif isinstance(op, tir.TStore):
                name, offs = v(op.ptr)
                flat = buffers[name].reshape(-1)  # 展平视图：写回就地生效
                value = np.asarray(v(op.value))
                offs = np.asarray(offs)
                if op.mask is not None:
                    mask = np.asarray(v(op.mask), dtype=np.bool_)
                    flat[offs[mask]] = value[mask]
                else:
                    flat[offs] = value
            elif isinstance(op, tir.TReturn):
                return
            else:  # pragma: no cover
                raise UnsupportedInterpreterFeature(f"op {op!r}")

    @staticmethod
    def _binop(op, lhs, rhs):
        import operator

        ops = {
            "+": operator.add, "-": operator.sub, "*": operator.mul,
            "/": operator.truediv,
            "<": operator.lt, "<=": operator.le, ">": operator.gt,
            ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
            "&": operator.and_, "|": operator.or_,
        }
        fn = ops[op.op]
        out = fn(lhs, rhs)
        l_arr = isinstance(lhs, np.ndarray)
        r_arr = isinstance(rhs, np.ndarray)
        if not l_arr and not r_arr:
            return out
        # 保持元素 dtype：numpy 2 的弱标量提升保证 tile ⊕ 标量不改变 dtype
        if isinstance(op, (tir.TCmp, tir.TLogic)):
            return np.asarray(out).astype(np.bool_)
        result_dtype = lhs.dtype if l_arr else rhs.dtype
        return np.asarray(out).astype(result_dtype)


def run_kernel(kernel: tir.TKernel, buffers, scalars=None, constexpr=None) -> None:
    Interpreter(kernel).run(buffers, scalars, constexpr)
