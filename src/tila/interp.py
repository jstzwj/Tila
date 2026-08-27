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
from .types.memory import RowMajor

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


def _element_strides(arr) -> list:
    """numpy 数组的逐轴元素步长（.strides 是字节，转元素）。"""
    return [s // arr.itemsize for s in arr.strides]


def _is_row_major(arr) -> bool:
    """实际布局是否行主序连续（与 launcher 的 RowMajor 断言同款契约）。

    size-0 轴的数组无元素可错读——布局契约空真（numpy 对空维的 strides 全零、
    torch 亦无意义；v0.4-kloop §12，零迭代恒等式测试的前提）。"""
    if any(d == 0 for d in arr.shape):
        return True
    expect = 1
    for d, s in zip(reversed(arr.shape), _element_strides(arr)[::-1]):
        if s != expect:
            return False
        expect *= d
    return True


class Interpreter:
    def __init__(self, kernel: tir.TKernel):
        self.kernel = kernel
        self._mems = {p.name: p.tila_type.mem for p in kernel.params
                      if p.kind == "buffer"}
        self._grid_dims = []

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

        # stride 符号取值 + 内存布局契约（v0.3-strides §5）：
        # RowMajor 断言实际连续；Strided 按声明绑定（共享/dim 复用断言一致）
        plan = self.kernel.launch_plan
        for name, mem in self._mems.items():
            if isinstance(mem, RowMajor) and not _is_row_major(buffers[name]):
                raise AssertionError(
                    f"interpreter: buffer '{name}' is declared row-major but the "
                    f"array is not contiguous (strides {_element_strides(buffers[name])}); "
                    f"declare strides=(...) instead")
        for b, i, sv in plan.stride_bindings:
            actual = _element_strides(buffers[b])[i]
            if isinstance(sv, int):
                assert actual == sv, \
                    f"interpreter: '{b}'.stride({i}) == {actual}, declared {sv}"
            elif sv in scalars:
                assert scalars[sv] == actual, \
                    f"interpreter: '{b}'.stride({i}) == {actual}, symbol '{sv}' " \
                    f"is bound to {scalars[sv]}"
            else:
                scalars[sv] = actual

        # 坐标 → 平面偏移：不做。既然 stride 契约已断言（声明值 == 实际元素步长），
        # "按声明步长访存"与逻辑多维索引严格等价——而逻辑索引对任意布局的
        # numpy 视图都正确（reshape(-1) 对非连续数组会按逻辑序拷贝，平面模型
        # 反而失真）。load/store 直接消费坐标元组（v0.3-strides §5）。
        self._buffers = buffers

        grid_dims = []
        for axis in plan.axes:
            dim = scalars[axis.dim] if isinstance(axis.dim, str) else int(axis.dim)
            tiling = constexpr_values[axis.tiling] if isinstance(axis.tiling, str) \
                else int(axis.tiling)
            grid_dims.append(max(1, -(-dim // tiling)))
        if not grid_dims:
            grid_dims = [1]
        self._grid_dims = grid_dims

        # 浮点异常不可观察（v0.5-reduce §1.3/§5）：where 只规定选中值，未选侧
        # 的 0/0 → NaN、f16 exp 溢出等 dead-lane 现象不产生可依赖的副作用——
        # 警告与 GPU 语义对齐，屏蔽之。
        with np.errstate(all="ignore"):
            for pids in itertools.product(*(range(g) for g in grid_dims)):
                self._run_program(pids, buffers, scalars, constexpr_values)

    def _gather(self, name: str, coords, mask, other):
        """按逻辑坐标取数；mask 假通道取 other（缺省 0）。"""
        arr = self._buffers[name]
        arrays = np.broadcast_arrays(*[np.asarray(c) for c in coords])
        if mask is not None:
            safe = tuple(np.where(mask, c, 0) for c in arrays)
            gathered = arr[safe]
            return np.where(mask, gathered, other if other is not None else 0)
        return arr[tuple(arrays)]

    def _scatter(self, name: str, coords, mask, value):
        """按逻辑坐标写回（就地）。"""
        arr = self._buffers[name]
        arrays = np.broadcast_arrays(*[np.asarray(c) for c in coords])
        if mask is not None:
            arr[tuple(c[mask] for c in arrays)] = value[mask]
        else:
            arr[tuple(arrays)] = value

    # ------------------------------------------------------------------

    def _run_program(self, pids, buffers, scalars, constexpr_values) -> None:
        vals: Dict[str, object] = {
            p.name: scalars[p.name]
            for p in self.kernel.params if p.kind == "scalar"
        }
        for op in self.kernel.ops:
            self._exec(op, vals, pids, scalars, constexpr_values)

    def _exec(self, op, vals: Dict[str, object], pids, scalars,
              constexpr_values) -> None:
        """单条 TIR 指令求值；TFor 递归进入 body（v0.4-kloop §5）。

        φ 链：迭代开始时 φ 取上一迭代的 back（首轮取 pre 种子）；零迭代时
        循环后回填种子——与 Triton loop-carried 变量语义一致。
        """
        def v(oid: str):
            return vals[oid]

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
        elif isinstance(op, tir.TZeros):
            shape = tuple(_eval_constexpr(s, constexpr_values) for s in op.shape)
            vals[op.id] = np.zeros(shape, dtype=np_dtype(op.dtype))
        elif isinstance(op, tir.TFull):
            shape = tuple(_eval_constexpr(s, constexpr_values) for s in op.shape)
            vals[op.id] = np.full(shape, op.value, dtype=np_dtype(op.dtype))
        elif isinstance(op, tir.TMaximum):
            vals[op.id] = np.maximum(v(op.lhs), v(op.rhs))
        elif isinstance(op, tir.TLaunchAssert):
            # host 启动断言：interpreter 与 launcher 同判据（名字 = 符号维/
            # constexpr，白名单求值——checker 已核验表达式合法性）
            env = dict(scalars)
            env.update(constexpr_values)
            try:
                ok = eval(op.cond, {"__builtins__": None}, env)
            except Exception as exc:  # 名字缺失等求值期故障 → 明确运行期错误
                raise RuntimeError(f"launch_assert could not be evaluated: "
                                   f"{op.cond} ({exc})") from None
            if not ok:
                raise RuntimeError(f"launch_assert failed: {op.cond}")
        elif isinstance(op, tir.TAddPtr):
            vals[op.id] = (op.base, [v(c) for c in op.coords])
        elif isinstance(op, (tir.TArith, tir.TCmp, tir.TLogic)):
            vals[op.id] = self._binop(op, v(op.lhs), v(op.rhs))
        elif isinstance(op, tir.TLoad):
            name, coords = v(op.ptr)
            out = self._gather(name, coords,
                               v(op.mask) if op.mask is not None else None,
                               v(op.other) if op.other is not None else None)
            vals[op.id] = np.asarray(out).astype(np_dtype(op.tila_type.dtype))
        elif isinstance(op, tir.TCast):
            x = v(op.operand)
            if isinstance(x, np.ndarray):
                vals[op.id] = x.astype(np_dtype(op.dtype))
            else:
                vals[op.id] = np_dtype(op.dtype)(x)
        elif isinstance(op, tir.TExpandDim):
            vals[op.id] = np.expand_dims(v(op.tile), op.axis)
        elif isinstance(op, tir.TReduce):
            # dtype 钉死：numpy 对 int<32 默认提升平台整数宽度（与 Triton 的
            # 默认策略同款陷阱）；结果 dtype = 输入 dtype（v0.5-reduce §5）
            x = np.asarray(v(op.tile))
            d = np_dtype(op.tila_type.dtype)
            if op.op == "sum":
                vals[op.id] = np.sum(x, axis=op.axis, dtype=d).astype(d)
            else:
                vals[op.id] = np.max(x, axis=op.axis).astype(d)
        elif isinstance(op, tir.TElem):
            x = np.asarray(v(op.operand))
            fn = {"exp": np.exp, "exp2": np.exp2, "sqrt": np.sqrt,
                  "abs": np.abs, "log2": np.log2}[op.op]
            vals[op.id] = fn(x).astype(np_dtype(op.tila_type.dtype))
        elif isinstance(op, tir.TWhere):
            out = np.where(np.asarray(v(op.cond)), v(op.a), v(op.b))
            vals[op.id] = np.asarray(out).astype(np_dtype(op.tila_type.dtype))
        elif isinstance(op, tir.TNumPrograms):
            # observational：读当前 grid 维；未建立的轴 = 1（Triton 补 1 语义）
            vals[op.id] = self._grid_dims[op.axis] \
                if op.axis < len(self._grid_dims) else 1
        elif isinstance(op, tir.TDot):
            x = np.asarray(v(op.lhs)).astype(np.float32)
            y = np.asarray(v(op.rhs)).astype(np.float32)
            vals[op.id] = x @ y  # fp16 乘积在 f32 中精确；累加序差异由容差覆盖
        elif isinstance(op, tir.TFor):
            end = vals[op.end]
            step = _eval_constexpr(op.step, constexpr_values)
            phis = [o for o in op.body if isinstance(o, tir.TPhi)]
            k = 0
            while k < end:
                vals[op.id] = k
                for ph in phis:
                    vals[ph.id] = vals[ph.back] if ph.back in vals else vals[ph.pre]
                for inner in op.body:
                    if isinstance(inner, tir.TPhi):
                        continue
                    self._exec(inner, vals, pids, scalars, constexpr_values)
                k += step
            for ph in phis:  # 零迭代回退：出口值 = 种子（φ 语义）
                if ph.back not in vals:
                    vals[ph.back] = vals[ph.pre]
        elif isinstance(op, tir.TStore):
            name, coords = v(op.ptr)
            value = np.asarray(v(op.value))
            self._scatter(name, coords,
                          np.asarray(v(op.mask)) if op.mask is not None else None,
                          value)
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
