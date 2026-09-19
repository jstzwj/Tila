"""Reference interpreter：TIR → NumPy（CPU 差分 oracle，docs/roadmap.md §8）。

以 SPMD 方式逐 program instance 执行 body；masked lanes 不访问内存
（clip 后以 where 恢复 other 语义），与 GPU 语义一致。
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from . import dtypes as D
from . import tir as T
from . import types as TY
from . import numeric


# Independent typed-TIR capability registry used by ADR-006 completeness gates.
INTERPRETER_TIR_HANDLERS = MappingProxyType({
    "TName": "operand.name",
    "TLit": "operand.literal",
    "TAssign": "stmt.assign",
    "TStore": "stmt.store",
    "TAssume": "stmt.assume",
    "TIf": "stmt.if",
    "TStaticIf": "stmt.static-if",
    "TFor": "stmt.for",
    "TReturn": "stmt.return",
    "TBin": "expr.binary",
    "TUna": "expr.unary",
    "TCast": "expr.cast",
    "TArange": "expr.arange",
    "TPid": "expr.program-id",
    "TNumPrograms": "expr.num-programs",
    "TZeros": "expr.zeros",
    "TReshape": "expr.reshape",
    "TExpand": "expr.expand",
    "TWhere": "expr.where",
    "TDot": "expr.dot",
    "TReduce": "expr.reduce",
    "TLoad": "expr.load",
    "TBufPtr": "expr.buffer-pointer",
    "TPAdd": "expr.pointer-add",
})
INTERPRETER_TIR_OPS = frozenset(INTERPRETER_TIR_HANDLERS)

_BF16 = None


class _ReturnSignal(Exception):
    """裸 return（TReturn）：提前结束当前 program instance 的 body 执行。

    由 Interp.stmt 抛出，穿透 TIf/TFor 的嵌套（它们不捕获），
    在 run() 的 per-instance 调用处被接住——该实例就此退出。"""


def _np_dtype(dt: D.DType):
    if dt is D.bf16:
        global _BF16
        if _BF16 is None:
            try:
                import ml_dtypes
                _BF16 = ml_dtypes.bfloat16
            except ImportError as e:
                raise RuntimeError(
                    "bf16 interpretation needs ml_dtypes (pip install "
                    "ml_dtypes)") from e
        return _BF16
    if dt.np_dtype is None:
        raise RuntimeError(f"dtype {dt.name} is storage-only (no arithmetic)")
    return dt.np_dtype


class Interp:
    def __init__(self, tk: T.TKernel, buffers: dict, scalars: dict,
                 consts: dict, grid: tuple, debug: bool = True):
        self.tk = tk
        self.buffers = {}      # name -> (ndarray, 元素步长)
        # Buffer 保留坐标布局；裸 Ptr 已由 launch 验证连续，在 interpreter
        # 中规范化成一维元素视图，匹配 GPU 的 flat pointer 语义。
        for param in tk.buffers:
            arr = buffers[param.name]
            assert isinstance(arr, np.ndarray), "interp 需要 numpy 数组"
            st = tuple(s // arr.itemsize for s in arr.strides)
            self.buffers[param.name] = (arr, st)
        for param in tk.ptr_params:
            arr = buffers[param.name]
            assert isinstance(arr, np.ndarray), "interp 需要 numpy 数组"
            arr = arr.reshape(-1)
            self.buffers[param.name] = (arr, (1,))
        self.scalars = {p.name: np.asarray(scalars[p.name], dtype=_np_dtype(p.dtype))[()]
                        for p in tk.scalars}
        self.consts = dict(consts)
        self.grid = grid
        self.debug = debug

    # ------------------------------------------------------------------

    def run(self):
        g = self.grid + (1, 1)
        for p2 in range(g[2]):
            for p1 in range(g[1]):
                for p0 in range(g[0]):
                    env = {p.name: ("ptr", p.name, 0)
                           for p in self.tk.ptr_params}
                    try:
                        self.stmts(self.tk.body, env, (p0, p1, p2))
                    except _ReturnSignal:
                        pass                    # 裸 return：本实例提前退出

    def stmts(self, stmts, env, pids):
        for s in stmts:
            self.stmt(s, env, pids)

    def stmt(self, s: T.TStmt, env, pids):
        if isinstance(s, T.TAssign):
            env[s.name] = self.e(s.value, env, pids)
        elif isinstance(s, T.TStore):
            value = self.o(s.value, env, pids)
            mask = self.o(s.mask, env, pids) if s.mask is not None else None
            if s.buffer is not None:
                arr, st = self.buffers[s.buffer]
                coords = [self.o(c, env, pids) for c in s.coords]
                self._store(arr, st, coords, value, mask)
            else:
                name, off = self._ptr_value(s.ptr, env, pids)
                arr, st = self.buffers[name]
                if len(st) != 1 or st[0] != 1:
                    raise RuntimeError("interp 的 Ptr 形式仅支持连续 1D buffer"
                                       "（Buffer 形式无此限制）")
                shape = np.broadcast_shapes(np.shape(off), np.shape(value))
                coord = (np.broadcast_to(np.asarray(off), shape)
                         if shape else int(off))
                self._store(arr, st, [coord],
                            np.broadcast_to(np.asarray(value), shape), mask)
        elif isinstance(s, T.TAssume):
            if self.debug:
                v = self.o(s.pred, env, pids)
                if isinstance(v, np.ndarray):
                    assert bool(v.all()), "device_assert failed"
                else:
                    assert bool(v), "device_assert failed"
        elif isinstance(s, (T.TIf, T.TStaticIf)):
            # TStaticIf：条件由 Const 参数构成——o() 经 env/self.consts 求得
            # 具体 python bool 后走普通分支（与 Triton constexpr 同语义）。
            c = self.o(s.cond, env, pids)
            if isinstance(c, np.ndarray):
                raise RuntimeError("mask-condition if reached interpreter")
            if c:
                self.stmts(s.then_body, env, pids)
            else:
                self.stmts(s.else_body, env, pids)
        elif isinstance(s, T.TFor):
            start = int(self.o(s.start, env, pids)) if s.start is not None else 0
            end = int(self.o(s.end, env, pids))
            step = int(self.o(s.step, env, pids))
            for i in range(start, end, step):
                env[s.var] = np.int32(i)
                self.stmts(s.body, env, pids)
        elif isinstance(s, T.TReturn):
            raise _ReturnSignal()

    # ------------------------------------------------------------------

    def _store(self, arr, strides, coords, value, mask):
        # 只写有效 lane：布尔筛选后散射写（clip 产生的重复索引不参与写，
        # 与 GPU 不访问 masked lane 的语义一致）。
        shape = np.broadcast_shapes(*[np.shape(c) for c in coords],
                                    np.shape(value))
        bc = []
        for c, n in zip(coords, arr.shape):
            c = np.asarray(c)
            if c.ndim == 0:
                bc.append(int(c))
                continue
            cc = np.broadcast_to(c, shape).astype(np.int64).copy()
            np.clip(cc, 0, max(n - 1, 0), out=cc)
            bc.append(cc)
        val = np.broadcast_to(np.asarray(value), shape)
        if mask is None:
            arr[tuple(bc)] = val
            return
        sel = np.broadcast_to(np.asarray(mask, dtype=bool), shape)
        idx = tuple(c[sel] if isinstance(c, np.ndarray) else c for c in bc)
        arr[idx] = val[sel]

    def _load(self, arr, strides, coords, mask, other):
        idx = self._index(arr, coords)
        gathered = arr[idx]
        if mask is not None:
            m = np.broadcast_to(np.asarray(mask, dtype=bool),
                                np.shape(gathered))
            # checker 已保证 other 与 buffer 元素 dtype 精确一致；解释器也要
            # 在值层落实该语义。特别是 ml_dtypes.bfloat16 与 Python int 0
            # 不存在 NumPy 公共 promotion，直接 np.where 会报错。
            o = np.asarray(0 if other is None else other, dtype=arr.dtype)
            return np.where(m, gathered, o)
        return gathered

    def _index(self, arr, coords):
        """坐标 → 可安全索引的多元组（masked lanes 的越界坐标裁剪到界内，
        结果由 where(mask, ...) 恢复语义——与 GPU 不访问 masked lane 一致）。"""
        out = []
        for c, n in zip(coords, arr.shape):
            c = np.asarray(c)
            if c.ndim == 0:
                out.append(int(min(max(int(c), 0), n - 1)) if n else 0)
                continue
            cc = c.astype(np.int64)
            np.clip(cc, 0, max(n - 1, 0), out=cc)
            out.append(cc)
        return tuple(out)

    # ------------------------------------------------------------------

    def o(self, x, env, pids):
        if isinstance(x, T.TName):
            if x.name in env:
                return env[x.name]
            if x.name in self.consts:
                return self.consts[x.name]
            if x.name in self.scalars:
                return self.scalars[x.name]
            raise KeyError(x.name)
        if isinstance(x, T.TLit):
            if x.dtype and x.dtype.is_float:
                return np.asarray(x.value, dtype=_np_dtype(x.dtype))[()]
            return x.value
        return self.e(x, env, pids)

    def e(self, x: T.TExpr, env, pids):
        # 赋值值可为纯操作数形态（名字拷贝 x = y、字面量）——TName/TLit
        # 由 o() 分派，其余委托下方表达式分支。
        if isinstance(x, (T.TName, T.TLit)):
            return self.o(x, env, pids)
        if isinstance(x, T.TBin):
            l = self.o(x.left, env, pids)
            if x.op == "and" and not bool(l):
                return False
            if x.op == "or" and bool(l):
                return True
            r = self.o(x.right, env, pids)
            dt = numeric.dtype(x.vt)
            if x.staged:
                l, r = np.asarray(l, dtype=object), np.asarray(r, dtype=object)
                value = self._bin(x.op, l, r)
                return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value
            if dt and dt.is_int:
                return numeric.integer_binary(x.op, l, r, dt)
            if x.operand_dtype and x.operand_dtype.is_int:
                l, r = numeric.wrap(l, x.operand_dtype), numeric.wrap(r, x.operand_dtype)
            return self._bin(x.op, l, r)
        if isinstance(x, T.TUna):
            v = self.o(x.operand, env, pids)
            dt = numeric.dtype(x.vt)
            if dt and dt.is_int and not x.staged:
                return numeric.wrap(-np.asarray(v).astype(object) if x.op == "-"
                                    else ~np.asarray(v).astype(object), dt)
            if x.op == "-":
                return -v
            if x.op == "~":
                return ~v
            if x.op == "not":
                return np.logical_not(v)
            if x.op == "any":      # Mask lane 归约 → 标量 bool
                return bool(np.any(v))
            if x.op == "all":
                return bool(np.all(v))
            if x.op == "exp":
                return np.exp(v.astype(np.float32) if not
                              isinstance(v, np.ndarray) or
                              v.dtype != np.float64 else v)
            if x.op == "exp2":
                return np.exp2(v.astype(np.float32) if not
                               isinstance(v, np.ndarray) or
                               v.dtype != np.float64 else v)
            raise RuntimeError(x.op)
        if isinstance(x, T.TCast):
            v = self.o(x.operand, env, pids)
            if x.dtype.is_int:
                a = np.asarray(v)
                if a.dtype.kind == "f":
                    lo, hi = numeric.limits(x.dtype)
                    values = np.asarray(a, dtype=object)
                    import math
                    if any(not math.isfinite(z) or not lo <= math.trunc(z) <= hi
                           for z in values.flat):
                        from .errors import TilaError
                        raise TilaError("TILA-NUM-001", "invalid float-to-integer cast")
                    v = np.vectorize(math.trunc, otypes=[object])(values)
                return numeric.wrap(v, x.dtype)
            return np.asarray(v).astype(_np_dtype(x.dtype)) if isinstance(v,
                                                                         np.ndarray) else \
                np.asarray(v, dtype=_np_dtype(x.dtype))
        if isinstance(x, T.TArange):
            end = int(self.o(x.end, env, pids))
            return np.arange(x.start, end, dtype=np.int32)
        if isinstance(x, T.TPid):
            return np.int32(pids[x.axis])
        if isinstance(x, T.TNumPrograms):
            return np.int32(self.grid[x.axis])
        if isinstance(x, T.TZeros):
            shape = tuple(int(self.o(d, env, pids)) for d in x.shape)
            dt = x.vt.elem.dtype if isinstance(x.vt, TY.BlockT) else x.vt.dtype
            return np.zeros(shape, dtype=_np_dtype(dt))
        if isinstance(x, T.TReshape):
            v = self.o(x.operand, env, pids)
            shape = tuple(int(self.o(d, env, pids)) for d in x.shape)
            return np.reshape(v, shape)
        if isinstance(x, T.TExpand):
            v = self.o(x.operand, env, pids)
            return v[:, None] if x.axis == 1 else v[None, :]
        if isinstance(x, T.TWhere):
            return np.where(self.o(x.cond, env, pids),
                            self.o(x.a, env, pids), self.o(x.b, env, pids))
        if isinstance(x, T.TDot):
            a = self.o(x.a, env, pids)
            b = self.o(x.b, env, pids)
            dt = x.vt.elem.dtype if isinstance(x.vt, TY.BlockT) else x.vt.dtype
            acc = self.o(x.acc, env, pids) if x.acc is not None else None
            r = np.matmul(a.astype(np.float32), b.astype(np.float32),
                          dtype=np.float32)
            if acc is not None:
                r = r + acc.astype(np.float32)
            return r.astype(_np_dtype(dt))
        if isinstance(x, T.TReduce):
            v = self.o(x.operand, env, pids)
            dt = x.vt.elem.dtype if isinstance(x.vt, TY.BlockT) else x.vt.dtype
            if x.op == "sum":
                if dt.is_int:
                    return numeric.wrap(np.sum(v.astype(object), axis=x.axis), dt)
                r = np.sum(v, axis=x.axis, dtype=np.float32)
                return np.asarray(r).astype(_np_dtype(dt))
            r = np.max(v, axis=x.axis)
            return np.asarray(r).astype(_np_dtype(dt))
        if isinstance(x, T.TLoad):
            if x.buffer is not None:
                arr, st = self.buffers[x.buffer]
                coords = [self.o(c, env, pids) for c in x.coords]
                mask = self.o(x.mask, env, pids) if x.mask is not None else None
                other = self.o(x.other, env, pids) if x.other is not None else None
                return self._load(arr, st, coords, mask, other)
            # Ptr 形式：指针值 = (buffer 名, flat offset)——仅连续 1D 支持
            name, off = self._ptr_value(x.ptr, env, pids)
            arr, st = self.buffers[name]
            if len(st) != 1 or st[0] != 1:
                raise RuntimeError("interp 的 Ptr 形式仅支持连续 1D buffer"
                                   "（Buffer 形式无此限制）")
            shape = np.shape(off)
            mask = self.o(x.mask, env, pids) if x.mask is not None else None
            other = self.o(x.other, env, pids) if x.other is not None else None
            coord = (np.broadcast_to(np.asarray(off), shape)
                     if shape else int(off))
            return self._load(arr, st, [coord], mask, other)
        if isinstance(x, T.TBufPtr):
            return ("ptr", x.buffer, 0)
        if isinstance(x, T.TPAdd):
            name, off = self._ptr_value(x.ptr, env, pids)
            add = self.o(x.offset, env, pids)
            return ("ptr", name, np.asarray(off) + np.asarray(add)
                    if isinstance(off, np.ndarray) or isinstance(add, np.ndarray)
                    else off + add)
        raise RuntimeError(f"?{x!r}")

    def _ptr_value(self, node, env, pids):
        v = self.o(node, env, pids)
        if isinstance(v, tuple) and len(v) == 3 and v[0] == "ptr":
            return v[1], v[2]
        raise RuntimeError("expected a pointer value in interp")

    def _bin(self, op, l, r):
        import operator as op_
        fn = {"+": op_.add, "-": op_.sub, "*": op_.mul, "%": op_.mod,
              "&": op_.and_, "|": op_.or_, "^": op_.xor,
              "<<": op_.lshift, ">>": op_.rshift,
              "<": op_.lt, "<=": op_.le, ">": op_.gt, ">=": op_.ge,
              "==": op_.eq, "!=": op_.ne,
              "and": np.logical_and, "or": np.logical_or}
        if op == "/":
            return l / r
        if op == "//":
            return l // r
        return fn[op](l, r)


def run_kernel(tk: T.TKernel, buffers: dict, scalars: dict, consts: dict,
               grid: tuple, debug: bool = True):
    Interp(tk, buffers, scalars, consts, grid, debug).run()
