"""TIR → Triton 源码 lowering（total、确定性——对通过检查的 TIR 全函数）。

- Const 参数保持符号名（tl.constexpr），同一份源码跨 Const 取值复用；
- 事实自动发射（refinements.md §4）：
  - contiguous(offs, span) ⇒ tl.max_contiguous(offs, span)；
  - offs = pid*STEP + arange(0, E) ⇒ 拆分 base = pid*STEP 并发射
    tl.multiple_of(base, STEP)——base 恒被 STEP 整除（结构性成立，
    与 STEP 取值无关；STEP 为 Const 参数名或整数字面量）。
- 归约 dtype 适配：tl.sum 恒传 dtype=，tl.max 窄 dtype 显式 cast 恢复
  （backend 实现 Tila 语义，不继承 Triton 默认提升）。
"""

from __future__ import annotations

from types import MappingProxyType

from . import dtypes as D
from . import tir as T
from . import types as TY
from . import numeric


_INTEGER_DIVISION = '''@triton.jit
def _tila_idiv(a, b, SIGNED: tl.constexpr, MIN: tl.constexpr, MOD: tl.constexpr):
    if SIGNED:
        overflow = (a == MIN) & (b == -1)
        safe_b = tl.where(overflow, 1, b).to(b.dtype)
        q = a // safe_b
        r = a % safe_b
        adjust = (r != 0) & ((a < 0) != (b < 0))
        if MOD:
            return tl.where(overflow, 0, tl.where(adjust, r + b, r)).to(a.dtype)
        return tl.where(overflow, MIN, q - adjust.to(q.dtype)).to(a.dtype)
    if MOD:
        return (a % b).to(a.dtype)
    return (a // b).to(a.dtype)
'''


# Typed TIR is the backend boundary (ADR-006). Values are stable symbolic
# handler IDs; the implementation remains grouped in stmt()/e() for now.
TRITON_TIR_HANDLERS = MappingProxyType({
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
TRITON_TIR_OPS = frozenset(TRITON_TIR_HANDLERS)


class Lowering:
    def __init__(self, tk: T.TKernel, debug_asserts: bool = False):
        self.tk = tk
        self.debug = debug_asserts
        self.hint_after = {var: span for var, span in tk.hints}
        # 裸指针参数名 → Triton 实参名（checker 正常物化为 TBufPtr；
        # 此映射兜底防御直引参数名的 TName）。
        self.ptr_args = {p.name: f"{p.name}_ptr" for p in tk.ptr_params}
        # multiple_of 拆分模式识别状态（refinements.md §4）：
        # - _pid_names：值恰为 tl.program_id 的赋值名（TIR 中 pid 以
        #   TName 引用——checker 把单值绑定到名字后才使用）；
        # - _const_names：Const 参数名（tl.multiple_of 的除数须为
        #   tl.constexpr 或字面量，运行期标量不可用）。
        self._pid_names = set()
        self._const_names = {c.name for c in tk.consts}
        self.hint_audit = []

    # ------------------------------------------------------------------

    def kernel_source(self) -> str:
        k = self.tk
        params = [f"{b.name}_ptr" for b in k.buffers]
        params += [f"{p.name}_ptr" for p in k.ptr_params]   # 裸指针参数
        params += [f"{s.name}: {s.dtype.tl_name}" for s in k.scalars]
        params += [f"{c.name}: tl.constexpr" for c in k.consts]

        out = []
        out.append("import triton")
        out.append("import triton.language as tl")
        out.append("")
        out.append("")
        # The helper is emitted only when an integer division actually needs it.
        self.needs_integer_division = False
        body = self.stmts(k.body, 1)
        if self.needs_integer_division:
            out.extend(_INTEGER_DIVISION.rstrip().splitlines())
            out.append("")
            out.append("")
        out.append("@triton.jit")
        out.append(f"def {k.name}(")
        for i, p in enumerate(params):
            comma = "," if i < len(params) - 1 else ""
            out.append(f"    {p}{comma}")
        out.append("):")
        casts = [f"    {s.name} = tl.cast({s.name}, {s.dtype.tl_name})"
                 for s in k.scalars]
        body = casts + body
        if not body:
            body = ["    pass"]
        out.extend(body)
        return "\n".join(out) + "\n"

    def launch_args(self):
        """launch 调用的实参名模板（runtime 填值）。

        参数序：buffer {name}_ptr → ptr {name}_ptr → 标量 → Const——
        与 kernel_source 签名及 runtime._execute 的实参构造严格一致。
        """
        return ([f"{b.name}_ptr" for b in self.tk.buffers] +
                [f"{p.name}_ptr" for p in self.tk.ptr_params] +
                [s.name for s in self.tk.scalars] +
                [c.name for c in self.tk.consts])

    # -- 语句 --------------------------------------------------------------

    def stmts(self, stmts, depth) -> list[str]:
        out = []
        pad = "    " * depth
        for s in stmts:
            out.extend(self.stmt(s, depth, pad))
        return out

    def stmt(self, s: T.TStmt, depth, pad) -> list[str]:
        if isinstance(s, T.TAssign):
            # pid 名字追踪：name = program_id(...) ⇒ 此后 name 可安全地
            # 视作 pid 参与 multiple_of 模式；改绑非 pid 值则失效。
            if isinstance(s.value, T.TPid):
                self._pid_names.add(s.name)
            else:
                self._pid_names.discard(s.name)
            split = self._multiple_of_split(s.name, s.value)
            if split is not None:
                base, mul, rng, step = split
                from .predicates import Origin, STATIC
                self.hint_audit.append((f"multiple_of({base}, {self.o(step)})",
                    (Origin(STATIC, s.line, "structural pid * step; integer launch guards required"),)))
                add = T.TBin(s.value.vt, "+", T.TName(base), rng,
                             checked_index=s.value.checked_index)
                out = [
                    f"{pad}{base} = {self.e(mul)}",
                    f"{pad}{s.name} = {self.e(add)}",
                    f"{pad}tl.multiple_of({base}, {self.o(step)})",
                ]
            else:
                out = [f"{pad}{s.name} = {self.e(s.value)}"]
            if s.name in self.hint_after:
                span = self.hint_after[s.name]
                self.hint_audit.append((f"max_contiguous({s.name}, {self.dim(span)})",
                                       tuple(sorted(self.tk.hint_origins.get(s.name, ())))))
                out.append(f"{pad}tl.max_contiguous({s.name}, {self.dim(span)})")
            return out
        if isinstance(s, T.TStore):
            dst = self.store_dst(s)
            m = f", mask={self.o(s.mask)}" if s.mask is not None else ""
            return [f"{pad}tl.store({dst}, {self.o(s.value)}{m})"]
        if isinstance(s, T.TAssume):
            if not self.debug:
                return []
            return [f"{pad}tl.device_assert({self.o(s.pred)})"]
        if isinstance(s, (T.TIf, T.TStaticIf)):
            # TStaticIf 与 TIf 同形 lowering：条件引用 tl.constexpr 实参，
            # Triton 在 trace 期解析该 python if（每特化取一分支）。
            out = [f"{pad}if {self.o(s.cond)}:"]
            out.extend(self.stmts(s.then_body, depth + 1) or [pad + "    pass"])
            if s.else_body:
                out.append(f"{pad}else:")
                out.extend(self.stmts(s.else_body, depth + 1))
            return out
        if isinstance(s, T.TFor):
            start = self.o(s.start) if s.start is not None else "0"
            induction = f"_tila_loop_{s.var}"
            out = [f"{pad}for {induction} in range(tl.cast({start}, tl.int64), "
                   f"tl.cast({self.o(s.end)}, tl.int64), tl.cast({self.o(s.step)}, tl.int64)):",
                   f"{pad}    {s.var} = tl.cast({induction}, tl.int32)"]
            body = self.stmts(s.body, depth + 1)
            out.extend(body or [pad + "    pass"])
            return out
        if isinstance(s, T.TReturn):
            # 裸 return：Triton jit 支持提前 return（program instance 级退出）。
            return [f"{pad}return"]
        return [f"{pad}# ?{s!r}"]

    def _multiple_of_split(self, name: str, value):
        """offs = pid*STEP + arange(0, E) ⇒ (base 名, mul, arange, STEP)。

        refinements.md §4 整除事实：base = pid*STEP 的每个取值都是 STEP
        的倍数（结构性成立，与 STEP 的具体取值无关），故可拆分赋值并对
        base 发射 tl.multiple_of(base, STEP)。STEP 取 Mul 中的因子——
        整除是按该因子证明的（非 arange 跨度 E）。

        模式：TBin('+', mul, arange) 两侧可交换；mul = TBin('*', pid, STEP)
        内部亦可交换。pid 侧为 TPid 节点，或先前由 program_id 赋值的名字
        （_pid_names）；STEP 侧为 Const 参数名（tl.constexpr 在 scope 内）
        或整数字面量——运行期标量不可作 multiple_of 除数。

        base 名 "{name}_base" 确定性派生（golden 逐字节稳定）。v0 命名
        约定：用户变量不应占用 *_base 名（冲突时行为未定义）。
        """
        if not isinstance(value, T.TBin) or value.op != "+":
            return None
        for add_l, add_r in ((value.left, value.right),
                             (value.right, value.left)):
            if not isinstance(add_r, T.TArange):
                continue
            if not isinstance(add_l, T.TBin) or add_l.op != "*":
                continue
            for x, y in ((add_l.left, add_l.right),
                         (add_l.right, add_l.left)):
                if self._is_pid_side(x) and self._is_step_side(y):
                    return f"{name}_base", add_l, add_r, y
        return None

    def _is_pid_side(self, x) -> bool:
        if isinstance(x, T.TPid):
            return True
        return isinstance(x, T.TName) and x.name in self._pid_names

    def _is_step_side(self, x) -> bool:
        if isinstance(x, T.TLit):
            return isinstance(x.value, int) and not isinstance(x.value, bool)
        return isinstance(x, T.TName) and x.name in self._const_names

    def store_dst(self, s: T.TStore):
        if s.buffer is not None:
            return self.addr_expr(s.buffer, s.coords)
        return self.o(s.ptr)

    def addr_expr(self, buf: str, coords) -> str:
        terms = []
        for i, c in enumerate(coords):
            terms.append(f"tl.cast({buf}_stride{i}, tl.int64) * tl.cast({self.o(c)}, tl.int64)")
        return f"{buf}_ptr + " + " + ".join(terms)

    # -- 表达式 ------------------------------------------------------------

    def dim(self, d) -> str:
        if isinstance(d, T.TOperand):
            return self.o(d)
        from .dims import Cst, Sym
        if isinstance(d, Cst):
            return str(d.value)
        if isinstance(d, Sym):
            return d.name
        return str(d)

    def o(self, x: T.TOperand) -> str:
        if isinstance(x, T.TName):
            if x.name in self.ptr_args:
                return self.ptr_args[x.name]
            return x.name
        if isinstance(x, T.TLit):
            v = x.value
            if isinstance(v, float):
                return repr(v)
            if isinstance(v, bool):
                return "True" if v else "False"
            return str(v)
        return self.e(x)

    def e(self, x: T.TExpr) -> str:
        # 赋值值可为纯操作数形态（名字拷贝 x = y）——TName/TLit 走 o()。
        if isinstance(x, (T.TName, T.TLit)):
            return self.o(x)
        if isinstance(x, T.TBin):
            left, right = self.o(x.left), self.o(x.right)
            dt = x.operand_dtype or numeric.dtype(x.vt)
            if x.checked_index and x.op in ("+", "-", "*"):
                return f"({left} {x.op} {right})"
            if dt and dt.is_int and not x.staged:
                left = f"tl.cast({left}, {dt.tl_name})"
                right = f"tl.cast({right}, {dt.tl_name})"
                if x.op in ("//", "%"):
                    self.needs_integer_division = True
                    return (f"_tila_idiv({left}, {right}, {dt.kind == 'int'}, "
                            f"{numeric.limits(dt)[0]}, {x.op == '%'})")
                value = f"({left} {x.op} {right})"
                if numeric.dtype(x.vt) is dt:
                    return f"tl.cast({value}, {dt.tl_name})"
                return value
            return f"({left} {x.op} {right})"
        if isinstance(x, T.TUna):
            dt = numeric.dtype(x.vt)
            if x.op == "~" and dt is D.bool_:
                return f"(~tl.cast({self.o(x.operand)}, tl.int1))"
            if dt and dt.is_int and not x.staged:
                return f"tl.cast(({x.op}tl.cast({self.o(x.operand)}, {dt.tl_name})), {dt.tl_name})"
            if x.op == "exp":
                return f"tl.exp({self.o(x.operand)})"
            if x.op == "exp2":
                return f"tl.math.exp2({self.o(x.operand)})"
            # Mask lane 归约 → 标量 bool（int1 块经 cast 求和；tl.sum 无
            # axis 时归约全部元素为标量）——type-system.md §3.3
            if x.op == "any":
                return f"(tl.sum(tl.cast({self.o(x.operand)}, tl.int32)) > 0)"
            if x.op == "all":
                return f"(tl.sum(tl.cast(~{self.o(x.operand)}, tl.int32)) == 0)"
            return f"({x.op}{self.o(x.operand)})"
        if isinstance(x, T.TCast):
            if x.dtype.is_int and (isinstance(x.operand, T.TLit) and type(x.operand.value) is int
                                  or isinstance(x.operand, T.TName) and x.operand.name in self._const_names
                                  or isinstance(x.operand, (T.TBin, T.TUna)) and x.operand.staged):
                # Fold arbitrary-precision staged integers before tensor conversion.
                return f"tl.cast(({self.o(x.operand)} % {1 << x.dtype.bits}), {x.dtype.tl_name})"
            return f"tl.cast({self.o(x.operand)}, {x.dtype.tl_name})"
        if isinstance(x, T.TArange):
            return f"tl.arange({x.start}, {self.o(x.end)})"
        if isinstance(x, T.TPid):
            return f"tl.program_id({x.axis})"
        if isinstance(x, T.TNumPrograms):
            return f"tl.num_programs({x.axis})"
        if isinstance(x, T.TZeros):
            shape = T.shape_tuple_text(x.shape, self.o)
            dt = x.vt.elem.dtype if isinstance(x.vt, TY.BlockT) else x.vt.dtype
            return f"tl.zeros(({shape}), {dt.tl_name})"
        if isinstance(x, T.TReshape):
            # 形状项经 o()：TLit → 整数，TName → Const 参数名（tl.constexpr
            # 在 scope 内，Triton reshape 接受）；单项补尾逗号保持 tuple。
            shape = T.shape_tuple_text(x.shape, self.o)
            return f"tl.reshape({self.o(x.operand)}, ({shape}))"
        if isinstance(x, T.TExpand):
            return f"{self.o(x.operand)}[:, None]" if x.axis == 1 else \
                f"{self.o(x.operand)}[None, :]"
        if isinstance(x, T.TWhere):
            return f"tl.where({self.o(x.cond)}, {self.o(x.a)}, {self.o(x.b)})"
        if isinstance(x, T.TDot):
            if x.acc is not None:
                return f"tl.dot({self.o(x.a)}, {self.o(x.b)}, {self.o(x.acc)})"
            return f"tl.dot({self.o(x.a)}, {self.o(x.b)})"
        if isinstance(x, T.TReduce):
            return self._reduce(x)
        if isinstance(x, T.TLoad):
            ptr = self.addr_expr(x.buffer, x.coords) if x.buffer is not None \
                else self.o(x.ptr)
            parts = [ptr]
            if x.mask is not None:
                parts.append(f"mask={self.o(x.mask)}")
                dt = self._elem_dtype(x.vt)
                other = self.o(x.other) if x.other is not None else \
                    self._zero_lit(dt)
                parts.append(f"other={other}")
            elif x.other is not None:
                parts.append(f"other={self.o(x.other)}")
            return f"tl.load({', '.join(parts)})"
        if isinstance(x, T.TBufPtr):
            return f"{x.buffer}_ptr"
        if isinstance(x, T.TPAdd):
            return f"({self.o(x.ptr)} + {self.o(x.offset)})"
        return f"# ?{x!r}"

    def _reduce(self, x: T.TReduce) -> str:
        dt = self._elem_dtype(x.vt)
        inner = self.o(x.operand)
        if x.op == "sum":
            if dt in (D.f16, D.bf16):
                return (f"tl.cast(tl.sum({inner}, axis={x.axis}, "
                        f"dtype=tl.float32), {dt.tl_name})")
            if dt is D.f32 or dt is D.f64:
                return f"tl.sum({inner}, axis={x.axis}, dtype=tl.float32)" \
                    if dt is D.f32 else \
                    f"tl.sum({inner}, axis={x.axis}, dtype=tl.float64)"
            return f"tl.cast(tl.sum({inner}, axis={x.axis}, dtype={dt.tl_name}), {dt.tl_name})"
        # max：Triton 对窄 dtype 以宽 dtype 返回——显式恢复
        if dt in (D.f16, D.bf16):
            return f"tl.cast(tl.max({inner}, axis={x.axis}), {dt.tl_name})"
        return f"tl.max({inner}, axis={x.axis})"

    def _elem_dtype(self, vt):
        if isinstance(vt, TY.BlockT) and isinstance(vt.elem, TY.ScalarT):
            return vt.elem.dtype
        if isinstance(vt, TY.ScalarT):
            return vt.dtype
        return D.f32

    def _zero_lit(self, dt: D.DType) -> str:
        return "0.0" if dt.is_float else "0"


def launcher_source(tk: T.TKernel, grid_hint: str = "(...)") -> str:
    """CLI build 输出的参考 launcher（runtime 用等价的 Python 逻辑）。"""
    b = tk.buffers[0] if tk.buffers else None
    args = ", ".join([p.name for p in tk.buffers] +
                     [p.name for p in tk.ptr_params] +
                     [s.name for s in tk.scalars] +
                     [f"{c.name}={c.default!r}" for c in tk.consts])
    out = [
        "import torch", "import triton", "",
        "",
        f"def launch({args}):",
    ]
    for p in tk.buffers:
        out.append(f"    {p.name}_ptr = {p.name}.data_ptr()")
        out.append(f"    # dtype/shape/stride 契约检查由 tila launcher 完成")
    for p in tk.ptr_params:
        out.append(f"    {p.name}_ptr = {p.name}   # 裸指针：dtype/对齐契约由 tila launcher 完成")
    out.append(f"    grid = {grid_hint}")
    out.append(f"    from {tk.name}_kernel import {tk.name}   # 生成的内核")
    out.append(f"    {tk.name}[grid](")
    names = Lowering(tk).launch_args()
    for i, n in enumerate(names):
        comma = "," if i < len(names) - 1 else ""
        out.append(f"        {n}{comma}")
    out.append("    )")
    return "\n".join(out) + "\n"
