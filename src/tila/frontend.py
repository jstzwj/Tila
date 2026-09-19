"""Frontend：源码提取、Python 子集校验、注解解析、Python AST → HIR
（docs/surface-language.md §1–§2）。

不执行被装饰函数；inspect.getsource + ast.parse，建立 Tila 自己的 IR。
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from . import dtypes as D
from .dims import DimExpr, Sym
from .errors import Loc, TilaError
from .intrinsics import (METHOD_INTRINSIC_NAMES, PUBLIC_INTRINSIC_NAMES,
                         SurfaceForm, call_shape_problem, get_intrinsic)
from . import types as TY
from .types import BufferT, ConstT, PtrT, RefinedScalar, ScalarT
from .hir import (Assign, BinOp, BoolOp, BufPtr, Call, Cmp, Expand, ExprStmt,
                  For, HDtype, HTuple, If, Kernel, Lit, Name, Param, Return,
                  UnaOp)

# tila 命名空间里可出现的内建名（与公共占位符共用唯一名字事实源）。
INTRINSIC_NAMES = frozenset(PUBLIC_INTRINSIC_NAMES)

_BIN_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.FloorDiv: "//", ast.Mod: "%", ast.BitAnd: "&",
            ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}
_CMP_OPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!="}

_ALLOWED_STMT = (ast.Assign, ast.Expr, ast.If, ast.For)
_ALLOWED_EXPR = (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Compare,
                 ast.BoolOp, ast.Call, ast.Subscript, ast.Attribute)


def _syn(code: str, title: str, node, fixes=None):
    loc = Loc(getattr(node, "lineno", 0), getattr(node, "col_offset", 0),
              ast.unparse(node) if hasattr(ast, "unparse") else "")
    raise TilaError(code, title, loc,
                    fixes=fixes or ["参考 docs/surface-language.md §2 的子集清单"])


def compile_stage1(fn) -> Kernel:
    """@ti.jit 装饰时调用：Python 函数 → HIR（未类型检查）。"""
    try:
        raw = inspect.getsource(fn)
    except OSError as e:
        raise TilaError("TILA-SYN-000", f"cannot retrieve kernel source: {e}")
    src = textwrap.dedent(raw)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise TilaError("TILA-SYN-001", f"invalid syntax: {e.msg}",
                        Loc(e.lineno or 0, e.offset or 0, e.text or ""))

    lines = src.splitlines()
    fndef = tree.body[0]
    if not isinstance(fndef, ast.FunctionDef):
        _syn("TILA-SYN-002", "a @ti.jit unit must be a single function definition", tree)

    # `import tila` / `import tila as ti`：绑定到 tila 模块对象的全部名字
    aliases = {name for name, val in fn.__globals__.items()
               if getattr(val, "__name__", "") == "tila"}
    if not aliases:
        aliases = {"tila"}

    args = fndef.args.args
    defaults = [None] * (len(args) - len(fndef.args.defaults)) + \
        list(fndef.args.defaults)
    params = [_parse_param(a, d, fn.__globals__, lines)
              for a, d in zip(args, defaults)]
    if fndef.args.kwonlyargs or fndef.args.vararg or fndef.args.kwarg:
        _syn("TILA-SYN-004", "kernel parameters: positional-only subset (no *args/**kwargs)", fndef)

    builder = _Builder(fn.__globals__, aliases, lines, [p.name for p in params])
    body = [builder.stmt(s) for s in fndef.body]
    return Kernel(fndef.name, params, body, src, lines)


# ---------------------------------------------------------------------------
# 参数注解解析
# ---------------------------------------------------------------------------

def _eval_annotation(anno, g: dict):
    """注解求值：正常路径拿到的是 AST 节点（我们重新 parse 源码）；
    `from __future__ import annotations` 时是字符串。两种都在 kernel
    模块的全局命名空间里求值（ti / N 等在那里绑定）。"""
    try:
        if isinstance(anno, ast.AST):
            return eval(compile(ast.Expression(anno), "<annotation>", "eval"),
                        g)
        if isinstance(anno, str):
            return eval(compile(anno, "<annotation>", "eval"), g)
        return anno
    except TilaError:
        raise
    except Exception as e:
        raise TilaError("TILA-SYN-010",
                        f"cannot evaluate annotation: {e}")


def _parse_param(a: ast.arg, default_node, g: dict, lines: list) -> Param:
    loc = Loc(a.lineno, a.col_offset, lines[a.lineno - 1] if a.lineno else "")
    name = a.arg
    spec = _eval_annotation(a.annotation, g) if a.annotation is not None else None
    default = None
    if default_node is not None:
        if not isinstance(default_node, ast.Constant):
            _syn("TILA-SYN-011", "parameter default must be a literal",
                 default_node)
        default = default_node.value

    if spec is None:
        _syn("TILA-SYN-012", f"parameter '{name}' needs a Tila type annotation "
            "(kernel boundary types are required)", a,
            [f"例如 x: ti.Buffer[ti.f32, (N,), ti.ReadOnly] 或 BLOCK: ti.Const[int, ti.PowerOfTwo]"])

    # ti.Const[...] 的 spec 是 ("const", refinements) 元组
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] in ("const", "const_bool"):
        refins = spec[1]
        expected = bool if spec[0] == "const_bool" else int
        if default_node is not None and type(default) is not expected:
            raise TilaError(
                "TILA-CONST-008",
                f"Const parameter '{name}' default must be an exact Python {expected.__name__}",
                loc,
                [f"found: {type(default).__name__} value {default!r}",
                 f"required: type(value) is {expected.__name__}"],
                [f"write {name}: ti.Const[{expected.__name__}] = <{expected.__name__} literal>"],
            )
        return Param(name, ConstT(name, refins, default,
                                 D.bool_ if expected is bool else D.i32), default)
    if isinstance(spec, D.DType):                    # 裸 dtype：ti.f32
        spec = ScalarT(spec)
    if isinstance(spec, (BufferT, PtrT, RefinedScalar, ScalarT, ConstT)):
        if default is not None:
            _syn("TILA-SYN-014", "only Const[int] parameters may have defaults", a)
        return Param(name, spec, None)
    _syn("TILA-SYN-015", f"parameter '{name}': unsupported annotation {spec!r}", a)
    raise AssertionError


# ---------------------------------------------------------------------------
# AST → HIR
# ---------------------------------------------------------------------------

class _Builder:
    def __init__(self, g: dict, aliases: set, lines: list, param_names: list):
        self.g = g
        self.aliases = aliases
        self.lines = lines
        self.param_names = set(param_names)
        self.local_names: set[str] = set()

    def _loc(self, node) -> Loc:
        return Loc(node.lineno, node.col_offset,
                   self.lines[node.lineno - 1] if node.lineno else "")

    def _syn(self, code, title, node, fixes=None):
        raise TilaError(code, title, self._loc(node), fixes=fixes)

    # -- 语句 ---------------------------------------------------------------

    def stmt(self, s):
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                self._syn("TILA-SYN-020", "assignment target must be a single name", s)
            self.local_names.add(s.targets[0].id)
            return Assign(self._loc(s), s.targets[0].id, self.expr(s.value))
        if isinstance(s, ast.Expr):
            return ExprStmt(self._loc(s), self.expr(s.value))
        if isinstance(s, ast.If):
            return If(self._loc(s), self.expr(s.test),
                      [self.stmt(x) for x in s.body],
                      [self.stmt(x) for x in s.orelse])
        if isinstance(s, ast.For):
            return self._for(s)
        if isinstance(s, ast.Return):
            # 裸 return = 提前退出当前 program instance（§2.1 允许表）；
            # 带值 return 在 v0 拒绝（Triton kernel 不向 host 返回值）。
            if s.value is not None:
                self._syn("TILA-SYN-021",
                          "kernel entry points cannot return values in v0 "
                          "(Triton kernels return nothing to the host)",
                          s,
                          ["store results to an output buffer "
                           "(ti.store(out, ...)); a bare `return` (no value) "
                           "may be used to exit early"])
            return Return(self._loc(s))
        if isinstance(s, (ast.Pass,)):
            return ExprStmt(self._loc(s), None)
        if isinstance(s, ast.AnnAssign):
            self._syn("TILA-SYN-022", "local variable annotations are not part of "
                      "the subset (locals are inferred)", s)
        if isinstance(s, ast.AugAssign):
            self._syn("TILA-SYN-002", "augmented assignment is not supported; "
                      "write x = x + value instead of x += value", s)
        self._syn("TILA-SYN-002",
                  f"'{type(s).__name__}' is not part of the Tila subset", s)

    def _for(self, s: ast.For):
        if not isinstance(s.target, ast.Name):
            self._syn("TILA-SYN-023", "loop variable must be a name", s)
        self.local_names.add(s.target.id)
        it = s.iter
        if not (isinstance(it, ast.Call) and self._intrinsic_name(it.func) == "range"):
            self._syn("TILA-SYN-003", "the only loop form is: for i in ti.range(0, end, step):",
                      s, [f"got: {ast.unparse(it)}"])
        spec = get_intrinsic("range")
        assert spec is not None and SurfaceForm.LOOP_FORM in spec.surface_forms
        self._validate_call_shape(spec, it, positional_count=len(it.args),
                                  code="TILA-SYN-024")
        step = self.expr(it.args[2]) if len(it.args) == 3 else Lit(self._loc(it), 1)
        return For(self._loc(s), s.target.id,
                   self.expr(it.args[0]), self.expr(it.args[1]), step,
                   [self.stmt(x) for x in s.body])

    # -- 表达式 -------------------------------------------------------------

    def _intrinsic_name(self, func) -> str | None:
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in self.aliases:
                if func.attr in INTRINSIC_NAMES:
                    return func.attr
                self._syn("TILA-SYN-030",
                          f"'{func.attr}' is not a tila intrinsic", func,
                          [f"tila 命名空间清单见 docs/intrinsics.md §2"])
            return None
        return None

    def _validate_call_shape(self, spec, node, positional_count=None,
                             code="TILA-SYN-036"):
        if any(keyword.arg is None for keyword in node.keywords):
            self._syn("TILA-SYN-037", "**kwargs is not supported", node)
        problem = call_shape_problem(
            spec,
            len(node.args) if positional_count is None else positional_count,
            [keyword.arg for keyword in node.keywords],
        )
        if problem is not None:
            self._syn(code, problem, node)

    def _require_surface(self, spec, surface, node):
        if surface in spec.surface_forms:
            return
        if spec.name == "range":
            self._syn("TILA-SYN-028",
                      "ti.range is only used as: for i in ti.range(0, end, step)",
                      node)
        if spec.name == "cast":
            self._syn("TILA-SYN-035",
                      "cast must use the subscript form ti.cast[dtype](x)", node)
        self._syn("TILA-SYN-036",
                  f"intrinsic '{spec.name}' does not support {surface.value}",
                  node)

    def expr(self, e):
        loc = self._loc(e)
        if isinstance(e, ast.Constant):
            if isinstance(e.value, (int, float, bool)):
                return Lit(loc, e.value)
            self._syn("TILA-SYN-031", f"literal {e.value!r} is not supported", e)
        if isinstance(e, ast.Name):
            kind = self._name_kind(e.id, e)
            if kind == "const_int":
                return Lit(loc, self.g[e.id])
            if kind == "const_float":
                return Lit(loc, self.g[e.id])
            if kind == "const_bool":
                return Lit(loc, self.g[e.id])
            return Name(loc, e.id, kind)
        if isinstance(e, ast.BinOp):
            op = _BIN_OPS.get(type(e.op))
            if op is None:
                self._syn("TILA-SYN-032",
                          f"operator {type(e.op).__name__} is not supported", e)
            return BinOp(loc, op, self.expr(e.left), self.expr(e.right))
        if isinstance(e, ast.UnaryOp):
            op = {ast.USub: "-", ast.Invert: "~", ast.Not: "not"}[type(e.op)]
            return UnaOp(loc, op, self.expr(e.operand))
        if isinstance(e, ast.Compare):
            if len(e.ops) != 1:
                self._syn("TILA-SYN-033", "chained comparisons are not supported", e)
            op = _CMP_OPS.get(type(e.ops[0]))
            if op is None:
                self._syn("TILA-SYN-033",
                          f"comparison {type(e.ops[0]).__name__} is not supported", e)
            return Cmp(loc, op, self.expr(e.left), self.expr(e.comparators[0]))
        if isinstance(e, ast.BoolOp):
            op = "and" if isinstance(e.op, ast.And) else "or"
            return BoolOp(loc, op, [self.expr(v) for v in e.values])
        if isinstance(e, ast.Call):
            return self._call(e, loc)
        if isinstance(e, ast.Subscript):
            return self._subscript(e, loc)
        if isinstance(e, ast.Attribute):
            if e.attr == "ptr" and isinstance(e.value, ast.Name):
                return BufPtr(loc, e.value.id)
            # 表达式位置的 dtype 引用（zeros((128,), ti.f16) 等）
            if isinstance(e.value, ast.Name) and e.value.id in self.aliases:
                mod = self.g.get(e.value.id)
                v = getattr(mod, e.attr, None) if mod is not None else None
                from . import dtypes as _D
                if isinstance(v, _D.DType):
                    return HDtype(loc, v)
            self._syn("TILA-SYN-034", "the only attribute forms are buf.ptr "
                      "and dtype references (ti.f16)", e)
        self._syn("TILA-SYN-002",
                  f"'{type(e).__name__}' is not part of the Tila subset", e)

    def _name_kind(self, id: str, node) -> str:
        if id in self.local_names or id in self.param_names:
            return "local"
        v = self.g.get(id)
        if isinstance(v, Sym):
            return "dim"
        if isinstance(v, bool):
            return "const_bool"
        if isinstance(v, int):
            return "const_int"
        if isinstance(v, float):
            return "const_float"
        if v is None:
            self._syn("TILA-SYN-040", f"name '{id}' is not defined", node,
                      [f"kernel 内可见：参数、局部变量、模块级 ti.Dim 与 int/float 常量"])
        self._syn("TILA-SYN-041",
                  f"name '{id}' refers to a Python object of type "
                  f"{type(v).__name__}; only ti.Dim / int / float consts are visible", node)

    def _call(self, e: ast.Call, loc):
        func = e.func
        # cast[U](x)：Subscript(Attribute(alias, 'cast'), dtype)
        if isinstance(func, ast.Subscript):
            base = func.value
            name = self._intrinsic_name(base)
            spec = get_intrinsic(name) if name is not None else None
            if spec is not None and SurfaceForm.SUBSCRIPT_CALL in spec.surface_forms:
                self._validate_call_shape(spec, e)
                dt = self._dtype_from_annotation(func.slice)
                if name == "constant":
                    arg = e.args[0]
                    atom = arg.operand if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) else arg
                    if isinstance(atom, ast.Constant):
                        value = atom.value
                    elif (isinstance(atom, ast.Name) and atom.id not in self.local_names
                          and atom.id not in self.param_names and atom.id in self.g):
                        value = self.g[atom.id]
                    else:
                        self._syn("TILA-CONST-011", "constant source must be an exact int/float literal or captured module constant", arg)
                    if type(value) is not int and type(value) is not float:
                        self._syn("TILA-CONST-011", "constant source must be exact Python int or float", arg)
                kw = {k.arg: self.expr(k.value) for k in e.keywords if k.arg}
                return Call(loc, name, [self.expr(a) for a in e.args], kw,
                            cast_dtype=dt)
            self._syn("TILA-SYN-035", "subscripted calls are only cast[dt](x)", e)
        # m.any() / m.all()：Mask 归约的方法形态（type-system.md §3.3、
        # intrinsics.md §2.5）——重写为普通调用 Call('any'|'all', [receiver])。
        # 仅接受 Name 接收者 + 零实参零 kwargs；别名形态 ti.any(...) 不在此
        # 路径（'any'/'all' 不进 INTRINSIC_NAMES，仍走 SYN-030 拒绝）。
        if (isinstance(func, ast.Attribute) and func.attr in METHOD_INTRINSIC_NAMES
                and isinstance(func.value, ast.Name)
                and func.value.id not in self.aliases):
            spec = get_intrinsic(func.attr)
            assert spec is not None
            self._require_surface(spec, SurfaceForm.METHOD_CALL, e)
            # registry arity counts the rewritten receiver as the first arg.
            self._validate_call_shape(spec, e, positional_count=1 + len(e.args))
            return Call(loc, func.attr,
                        [self.expr(func.value), *[self.expr(a) for a in e.args]],
                        {k.arg: self.expr(k.value) for k in e.keywords})
        name = self._intrinsic_name(func)
        if name is None:
            if isinstance(func, ast.Attribute):
                self._syn("TILA-SYN-036",
                          f"call to '{ast.unparse(func)}': only tila.* intrinsics "
                          "can be called", e)
            self._syn("TILA-SYN-036",
                      f"call to Python name '{ast.unparse(func)}' is not allowed "
                      "(only tila.* intrinsics)", e)
        spec = get_intrinsic(name)
        assert spec is not None
        self._require_surface(spec, SurfaceForm.PUBLIC_CALL, e)
        self._validate_call_shape(spec, e)
        args = []
        for a in e.args:
            if isinstance(a, ast.Tuple):
                args.append(HTuple(loc, [self.expr(x) for x in a.elts]))
            else:
                args.append(self.expr(a))
        kw = {}
        for k in e.keywords:
            kw[k.arg] = self.expr(k.value)
        return Call(loc, name, args, kw)

    def _subscript(self, e: ast.Subscript, loc):
        # 仅 [:, None] / [None, :]（expand_dims 语法糖，两个轴都支持）
        sl = e.slice
        items = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        if len(items) != 2:
            self._syn("TILA-SYN-038", "subscripts are only the "
                      "x[:, None] / x[None, :] expand_dims sugar", e)
        axis = None
        for i, it in enumerate(items):
            if isinstance(it, ast.Slice):
                continue
            if isinstance(it, ast.Constant) and it.value is None:
                axis = i       # None 所在位置 = 插入的轴
            else:
                self._syn("TILA-SYN-038", "subscripts are only the "
                          "x[:, None] / x[None, :] expand_dims sugar", e)
        if axis is None:
            self._syn("TILA-SYN-038", "subscripts are only the "
                      "x[:, None] / x[None, :] expand_dims sugar", e)
        return Expand(loc, self.expr(e.value), axis)

    def _dtype_from_annotation(self, node):
        if isinstance(node, ast.Name):
            v = self.g.get(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in self.aliases:
            mod = self.g.get(node.value.id)
            v = getattr(mod, node.attr, None) if mod is not None else None
        else:
            v = None
        if isinstance(v, D.DType):
            return v
        self._syn("TILA-SYN-039",
                  f"cast[..] needs a dtype (ti.f32 etc.), got {ast.unparse(node)}", node)
