"""pyast → tila_ast 转换（docs/ast.md §3）：子集校验 + 脱糖 + intrinsic resolution。

转换期不做任何类型判断（arange 参数个数等留给 checker）；职责单一便于测试。
拒绝项抛 E11–E15，附"Python 语法 X 不属于 Tila v0.1 子集"式说明。
"""

from __future__ import annotations

import ast as py

from .. import intrinsic
from ..ast import nodes as t
from ..diagnostics import Loc, err
from ..types import SURFACE_DTYPE_NAMES

RESERVED_NAMES = ("tila", "tl", "triton")

_BINOPS = {
    py.Add: "+",
    py.Sub: "-",
    py.Mult: "*",
    py.Div: "/",
    py.BitAnd: "&",
    py.BitOr: "|",
}
_CMPOPS = {
    py.Lt: "<",
    py.LtE: "<=",
    py.Gt: ">",
    py.GtE: ">=",
    py.Eq: "==",
    py.NotEq: "!=",
}


def _loc(node: py.AST) -> Loc:
    return Loc(getattr(node, "lineno", 1), getattr(node, "col_offset", 1))


def _not_in_subset(node: py.AST, what: str = None) -> TilaError:
    name = what or type(node).__name__
    return err(_loc(node), "E11", f"Python syntax {name} is not in the Tila v0.1 subset")


# ---------------------------------------------------------------------------
# module / kernel / signature
# ---------------------------------------------------------------------------


def convert(module: py.Module) -> t.KernelDef:
    """Module → KernelDef；模块结构必须恰为 `import tila` + 一个 @tila.jit 函数。"""
    body = module.body
    if len(body) < 2 or not isinstance(body[0], py.Import):
        raise err(Loc(1, 1), "E11", "a module must be exactly `import tila` followed by one @tila.jit kernel")
    imp = body[0]
    if len(imp.names) != 1 or imp.names[0].name != "tila" or imp.names[0].asname is not None:
        raise err(_loc(imp), "E11", 'the only legal import is a plain `import tila` '
                                '(`from tila import ...` / `import tila as tl` are rejected)')
    if len(body) != 2:
        raise err(_loc(body[2] if len(body) > 2 else body[1]), "E11",
                  "a module must contain exactly one kernel definition and nothing else")
    fn = body[1]
    if not isinstance(fn, py.FunctionDef):
        raise _not_in_subset(fn)
    return _convert_kernel(fn)


def _convert_kernel(fn: py.FunctionDef) -> t.KernelDef:
    if len(fn.decorator_list) != 1:
        raise err(_loc(fn), "E11", "the kernel must have exactly one decorator: @tila.jit")
    dec = fn.decorator_list[0]
    if not (isinstance(dec, py.Attribute) and isinstance(dec.value, py.Name)
            and dec.value.id == "tila" and dec.attr == "jit"):
        raise err(_loc(dec), "E11", "the only legal decorator is @tila.jit")
    if fn.name in RESERVED_NAMES:
        raise err(_loc(fn), "E12", f"kernel name '{fn.name}' is reserved "
                                   f"(tila/tl/triton would shadow generated module imports)")
    if fn.returns is not None:
        raise err(_loc(fn), "E11", "return annotations are not in the Tila v0.1 subset")

    a = fn.args
    if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.kw_defaults:
        raise err(_loc(fn), "E11", "positional-only / keyword-only / *args / **kwargs parameters "
                                   "are not in the Tila v0.1 subset")
    params = _convert_params(a.args, a.defaults, fn)
    if not fn.body:
        raise err(_loc(fn), "E11", "kernel body must not be empty")
    body = tuple(_convert_stmt(s) for s in fn.body)
    return t.KernelDef(_loc(fn), fn.name, params, body)


def _convert_params(args, defaults, fn) -> tuple:
    # defaults 右对齐于最后 len(defaults) 个参数
    n_def = len(defaults)
    params = []
    for i, arg in enumerate(args):
        has_default = i >= len(args) - n_def
        default_node = defaults[i - (len(args) - n_def)] if has_default else None
        ann = _convert_annotation(arg.annotation, arg)
        if ann is None and default_node is not None:
            raise err(_loc(default_node), "E11", "only `tila.constexpr` parameters may have defaults")
        if arg.arg in RESERVED_NAMES:
            raise err(_loc(arg), "E12", f"parameter name '{arg.arg}' is reserved "
                                        f"(tila/tl/triton would shadow generated module imports)")
        default = None
        if default_node is not None:
            if not isinstance(ann, t.ConstexprAnn):
                raise err(_loc(default_node), "E11", "only `tila.constexpr` parameters may have defaults")
            default = _const_int_default(default_node)
        params.append(t.Param(_loc(arg), arg.arg, ann, default))
    names = [p.name for p in params]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise err(_loc(fn), "E12", f"duplicate parameter name(s): {sorted(dup)}")
    return tuple(params)


def _const_int_default(node) -> int:
    """constexpr 默认值：整数字面量（允许 -INT 负字面量脱糖）；否则 E15。"""
    value = _int_literal(node)
    if value is None:
        raise err(_loc(node), "E15", "a `tila.constexpr` parameter must have an integer literal default "
                                     "(e.g. BLOCK: tila.constexpr = 128)")
    return value


def _int_literal(node):
    if isinstance(node, py.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, py.UnaryOp) and isinstance(node.op, py.USub):
        inner = _int_literal(node.operand)
        return None if inner is None else -inner
    return None


def _convert_annotation(ann, arg) -> t.Ann:
    if ann is None:
        return None
    loc = _loc(ann)
    if isinstance(ann, py.Attribute):
        if isinstance(ann.value, py.Name) and ann.value.id == "tila" and ann.attr == "constexpr":
            return t.ConstexprAnn(loc)
        raise err(loc, "E12", "parameter annotation must be tila.Tensor[dt, dims] or tila.constexpr")
    if isinstance(ann, py.Subscript):
        val = ann.value
        if isinstance(val, py.Attribute) and isinstance(val.value, py.Name) and val.value.id == "tila" \
                and val.attr == "Tensor":
            return _convert_tensor_ann(ann, loc)
        raise err(loc, "E12", "parameter annotation must be tila.Tensor[dt, dims] or tila.constexpr")
    raise err(loc, "E12", "parameter annotation must be tila.Tensor[dt, dims] or tila.constexpr")


def _convert_tensor_ann(ann: py.Subscript, loc: Loc) -> t.TensorAnn:
    sl = ann.slice
    if isinstance(sl, py.Index):  # Python < 3.9 兼容
        sl = sl.value
    if isinstance(sl, (py.Tuple,)):
        items = sl.elts
    else:
        items = [sl]
    if not items:
        raise err(loc, "E12", "tila.Tensor[...] needs a dtype and at least one dimension")
    dt_node = items[0]
    if not (isinstance(dt_node, py.Attribute) and isinstance(dt_node.value, py.Name)
            and dt_node.value.id == "tila"):
        raise err(loc, "E12", "the first subscript entry of tila.Tensor[...] must be a dtype "
                              "reference like tila.float32")
    dt = SURFACE_DTYPE_NAMES.get(dt_node.attr)
    if dt is None:
        raise err(_loc(dt_node), "E12", f"'tila.{dt_node.attr}' is not a dtype "
                                        f"(17 dtypes aligned with triton.language)")
    dims = []
    for d in items[1:]:
        if isinstance(d, py.Constant) and isinstance(d.value, int) and not isinstance(d.value, bool):
            dims.append(t.StaticDim(_loc(d), d.value))
        elif isinstance(d, py.Name):
            dims.append(t.SymDim(_loc(d), d.id))
        else:
            raise err(_loc(d), "E12", "dimensions in tila.Tensor[dt, dims] must be integer literals "
                                      "or symbol names (e.g. N)")
    return t.TensorAnn(loc, dt, tuple(dims))


# ---------------------------------------------------------------------------
# statements
# ---------------------------------------------------------------------------


def _convert_stmt(s: py.stmt) -> t.Stmt:
    if isinstance(s, py.Assign):
        if len(s.targets) != 1 or not isinstance(s.targets[0], py.Name):
            raise _not_in_subset(s, "multi-target or non-name assignment")
        value = convert_expr(s.value)
        return t.Assign(_loc(s), s.targets[0].id, value)
    if isinstance(s, py.Expr):
        if not isinstance(s.value, py.Call):
            raise _not_in_subset(s, "expression statement that is not a call")
        call = convert_expr(s.value)
        if not isinstance(call, t.Call):
            raise err(_loc(s), "E08", "the only legal expression statement is a tila.store(...) call")
        return t.ExprStmt(_loc(s), call)
    raise _not_in_subset(s)


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------


def convert_expr(node: py.expr, const_ctx: bool = False) -> t.Expr:
    """pyast 表达式 → tila_ast。

    const_ctx=True 仅用于 arange 边界这类编译期常量表达式位置：那里额外允许
    整数 `//`（`BLOCK // 2` 是合法 arange 边界，docs/language-spec.md §7）；
    其余位置的 FloorDiv 一律 E11（运算符集不含 `//`）。
    """
    loc = _loc(node)

    if isinstance(node, py.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _not_in_subset(node, f"constant {type(node.value).__name__}")
        if isinstance(node.value, int):
            return t.IntLit(loc, node.value)
        return t.FloatLit(loc, node.value)

    if isinstance(node, py.UnaryOp):
        if isinstance(node.op, py.USub) and isinstance(node.operand, py.Constant) \
                and isinstance(node.operand.value, (int, float)) \
                and not isinstance(node.operand.value, bool):
            # 负字面量脱糖：-1 是字面量前缀，不是一元运算
            v = -node.operand.value
            if isinstance(v, int):
                return t.IntLit(loc, v)
            return t.FloatLit(loc, v)
        raise _not_in_subset(node, "UnaryOp")

    if isinstance(node, py.Name):
        if node.id in RESERVED_NAMES:
            raise err(loc, "E12", f"reserved name '{node.id}' cannot appear in an expression")
        return t.NameRef(loc, node.id)

    if isinstance(node, py.Attribute):
        # tila.<name> 出现在非 call、非 cast-arg、非注解位置（docs/ast.md §3）
        if isinstance(node.value, py.Name) and node.value.id == "tila":
            name = node.attr
            if name in SURFACE_DTYPE_NAMES:
                raise err(loc, "E13", f"dtype reference tila.{name} is only legal as the 2nd "
                                      f"argument of tila.cast(x, tila.<dt>) or inside tila.Tensor[...]")
            if name == "Tensor":
                raise err(loc, "E14", "tila.Tensor may only appear in a parameter annotation position")
            if name in intrinsic.INTRINSICS or name in intrinsic.MARKER_NAMES:
                raise err(loc, "E13", f"tila.{name} may only be used in call position")
            raise err(loc, "E13", f"unknown tila.{name} (v0.1 builtins: "
                                  f"{', '.join(intrinsic.INTRINSICS)})")
        raise _not_in_subset(node, "attribute access")

    if isinstance(node, py.BinOp):
        if const_ctx and isinstance(node.op, py.FloorDiv):
            op = "//"
        else:
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise _not_in_subset(node, f"binary operator {type(node.op).__name__}")
        return t.BinOp(loc, op,
                       convert_expr(node.left, const_ctx),
                       convert_expr(node.right, const_ctx))

    if isinstance(node, py.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise _not_in_subset(node, "chained comparison")
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise _not_in_subset(node, "comparison operator")
        return t.BinOp(loc, op,
                       convert_expr(node.left, const_ctx),
                       convert_expr(node.comparators[0], const_ctx))

    if isinstance(node, py.Call):
        return _convert_call(node)

    if isinstance(node, py.Subscript):
        # tila.Tensor[...] 只出现在参数注解位置；出现在表达式位置 → E14（ast.md §3）
        val = node.value
        if isinstance(val, py.Attribute) and isinstance(val.value, py.Name) \
                and val.value.id == "tila" and val.attr == "Tensor":
            raise err(loc, "E14", "tila.Tensor may only appear in a parameter "
                                  "annotation position")
        raise _not_in_subset(node, "subscript")

    raise _not_in_subset(node)


def _convert_call(node: py.Call) -> t.Expr:
    loc = _loc(node)
    func = node.func
    if not (isinstance(func, py.Attribute) and isinstance(func.value, py.Name)
            and func.value.id == "tila"):
        if isinstance(func, py.Name):
            raise _not_in_subset(node, "call to a plain name (user functions do not exist in v0.1)")
        if isinstance(func, py.Attribute):
            raise _not_in_subset(node, "call to a non-tila attribute")
        raise _not_in_subset(node, "call")

    if node.keywords:
        for kw in node.keywords:
            if kw.arg is None:
                raise _not_in_subset(node, "**kwargs argument")
            if kw.arg not in intrinsic.KWARG_NAMES:
                raise err(_loc(kw), "E13",
                          f"unknown keyword argument '{kw.arg}' (only 'mask' and 'other' exist)")
    kwargs = tuple(
        (kw.arg, convert_expr(kw.value)) for kw in node.keywords
    )

    name = func.attr
    if name in intrinsic.INTRINSICS:
        const_ctx = name == "arange"  # arange 边界是编译期常量表达式位置
        args = []
        for i, a in enumerate(node.args):
            # cast 的第二位置实参是 DTypeRef 位置：tila.<dt> → DTypeRef
            if name == "cast" and i == 1 and isinstance(a, py.Attribute) \
                    and isinstance(a.value, py.Name) and a.value.id == "tila" \
                    and a.attr in SURFACE_DTYPE_NAMES:
                args.append(t.DTypeRef(_loc(a), SURFACE_DTYPE_NAMES[a.attr]))
            else:
                args.append(convert_expr(a, const_ctx))
        args = tuple(args)
        if name == "cast" and len(args) == 2 and isinstance(args[1], t.DTypeRef) and not kwargs:
            return t.Cast(loc, args[1].name, args[0])
        return t.Call(loc, name, args, kwargs)

    if name in SURFACE_DTYPE_NAMES:
        raise err(loc, "E13", f"dtype reference tila.{name} is only legal as the 2nd argument "
                              f"of tila.cast(x, tila.<dt>) or inside tila.Tensor[...]")
    if name == "Tensor":
        raise err(loc, "E14", "tila.Tensor may only appear in a parameter annotation position")
    if name in intrinsic.MARKER_NAMES:
        raise err(loc, "E13", f"tila.{name} may not be called (it is a decorator/annotation marker)")
    raise err(loc, "E13", f"unknown tila.{name} (v0.1 builtins: "
                          f"{', '.join(intrinsic.INTRINSICS)})")
