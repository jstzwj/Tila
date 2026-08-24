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


def _convert_dim(d, loc: Loc, what: str):
    """INT | IDENT → StaticDim | SymDim（维度与 strides 元素共用；v0.3-strides §1.1）。"""
    if isinstance(d, py.Constant) and isinstance(d.value, int) and not isinstance(d.value, bool):
        return t.StaticDim(_loc(d), d.value)
    if isinstance(d, py.Name):
        return t.SymDim(_loc(d), d.id)
    raise err(_loc(d), "E12", f"{what} must be integer literals or symbol names (e.g. N)")


def _convert_zeros_shape(a: py.Tuple) -> t.ShapeLit:
    """zeros 的静态 shape 元组（v0.4-kloop §1.2）：INT 字面量或 constexpr 名。"""
    if not a.elts:
        raise err(_loc(a), "E20", "the zeros shape tuple must not be empty",
                  subcode="ZerosForm")
    items = []
    for x in a.elts:
        if isinstance(x, py.Constant) and isinstance(x.value, int) \
                and not isinstance(x.value, bool):
            items.append(t.StaticDim(_loc(x), x.value))
        elif isinstance(x, py.Name):
            items.append(t.SymDim(_loc(x), x.id))
        else:
            raise err(_loc(x), "E20", "zeros shape entries must be integer literals "
                                      "or constexpr names (static extents only)",
                      subcode="ZerosForm")
    return t.ShapeLit(_loc(a), tuple(items))


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
    # v0.3：尾随 strides 元组（只允许最后一项；strides= 关键字形式是 Python
    # SyntaxError——下标不支持关键字参数，PEP 637 从未落地）
    strides = None
    if items and isinstance(items[-1], py.Tuple):
        stride_items = items[-1].elts
        if not stride_items:
            raise err(_loc(items[-1]), "E12", "the strides tuple of tila.Tensor[...] "
                                              "must list one stride per dimension")
        strides = tuple(_convert_dim(s, loc, "strides entries") for s in stride_items)
        items = items[:-1]
        if not items[1:]:
            raise err(loc, "E12", "tila.Tensor[...] needs a dtype and at least one dimension")
    dims = tuple(_convert_dim(d, loc, "dimensions in tila.Tensor[dt, dims]")
                 for d in items[1:])
    if strides is not None and len(strides) != len(dims):
        raise err(loc, "E12", f"strides tuple has {len(strides)} entries but the "
                              f"annotation has rank {len(dims)} (one stride per dimension)")
    for s in strides or ():
        if isinstance(s, t.StaticDim) and s.value < 1:
            raise err(s.loc, "E12", f"static strides must be >= 1 (got {s.value}); "
                                    f"use a symbol name for variable strides")
    return t.TensorAnn(loc, dt, dims, strides)


# ---------------------------------------------------------------------------
# statements
# ---------------------------------------------------------------------------


def _convert_stmt(s: py.stmt) -> t.Stmt:
    if isinstance(s, py.Assign):
        if len(s.targets) != 1 or not isinstance(s.targets[0], py.Name):
            raise _not_in_subset(s, "multi-target or non-name assignment")
        value = convert_expr(s.value)
        return t.Assign(_loc(s), s.targets[0].id, value)
    if isinstance(s, py.AugAssign):
        # v0.4（docs/v0.4-kloop.md §1.3）：只有 '+=' 进子集——累加器的显式标记
        if not isinstance(s.target, py.Name):
            raise _not_in_subset(s, "augmented assignment to a non-name target")
        if not isinstance(s.op, py.Add):
            raise err(_loc(s), "E20", f"only '+=' exists in the Tila v0.4 subset "
                                      f"(accumulator update); {type(s.op).__name__} "
                                      f"is rejected", subcode="AccumForm")
        return t.AugAssign(_loc(s), s.target.id, convert_expr(s.value))
    if isinstance(s, py.For):
        if s.orelse:
            raise _not_in_subset(s, "for ... else")
        if not isinstance(s.target, py.Name):
            raise _not_in_subset(s, "for target that is not a plain name")
        it = s.iter
        if not (isinstance(it, py.Call) and isinstance(it.func, py.Attribute)
                and isinstance(it.func.value, py.Name)
                and it.func.value.id == "tila" and it.func.attr == "range"):
            # 不走 convert_expr：裸 range / 列表迭代器会报误导性的 E11
            raise err(_loc(it), "E20",
                      "the only iterable is tila.range(0, end, step) "
                      "(tila.arange is a tile constructor, not an iterable)",
                      subcode="NotRange")
        call = _convert_call(it)
        return t.For(_loc(s), s.target.id, call,
                     tuple(_convert_stmt(b) for b in s.body))
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
            if name in intrinsic.CONSTANT_NAMES:
                # v0.5：tila.neg_inf ——max 的归约零元（−∞），语境定型同浮点字面量
                return t.FloatLit(loc, float("-inf"))
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
                          f"unknown keyword argument '{kw.arg}' (only 'mask', "
                          f"'other' and 'axis' exist)")
    if func.attr in ("range", "zeros") and node.keywords:
        # 先于通用 kwarg 门：range/zeros 不接受任何关键字实参（v0.4-kloop §7）
        raise err(_loc(node.keywords[0]), "E20",
                  f"tila.{func.attr} takes no keyword arguments",
                  subcode="RangeForm" if func.attr == "range" else "ZerosForm")
    kwargs = tuple(
        (kw.arg, convert_expr(kw.value)) for kw in node.keywords
    )

    name = func.attr
    if name in intrinsic.INTRINSICS:
        args = []
        for i, a in enumerate(node.args):
            # range 的 step（第 3 实参）是 ConstExpr 位置：额外允许整数 //
            const_ctx = name == "arange" or (name == "range" and i == 2)
        for i, a in enumerate(node.args):
            # cast 的第二位置实参是 DTypeRef 位置：tila.<dt> → DTypeRef
            if name == "cast" and i == 1 and isinstance(a, py.Attribute) \
                    and isinstance(a.value, py.Name) and a.value.id == "tila" \
                    and a.attr in SURFACE_DTYPE_NAMES:
                args.append(t.DTypeRef(_loc(a), SURFACE_DTYPE_NAMES[a.attr]))
            elif name == "zeros" and i == 1 and isinstance(a, py.Attribute) \
                    and isinstance(a.value, py.Name) and a.value.id == "tila" \
                    and a.attr in SURFACE_DTYPE_NAMES:
                args.append(t.DTypeRef(_loc(a), SURFACE_DTYPE_NAMES[a.attr]))
            elif name in ("load", "store") and i == 1 and isinstance(a, py.Tuple):
                # v0.3 定稿：坐标元组只作为 load/store 的第 2 位置实参
                # （tila.load(buffer, (c0, …))）；其余位置的元组一律 E11
                if not a.elts:
                    raise err(_loc(a), "E13", "the coordinates argument must be a "
                                              "non-empty tuple of index tiles")
                args.append(t.IndexTuple(_loc(a),
                                         tuple(convert_expr(x) for x in a.elts)))
            elif name == "zeros" and i == 0 and isinstance(a, py.Tuple):
                # v0.4：静态 shape 元组——INT 字面量或 constexpr 名（§1.2）
                args.append(_convert_zeros_shape(a))
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
    if name in intrinsic.CONSTANT_NAMES:
        raise err(loc, "E13", f"tila.{name} is a constant, not a callable — use it "
                              f"bare (e.g. tila.where(mask, x, tila.{name}))")
    if name in intrinsic.MARKER_NAMES:
        raise err(loc, "E13", f"tila.{name} may not be called (it is a decorator/annotation marker)")
    raise err(loc, "E13", f"unknown tila.{name} (v0.1 builtins: "
                          f"{', '.join(intrinsic.INTRINSICS)})")
