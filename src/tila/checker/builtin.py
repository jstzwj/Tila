"""内建检查流程（docs/type-checker.md §4.2）：program_id / arange / load / store / cast。

每个流程消费 checker 上下文 ctx（提供 infer / emit / env / next_id），
按语义合同（docs/semantic-model.md §5）静态检查并发射 TIR 指令。
"""

from __future__ import annotations

from .. import tir
from ..ast import nodes as t
from ..diagnostics import Loc, err
from ..types import (
    ROW_MAJOR,
    AddressType,
    BufferType,
    CastL,
    Const,
    Identity,
    LoadL,
    Mma,
    ScalarType,
    TileType,
    UnitType,
    numel,
    normalize,
)
from ..types import dtype as dt
from ..types import type_str  # noqa: F401
from ..types.layout import Zeros
from . import broadcast as bcast
from . import layout as lay

TRITON_MAX_TENSOR_NUMEL = 2 ** 20


def _type_note(what: str, ty, defined: Loc = None) -> str:
    s = f"{what} : {type_str(ty)}"
    if defined is not None:
        s += f"  (defined at line {defined.line})"
    return s


# ---------------------------------------------------------------------------
# 5.1  program_id
# ---------------------------------------------------------------------------


def check_program_id(ctx, e: t.Call):
    if len(e.args) != 1 or e.kwargs:
        raise err(e.loc, "E13", "tila.program_id takes exactly one positional argument: "
                                "tila.program_id(0)")
    arg = e.args[0]
    axis = None
    if isinstance(arg, t.IntLit):
        axis = arg.value
    elif isinstance(arg, t.NameRef) and arg.name in ctx.env.constexprs:
        axis = ctx.env.constexprs[arg.name]
    else:
        raise err(e.loc, "E13", "program_id axis must be a compile-time constant "
                                "(literal or constexpr value)")
    if axis not in (0, 1):
        raise err(e.loc, "E13", f"program_id axis {axis} is not supported "
                                f"(axes {{0, 1}} as of the v0.2 2D preview)")
    ty = ScalarType("i32")
    return ty, ctx.emit(tir.TProgramId(ctx.next_id("t"), ty, None, None, axis)).id


# ---------------------------------------------------------------------------
# 5.2  arange
# ---------------------------------------------------------------------------


def check_arange(ctx, e: t.Call):
    if len(e.args) != 2 or e.kwargs:
        raise err(e.loc, "E13", "tila.arange takes exactly two positional arguments: "
                                "tila.arange(0, end)")
    start, end = e.args
    if not isinstance(start, t.IntLit) or start.value != 0:
        raise err(e.loc, "E06", "arange start must be the literal 0 "
                                "(Tila requires tila.arange(0, end))")
    tree, value = ctx.eval_constexpr(end)
    if value < 2 or value & (value - 1) or value > TRITON_MAX_TENSOR_NUMEL:
        kind = "is less than 2" if value < 2 else (
            "is not a power of two" if value & (value - 1) else f"exceeds 2^20"
        )
        raise err(e.loc, "E06", f"arange extent {value} {kind} "
                                f"(Tila requires 2^k, 1 <= k <= 20)")
    shape = (Const(value),)
    ty = TileType("i32", shape, Identity(shape))
    return ty, ctx.emit(tir.TArange(ctx.next_id("r"), ty, None, None, 0, tree)).id


# ---------------------------------------------------------------------------
# 5.3  load
# ---------------------------------------------------------------------------


def check_load(ctx, e: t.Call):
    if len(e.args) != 2:
        raise err(e.loc, "E13", "tila.load takes exactly two positional arguments: "
                                "tila.load(buffer, (coord, …)); the v0.2 form "
                                "`tila.load(a + offs)` was removed — addressing is "
                                "load/store-internal (docs/v0.3-strides.md §1.2)")
    kw = dict(e.kwargs)
    if set(kw) - {"mask", "other"}:
        raise err(e.loc, "E13", "tila.load accepts only 'mask' and 'other' keyword arguments")

    ptr_ty, ptr_id = ctx.buffer_address(e.args[0], e.args[1])

    mask_id = None
    if "mask" in kw:
        m_ty, mask_id = ctx.infer(kw["mask"])
        if not isinstance(m_ty, TileType) or m_ty.dtype != "bool":
            raise err(kw["mask"].loc, "E02", "load mask must be Tile[bool] "
                                             "(produced by comparisons)",
                      _type_note("mask", m_ty))
        bcast.require_same_shape(m_ty.shape, ptr_ty.shape, kw["mask"].loc, "load mask")
        lay.require_equiv(m_ty.layout, ptr_ty.layout, kw["mask"].loc, "load mask")

    other_id = None
    if "other" in kw:
        o_expr = kw["other"]
        if mask_id is None:
            raise err(o_expr.loc, "E13", "'other' requires 'mask': without a mask there are "
                                         "no masked-out lanes")
        if not isinstance(o_expr, (t.IntLit, t.FloatLit)):
            raise err(o_expr.loc, "E13", "'other' must be a literal matching the element "
                                         "dtype category (e.g. other=0 / other=0.0)")
        is_float_lit = isinstance(o_expr, t.FloatLit)
        if dt.is_float(ptr_ty.dtype) != is_float_lit:
            want = "a FLOAT literal (e.g. 0.0)" if dt.is_float(ptr_ty.dtype) \
                else "an INT literal (e.g. 0)"
            raise err(o_expr.loc, "E02", f"'other' literal category mismatch for element "
                                         f"dtype {ptr_ty.dtype}; write {want}")
        _, other_id = ctx.infer(o_expr)

    origin = LoadL(ROW_MAJOR, ptr_ty.layout)
    ty = TileType(ptr_ty.dtype, ptr_ty.shape, normalize(origin))
    return ty, ctx.emit(tir.TLoad(ctx.next_id("t"), ty, None, origin,
                                  ptr_id, mask_id, other_id)).id


# ---------------------------------------------------------------------------
# 5.4  store（表达式语句语境由 checker 保证；这里做 0 之外的检查）
# ---------------------------------------------------------------------------


def check_store(ctx, e: t.Call):
    if len(e.args) != 3:
        raise err(e.loc, "E13", "tila.store takes exactly three positional arguments: "
                                "tila.store(buffer, (coord, …), value); the v0.2 form "
                                "`tila.store(ptr, value)` was removed — addressing is "
                                "load/store-internal (docs/v0.3-strides.md §1.2)")
    kw = dict(e.kwargs)
    if set(kw) - {"mask"}:
        raise err(e.loc, "E13", "tila.store accepts only the 'mask' keyword argument "
                                "('other' is meaningless for stores)")

    ptr_ty, ptr_id = ctx.buffer_address(e.args[0], e.args[1])
    val_expr = e.args[2]

    val_ty, val_id = ctx.infer(val_expr)
    if not isinstance(val_ty, TileType):
        raise err(val_expr.loc, "E07", "tila.store value must be a Tile",
                  _type_note("value", val_ty))

    # 检查顺序（§4.2）：rank → shape → dtype（严格相等，无隐式收窄）
    # layout 前提已按 R9'（v0.2 matmul fragment）放宽：store 逐坐标写
    # v[i,j] → ptr[i,j]，分布只影响代价不影响坐标语义——内存边界布局无关。
    if len(val_ty.shape) != len(ptr_ty.shape):
        raise err(e.loc, "E04", f"store rank mismatch: value rank {len(val_ty.shape)} "
                                f"vs address rank {len(ptr_ty.shape)}")
    bcast.require_same_shape(val_ty.shape, ptr_ty.shape, e.loc, "store value")
    if val_ty.dtype != ptr_ty.dtype:
        raise err(e.loc, "E02",
                  f"cannot store Tile<{val_ty.dtype}> into Buffer<{ptr_ty.dtype}>",
                  _type_note("value", val_ty),
                  "implicit narrowing is not allowed; write "
                  f"tila.cast(value, tila.{dt.TILA_SURFACE_NAME.get(ptr_ty.dtype, ptr_ty.dtype)}) "
                  f"if intended")

    mask_id = None
    if "mask" in kw:
        m_ty, mask_id = ctx.infer(kw["mask"])
        if not isinstance(m_ty, TileType) or m_ty.dtype != "bool":
            raise err(kw["mask"].loc, "E02", "store mask must be Tile[bool]",
                      _type_note("mask", m_ty))
        bcast.require_same_shape(m_ty.shape, ptr_ty.shape, kw["mask"].loc, "store mask")

    ctx.emit(tir.TStore(None, None, None, None, ptr_id, val_id, mask_id))
    return UnitType(), None


# ---------------------------------------------------------------------------
# 5.5  cast
# ---------------------------------------------------------------------------


def check_cast(ctx, loc: Loc, operand_expr: t.Expr, dtype_name: str):
    if isinstance(operand_expr, (t.IntLit, t.FloatLit)):
        raise err(loc, "E09", "a literal cannot be cast directly; use it in a matching "
                              "context or bind it to a name first")
    x_ty, x_id = ctx.infer(operand_expr)
    if not isinstance(x_ty, (TileType, ScalarType)):
        raise err(loc, "E09", "tila.cast operand must be a Tile or a Scalar",
                  _type_note("operand", x_ty))
    if isinstance(x_ty, TileType):
        origin = x_ty.layout
        ty = TileType(dtype_name, x_ty.shape, normalize(CastL(origin)))
    else:
        ty = ScalarType(dtype_name)
    return ty, ctx.emit(tir.TCast(ctx.next_id("t"), ty, None, None, dtype_name, x_id)).id


def check_cast_call(ctx, e: t.Call):
    """未脱糖成 Cast 节点的 tila.cast(...) 调用：必然形态非法（E09/E13）。"""
    if e.kwargs:
        raise err(e.loc, "E13", "tila.cast takes no keyword arguments")
    if len(e.args) != 2:
        raise err(e.loc, "E09", "tila.cast takes exactly two positional arguments: "
                                "tila.cast(x, tila.<dtype>)")
    from ..ast import nodes as tn

    if not isinstance(e.args[1], tn.DTypeRef):
        raise err(e.loc, "E09", "the 2nd argument of tila.cast must be a dtype reference "
                                "like tila.float32")
    return check_cast(ctx, e.loc, e.args[0], e.args[1].name)


# ---------------------------------------------------------------------------
# R13  expand_dim（v0.2 预览，docs/v0.2-preview-2d.md §3）
# ---------------------------------------------------------------------------


def check_expand_dim(ctx, e: t.Call):
    if len(e.args) != 2 or e.kwargs:
        raise err(e.loc, "E13", "tila.expand_dim takes exactly two positional arguments: "
                                "tila.expand_dim(tile, axis)")
    tile_expr, axis_expr = e.args
    if not isinstance(axis_expr, t.IntLit):
        raise err(axis_expr.loc, "E04", "expand_dim axis must be an integer literal "
                                        "(numbering the result tensor's axes)")
    t_ty, t_id = ctx.infer(tile_expr)
    if not isinstance(t_ty, TileType):
        raise err(tile_expr.loc, "E07", "tila.expand_dim operates on a Tile",
                  _type_note("operand", t_ty))
    axis = axis_expr.value
    rank = len(t_ty.shape)
    if not 0 <= axis <= rank:
        raise err(axis_expr.loc, "E04",
                  f"expand_dim axis {axis} is out of range for rank {rank} "
                  f"(axis numbers the result tensor: 0..{rank})")
    # layout 不变：长度 1 的轴，其元素分布是唯一的平凡映射——layout 代数对
    # size-1 轴不可见（R13）。shape 与 layout 是独立字段，互不隐含。
    new_shape = t_ty.shape[:axis] + (Const(1),) + t_ty.shape[axis:]
    ty = TileType(t_ty.dtype, new_shape, t_ty.layout)
    return ty, ctx.emit(tir.TExpandDim(ctx.next_id("t"), ty, None, t_ty.layout,
                                       t_id, axis)).id


# ---------------------------------------------------------------------------
# R16  dot（v0.2 matmul fragment，docs/v0.2-matmul-fragment.md §2）
# ---------------------------------------------------------------------------

MMA_MIN_EXTENT = 16
MMA_OPERAND_DTYPES = ("f16", "bf16")


def check_dot(ctx, e: t.Call):
    if len(e.args) != 2 or e.kwargs:
        raise err(e.loc, "E13", "tila.dot takes exactly two positional arguments: "
                                "tila.dot(x, y)  with  x: Tile[dt,(M,K)], y: Tile[dt,(K,N)]")
    x_ty, x_id = ctx.infer(e.args[0])
    y_ty, y_id = ctx.infer(e.args[1])
    notes = [_type_note("lhs", x_ty, ctx._def_loc(e.args[0])),
             _type_note("rhs", y_ty, ctx._def_loc(e.args[1]))]
    if not (isinstance(x_ty, TileType) and isinstance(y_ty, TileType)):
        raise err(e.loc, "E18", "tila.dot operands must be rank-2 Tiles",
                  *notes, "expected shapes: (M, K) @ (K, N) -> (M, N)")
    if len(x_ty.shape) != 2 or len(y_ty.shape) != 2:
        raise err(e.loc, "E18", "tila.dot operands must be rank-2 Tiles",
                  *notes, "expected shapes: (M, K) @ (K, N) -> (M, N)")
    (m, kx), (ky, n) = x_ty.shape, y_ty.shape
    if kx != ky:
        raise err(e.loc, "E18",
                  f"dot contraction mismatch: lhs K dim {shape_str_of(kx)} vs "
                  f"rhs K dim {shape_str_of(ky)}",
                  *notes, "expected shapes: (M, K) @ (K, N) -> (M, N)")
    if x_ty.dtype != y_ty.dtype:
        raise err(e.loc, "E18",
                  f"dot operands must share one dtype (f16 or bf16); got "
                  f"{x_ty.dtype} and {y_ty.dtype}",
                  *notes, "mixed-precision dot is not in the fragment; cast both "
                          "operands to the same MMA dtype first")
    if x_ty.dtype not in MMA_OPERAND_DTYPES:
        surface = dt.TILA_SURFACE_NAME.get(x_ty.dtype, x_ty.dtype)
        raise err(e.loc, "E18",
                  f"dot operand dtype {x_ty.dtype} is not MMA-able in the v0.2 "
                  f"fragment (f16/bf16 only, f32 accumulation)",
                  *notes, "hint: tila.cast(x, tila.float16) if precision allows; "
                          "tf32/fp8 operands are planned")
    for label, d in (("M", m), ("K", kx), ("N", n)):
        v = numel((d,))
        if v is None:
            raise err(e.loc, "E18", f"dot {label} extent must be static "
                                    f"(constexpr-derived)", *notes)
        if v < MMA_MIN_EXTENT:
            raise err(e.loc, "E18",
                      f"dot {label} extent {v} is below the MMA minimum "
                      f"{MMA_MIN_EXTENT}", *notes,
                      "MMA instructions require every dot dimension >= 16")

    ty = TileType("f32", (m, n), Mma(numel((m,)), numel((n,)), numel((kx,))))
    return ty, ctx.emit(tir.TDot(ctx.next_id("t"), ty, None, None,
                                 x_id, y_id)).id


def shape_str_of(d) -> str:
    v = getattr(d, "value", None)
    return str(v) if v is not None else getattr(d, "name", "?")


# ---------------------------------------------------------------------------
# R17  zeros（v0.4，docs/v0.4-kloop.md §2.1）
# ---------------------------------------------------------------------------


def check_zeros(ctx, e: t.Call):
    """常量分布种子：Tile[dt, (Const,…), Zeros(Σ)]。

    shape 项是 INT 字面量或 constexpr 名（静态量——累加器 shape 参与类型
    等价判定）；dtype 走 DTypeRef 位置（cast 第二实参同款解析）。
    """
    if len(e.args) != 2:
        raise err(e.loc, "E20", "tila.zeros takes exactly two positional arguments: "
                                "tila.zeros((BM, BN), tila.float32)",
                  subcode="ZerosForm")
    shape_arg, dtype_arg = e.args
    if not isinstance(shape_arg, t.ShapeLit):
        raise err(shape_arg.loc, "E20",
                  "the zeros shape must be a tuple of integer literals or constexpr "
                  "names: tila.zeros((BM, BN), tila.float32)",
                  subcode="ZerosForm")
    if not isinstance(dtype_arg, t.DTypeRef):
        raise err(dtype_arg.loc, "E20", "the 2nd argument of tila.zeros must be a "
                                        "dtype reference like tila.float32",
                  subcode="ZerosForm")
    if len(shape_arg.items) >= 3:
        raise err(shape_arg.loc, "E20", "rank >= 3 is rejected "
                                        "(rank <= 2 as of the v0.2 2D preview)",
                  subcode="ZerosForm")
    if not dt.can_be_tensor_element(dtype_arg.name):
        raise err(dtype_arg.loc, "E20", "bool cannot be a zeros element dtype "
                                        "(masks come from comparisons; bool storage "
                                        "has no use)",
                  subcode="ZerosForm")
    shape_ce = []
    shape_ty = []
    for d in shape_arg.items:
        if isinstance(d, t.StaticDim):
            shape_ce.append(d.value)
            shape_ty.append(Const(d.value))
        else:  # SymDim：此处语义 = constexpr 名（静态量）
            if d.name not in ctx.env.constexprs:
                raise err(d.loc, "E20",
                          f"zeros shape entries must be integer literals or "
                          f"constexpr names; '{d.name}' is not a constexpr "
                          f"(accumulator shapes are static)",
                          subcode="ZerosForm")
            shape_ce.append(d.name)
            shape_ty.append(Const(ctx.env.constexprs[d.name]))
    ty = TileType(dtype_arg.name, tuple(shape_ty), Zeros(tuple(shape_ty)))
    return ty, ctx.emit(tir.TZeros(ctx.next_id("t"), ty, None, None,
                                   tuple(shape_ce), dtype_arg.name)).id


# ---------------------------------------------------------------------------
# R20  sum / max（v0.5，docs/v0.5-reduce.md §2.1——归约 = 分布的边缘化）
# ---------------------------------------------------------------------------


def _reduce_axis_arg(ctx, e, args, kwargs):
    """axis 实参：位置实参或唯一 kwargs `axis=`；INT 字面量 / constexpr 特化值。"""
    kw = dict(kwargs)
    if set(kw) - {"axis"}:
        raise err(e.loc, "E13", "tila.sum/tila.max accept only the 'axis' keyword argument")
    axis_expr = None
    if len(args) == 2 and not kw:
        axis_expr = args[1]
    elif len(args) == 1 and set(kw) == {"axis"}:
        axis_expr = kw["axis"]
    if axis_expr is None:
        raise err(e.loc, "E13", "tila.sum/tila.max take (tile, axis) — axis is an "
                                "integer literal 0 or 1, positional or axis=…")
    if isinstance(axis_expr, t.IntLit):
        return axis_expr.value
    if isinstance(axis_expr, t.NameRef) and axis_expr.name in ctx.env.constexprs:
        return ctx.env.constexprs[axis_expr.name]
    raise err(axis_expr.loc, "E13", "reduction axis must be a compile-time constant "
                                    "(literal or constexpr value) in {0, 1}")


def check_reduce(ctx, e: t.Call, op: str):
    if not e.args:
        raise err(e.loc, "E13", "tila.sum/tila.max take (tile, axis)")
    axis = _reduce_axis_arg(ctx, e, e.args, e.kwargs)
    if axis not in (0, 1):
        raise err(e.loc, "E13", f"reduction axis {axis} is not supported "
                                f"(drop-axis reductions over {0, 1} as of v0.5)")
    x_ty, x_id = ctx.infer(e.args[0])
    if not isinstance(x_ty, TileType):
        raise err(e.args[0].loc, "E07", f"tila.{op} operates on a Tile",
                  _type_note("operand", x_ty))
    if len(x_ty.shape) != 2:
        raise err(e.loc, "E21",
                  f"reductions are rank-2 in v0.5 (drop-axis only; full-reduce "
                  f"to Scalar is deferred) — operand rank is {len(x_ty.shape)}",
                  _type_note("operand", x_ty), subcode="ReduceRank")
    # 能力门与普通运算同源：sum 走 '+' 能力、max 走比较能力（fp8 storage-only、
    # bool 无序 → E16）
    ok = ("+" in dt.arith_ops(x_ty.dtype)) if op == "sum" \
        else ("<" in dt.cmp_ops(x_ty.dtype))
    if not ok:
        raise err(e.loc, "E16", f"'{op}' is not in the capability set of dtype "
                                f"{x_ty.dtype} (reductions follow the same capability "
                                f"table as arithmetic/comparisons; fp8 is storage-only, "
                                f"bool masks are not ordered values)")
    out_layout = lay.marginal(x_ty.layout, x_ty.shape, axis, e.loc, f"'{op}'")
    out_shape = x_ty.shape[:axis] + x_ty.shape[axis + 1:]
    ty = TileType(x_ty.dtype, out_shape, out_layout)
    return ty, ctx.emit(tir.TReduce(ctx.next_id("t"), ty, None, x_ty.layout,
                                    op, x_id, axis)).id


# ---------------------------------------------------------------------------
# R21  exp / exp2 / sqrt / abs（v0.5，docs/v0.5-reduce.md §2.2——律 L8 记法）
# ---------------------------------------------------------------------------


def check_elem(ctx, e: t.Call, op: str):
    if len(e.args) != 1 or e.kwargs:
        raise err(e.loc, "E13", f"tila.{op} takes exactly one positional argument: "
                                f"tila.{op}(tile)")
    x_ty, x_id = ctx.infer(e.args[0])
    if not isinstance(x_ty, TileType):
        raise err(e.args[0].loc, "E07", f"tila.{op} operates on a Tile",
                  _type_note("operand", x_ty))
    if not dt.unary_ok(x_ty.dtype, op):
        domain = "float" if op != "abs" else "int or float"
        raise err(e.loc, "E16", f"'{op}' is not in the capability set of dtype "
                                f"{x_ty.dtype} (elementwise math requires {domain} "
                                f"element dtypes; cast to tila.float32 first if "
                                f"intended)")
    # 律 L8（记法 ElemL(L) ≡ L）：逐元素一元不改变分布——不物化任何新 term
    ty = TileType(x_ty.dtype, x_ty.shape, x_ty.layout)
    return ty, ctx.emit(tir.TElem(ctx.next_id("t"), ty, None, x_ty.layout,
                                  op, x_id)).id


# ---------------------------------------------------------------------------
# R22  where（v0.5，docs/v0.5-reduce.md §2.3——R14 的 n 元推广）
# ---------------------------------------------------------------------------


def check_where(ctx, e: t.Call):
    if len(e.args) != 3 or e.kwargs:
        raise err(e.loc, "E13", "tila.where takes exactly three positional arguments: "
                                "tila.where(cond, a, b)")
    cond_expr, a_expr, b_expr = e.args
    c_ty, c_id = ctx.infer(cond_expr)
    if not isinstance(c_ty, TileType) or c_ty.dtype != "bool":
        raise err(cond_expr.loc, "E02", "where condition must be Tile[bool] "
                                        "(produced by comparisons)",
                  _type_note("cond", c_ty))

    sides = []
    for expr in (a_expr, b_expr):
        ty, oid = ctx.infer(expr)
        if isinstance(ty, TileType):
            sides.append(("tile", expr, ty, oid))
        elif isinstance(expr, (t.IntLit, t.FloatLit)):
            sides.append(("lit", expr, ty, oid))
        else:
            raise err(expr.loc, "E13", "where branches must be Tiles or literals "
                                       "(scalar names have no broadcast select "
                                       "meaning; bind to a name or use a literal)",
                      _type_note("branch", ty))
    if sides[0][0] == "lit" and sides[1][0] == "lit":
        raise err(e.loc, "E13", "where branches cannot both be literals — at least "
                                "one Tile is needed to fix shape and dtype")

    # dtype：Tile 侧严格相等（E02）；字面量侧按语境定型（R24——类别匹配，
    # 与 R4/R12 同款：float 字面量配浮点类、int 字面量配整数类）
    tile_sides = [s for s in sides if s[0] == "tile"]
    dtype = tile_sides[0][2].dtype
    for _, expr, ty, _ in tile_sides[1:]:
        if ty.dtype != dtype:
            raise err(e.loc, "E02", "where branch dtypes must match exactly",
                      _type_note("a-branch" if expr is a_expr else "b-branch", ty),
                      f"expected element dtype {dtype} (no implicit conversion; "
                      f"write tila.cast(branch, tila.{dt.TILA_SURFACE_NAME.get(dtype, dtype)}))")
    for kind, expr, _, _ in sides:
        if kind == "lit":
            ctx._literal_category(expr, dtype, "where")
    if dt.is_storage_only(dtype):
        raise err(e.loc, "E16", f"fp8 dtypes are storage-only: no arithmetic, no "
                                f"selects (element dtype {dtype}); hint: "
                                f"tila.cast(x, tila.float16) first")

    # shape / layout：三方广播 + merge_nontrivial（pairwise 折叠严格退化为
    # R14 的 broadcast_layout；双侧非平凡 → equiv，否则 E05）；字面量侧
    # 不携带 shape/分布，不参与折叠
    shape = c_ty.shape
    layout = c_ty.layout
    for _, expr, ty, _ in tile_sides:
        new_shape = bcast.require_broadcast(shape, ty.shape, expr.loc, "'where'")
        layout = lay.broadcast_layout(layout, shape, ty.layout, ty.shape,
                                      expr.loc, "'where'")
        shape = new_shape
    ty = TileType(dtype, shape, layout)
    return ty, ctx.emit(tir.TWhere(ctx.next_id("t"), ty, None, layout,
                                   c_id, sides[0][3], sides[1][3])).id


# ---------------------------------------------------------------------------
# R23  num_programs（v0.5，docs/v0.5-reduce.md §2.4——program-context query）
# ---------------------------------------------------------------------------


def check_num_programs(ctx, e: t.Call):
    if len(e.args) != 1 or e.kwargs:
        raise err(e.loc, "E13", "tila.num_programs takes exactly one positional "
                                "argument: tila.num_programs(0)")
    arg = e.args[0]
    axis = None
    if isinstance(arg, t.IntLit):
        axis = arg.value
    elif isinstance(arg, t.NameRef) and arg.name in ctx.env.constexprs:
        axis = ctx.env.constexprs[arg.name]
    else:
        raise err(arg.loc, "E13", "num_programs axis must be a compile-time constant "
                                  "(literal or constexpr value)")
    if axis not in (0, 1):
        raise err(e.loc, "E13", f"num_programs axis {axis} is not supported "
                                f"(axes {{0, 1}} as of the v0.2 2D preview)")
    ty = ScalarType("i32")
    return ty, ctx.emit(tir.TNumPrograms(ctx.next_id("t"), ty, None, None, axis)).id
