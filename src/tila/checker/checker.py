"""check_kernel：规则驱动的 tila_ast → TIR 通道（docs/type-checker.md）。

内部固定两个 phase：typing（产出 typed TIR 指令序列）→ launch analysis
（在 typed TIR 上推导 LaunchPlan，E17）。所有失败抛 TilaError，绝不以
Python 原生异常泄漏。检查顺序固定：dtype → shape → layout。
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

from .. import tir
from ..ast import nodes as t
from ..diagnostics import Loc, err
from ..launch import analysis as launch
from ..types import (
    ARITH_OPS,
    CMP_OPS,
    LOGIC_OPS,
    ROW_MAJOR,
    AddressType,
    BcastScalarL,
    BufferType,
    Const,
    JoinL,
    ScalarType,
    Symbol,
    TileType,
    UnitType,
    is_float,
    is_int,
    normalize,
    shape_str,
    type_str,
)
from ..types import dtype as dt
from ..types.layout import CastL, layout_str
from . import arithmetic as arith
from . import builtin
from . import broadcast as bcast
from . import layout as lay

RESERVED_NAMES = ("tila", "tl", "triton")


def _note(what: str, ty, loc: Loc = None) -> str:
    s = f"{what} : {type_str(ty)}"
    if loc is not None:
        s += f"  (defined at line {loc.line})"
    return s


def _surface(dt_name: str) -> str:
    return f"tila.{dt.TILA_SURFACE_NAME.get(dt_name, dt_name)}"


@dataclasses.dataclass
class Binding:
    tila_type: object
    loc: Loc
    id: str


@dataclasses.dataclass
class Env:
    buffers: Dict[str, BufferType]
    syms: Dict[str, str]                       # 符号维名 → 名（值即名字）
    constexprs: Dict[str, int]                 # 已特化的值
    scalar_params: Dict[str, str]              # 未注解参数名 → "i32"
    locals: Dict[str, Binding]


class Checker:
    def __init__(self, kernel: t.KernelDef, overrides: Dict[str, int]):
        self.kernel = kernel
        self.overrides = dict(overrides)
        self.ops: list = []
        self.env = Env({}, {}, {}, {}, {})
        self._counters = {"t": 0, "r": 0, "m": 0, "p": 0}
        self._sym_ids: Dict[str, str] = {}
        self._constexpr_ids: Dict[str, str] = {}
        self._param_locs: Dict[str, Loc] = {}
        self.explain: list = []

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def next_id(self, prefix: str) -> str:
        """求值期的临时 id：带 '#' 前缀（绝不与用户名/参数名冲突）。

        终态命名在 _finalize_ids 后处理中完成：匿名指令按最终出现顺序编号
        %t0/%r0/%m0/%p0…（被赋值改名的 op 不再占用匿名编号）。
        """
        i = self._counters[prefix]
        self._counters[prefix] = i + 1
        return f"#{prefix}{i}"

    def _finalize_ids(self) -> None:
        """把 '#' 临时 id 重写为最终 id，并同步重写全部操作数引用。"""
        taken = {p.name for p in self.kernel.params}
        taken |= {op.src_name for op in self.ops if op.src_name is not None}
        counters = {"t": 0, "r": 0, "m": 0, "p": 0}
        remap: Dict[str, str] = {}

        def anon_prefix(op) -> str:
            if isinstance(op, tir.TArange):
                return "r"
            if isinstance(op, tir.TAddPtr):
                return "p"
            if isinstance(op, (tir.TCmp, tir.TLogic)):
                return "m"
            return "t"

        def r(oid):
            return remap.get(oid, oid)

        new_ops = []
        for op in self.ops:
            if op.id is not None and op.id.startswith("#"):
                prefix = anon_prefix(op)
                nid = f"{prefix}{counters[prefix]}"
                while nid in taken:
                    counters[prefix] += 1
                    nid = f"{prefix}{counters[prefix]}"
                counters[prefix] += 1
                remap[op.id] = nid
                op = dataclasses.replace(op, id=nid)
            if isinstance(op, (tir.TArith, tir.TCmp, tir.TLogic)):
                op = dataclasses.replace(op, lhs=r(op.lhs), rhs=r(op.rhs))
            elif isinstance(op, tir.TAddPtr):
                op = dataclasses.replace(op, offs=r(op.offs))
            elif isinstance(op, tir.TLoad):
                op = dataclasses.replace(
                    op, ptr=r(op.ptr),
                    mask=r(op.mask) if op.mask is not None else None,
                    other=r(op.other) if op.other is not None else None)
            elif isinstance(op, tir.TStore):
                op = dataclasses.replace(
                    op, ptr=r(op.ptr), value=r(op.value),
                    mask=r(op.mask) if op.mask is not None else None)
            elif isinstance(op, tir.TCast):
                op = dataclasses.replace(op, operand=r(op.operand))
            elif isinstance(op, tir.TExpandDim):
                op = dataclasses.replace(op, tile=r(op.tile))
            elif isinstance(op, tir.TDot):
                op = dataclasses.replace(op, lhs=r(op.lhs), rhs=r(op.rhs))
            new_ops.append(op)
        self.ops = new_ops
        for b in self.env.locals.values():
            if b.id in remap:
                b.id = remap[b.id]

    def emit(self, op):
        self.ops.append(op)
        return op

    def _is_bound(self, name: str) -> bool:
        e = self.env
        return name in e.locals or name in e.buffers or name in e.syms \
            or name in e.constexprs or name in e.scalar_params

    # ------------------------------------------------------------------
    # 签名检查（§5）
    # ------------------------------------------------------------------

    def check_signature(self) -> tuple:
        k = self.kernel
        for name in self.overrides:
            if name not in {p.name for p in k.params if isinstance(p.ann, t.ConstexprAnn)}:
                raise err(k.loc, "E13", f"constexpr override for '{name}' does not match "
                                        f"any `tila.constexpr` parameter")

        buffer_order, sym_order, scalar_order, constexpr_order = [], [], [], []
        for p in k.params:
            self._param_locs[p.name] = p.loc
            if p.ann is None:
                self.env.scalar_params[p.name] = "i32"
                scalar_order.append(p)
                continue
            if isinstance(p.ann, t.ConstexprAnn):
                value = self.overrides.get(p.name, p.default)
                if value is None:
                    raise err(p.loc, "E15", f"constexpr parameter '{p.name}' has no integer "
                                            f"literal default and no call-site override")
                self.env.constexprs[p.name] = value
                constexpr_order.append(p)
                continue
            # TensorAnn
            ann = p.ann
            if not dt.can_be_tensor_element(ann.dtype):
                raise err(p.loc, "E12", f"bool cannot be a Buffer element dtype "
                                        f"(masks come from comparisons; bool storage has no use)")
            if len(ann.dims) >= 3:
                raise err(p.loc, "E12", f"rank >= 3 annotations are rejected "
                                        f"(rank <= 2 as of the v0.2 2D preview; "
                                        f"see docs/v0.2-preview-2d.md)")
            shape = tuple(
                Const(d.value) if isinstance(d, t.StaticDim) else Symbol(d.name)
                for d in ann.dims
            )
            self.env.buffers[p.name] = BufferType(ann.dtype, shape, ROW_MAJOR)
            buffer_order.append(p)
            for d in ann.dims:
                if isinstance(d, t.SymDim) and d.name not in self.env.syms:
                    self.env.syms[d.name] = d.name
                    sym_order.append(d.name)
                    self._param_locs.setdefault(d.name, d.loc)

        all_param_names = [p.name for p in k.params]
        for s in self.env.syms:
            if s in all_param_names:
                raise err(k.loc, "E12", f"symbol dim '{s}' conflicts with a parameter name "
                                        f"(it is auto-bound as a runtime scalar parameter)")
            if s in RESERVED_NAMES:
                raise err(k.loc, "E12", f"symbol dim '{s}' is a reserved name "
                                        f"(tila/tl/triton would shadow generated imports)")

        params = [tir.TParam(p.name, "buffer", self.env.buffers[p.name], None)
                  for p in buffer_order]
        params += [tir.TParam(s, "sym", ScalarType("i32"), None) for s in sym_order]
        params += [tir.TParam(p.name, "scalar", ScalarType("i32"), None)
                   for p in scalar_order]
        params += [tir.TParam(p.name, "constexpr", ScalarType("i32"),
                              p.default if p.default is not None
                              else self.overrides.get(p.name))
                   for p in constexpr_order]
        return tuple(params)

    # ------------------------------------------------------------------
    # 语句检查（§6）
    # ------------------------------------------------------------------

    def check_stmt(self, stmt: t.Stmt) -> None:
        if isinstance(stmt, t.Assign):
            self.check_assign(stmt)
        else:
            self.check_expr_stmt(stmt)

    def check_assign(self, stmt: t.Assign) -> None:
        if stmt.target in RESERVED_NAMES:
            raise err(stmt.loc, "E12", f"assignment target '{stmt.target}' is reserved "
                                       f"(tila/tl/triton would shadow generated module imports)")
        if self._is_bound(stmt.target):
            where = self.env.locals.get(stmt.target)
            defined = f"  (defined at line {where.loc.line})" if where else ""
            raise err(stmt.loc, "E14",
                      f"name '{stmt.target}' is already bound{defined}; "
                      f"Tila is single-assignment — bind a new name instead "
                      f"(params and symbol dims count as bindings)")
        before = len(self.ops)
        ty, oid = self.infer(stmt.value)
        if isinstance(ty, UnitType):
            raise err(stmt.loc, "E08", "store returns () and cannot be bound to a name")
        if len(self.ops) > before:
            root = self.ops[-1]
            self.ops[-1] = dataclasses.replace(root, id=stmt.target, src_name=stmt.target)
            oid = stmt.target
        self.env.locals[stmt.target] = Binding(ty, stmt.loc, oid)

    def check_expr_stmt(self, stmt: t.ExprStmt) -> None:
        if stmt.call.intrinsic != "store":
            raise err(stmt.loc, "E08", "the only legal expression statement is a "
                                      "tila.store(ptr, value, mask=...) call")
        builtin.check_store(self, stmt.call)

    # ------------------------------------------------------------------
    # 表达式推导（§4，规则驱动分派）
    # ------------------------------------------------------------------

    def infer(self, e: t.Expr) -> Tuple[object, str]:
        rule = RULES[type(e)]
        return rule(self, e)

    def check_int_lit(self, e: t.IntLit):
        ty = ScalarType("i32")
        return ty, self.emit(tir.TConstInt(self.next_id("t"), ty, None, None, e.value)).id

    def check_float_lit(self, e: t.FloatLit):
        ty = ScalarType("f32")
        return ty, self.emit(tir.TConstFloat(self.next_id("t"), ty, None, None, e.value)).id

    def check_name(self, e: t.NameRef):
        env = self.env
        if e.name in env.locals:
            b = env.locals[e.name]
            return b.tila_type, b.id
        if e.name in env.buffers:
            return env.buffers[e.name], e.name
        if e.name in env.syms:
            return ScalarType("i32"), self._sym_ref(e)
        if e.name in env.constexprs:
            return ScalarType("i32"), self._constexpr_ref(e)
        if e.name in env.scalar_params:
            return ScalarType(env.scalar_params[e.name]), e.name
        raise err(e.loc, "E01", f"name '{e.name}' is not bound (define before use)")

    def _sym_ref(self, e: t.NameRef) -> str:
        if e.name not in self._sym_ids:
            self.emit(tir.TSymRef(e.name, ScalarType("i32"), None, None, e.name))
            self._sym_ids[e.name] = e.name
        return self._sym_ids[e.name]

    def _constexpr_ref(self, e: t.NameRef) -> str:
        if e.name not in self._constexpr_ids:
            self.emit(tir.TConstParamRef(e.name, ScalarType("i32"), None, None, e.name))
            self._constexpr_ids[e.name] = e.name
        return self._constexpr_ids[e.name]

    def check_dtype_ref(self, e: t.DTypeRef):
        raise err(e.loc, "E13", "a dtype reference is only legal as the 2nd argument of "
                                "tila.cast(x, tila.<dt>) or inside tila.Tensor[...]")

    def check_call(self, e: t.Call):
        if e.intrinsic == "store":
            raise err(e.loc, "E08", "store returns () — it may only appear as an "
                                    "expression statement, not in a value position")
        if e.intrinsic == "program_id":
            return builtin.check_program_id(self, e)
        if e.intrinsic == "arange":
            return builtin.check_arange(self, e)
        if e.intrinsic == "load":
            return builtin.check_load(self, e)
        if e.intrinsic == "cast":
            return builtin.check_cast_call(self, e)
        if e.intrinsic == "expand_dim":
            return builtin.check_expand_dim(self, e)
        if e.intrinsic == "dot":
            return builtin.check_dot(self, e)
        raise AssertionError(f"unknown intrinsic {e.intrinsic}")  # pragma: no cover

    def check_cast(self, e: t.Cast):
        return builtin.check_cast(self, e.loc, e.operand, e.dtype)

    # ------------------------------------------------------------------
    # 二元运算（§4.1 分派矩阵）
    # ------------------------------------------------------------------

    def check_binop(self, e: t.BinOp):
        lt, lid = self.infer(e.lhs)
        rt, rid = self.infer(e.rhs)
        op = e.op

        # R7：Buffer + Tile[i32] → Address（表面语言中指针出现的唯一形式）
        if op == "+" and isinstance(lt, BufferType):
            if not isinstance(rt, TileType):
                raise err(e.loc, "E07", "the right operand of `buffer + ...` must be an "
                                        "i32 index tile (e.g. pid * BLOCK + tila.arange(...))",
                          _note("rhs", rt))
            if rt.dtype != "i32":
                raise err(e.loc, "E02", "the index tile of `buffer + index` must have "
                                        "dtype i32",
                          _note("rhs", rt))
            if len(lt.shape) != len(rt.shape):
                raise err(e.loc, "E04",
                          f"rank mismatch in `buffer + index`: buffer rank "
                          f"{len(lt.shape)} vs index tile rank {len(rt.shape)} "
                          f"(rank(idx) must equal rank(buffer))",
                          _note("buffer", lt), _note("index", rt))
            ty = AddressType(lt.dtype, rt.shape, normalize(rt.layout))
            return ty, self.emit(
                tir.TAddPtr(self.next_id("p"), ty, None, rt.layout, lid, rid)).id

        for side, ty in (("lhs", lt), ("rhs", rt)):
            if isinstance(ty, (BufferType, AddressType)):
                raise err(e.loc, "E07",
                          f"invalid operand for '{op}': a buffer may only appear as the "
                          f"left operand of `buffer + index_tile`; an address is only "
                          f"consumed by tila.load / tila.store",
                          _note(side, ty))

        if op in ARITH_OPS:
            return self._arith(e, op, lt, lid, rt, rid)
        if op in CMP_OPS:
            return self._cmp(e, op, lt, lid, rt, rid)
        if op in LOGIC_OPS:
            return self._logic(e, op, lt, lid, rt, rid)
        raise err(e.loc, "E11", f"operator '{op}' is only legal inside compile-time "
                                f"constant expressions (arange bounds)")  # pragma: no cover

    # ---- R3 / R4 / R5 ----
    def _arith(self, e, op, lt, lid, rt, rid):
        lhs_lit = isinstance(e.lhs, (t.IntLit, t.FloatLit))
        rhs_lit = isinstance(e.rhs, (t.IntLit, t.FloatLit))

        if isinstance(lt, ScalarType) and isinstance(rt, ScalarType):
            # R3 标量 ⊕ 标量（constexpr 与整数字面量自动提升为 i32）
            if lhs_lit and rhs_lit:
                both_int = isinstance(e.lhs, t.IntLit) and isinstance(e.rhs, t.IntLit)
                both_float = isinstance(e.lhs, t.FloatLit) and isinstance(e.rhs, t.FloatLit)
                if not (both_int or both_float):
                    raise err(e.loc, "E02", f"cannot apply '{op}' to literals of "
                                            f"different categories (i32 vs f32)")
                rdt = "i32" if both_int else "f32"
            elif rhs_lit:
                self._literal_category(e.rhs, lt.dtype, op)
                rdt = lt.dtype
            elif lhs_lit:
                self._literal_category(e.lhs, rt.dtype, op)
                rdt = rt.dtype
            elif lt.dtype != rt.dtype:
                raise err(e.loc, "E02", f"cannot apply '{op}' to operands",
                          _note("lhs", lt), _note("rhs", rt),
                          f"expected: same arithmetic dtype; hint: use "
                          f"tila.cast(rhs, {_surface(lt.dtype)})")
            else:
                rdt = lt.dtype
            self._capability_arith(e, op, rdt)
            ty = ScalarType(rdt)
            return ty, self.emit(tir.TArith(self.next_id("t"), ty, None, None,
                                            op, lid, rid)).id

        if isinstance(lt, TileType) and isinstance(rt, TileType):
            # R5 Tile ⊕ Tile：dtype → shape → layout
            if lt.dtype != rt.dtype:
                raise err(e.loc, "E02", f"cannot apply '{op}' to operands",
                          _note("lhs", lt, self._def_loc(e.lhs)),
                          _note("rhs", rt, self._def_loc(e.rhs)),
                          f"expected: same arithmetic dtype; hint: use "
                          f"tila.cast(rhs, {_surface(lt.dtype)})")
            self._capability_arith(e, op, lt.dtype)
            shape = bcast.require_broadcast(lt.shape, rt.shape, e.loc, f"'{op}'")
            origin, layout = self._shape_layout(lt, rt, shape, e.loc, op)
            ty = TileType(lt.dtype, shape, layout)
            self._log_judgment(e, op, lt, rt, shape)
            return ty, self.emit(tir.TArith(self.next_id("t"), ty, None, origin,
                                            op, lid, rid)).id

        tile, scalar, tile_id, scalar_id, lit_node, tile_side = (
            (lt, rt, lid, rid, e.rhs, "lhs") if isinstance(lt, TileType)
            else (rt, lt, rid, lid, e.lhs, "rhs")
        )
        if isinstance(tile, TileType) and isinstance(scalar, ScalarType):
            # R4 标量 ⊕ Tile（layout 由律 L2 保持）
            if isinstance(lit_node, (t.IntLit, t.FloatLit)):
                self._literal_category(lit_node, tile.dtype, op)
            elif scalar.dtype != tile.dtype:
                raise err(e.loc, "E02", f"cannot apply '{op}' to operands",
                          _note("lhs", lt, self._def_loc(e.lhs)),
                          _note("rhs", rt, self._def_loc(e.rhs)),
                          f"expected: same arithmetic dtype; hint: use "
                          f"tila.cast(x, {_surface(tile.dtype)}) or a category-matching "
                          f"literal (1.0 for float element dtypes)")
            self._capability_arith(e, op, tile.dtype)
            origin = BcastScalarL(tile.layout)
            ty = TileType(tile.dtype, tile.shape, normalize(origin))
            return ty, self.emit(tir.TArith(self.next_id("t"), ty, None, origin,
                                            op, lid, rid)).id

        raise err(e.loc, "E07", f"operand kinds for '{op}' have no defined combination",
                  _note("lhs", lt), _note("rhs", rt))

    # ---- R6 ----
    def _cmp(self, e, op, lt, lid, rt, rid):
        if not isinstance(lt, TileType):
            raise err(e.loc, "E07", "comparisons must have a Tile on the left "
                                    "(there is no scalar boolean type; `scalar < scalar` "
                                    "has no defined meaning)",
                      _note("lhs", lt))
        if not arith.capability_cmp(lt.dtype, op):
            raise err(e.loc, "E16", self._capability_msg(lt.dtype, op))
        if isinstance(rt, ScalarType):
            if isinstance(e.rhs, (t.IntLit, t.FloatLit)):
                self._literal_category(e.rhs, lt.dtype, op)
            elif rt.dtype != lt.dtype:
                raise err(e.loc, "E02", f"cannot compare {lt.dtype} with {rt.dtype}",
                          _note("lhs", lt), _note("rhs", rt),
                          f"expected: same dtype (or a category-matching literal)")
            origin = JoinL(lt.layout, BcastScalarL(lt.layout))
            ty = TileType("bool", lt.shape, normalize(origin))
            return ty, self.emit(tir.TCmp(self.next_id("m"), ty, None, origin,
                                          op, lid, rid)).id
        if isinstance(rt, TileType):
            if rt.dtype != lt.dtype:
                raise err(e.loc, "E02", f"cannot compare {lt.dtype} with {rt.dtype}",
                          _note("lhs", lt), _note("rhs", rt),
                          f"expected: same dtype")
            shape = bcast.require_broadcast(lt.shape, rt.shape, e.loc, f"'{op}'")
            origin, layout = self._shape_layout(lt, rt, shape, e.loc, op)
            ty = TileType("bool", shape, layout)
            self._log_judgment(e, op, lt, rt, shape)
            return ty, self.emit(tir.TCmp(self.next_id("m"), ty, None, origin,
                                          op, lid, rid)).id
        raise err(e.loc, "E07", f"invalid comparison operand",
                  _note("rhs", rt))

    # ---- R10 ----
    def _logic(self, e, op, lt, lid, rt, rid):
        if not (isinstance(lt, TileType) and isinstance(rt, TileType)):
            raise err(e.loc, "E07", f"'{op}' operates on Tile[bool] masks",
                      _note("lhs", lt), _note("rhs", rt))
        if lt.dtype != rt.dtype:
            raise err(e.loc, "E02", f"cannot apply '{op}' to {lt.dtype} and {rt.dtype}",
                      _note("lhs", lt), _note("rhs", rt))
        if lt.dtype != "bool":
            raise err(e.loc, "E16", f"'{op}' requires bool masks, but element dtype is "
                                    f"{lt.dtype} (masks come from comparisons)")
        shape = bcast.require_broadcast(lt.shape, rt.shape, e.loc, f"'{op}'")
        origin, layout = self._shape_layout(lt, rt, shape, e.loc, op)
        ty = TileType("bool", shape, layout)
        return ty, self.emit(tir.TLogic(self.next_id("m"), ty, None, origin,
                                        op, lid, rid)).id

    # ------------------------------------------------------------------
    # 能力门与字面量类别（R12）
    # ------------------------------------------------------------------

    def _capability_arith(self, e, op, d: str) -> None:
        if not arith.capability_arith(d, op):
            raise err(e.loc, "E16", self._capability_msg(d, op))

    def _capability_msg(self, d: str, op: str) -> str:
        if dt.is_storage_only(d):
            return (f"fp8 dtypes are storage-only: no arithmetic, no comparisons "
                    f"(element dtype {d}); hint: tila.cast(x, tila.float16) + "
                    f"tila.cast(y, tila.float16)")
        if d == "bool":
            return f"bool has no arithmetic and supports only '==' and '!=' (got '{op}')"
        if op == "/" and dt.is_int(d):
            return (f"integer '/' is not in the v0.1 capability set (no implicit float "
                    f"promotion); cast an operand to a float dtype first")
        if op in ("&", "|") and dt.is_int(d):
            return f"'{op}' requires bool masks, but element dtype is {d}"
        return f"operator '{op}' is not in the capability set of dtype {d}"

    def _literal_category(self, lit: t.Expr, d: str, op: str) -> None:
        """R12：INT 配整数类、FLOAT 配浮点类；类别不一致 → E02。"""
        is_float_lit = isinstance(lit, t.FloatLit)
        if is_float(d) != is_float_lit:
            want = "a FLOAT literal (e.g. 1.0)" if is_float(d) else "an INT literal (e.g. 1)"
            raise err(lit.loc, "E02",
                      f"literal category mismatch for element dtype {d} in '{op}'",
                      f"element dtype {d} is {'float' if is_float(d) else 'integer'}-kind; "
                      f"write {want}")

    def _shape_layout(self, lt, rt, shape, loc, op):
        """tile ⊕ tile 的 layout 推导：严格同形走 v0.1 规则（等价 + JoinL 擦除）；
        发生 size-1 广播走 R14（v0.2 预览）——逐轴取非平凡侧的分布构成 Product。"""
        if shape == lt.shape and shape == rt.shape:
            lay.require_equiv(lt.layout, rt.layout, loc, f"'{op}'")
            origin = JoinL(lt.layout, rt.layout)
            return origin, normalize(origin)
        layout = lay.broadcast_layout(lt.layout, lt.shape, rt.layout, rt.shape,
                                      loc, f"'{op}'")
        return layout, layout

    def _def_loc(self, e: t.Expr) -> Optional[Loc]:
        if isinstance(e, t.NameRef):
            b = self.env.locals.get(e.name)
            if b is not None:
                return b.loc
        return None

    def _log_judgment(self, e, op, lt, rt, result_shape) -> None:
        """判定日志（--explain；非规范，仅供诊断）。"""
        self.explain.append({
            "loc": e.loc,
            "op": op,
            "dtype": f"{lt.dtype} == {rt.dtype}",
            "shape": f"{shape_str(lt.shape)} ⊗ {shape_str(rt.shape)} = "
                     f"{shape_str(result_shape)}",
            "layout": f"{layout_str(normalize(lt.layout))} ~ "
                      f"{layout_str(normalize(rt.layout))}",
        })

    # ------------------------------------------------------------------
    # constexpr 求值器（检查期求值；发射不折叠——模型 B）
    # ------------------------------------------------------------------

    def eval_constexpr(self, e: t.Expr) -> Tuple[object, int]:
        if isinstance(e, t.IntLit):
            return e.value, e.value
        if isinstance(e, t.NameRef):
            if e.name in self.env.constexprs:
                return e.name, self.env.constexprs[e.name]
            if self._is_bound(e.name):
                raise err(e.loc, "E10", f"'{e.name}' is a runtime value; this position "
                                        f"needs a compile-time constant expression "
                                        f"(literals / constexpr names / integer arithmetic)")
            raise err(e.loc, "E01", f"name '{e.name}' is not bound")
        if isinstance(e, t.BinOp) and e.op in ("+", "-", "*", "//"):
            lhs_tree, lv = self.eval_constexpr(e.lhs)
            rhs_tree, rv = self.eval_constexpr(e.rhs)
            try:
                value = {"+": lambda: lv + rv, "-": lambda: lv - rv,
                         "*": lambda: lv * rv, "//": lambda: lv // rv}[e.op]()
            except ZeroDivisionError:
                raise err(e.loc, "E10", "constant expression evaluation failed "
                                        "(division by zero)")
            return tir.CBinOp(e.op, lhs_tree, rhs_tree), value
        raise err(e.loc, "E10", "this position needs a compile-time constant expression "
                                "(literals / constexpr names / integer arithmetic)")


RULES = {
    t.IntLit: Checker.check_int_lit,
    t.FloatLit: Checker.check_float_lit,
    t.NameRef: Checker.check_name,
    t.BinOp: Checker.check_binop,
    t.Call: Checker.check_call,
    t.Cast: Checker.check_cast,
    t.DTypeRef: Checker.check_dtype_ref,
}


def check_kernel(kernel_def: t.KernelDef,
                 constexpr_overrides: Optional[Dict[str, int]] = None) -> tir.TKernel:
    """tila_ast → TIR；失败抛 TilaError。launch analysis 是 typing 之后的独立 phase。"""
    return check_kernel_verbose(kernel_def, constexpr_overrides)[0]


def check_kernel_verbose(kernel_def: t.KernelDef,
                         constexpr_overrides: Optional[Dict[str, int]] = None):
    """同 check_kernel，但一并返回 (TKernel, explain 判定日志)。"""
    ck = Checker(kernel_def, constexpr_overrides or {})
    params = ck.check_signature()
    for stmt in kernel_def.body:
        ck.check_stmt(stmt)
    ck.emit(tir.TReturn(None, None, None, None))
    ck._finalize_ids()
    plan = launch.plan_launch(ck.ops, ck.env, ck.kernel.loc)
    return tir.TKernel(kernel_def.name, params, tuple(ck.ops), plan, kernel_def.loc), ck.explain
