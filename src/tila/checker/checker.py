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
from ..diagnostics import Loc, TilaError, err
from ..launch import analysis as launch
from ..types import (
    ARITH_OPS,
    CMP_OPS,
    LOGIC_OPS,
    ROW_MAJOR,
    AddressType,
    BufferType,
    Const,
    ScalarType,
    Symbol,
    TileType,
    UnitType,
    is_float,
    is_int,
    shape_str,
    type_str,
)
from ..types import dtype as dt
from ..types.dist import NODIST, Alpha, Seed, dist_str, normalize_dist
from ..types.memory import Strided
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


@dataclasses.dataclass(frozen=True)
class Binding:
    tila_type: object
    loc: Loc
    id: str


@dataclasses.dataclass(frozen=True)
class SymBinding:
    """符号表条目（v0.3 评审 §9/§11 采纳）：Dim 符号与 Memory(stride) 符号共享
    同一 infrastructure——都是"运行期需物化的编译期符号"；kind 记录来源类别，
    供诊断与将来的 assume/约束系统使用。stride = N（维符号复用）不产生新条目：
    那是 symbol identity relation，不是两个符号。"""

    kind: str        # "dim" | "stride"
    buffer: str      # 首次绑定来源：buffer 名
    axis: int        # 对应 shape[axis] 或 stride[axis]


@dataclasses.dataclass
class Env:
    buffers: Dict[str, BufferType]
    syms: Dict[str, SymBinding]              # 符号名 → 绑定（Dim / Memory 同表）
    constexprs: Dict[str, int]               # 已特化的值
    scalar_params: Dict[str, str]            # 未注解参数名 → "i32"
    locals: Dict[str, Binding]


@dataclasses.dataclass
class LoopCtx:
    """单个 for 的检查语境（v0.4-kloop §2.4 三条围栏的载体）。

    snapshot 是循环入口的 locals 快照（外层名只读 + 累加器种子判定）；
    carried 记录被本循环 update 的累加器的当前体内绑定（φ → next 链）；
    phis 按 update/read 首次出现序登记 (名, φ id, 入口 id)，循环收尾时
    构造 TPhi 前插到 body 顶部。

    v0.6b（docs/v0.6b-flash.md §3 B）：
    - states  = 累加状态机 [kind, dist, seed_value]（kind ∈ {"seed","mat"}）；
    - alpha_constraints = StateRead 的布局元变量约束 (name, L, loc, what)；
    - lax     = 第一遍宽松 typing（读累加器给 Alpha 占位类型）；
    - exit_dists = 第一遍收尾解出的各累加器出口分布（第二遍读类型用）。
    """

    snapshot: Dict[str, Binding]
    carried: Dict[str, Binding] = dataclasses.field(default_factory=dict)
    phis: list = dataclasses.field(default_factory=list)
    states: Dict[str, list] = dataclasses.field(default_factory=dict)
    alpha_constraints: list = dataclasses.field(default_factory=list)
    exit_dists: Dict[str, object] = dataclasses.field(default_factory=dict)
    lax: bool = False


def _collect_update_targets(stmts) -> set:
    """kernel 全体中 update 目标名集合：累加器认定的第二条件（zeros/full
    播种 + 被 update）。覆盖 '+='、'*=' 与循环体内的 '='（handoff 候选）。"""
    out = set()
    for s in stmts:
        if isinstance(s, t.AugAssign):
            out.add(s.target)
        elif isinstance(s, t.For):
            out |= _collect_update_targets(s.body)
            out |= {a.target for a in s.body if isinstance(a, t.Assign)}
    return out


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
        # v0.4 循环语境（docs/v0.4-kloop.md §2.3/§2.4）
        self._loop_stack: list = []
        self._aug_targets: set = set()
        self._accumulators: set = set()      # zeros/full 播种且被 update 的名字
        self._seed_values: Dict[str, object] = {}   # 累加器种子值（构造点记录；
                                                    # 值属于指令、状态属于类型态）
        self._accum_counters: Dict[str, int] = {}   # '{acc}.next{N}' 命名
        self._loop_counters: Dict[str, int] = {}    # '{acc}.loop{N}' 命名
        self._out_of_scope: Dict[str, Loc] = {}     # 循环局部出作用域记录

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
        """把 '#' 临时 id 重写为最终 id，并同步重写全部操作数引用
        （v0.4 起递归进入 TFor.body；# 命名的匿名空间跨层唯一）。"""
        taken = {p.name for p in self.kernel.params}

        def collect_names(ops_seq) -> None:
            for op in ops_seq:
                if op.src_name is not None:
                    taken.add(op.src_name)
                if isinstance(op, tir.TFor):
                    collect_names(op.body)

        collect_names(self.ops)
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

        def rewrite(op):
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
                op = dataclasses.replace(op, coords=tuple(r(c) for c in op.coords))
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
            elif isinstance(op, tir.TMaximum):
                op = dataclasses.replace(op, lhs=r(op.lhs), rhs=r(op.rhs))
            elif isinstance(op, tir.TDot):
                op = dataclasses.replace(op, lhs=r(op.lhs), rhs=r(op.rhs))
            elif isinstance(op, tir.TPhi):
                op = dataclasses.replace(op, pre=r(op.pre), back=r(op.back))
            elif isinstance(op, tir.TReduce):
                op = dataclasses.replace(op, tile=r(op.tile))
            elif isinstance(op, tir.TElem):
                op = dataclasses.replace(op, operand=r(op.operand))
            elif isinstance(op, tir.TWhere):
                op = dataclasses.replace(op, cond=r(op.cond), a=r(op.a), b=r(op.b))
            elif isinstance(op, tir.TFor):
                op = dataclasses.replace(op, end=r(op.end),
                                         body=tuple(rewrite(b) for b in op.body))
            return op

        self.ops = [rewrite(op) for op in self.ops]
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
            mem = ROW_MAJOR
            if ann.strides is not None:
                mem = Strided(tuple(
                    Const(d.value) if isinstance(d, t.StaticDim) else Symbol(d.name)
                    for d in ann.strides
                ))
            self.env.buffers[p.name] = BufferType(ann.dtype, shape, mem)
            buffer_order.append(p)
            for axis, d in enumerate(ann.dims):
                if isinstance(d, t.SymDim) and d.name not in self.env.syms:
                    self.env.syms[d.name] = SymBinding("dim", p.name, axis)
                    sym_order.append(d.name)
                    self._param_locs.setdefault(d.name, d.loc)
            for axis, d in enumerate(ann.strides or ()):
                if isinstance(d, t.SymDim) and d.name not in self.env.syms:
                    self.env.syms[d.name] = SymBinding("stride", p.name, axis)
                    sym_order.append(d.name)
                    self._param_locs.setdefault(d.name, d.loc)

        all_param_names = [p.name for p in k.params]
        for s, sb in self.env.syms.items():
            if s in all_param_names:
                raise err(k.loc, "E12", f"symbol '{s}' ({sb.kind}) conflicts with a "
                                        f"parameter name "
                                        f"(it is auto-bound as a runtime scalar parameter)")
            if s in RESERVED_NAMES:
                raise err(k.loc, "E12", f"symbol '{s}' ({sb.kind}) is a reserved name "
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
        elif isinstance(stmt, t.AugAssign):
            self.check_augassign(stmt)
        elif isinstance(stmt, t.For):
            self.check_for(stmt)
        else:
            self.check_expr_stmt(stmt)

    def check_assign(self, stmt: t.Assign) -> None:
        if stmt.target in RESERVED_NAMES:
            raise err(stmt.loc, "E12", f"assignment target '{stmt.target}' is reserved "
                                       f"(tila/tl/triton would shadow generated module imports)")
        # v0.6b handoff（HandoffUpdate）：循环体内对累加器赋值 = 状态交接
        # （semantic move of state），不是普通 SSA 定义（docs/v0.6-attention
        # §1.4）——检查在 _handoff_update；普通 Assign 保持"SSA 定义"唯一含义
        if self._loop_stack and stmt.target in self._accumulators:
            self._handoff_update(stmt)
            return
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
        # v0.4/v0.6b：zeros/full 播种 + 预扫描确认被 update（'+='、'*=' 或
        # 循环内 '=' 交接）→ 认定为累加器（R19 前提）。**播种必须在循环外**
        # （顶层语句序保证 snapshot 可达；循环内 zeros 只是普通循环局部，
        # 审核修复 P0-①：否则体内读走 StateRead 分支时 snapshot 缺名崩溃）。
        # 种子值记入状态机（值属于指令、状态属于类型态）。
        if not self._loop_stack \
                and isinstance(stmt.value, t.Call) \
                and stmt.value.intrinsic in ("zeros", "full") \
                and stmt.target in self._aug_targets:
            self._accumulators.add(stmt.target)
            if stmt.value.intrinsic == "zeros":
                self._seed_values[stmt.target] = 0
            else:
                self._seed_values[stmt.target] = stmt.value.args[1].value

    # ------------------------------------------------------------------
    # v0.4/v0.6b 循环与累加器（docs/v0.4-kloop.md §2.2–§2.4、
    # v0.6-attention §1.4/§2.6——AccumulatorUpdate 状态机）
    # ------------------------------------------------------------------

    def _ensure_state(self, ctx: LoopCtx, name: str, entry_ty) -> list:
        """惰性初始化累加状态 [kind, dist, seed_value]：入口 binding 的 dist
        是 Seed(shape) → "seed" 态（种子值来自构造点）；否则 "mat" 态。"""
        if name in ctx.states:
            return ctx.states[name]
        d = normalize_dist(entry_ty.dist)
        if isinstance(d, Seed):
            ctx.states[name] = ["seed", d, self._seed_values.get(name, 0)]
        else:  # pragma: no cover - 累加器必由 zeros/full 播种（围栏保证）
            ctx.states[name] = ["mat", d, None]
        return ctx.states[name]

    def _phi_of(self, ctx: LoopCtx, name: str, entry: Binding) -> str:
        """累加器 φ id（入口值引用）：update 或 StateRead 都可能先于对方
        出现——首次引用创建，其后复用（docs/v0.6-attention §2.6）。"""
        for (n, pid, _) in ctx.phis:
            if n == name:
                return pid
        self._loop_counters[name] = self._loop_counters.get(name, 0) + 1
        m = self._loop_counters[name]
        pid = f"{name}.loop" if m == 1 else f"{name}.loop{m}"
        ctx.phis.append((name, pid, entry.id))
        return pid

    def _accumulator_guard(self, stmt: t.Stmt, ctx: LoopCtx, name: str,
                           want: str) -> Binding:
        """update 前提（AccumForm 门）：循环体内 + zeros/full 播种累加器。"""
        entry = ctx.snapshot.get(name)
        if entry is None or name not in self._accumulators:
            why = ("it is not bound before the loop" if entry is None else
                   f"it was not seeded by tila.zeros/tila.full "
                   f"({type_str(entry.tila_type)})")
            raise err(stmt.loc, "E20",
                      f"'{name}' cannot be updated with '{want}' ({why}); only "
                      f"accumulators seeded with acc = tila.zeros(...) before the "
                      f"loop may be updated inside it", subcode="AccumForm")
        return entry

    def check_augassign(self, stmt: t.AugAssign) -> None:
        if not self._loop_stack:
            raise err(stmt.loc, "E20",
                      f"'{stmt.op}=' is only legal inside a loop body, on a "
                      f"tila.zeros-seeded accumulator", subcode="AccumForm")
        ctx = self._loop_stack[-1]
        name = stmt.target
        entry = self._accumulator_guard(stmt, ctx, name, stmt.op)
        ty, val_id = self.infer(stmt.value)
        if not isinstance(ty, TileType):
            raise err(stmt.loc, "E07", "an accumulator update needs a Tile on the "
                                       "right of '+='/ '*='", _note("value", ty))
        if stmt.op == "+":
            self._reduce_update(stmt, ctx, name, entry, ty, val_id)
        else:
            self._scale_update(stmt, ctx, name, entry, ty, val_id)

    def _reduce_update(self, stmt, ctx: LoopCtx, name: str, entry: Binding,
                       ty, val_id: str) -> None:
        """ReduceUpdate（'+='，L7 语义）：Seed(V) × L → Materialized(L)；
        Materialized(L) × L' → Materialized(L) require equiv（E05）。"""
        cur = ctx.carried.get(name, entry)
        cur_ty = cur.tila_type
        if ty.dtype != cur_ty.dtype:
            raise err(stmt.loc, "E02",
                      f"cannot accumulate Tile<{ty.dtype}> into accumulator "
                      f"'{name}' of dtype {cur_ty.dtype}",
                      _note("value", ty), _note("accumulator", cur_ty),
                      f"expected: same dtype; hint: use "
                      f"tila.cast(value, {_surface(cur_ty.dtype)})")
        self._capability_arith(stmt, "+", cur_ty.dtype)
        bcast.require_same_shape(ty.shape, cur_ty.shape, stmt.loc, f"'+=' on '{name}'")
        state = self._ensure_state(ctx, name, entry.tila_type)
        if state[0] == "seed":
            # 状态机：Seed + Tile[D] → Materialized(D)——常量分布是 join 单位元。
            # 宽松遍中 D 可能是 α（RHS 直接读另一累加器）：原样物化，出口
            # 求解时按 α := exits[另一累加器] 解链（审核修复 P0-③）
            origin = ty.dist
            result_d = normalize_dist(origin)
            state[:] = ["mat", result_d, state[2]]
        elif self._in_lax_pass() and self._has_alpha(state[1], ty.dist):
            # 物化态 × α（或反向）：记约束延迟到出口求解——立即 require_equiv
            # 会对 α 假失败（审核修复 P0-③）
            ctx.alpha_constraints.append((name, ty.dist, stmt.loc,
                                          f"'+=' on '{name}'"))
            origin = ty.dist
            result_d = state[1]
        else:
            lay.require_equiv(cur_ty.dist, ty.dist, stmt.loc, f"'+=' on '{name}'")
            origin = ty.dist
            result_d = state[1]
        result_ty = TileType(cur_ty.dtype, cur_ty.shape, result_d)
        self._accum_counters[name] = self._accum_counters.get(name, 0) + 1
        n = self._accum_counters[name]
        nid = f"{name}.next" if n == 1 else f"{name}.next{n}"
        if name not in ctx.carried:
            lhs_id = self._phi_of(ctx, name, entry)
        else:
            lhs_id = ctx.carried[name].id
        self.emit(tir.TArith(nid, result_ty, name, origin, "+", lhs_id, val_id))
        b = Binding(result_ty, stmt.loc, nid)
        self.env.locals[name] = b
        ctx.carried[name] = b

    def _scale_update(self, stmt, ctx: LoopCtx, name: str, entry: Binding,
                      ty, val_id: str) -> None:
        """ScaleUpdate（'*='，评审 B-⑨）：Seed(0) × X → Seed(0)（mul-identity
        特例）；Seed(V≠0) × X → E20 MulSeed；Materialized(L) × X →
        Materialized(L)（乘数 R-ST 透明——乘数分布免检）。

        与 '+=' 的 shape 前提刻意不对称：乘数可广播进累加器（⊗ 良式且
        结果 shape = acc 的 shape，不放大）；聚合 vs 逐行重定标
        （docs/v0.6-attention.md §1.4）。
        """
        cur = ctx.carried.get(name, entry)
        cur_ty = cur.tila_type
        if ty.dtype != cur_ty.dtype:
            raise err(stmt.loc, "E02",
                      f"cannot rescale accumulator '{name}' of dtype "
                      f"{cur_ty.dtype} by Tile<{ty.dtype}>",
                      _note("value", ty), _note("accumulator", cur_ty),
                      f"expected: same dtype; hint: use "
                      f"tila.cast(value, {_surface(cur_ty.dtype)})")
        self._capability_arith(stmt, "*", cur_ty.dtype)
        out_shape = bcast.require_broadcast(ty.shape, cur_ty.shape, stmt.loc,
                                            f"'*=' on '{name}'")
        if out_shape != cur_ty.shape:
            raise err(stmt.loc, "E03",
                      f"'*=' may only rescale: the multiplier must broadcast "
                      f"into the accumulator shape {shape_str(cur_ty.shape)} "
                      f"(it cannot grow the accumulator)",
                      _note("value", ty), _note("accumulator", cur_ty))
        state = self._ensure_state(ctx, name, entry.tila_type)
        if state[0] == "seed":
            if state[2] == 0:
                # mul-identity 特例：0·x = 0——值层事实，不是布局律（§2.4）
                result_d = normalize_dist(cur_ty.dist)
            else:
                raise err(stmt.loc, "E20",
                          "rescale-from-seed requires a zero-valued seed "
                          "(0·x = 0); nonzero full seeds are not multipliable "
                          "in v0.6b", subcode="MulSeed")
        else:
            result_d = state[1]          # 乘数 R-ST 读透明：分布不变
        result_ty = TileType(cur_ty.dtype, cur_ty.shape, result_d)
        self._accum_counters[name] = self._accum_counters.get(name, 0) + 1
        n = self._accum_counters[name]
        nid = f"{name}.next" if n == 1 else f"{name}.next{n}"
        if name not in ctx.carried:
            lhs_id = self._phi_of(ctx, name, entry)
        else:
            lhs_id = ctx.carried[name].id
        self.emit(tir.TArith(nid, result_ty, name, result_d, "*", lhs_id, val_id))
        b = Binding(result_ty, stmt.loc, nid)
        self.env.locals[name] = b
        ctx.carried[name] = b

    def _handoff_update(self, stmt: t.Assign) -> None:
        """HandoffUpdate（'='，评审 B-⑯）：acc = x——x 必须是当前迭代已定义
        且非累加器的单一局部名（semantic move of state，无额外计算）。
        φ 的 back 边直接指向 x 的 id；无新指令。"""
        ctx = self._loop_stack[-1]
        name = stmt.target
        entry = self._accumulator_guard(stmt, ctx, name, "=")
        rhs = stmt.value
        if not isinstance(rhs, t.NameRef):
            raise err(stmt.loc, "E20",
                      "handoff is `acc = <local>` on an accumulator inside a "
                      "loop — a semantic move of state, not a computation "
                      f"(`{name} = <expression>` is a computation; decompose "
                      f"into updates)", subcode="HandoffForm")
        if rhs.name == name:
            raise err(stmt.loc, "E20",
                      "handoff to itself (`{0} = {0}`) is a no-op: it neither "
                      "moves state nor computes — bind the new value under a "
                      "fresh name and hand it off".format(name),
                      subcode="HandoffForm")
        if rhs.name in self._accumulators:
            raise err(stmt.loc, "E20",
                      f"handoff RHS must be a non-accumulator local "
                      f"(`{name} = {rhs.name}` crosses accumulator states; "
                      f"give the value a fresh name first)",
                      subcode="HandoffForm")
        x_b = self.env.locals.get(rhs.name)
        if x_b is None or rhs.name in ctx.snapshot:
            # "当前迭代内已定义"（评审 B-⑯）：交接的是本迭代算出的值；
            # 循环前局部是常量携带，不是 state move（审核收紧 P2）
            raise err(rhs.loc, "E20",
                      f"handoff RHS '{rhs.name}' must be a local defined "
                      f"inside the loop body (the current iteration's value); "
                      f"pre-loop names are not state being moved",
                      subcode="HandoffForm")
        x_ty = x_b.tila_type
        if not isinstance(x_ty, TileType):
            raise err(rhs.loc, "E07", "handoff value must be a Tile",
                      _note("value", x_ty))
        cur = ctx.carried.get(name, entry)
        cur_ty = cur.tila_type
        if x_ty.dtype != cur_ty.dtype:
            raise err(stmt.loc, "E02",
                      f"cannot hand off Tile<{x_ty.dtype}> to accumulator "
                      f"'{name}' of dtype {cur_ty.dtype}",
                      _note("value", x_ty), _note("accumulator", cur_ty))
        bcast.require_same_shape(x_ty.shape, cur_ty.shape, stmt.loc,
                                 f"handoff to '{name}'")
        state = self._ensure_state(ctx, name, entry.tila_type)
        if state[0] == "seed":
            state[:] = ["mat", normalize_dist(x_ty.dist), state[2]]
        elif self._in_lax_pass() and self._has_alpha(state[1], x_ty.dist):
            # 物化态交接 α 值：记约束延迟判等（审核修复 P0-③，与 '+=' 同款）
            ctx.alpha_constraints.append((name, x_ty.dist, stmt.loc,
                                          f"handoff to '{name}'"))
        else:
            lay.require_equiv(cur_ty.dist, x_ty.dist, stmt.loc,
                              f"handoff to '{name}'")
        # 无指令：φ back 边 = x 的 id；名字绑定直通交接值
        if name not in ctx.carried:
            self._phi_of(ctx, name, entry)   # 首现 update：登记 φ（back 已就位）
        b = Binding(x_ty, stmt.loc, x_b.id)
        self.env.locals[name] = b
        ctx.carried[name] = b

    def check_for(self, stmt: t.For) -> None:
        if self._loop_stack:
            raise err(stmt.loc, "E20", "nested loops are not supported in v0.4 "
                                      "(one level of tila.range loops only)",
                      subcode="NestedLoop")
        if stmt.target in RESERVED_NAMES:
            raise err(stmt.loc, "E12", f"loop variable '{stmt.target}' is reserved "
                                       f"(tila/tl/triton would shadow generated module imports)")
        if self._is_bound(stmt.target):
            where = self.env.locals.get(stmt.target)
            defined = f"  (defined at line {where.loc.line})" if where else ""
            raise err(stmt.loc, "E14",
                      f"name '{stmt.target}' is already bound{defined}; "
                      f"Tila is single-assignment — bind a new name instead")
        call = stmt.iter
        if call.intrinsic != "range":  # 防御：转换期已保证（E20 NotRange）
            raise err(call.loc, "E20",
                      "the only iterable is tila.range(0, end, step)",
                      subcode="NotRange")
        if len(call.args) != 3 or call.kwargs:
            raise err(call.loc, "E20",
                      "tila.range takes exactly three positional arguments: "
                      "tila.range(0, end, step) — the literal 0, a runtime i32 "
                      "scalar end, and a compile-time constant step >= 1",
                      subcode="RangeForm")
        start, end_expr, step_expr = call.args
        if not isinstance(start, t.IntLit) or start.value != 0:
            raise err(start.loc, "E20", "tila.range start must be the literal 0 "
                                        "(Tila requires tila.range(0, end, step))",
                      subcode="RangeForm")
        end_ty, end_id = self.infer(end_expr)
        if not (isinstance(end_ty, ScalarType) and end_ty.dtype == "i32"):
            raise err(end_expr.loc, "E20",
                      "tila.range end must be a runtime i32 scalar (a symbol dim, "
                      "scalar arithmetic, or an integer literal)",
                      _note("end", end_ty), subcode="RangeForm")
        try:
            step_tree, step_value = self.eval_constexpr(step_expr)
        except TilaError:
            raise err(step_expr.loc, "E20",
                      "tila.range step must be a compile-time constant >= 1 "
                      "(an integer literal or a constexpr name)",
                      subcode="RangeForm") from None
        if step_value < 1:
            raise err(step_expr.loc, "E20",
                      f"tila.range step must be a compile-time constant >= 1 "
                      f"(got {step_value})", subcode="RangeForm")

        # v0.6b StateRead 定型 = 布局元变量 + 出口回填（docs/v0.6-attention
        # §2.6），实现为两遍 typing：第一遍宽松（累加器读给 Alpha 占位、
        # 同形 join 记约束、ops 丢弃）→ 解出口分布并复检约束；第二遍以
        # 真实出口类型正式重推（产物即最终 TIR）。
        outer_ops = self.ops
        self.ops = []
        snapshot = dict(self.env.locals)
        exit_dists = self._pass1_loop_body(stmt, snapshot)
        ctx = self._pass2_loop_body(stmt, snapshot, exit_dists)
        # 逃逸位检查（E20 AccumRead）：循环体 store 的 use-def 闭包不得
        # 触达累加器 φ（StateRead 只许纯值表达式位，含经局部名的传递）
        self._check_escape_reads(stmt, ctx, self.ops)
        body_ops = self.ops
        self.ops = outer_ops

        # 作用域收尾：循环局部（含归纳变量）出作用域；累加器出口绑定回写。
        # 零迭代的出口值由 φ 语义覆盖（interpreter 回填种子；Triton loop-carried 同款）。
        exit_bindings = {name: self.env.locals[name] for name in ctx.carried}
        added = [k for k in self.env.locals if k not in snapshot]
        self.env.locals = dict(snapshot)
        self.env.locals.update(exit_bindings)
        for k in added:
            self._out_of_scope.setdefault(k, stmt.loc)
        phi_ops = []
        for name, phi_id, pre_id in ctx.phis:
            b = ctx.carried[name]
            phi_ops.append(tir.TPhi(phi_id, b.tila_type, name, None, pre_id, b.id))
        self.emit(tir.TFor(stmt.target, ScalarType("i32"), stmt.target, None,
                           end_id, step_tree, tuple(phi_ops) + tuple(body_ops)))

    # ------------------------------------------------------------------
    # v0.6b StateRead 两遍 typing（docs/v0.6-attention.md §2.6）
    # ------------------------------------------------------------------

    def _pass1_loop_body(self, stmt: t.For, snapshot) -> Dict[str, object]:
        """第一遍（宽松）：跑循环体求状态机出口分布与 α 约束；产物丢弃。
        副作用缓存（sym/constexpr id、φ/next 计数器）事后恢复。"""
        saved = (self._sym_ids, self._constexpr_ids,
                 self._accum_counters, self._loop_counters)
        self._sym_ids, self._constexpr_ids = {}, {}
        self._accum_counters, self._loop_counters = {}, {}
        self.ops = []
        ctx = LoopCtx(snapshot=dict(snapshot), lax=True)
        self._loop_stack.append(ctx)
        self.env.locals = dict(snapshot)
        self.env.locals[stmt.target] = Binding(ScalarType("i32"), stmt.loc, stmt.target)
        for s in stmt.body:
            self.check_stmt(s)
        self._loop_stack.pop()
        self.env.locals = dict(snapshot)
        self.ops = []
        (self._sym_ids, self._constexpr_ids,
         self._accum_counters, self._loop_counters) = saved
        return self._solve_exit_dists(ctx)

    def _pass2_loop_body(self, stmt: t.For, snapshot,
                         exit_dists: Dict[str, object]) -> LoopCtx:
        """第二遍（正式）：读累加器用出口真实类型重推；产物 = 最终 TIR。"""
        ctx = LoopCtx(snapshot=dict(snapshot), lax=False)
        ctx.exit_dists = exit_dists
        self._loop_stack.append(ctx)
        self.env.locals = dict(snapshot)
        self.env.locals[stmt.target] = Binding(ScalarType("i32"), stmt.loc, stmt.target)
        for s in stmt.body:
            self.check_stmt(s)
        self._loop_stack.pop()
        return ctx

    @staticmethod
    def _has_alpha(*dists) -> bool:
        """dists 中是否存在 StateRead 元变量（宽松遍专用判据；rank ≤ 2 下
        α 只能作为完整 dist 出现——join 的宽松路径要么返回另一侧、要么
        返回 α 本身，不会嵌进 Product/Slice）。"""
        return any(isinstance(d, Alpha) for d in dists)

    def _solve_exit_dists(self, ctx: LoopCtx) -> Dict[str, object]:
        """α := L_exit 的求解与复检（审核修复 P0-③ 后的完整版）。

        出口分布 = 状态机终态（states[name][1]）。求解三步：
        1. 约束定值：终态是 α 且约束集里有 (name, 具体 L) → α := L；
        2. 跨累加器链：终态是 α(x) → α := exits[x]（x 未 update → 入口
           种子态）——`l += mm` 的 RHS 直接是另一累加器的读时走此步；
        3. 种子回退：仍是 α（自引链——只消费过自己，从未触及具体分布）
           → 入口种子态（L7 语义下 sound）。
        随后对每条约束 require_equiv（约束值里的 α 先解掉；任一侧是种子
        态则恒放行——单位元）。**出口值绝不含 α**（否则第二遍 TIR 类型
        违反"dist 已是正规形式"不变量）。"""
        def seed_exit_of(n):
            entry = ctx.snapshot.get(n)
            return normalize_dist(entry.tila_type.dist) if entry is not None else None

        exits = {name: state[1] for name, state in ctx.states.items()}

        changed = True  # 1. 约束定值
        while changed:
            changed = False
            for name, dist in list(exits.items()):
                if isinstance(dist, Alpha):
                    for (cn, cl, _, _) in ctx.alpha_constraints:
                        if cn == name and not isinstance(cl, Alpha):
                            exits[name] = cl
                            changed = True
                            break

        changed = True  # 2. 跨累加器链（α(x) := exits[x] / x 的入口种子）
        while changed:
            changed = False
            for name, dist in list(exits.items()):
                if isinstance(dist, Alpha):
                    src = exits.get(dist.name)
                    if src is None:
                        src = seed_exit_of(dist.name)
                    if src is not None and not isinstance(src, Alpha):
                        exits[name] = src
                        changed = True

        for name, dist in list(exits.items()):  # 3. 种子回退
            if isinstance(dist, Alpha):
                fallback = seed_exit_of(name)
                if fallback is None:  # 防御：播种围栏保证不可达
                    raise err(ctx.snapshot.get(name).loc if name in ctx.snapshot
                              else Loc(1, 1), "E20",
                              f"accumulator '{name}' is not bound at loop entry",
                              subcode="AccumForm")
                exits[name] = fallback

        for (cn, cl, loc, what) in ctx.alpha_constraints:  # 复检（含读未 update）
            d = exits.get(cn)
            if d is None:
                d = seed_exit_of(cn)
                if d is not None:
                    exits[cn] = d
            if isinstance(cl, Alpha):
                cl = exits.get(cl.name) or seed_exit_of(cl.name)
            if d is None or cl is None:
                continue  # 名字不可达（防御；播种围栏下不应发生）
            if isinstance(d, Seed) or isinstance(cl, Seed):
                continue  # 种子单位元：约束恒放行（L7 语义）
            lay.require_equiv(d, cl, loc, what)
        return exits

    def _check_escape_reads(self, stmt: t.For, ctx: LoopCtx, body_ops) -> None:
        """E20 AccumRead（v0.6b 重新定位）：循环体 store 的 use-def 依赖闭包
        内不得出现累加器 φ id——StateRead 只允许纯值表达式位；传递逃逸
        （x = m; store(..., x)）经 SSA def 链线性追溯同样抓获。"""
        by_id = {op.id: op for op in body_ops if op.id}
        phi_ids = {pid for (_, pid, _) in ctx.phis}

        def operands(op) -> list:
            if isinstance(op, tir.TAddPtr):
                return list(op.coords)
            if isinstance(op, (tir.TArith, tir.TCmp, tir.TLogic, tir.TMaximum)):
                return [op.lhs, op.rhs]
            if isinstance(op, tir.TLoad):
                out = [op.ptr]
                if op.mask is not None:
                    out.append(op.mask)
                if op.other is not None:
                    out.append(op.other)
                return out
            if isinstance(op, tir.TStore):
                out = [op.ptr, op.value]
                if op.mask is not None:
                    out.append(op.mask)
                return out
            if isinstance(op, tir.TCast):
                return [op.operand]
            if isinstance(op, tir.TExpandDim):
                return [op.tile]
            if isinstance(op, tir.TDot):
                return [op.lhs, op.rhs]
            if isinstance(op, tir.TReduce):
                return [op.tile]
            if isinstance(op, tir.TElem):
                return [op.operand]
            if isinstance(op, tir.TWhere):
                return [op.cond, op.a, op.b]
            if isinstance(op, tir.TPhi):
                return [op.pre, op.back]
            if isinstance(op, tir.TFor):
                return [op.end]
            return []

        for op in body_ops:
            if not isinstance(op, tir.TStore):
                continue
            stack = [op.ptr, op.value] + ([op.mask] if op.mask is not None else [])
            seen = set()
            while stack:
                oid = stack.pop()
                if oid in seen:
                    continue
                seen.add(oid)
                if oid in phi_ids:
                    raise err(
                        stmt.loc, "E20",
                        "an accumulator may be read only inside pure value "
                        "expressions in the loop body; escape via store "
                        "arguments (value/mask/coords, directly or through "
                        "bound names) is not allowed — read it after the loop",
                        subcode="AccumRead")
                src = by_id.get(oid)
                if src is None:
                    continue
                stack.extend(operands(src))

    def check_expr_stmt(self, stmt: t.ExprStmt) -> None:
        if stmt.call.intrinsic == "store":
            builtin.check_store(self, stmt.call)
            return
        if stmt.call.intrinsic == "launch_assert":
            self.check_launch_assert(stmt.call)
            return
        raise err(stmt.loc, "E08", "the only legal expression statements are "
                                  "tila.store(ptr, value, mask=...) and "
                                  "tila.launch_assert(host_condition)")

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
        if self._loop_stack and e.name in self._accumulators:
            # v0.6b StateRead（docs/v0.6-attention.md §2.6）：体内读解禁——
            # 读到的 = 当前迭代入口值（φ）。定型 = 布局元变量（第一遍宽松：
            # Alpha 占位 + 同形 join 记约束；第二遍：出口真实类型）。
            # 逃逸位（store 实参/mask/coords 及传递依赖）由循环收尾的
            # use-def 追溯检查（E20 AccumRead）。
            ctx = self._loop_stack[-1]
            entry = ctx.snapshot.get(e.name)
            if entry is None:  # 防御：播种围栏（循环外播种）保证不可达
                raise err(e.loc, "E20",
                          f"accumulator '{e.name}' is not bound at loop entry "
                          f"(accumulators must be seeded before the loop)",
                          subcode="AccumForm")
            if ctx.lax:
                d = Alpha(e.name)
            else:
                d = ctx.exit_dists.get(e.name)
                if d is None:  # 读了但从未 update：入口 = 种子态（单位元）
                    d = normalize_dist(entry.tila_type.dist)
            ty = TileType(entry.tila_type.dtype, entry.tila_type.shape, d)
            return ty, self._phi_of(ctx, e.name, entry)
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
        if e.name in self._out_of_scope:
            raise err(e.loc, "E20",
                      f"name '{e.name}' was defined inside a loop body; loop-local "
                      f"names (including the loop variable) are not visible after "
                      f"the loop", subcode="LoopScope")
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
        if e.intrinsic == "zeros":
            return builtin.check_zeros(self, e)
        if e.intrinsic == "full":
            return builtin.check_full(self, e)
        if e.intrinsic == "maximum":
            return builtin.check_maximum(self, e)
        if e.intrinsic in ("sum", "max"):
            return builtin.check_reduce(self, e, e.intrinsic)
        if e.intrinsic in ("exp", "exp2", "sqrt", "abs", "log2"):
            return builtin.check_elem(self, e, e.intrinsic)
        if e.intrinsic == "where":
            return builtin.check_where(self, e)
        if e.intrinsic == "num_programs":
            return builtin.check_num_programs(self, e)
        if e.intrinsic == "launch_assert":
            raise err(e.loc, "E08", "tila.launch_assert is a statement (host-side "
                                    "launch assertion) — it may only appear as "
                                    "an expression statement, not in a value "
                                    "position")
        if e.intrinsic == "range":
            raise err(e.loc, "E20", "tila.range may only appear as the iterable of "
                                    "a for loop: for k0 in tila.range(0, end, step)",
                      subcode="RangePosition")
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

        # v0.3 定稿：表面语言没有地址算术——Address 完全 compiler-internal，
        # 只由 checker 在 load/store 内部构造（buffer_address）。
        for side, ty in (("lhs", lt), ("rhs", rt)):
            if isinstance(ty, (BufferType, AddressType)):
                raise err(e.loc, "E07",
                          f"invalid operand for '{op}': buffers are accessed with "
                          f"tila.load(buffer, coords) / tila.store(buffer, coords, "
                          f"value); address calculation is compiler-owned "
                          f"(write tila.load(a, (offs,)) — one i32 tile per axis)",
                          _note(side, ty))

        if op in ARITH_OPS:
            return self._arith(e, op, lt, lid, rt, rid)
        if op in CMP_OPS:
            return self._cmp(e, op, lt, lid, rt, rid)
        if op in LOGIC_OPS:
            return self._logic(e, op, lt, lid, rt, rid)
        raise err(e.loc, "E11", f"operator '{op}' is only legal inside compile-time "
                                f"constant expressions (arange bounds)")  # pragma: no cover

    def check_index_tuple(self, e: t.IndexTuple):
        # 防御入口：转换期保证 IndexTuple 只出现在 load/store 的坐标实参位置
        raise err(e.loc, "E19", "a coordinate tuple may only appear as the coordinates "
                                "argument of tila.load(buffer, coords) / "
                                "tila.store(buffer, coords, value)",
                  subcode="CoordinatePosition")

    def check_shape_lit(self, e: t.ShapeLit):
        # 防御入口：转换期保证 ShapeLit 只出现在 zeros 的第 1 位置实参
        raise err(e.loc, "E20", "a shape tuple may only appear as the first "
                                "argument of tila.zeros((BM, BN), tila.float32)",
                  subcode="ZerosForm")

    def buffer_address(self, buf_expr, coords_expr):
        """R8/R9 的寻址前提（v0.3 定稿）：buffer + 坐标元组 → 内部 Address。

        表面语言没有地址算术——Address 由 checker 在 load/store 内部构造，
        用户不可见（docs/v0.3-strides.md §1.2）。坐标逐轴给出，长度 = buffer
        rank；Address shape/dist = infer_coordinate_dist（评审 §32/§33：坐标
        广播积 = n-ary Product，与 R14 同款机制）；线性化由 lowering 按 base
        的 MemoryLayout 发射（§1.3 Address Function）。两套系统正交：分布
        推导完全不看 MemoryLayout。
        """
        bty, bid = self.infer(buf_expr)
        if not isinstance(bty, BufferType):
            raise err(buf_expr.loc, "E07",
                      "the first argument of tila.load/tila.store must be a buffer",
                      _note("operand", bty))
        if not isinstance(coords_expr, t.IndexTuple):
            raise err(coords_expr.loc, "E19",
                      "coordinates must be a tuple literal with one i32 tile per "
                      "axis: tila.load(a, (offs,)) / tila.load(a, (rows, cols))",
                      subcode="CoordinateForm")
        coords = coords_expr.items
        if len(coords) != len(bty.shape):
            name = getattr(buf_expr, "name", "buffer")
            raise err(coords_expr.loc, "E19",
                      f"coordinate tuple has {len(coords)} entries but buffer "
                      f"'{name}' has rank {len(bty.shape)}",
                      _note("buffer", bty), subcode="CoordinateArity")
        coord_ids: list = []
        coord_tys = []
        for c in coords:
            cty, cid = self.infer(c)
            if not isinstance(cty, TileType):
                raise err(c.loc, "E19", "each coordinate must be an i32 index tile "
                                        "(scalar coordinates are not supported in v0.3)",
                          _note("coordinate", cty), subcode="CoordinateKind")
            if cty.dtype != "i32":
                raise err(c.loc, "E02", "addressing coordinates must have dtype i32",
                          _note("coordinate", cty))
            coord_ids.append(cid)
            coord_tys.append((cty.shape, cty.dist, c.loc))
        shape, layout = lay.infer_coordinate_dist(coord_tys, coords_expr.loc,
                                                  "coordinates")
        ty = AddressType(bty.dtype, shape, layout)
        return ty, self.emit(
            tir.TAddPtr(self.next_id("p"), ty, None, layout, bid,
                        tuple(coord_ids))).id

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
            # R4 标量 ⊕ Tile（评审 §25：标量 = NoDist；read_join(owner, NoDist)
            # = owner——标量不携带分布，不决定结果分布）
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
            origin = lay.read_join(tile.dist, tile.shape, NODIST, None,
                                   e.loc, f"'{op}'")
            ty = TileType(tile.dtype, tile.shape, normalize_dist(origin))
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
            origin = lay.read_join(lt.dist, lt.shape, NODIST, None, e.loc,
                                   f"'{op}'")
            ty = TileType("bool", lt.shape, normalize_dist(origin))
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
        """tile ⊕ tile 的分布推导：同形双侧走 strict_join（L5）；发生
        size-1 广播走 read_join（L6）/ 逐轴段联合（R14 的 v0.6a 形式）。
        v0.6b 宽松遍（StateRead α）经 _loose_join 记约束/透明。"""
        if shape == lt.shape and shape == rt.shape:
            if self._in_lax_pass():
                d = self._loose_strict(lt.dist, rt.dist, loc, f"'{op}'")
            else:
                d = lay.strict_join(lt.dist, rt.dist, loc, f"'{op}'")
            return d, d
        d = self._loose_join(lt.dist, lt.shape, rt.dist, rt.shape,
                             loc, f"'{op}'")
        return d, d

    # ---- v0.6b StateRead 宽松遍（第一遍 typing 的 join 路径）----

    def _in_lax_pass(self) -> bool:
        return bool(self._loop_stack) and self._loop_stack[-1].lax

    def _record_alpha(self, alpha: "Alpha", other, loc, what: str) -> None:
        ctx = self._loop_stack[-1]
        ctx.alpha_constraints.append((alpha.name, other, loc, what))

    def _loose_strict(self, d1, d2, loc, what: str):
        """宽松 strict join：Alpha（StateRead）同形记约束（α ~ L）、返回
        另一侧；双 Alpha 自反约束；两侧正常 → 正式 strict_join。"""
        if isinstance(d1, Alpha) and isinstance(d2, Alpha):
            if d1.name != d2.name:
                self._record_alpha(d1, d2, loc, what)
            return d1
        if isinstance(d1, Alpha):
            self._record_alpha(d1, d2, loc, what)
            return normalize_dist(d2)
        if isinstance(d2, Alpha):
            self._record_alpha(d2, d1, loc, what)
            return normalize_dist(d1)
        return lay.strict_join(d1, d2, loc, what)

    def _loose_join(self, d1, s1, d2, s2, loc, what: str):
        """宽松 join_distributions：同形 → _loose_strict；one-sided →
        read_join（形状判定与 dist 无关——Alpha 侧安全透明）；逐轴段联合
        中出现 Alpha → E20 AccumRead（读只能以同形或 one-sided 参与）。"""
        if s1 == s2:
            return self._loose_strict(d1, d2, loc, what)
        o = lay.read_join(d1, s1, d2, s2, loc, what)
        if o is not None:
            return o
        o = lay.read_join(d2, s2, d1, s1, loc, what)
        if o is not None:
            return o
        if isinstance(d1, Alpha) or isinstance(d2, Alpha):
            raise err(loc, "E20",
                      "a StateRead may only join as a same-shape owner or a "
                      "one-sided operand (per-axis mixing is not defined)",
                      subcode="AccumRead")
        return lay.join_distributions(d1, s1, d2, s2, loc, what)

    def check_launch_assert(self, e: t.Call) -> None:
        """tila.launch_assert（v0.6b）：host 侧启动断言——条件由符号维名 /
        constexpr 名 / 字面量经 `+ - * // %` 与 `== !=` 组成（v0.7-A 之前
        运算符集外按形态核验），名字必须已绑定且全为编译期/运行期标量。
        发射 TLaunchAssert（语句级：进 launcher/interpreter，不进 kernel、
        不参与 launch analysis）。"""
        if len(e.args) != 1 or e.kwargs:
            raise err(e.loc, "E13", "tila.launch_assert takes exactly one "
                                    "positional argument: "
                                    "tila.launch_assert(N % BM == 0)")
        if self._loop_stack:
            raise err(e.loc, "E13", "tila.launch_assert must appear at kernel "
                                    "top level (host-side assertion)")
        cond = e.args[0]
        names = self._launch_names(cond, e.loc)
        for n in names:
            if not (n in self.env.syms or n in self.env.constexprs
                    or n in self.env.scalar_params):
                raise err(e.loc, "E13",
                          f"launch_assert may only mention symbol dims, "
                          f"constexpr names and scalar parameters; '{n}' is "
                          f"none of these (buffers and locals are not "
                          f"host-launch quantities)")
        text = self._launch_str(cond)
        self.emit(tir.TLaunchAssert(None, None, None, None, text))

    def _launch_names(self, node, loc: Loc) -> list:
        if isinstance(node, t.NameRef):
            return [node.name]
        if isinstance(node, t.IntLit):
            return []
        if isinstance(node, t.BinOp):
            return self._launch_names(node.lhs, loc) \
                + self._launch_names(node.rhs, loc)
        raise err(loc, "E13", "malformed launch_assert condition")  # pragma: no cover

    def _launch_str(self, node) -> str:
        if isinstance(node, t.NameRef):
            return node.name
        if isinstance(node, t.IntLit):
            return str(node.value)
        return f"{self._launch_str(node.lhs)} {node.op} {self._launch_str(node.rhs)}"

    def _def_loc(self, e: t.Expr) -> Optional[Loc]:
        if isinstance(e, t.NameRef):
            b = self.env.locals.get(e.name)
            if b is not None:
                return b.loc
        return None

    def _log_judgment(self, e, op, lt, rt, result_shape) -> None:
        """判定日志（--explain；非规范，仅供诊断）。宽松遍跳过——两遍
        typing 会让循环体判定重复一次（审核修复 P2）。"""
        if self._in_lax_pass():
            return
        self.explain.append({
            "loc": e.loc,
            "op": op,
            "dtype": f"{lt.dtype} == {rt.dtype}",
            "shape": f"{shape_str(lt.shape)} ⊗ {shape_str(rt.shape)} = "
                     f"{shape_str(result_shape)}",
            "layout": f"{dist_str(normalize_dist(lt.dist))} ~ "
                      f"{dist_str(normalize_dist(rt.dist))}",
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
    t.IndexTuple: Checker.check_index_tuple,
    t.ShapeLit: Checker.check_shape_lit,
}


def check_kernel(kernel_def: t.KernelDef,
                 constexpr_overrides: Optional[Dict[str, int]] = None) -> tir.TKernel:
    """tila_ast → TIR；失败抛 TilaError。launch analysis 是 typing 之后的独立 phase。"""
    return check_kernel_verbose(kernel_def, constexpr_overrides)[0]


def check_kernel_verbose(kernel_def: t.KernelDef,
                         constexpr_overrides: Optional[Dict[str, int]] = None):
    """同 check_kernel，但一并返回 (TKernel, explain 判定日志)。"""
    ck = Checker(kernel_def, constexpr_overrides or {})
    ck._aug_targets = _collect_update_targets(kernel_def.body)
    params = ck.check_signature()
    for stmt in kernel_def.body:
        ck.check_stmt(stmt)
    ck.emit(tir.TReturn(None, None, None, None))
    ck._finalize_ids()
    plan = launch.plan_launch(ck.ops, ck.env, ck.kernel.loc)
    return tir.TKernel(kernel_def.name, params, tuple(ck.ops), plan, kernel_def.loc), ck.explain
