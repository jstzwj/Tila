"""Checker：双向类型推导 + 内建分派 + 控制流规则 + 义务生成
（docs/type-system.md、intrinsics.md、bounds-safety.md）。

单遍产出 TIR。Stage 1（装饰期）在符号层完成全部类型检查与可判定部分的
bounds 预证明；Unknown 义务与延迟 shape 约束留到特化期（runtime）求解。
"""

from __future__ import annotations
from dataclasses import replace
from . import predicates as P

from . import dtypes as D
from .dims import (Cst, DimExpr, FloorDiv, Mod, Sym, canon, equal, free_syms)
from .errors import Loc, TilaError, Warning_
from .facts import (Facts, Obligation, Pred, evaluate_obligation, audit_result_lines,
                    is_nonneg_expr, PROVEN_UNSAFE)
from . import types as TY
from . import tir as T
from .hir import (Assign, BinOp, BoolOp, BufPtr, Call, Cmp, Expand, ExprStmt,
                  For, HDtype, HTuple, If, Kernel, Lit, Name, Return, UnaOp)
from .intrinsics import call_shape_problem, get_intrinsic


class VarInfo:
    __slots__ = ("vtype", "expr", "predicate", "lit", "is_const",
                 "contiguous_span", "maybe_undefined", "tir", "static_variant")

    def __init__(self, vtype=None, expr=None, predicate=None, lit=None,
                 is_const=False, contiguous_span=None, maybe_undefined=False,
                 tir=None, static_variant=None):
        self.vtype = vtype
        self.expr = expr              # DimExpr：i32 值的符号式
        self.predicate = predicate if predicate is not None else P.unknown(
            shape=vtype.dims if isinstance(vtype, (TY.BlockT, TY.MaskT)) else ())
        self.lit = lit                # Python 字面量（语境多态）
        self.is_const = is_const
        self.contiguous_span = contiguous_span
        self.maybe_undefined = maybe_undefined
        self.tir = tir                # 本表达式对应的 TIR 值节点
        self.static_variant = static_variant   # (then_t, else_t)：static-if
        #                                          两分支类型不同的变体名

    def clone(self):
        return VarInfo(self.vtype, self.expr, self.predicate, self.lit,
                       self.is_const, self.contiguous_span,
                       self.maybe_undefined, self.tir, self.static_variant)


def shape_of(v: VarInfo) -> tuple:
    if isinstance(v.vtype, (TY.BlockT, TY.MaskT)):
        return v.vtype.dims
    return ()


def broadcast(s1: tuple, s2: tuple, loc=None, defer=None,
              what: str = "broadcast shapes"):
    """type-system.md §7：右对齐补 1，逐维相等或 size-1。

    defer（§4.3 约束层延迟）：语法不等时，若两侧自由符号全是 Const
    参数，回调 defer(x, y, loc, what) 登记延迟等价约束并乐观取 x，
    特化期（runtime._check_deferred）数值复核；回调拒绝（纯常量不等
    或含运行期符号）或无 defer（非 Checker 语境）⇒ 维持立即
    TILA-SHAPE-003。
    """
    r = max(len(s1), len(s2))
    a = (Cst(1),) * (r - len(s1)) + tuple(s1)
    b = (Cst(1),) * (r - len(s2)) + tuple(s2)
    out = []
    for x, y in zip(a, b):
        if equal(x, y):
            out.append(x)
        elif equal(x, Cst(1)):
            out.append(y)
        elif equal(y, Cst(1)):
            out.append(x)
        elif defer is not None and defer(x, y, loc, what):
            out.append(x)      # 乐观相等（§4.3）：取 x，Stage 2 复核
        else:
            if loc is not None:
                raise TilaError(
                    "TILA-SHAPE-003", "shapes are not broadcast-compatible",
                    loc,
                    [f"    dim pair ({x}, {y}): must be equal or size-1",
                     f"    shape A: ({', '.join(map(str, s1)) or 'scalar'})",
                     f"    shape B: ({', '.join(map(str, s2)) or 'scalar'})"],
                    ["用 [:, None] / ti.expand_dims 显式对齐，或修正 tile 形状"])
            return None
    return tuple(out)


def _lit_default_dtype(v):
    if isinstance(v, bool):
        return D.bool_
    if isinstance(v, float):
        return D.f32
    return D.i32


def check_kernel(kern: Kernel) -> T.TKernel:
    return Checker(kern).run()


class Checker:
    def __init__(self, kern: Kernel):
        self.kern = kern
        self.facts = Facts()
        self.vars: dict[str, VarInfo] = {}
        self.nonneg_syms: set[str] = set()
        self.lane_counter = 0
        self.lane_axes = {}  # symbol -> right-relative tile axes (scalar symbols absent)
        self.value_types = {}
        self.value_defs = {}
        self.execution_context_exact = True
        self.loop_stack: list[dict] = []
        self.static_guards = []
        self.regions: dict[str, TY.RegionId] = {}
        self.tk = T.TKernel(kern.name, [], [], [], [])
        self._collect_params()

    # ------------------------------------------------------------------
    # 参数环境
    # ------------------------------------------------------------------

    def _collect_params(self):
        implicit: list[str] = []
        order: list[tuple[str, str]] = []   # (kind, name)：声明序
        external_regions: list[TY.RegionId] = []
        for parameter_index, p in enumerate(self.kern.params):
            spec = p.spec
            if isinstance(spec, TY.BufferT):
                region_id = TY.BufferRegion(parameter_index, p.name)
                self.regions[p.name] = region_id
                external_regions.append(region_id)
                bound_spec = TY.BufferT(
                    element=spec.element,
                    shape=spec.shape,
                    access=spec.access,
                    alignment=spec.alignment,
                    address_space=spec.address_space,
                    strides=TY.BoundStrides(tuple(
                        Sym(f"{p.name}_stride{i}")
                        for i in range(len(spec.shape)))),
                )
                self.tk.buffers.append(T.TBufferParam(p.name, bound_spec,
                                                      region_id))
                order.append(("buffer", p.name))
                self.vars[p.name] = VarInfo(vtype=bound_spec)
                for d in bound_spec.shape:
                    for s in sorted(free_syms(d)):
                        if s not in implicit:
                            implicit.append(s)
                implicit.extend(st.name for st in bound_spec.strides)
            elif isinstance(spec, TY.PtrT):
                # 裸指针参数（type-system.md §8/§9.3）：进入 TKernel 的
                # ptr_params（launcher 可消费的位置实参）；引用处物化为
                # TBufPtr（与 buf.ptr 同一指针值形态，lowering → {name}_ptr，
                # interp → ("ptr", name, 0)）。Extent 符号与维符号同机制；
                # RegionId 由 checker 从参数来源生成，用户注解不能指定。
                bound_spec = TY.PtrT(
                    element=spec.element,
                    address_space=spec.address_space,
                    access=spec.access,
                    extent=spec.extent,
                    alignment=spec.alignment,
                    region_id=TY.ParamRegion(parameter_index, p.name),
                )
                self.regions[p.name] = bound_spec.region_id
                external_regions.append(bound_spec.region_id)
                self.tk.ptr_params.append(T.TPtrParam(p.name, bound_spec))
                order.append(("ptr", p.name))
                self.vars[p.name] = VarInfo(vtype=bound_spec, expr=Cst(0),
                                            tir=T.TBufPtr(vt=bound_spec,
                                                          buffer=p.name))
                extent_expr = TY.extent_expr(spec.extent)
                if extent_expr is not None:
                    for s in sorted(free_syms(extent_expr)):
                        if s not in implicit:
                            implicit.append(s)
            elif isinstance(spec, TY.ConstT):
                self.tk.consts.append(T.TConstParam(p.name, spec.refinements,
                                                    spec.default, spec.value_kind))
                order.append(("const", p.name))
                if spec.value_kind == "Bool":
                    self.vars[p.name] = VarInfo(vtype=TY.ScalarT(D.bool_),
                        predicate=P.Predicate("const_bool", atom=p.name), is_const=True)
                    continue
                self.vars[p.name] = VarInfo(vtype=TY.ScalarT(D.i32),
                                            expr=Sym(p.name), is_const=True)
                if spec.default is not None:
                    self.facts.num[p.name] = spec.default
                self._register_refinement_facts(p.name, D.i32, spec.refinements)
            elif isinstance(spec, (TY.RefinedScalar, TY.ScalarT)):
                if isinstance(spec, TY.RefinedScalar):
                    dt, refins = spec.dtype, spec.refinements
                else:
                    dt, refins = spec.dtype, ()
                self.tk.scalars.append(T.TScalarParam(p.name, dt, refins))
                order.append(("scalar", p.name))
                self.tk.explicit_scalars.add(p.name)
                self.vars[p.name] = VarInfo(vtype=TY.ScalarT(dt),
                                            expr=Sym(p.name))
                if dt.is_int:
                    self.value_types[p.name] = dt.name
                self._register_refinement_facts(p.name, dt, refins)
            else:
                raise TilaError(
                    "TILA-SYN-015",
                    f"parameter '{p.name}': unsupported annotation", Loc(0))

        named = {s.name for s in self.tk.scalars} | \
                {c.name for c in self.tk.consts}
        bool_names = {c.name for c in self.tk.consts if c.value_kind == "Bool"}
        if bool_names.intersection(implicit):
            raise TilaError("TILA-CONST-001", "Const[bool] cannot be used as a dimension or extent")
        for s in implicit:
            if s in named:
                continue
            self.tk.scalars.append(T.TScalarParam(s, D.i32, ()))
            self.vars[s] = VarInfo(vtype=TY.ScalarT(D.i32), expr=Sym(s))
            if "_stride" not in s:
                self.nonneg_syms.add(s)  # strides may be negative
        self.tk.implicit_names = {s for s in implicit}
        self.tk.param_order = order
        for i, left in enumerate(external_regions):
            for right in external_regions[i + 1:]:
                self.tk.aliases.append(TY.AliasFact(
                    left, right, TY.default_alias_relation(left, right),
                    "distinct external parameters are conservatively MayAlias",
                ))

    def _register_refinement_facts(self, name, dtype, refinements):
        """已由 launch 强制的整数 refinement 可安全进入 Stage-1 事实。"""
        if not dtype.is_int:
            return
        symbol = Sym(name)
        lower = None
        upper = None
        for refinement in refinements:
            if isinstance(refinement, TY.Positive):
                self.facts.add_pred(Pred(">", symbol, Cst(0)))
                lower = 1 if lower is None else max(lower, 1)
            elif isinstance(refinement, TY.NonNegative):
                self.facts.add_pred(Pred(">=", symbol, Cst(0)))
                lower = 0 if lower is None else max(lower, 0)
            elif isinstance(refinement, TY.RangeRefinement):
                self.facts.add_pred(Pred(">=", symbol, Cst(refinement.lo)))
                self.facts.add_pred(Pred("<=", symbol, Cst(refinement.hi)))
                lower = refinement.lo if lower is None else \
                    max(lower, refinement.lo)
                upper = refinement.hi if upper is None else \
                    min(upper, refinement.hi)
            elif isinstance(refinement, TY.MultipleOfRefinement):
                self.facts.add_pred(Pred(
                    "==", Mod(symbol, Cst(refinement.k)), Cst(0)))
            elif isinstance(refinement, TY.PowerOfTwo):
                self.facts.add_pred(Pred(">", symbol, Cst(0)))
                lower = 1 if lower is None else max(lower, 1)
        if lower is not None:
            self.facts.sym_lo[name] = Cst(lower)
            if lower >= 0:
                self.nonneg_syms.add(name)
        if upper is not None:
            self.facts.sym_hi[name] = Cst(upper + 1)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> T.TKernel:
        self.stmts(self.kern.body, self.tk.body)
        proof_facts = self.facts.clone()
        proof_facts.preds = {}  # Only access-point snapshots may supply assumptions.
        from .solver import ProofSession
        proof_session = ProofSession()
        for ob in self.tk.obligations:
            result = evaluate_obligation(ob, proof_facts, self.nonneg_syms, proof_session)
            if result.verdict == PROVEN_UNSAFE:
                raise TilaError(
                    "TILA-BOUNDS-003", "out-of-bounds access is provable",
                    Loc(ob.loc_line), [ob.describe(), *audit_result_lines(ob, result, include_fix=False)],
                    ["修正坐标、补 mask（mask = offs < N），或 tila.unsafe_load"],
                    proof_result=result)
        self.tk.types = {n: v.vtype for n, v in self.vars.items()
                         if v.vtype is not None}
        self.tk.nonneg_syms = set(self.nonneg_syms)
        self.tk.sym_hi = dict(self.facts.sym_hi)
        self.tk.sym_lo = dict(self.facts.sym_lo)
        return self.tk

    def stmts(self, stmts, out) -> bool:
        """检查一个直线语句块。

        返回该块路径是否已终止：裸 return，或（static-resolved 分支 /
        双分支均终止的 if）。终止后的剩余语句不可达——记录 note 并跳过
        （不检查：死代码中的类型错误不报告，第二个 return 也永不检查）。
        """
        for i, s in enumerate(stmts):
            if self.stmt(s, out):
                if i + 1 < len(stmts):
                    what = ("return" if isinstance(s, Return)
                            else "if (both branches return)")
                    self.tk.notes.append(
                        f"line {s.loc.line}: unreachable code after {what} "
                        "skipped")
                return True
        return False

    def stmt(self, s, out):
        if isinstance(s, Assign):
            v = self.synth(s.value, out)
            if v.vtype is None and v.lit is not None:
                v = VarInfo(vtype=TY.ScalarT(_lit_default_dtype(v.lit)),
                            lit=v.lit, is_const=True)
                if isinstance(v.lit, int) and not isinstance(v.lit, bool):
                    v.expr = Cst(v.lit)
            if v.vtype is None:
                raise TilaError("TILA-TYPE-018", "cannot type this expression",
                                s.loc)
            if isinstance(v.vtype, TY.UnitT):
                raise TilaError(
                    "TILA-TYPE-017",
                    "store/assume produce Unit and cannot be assigned", s.loc)
            # int 值统一获得符号身份：加载值/计算值可进入谓词与 assume
            # （refinements.md §5.1：assume 注入的事实作用于这些符号）。
            if v.expr is None:
                dt = self._dtype_of(v)
                if dt is not None and dt.is_int:
                    self.lane_counter += 1
                    v.expr = Sym(f"__v{self.lane_counter}")
                    self.value_types[v.expr.name] = dt.name
                    if shape_of(v):
                        self.lane_axes[v.expr.name] = tuple(range(-len(shape_of(v)), 0))
            # loop-carried 类型稳定（type-system.md §10.2）
            for ctx in self.loop_stack:
                if s.target in ctx and not self._same_type(
                        ctx[s.target].vtype, v.vtype):
                    raise TilaError(
                        "TILA-TYPE-021",
                        f"loop-carried variable '{s.target}' changes type",
                        s.loc,
                        [f"    before loop: {TY.describe(ctx[s.target].vtype)}",
                         f"    in loop:     {TY.describe(v.vtype)}"])
            self.vars[s.target] = VarInfo(v.vtype, v.expr,
                                          v.predicate, v.lit,
                                          v.is_const, v.contiguous_span)
            if v.is_const and self._dtype_of(v) is D.bool_:
                self.vars[s.target].tir = self._operand_of(v, s.value, out)
            out.append(T.TAssign(s.target, self._operand_of(v, s.value, out),
                                 line=s.loc.line))
            if v.contiguous_span is not None:
                self.tk.hints.append((s.target, v.contiguous_span))
                self.tk.hint_origins.setdefault(s.target, set()).add(
                    P.Origin(P.STATIC, s.loc.line,
                             "structural contiguous span; integer launch guards required"))
        elif isinstance(s, ExprStmt):
            if s.value is None:
                return
            v = self.synth(s.value, out)
            if not isinstance(v.vtype, TY.UnitT):
                raise TilaError(
                    "TILA-SYN-026",
                    "expression statements must be store/assume (value unused)",
                    s.loc)
        elif isinstance(s, If):
            return self._if(s, out)
        elif isinstance(s, For):
            return self._for(s, out)
        elif isinstance(s, Return):
            # 裸 return（HIR 保证无值）：提前退出当前 program instance。
            out.append(T.TReturn(line=s.loc.line))
            return True
        else:
            raise TilaError("TILA-SYN-002", "unsupported statement", s.loc)

    # -- If -----------------------------------------------------------------

    def _if(self, s: If, out) -> bool:
        cval = self._const_eval_bool(s.cond)
        if cval is not None:
            self.tk.notes.append(f"line {s.loc.line}: static branch resolved "
                                 f"to {'then' if cval else 'else'}")
            # 实际执行的分支终止 ⇒ if 之后不可达（沿 stmts 上抛）。
            return self.stmts(s.then_body if cval else s.else_body, out)
        cv = self.synth(s.cond, out)
        if isinstance(cv.vtype, TY.MaskT):
            raise TilaError(
                "TILA-TYPE-022",
                "if-condition must be a scalar bool, found a block predicate",
                s.loc, [f"    found: {cv.vtype.describe()}"],
                ["块级条件执行只有 masked load/store 与 ti.where；"
                 "标量化请用 mask.any() / mask.all()"])
        if not (isinstance(cv.vtype, TY.ScalarT) and
                cv.vtype.dtype is D.bool_):
            raise TilaError("TILA-TYPE-019",
                            f"if-condition must be bool, found "
                            f"{TY.describe(cv.vtype)}", s.loc)

        # StaticIf / If 彻底分离（type-system.md §5.2）：条件全部由 Const
        # 参数构成（模块常量已在上方折叠为字面量）⇒ static-deferred if，
        # 两分支入 IR，特化期定值；混入任何运行期值 ⇒ 真 runtime if。
        if self._static_cond_syms(s.cond) is not None:
            return self._static_if(s, cv, out)

        snap_vars = {k: v.clone() for k, v in self.vars.items()}
        snap_facts = self.facts.clone()
        self.facts.path = P.conjunction(snap_facts.path, cv.predicate)
        then_out: list = []
        then_term = self.stmts(s.then_body, then_out)
        then_vars, then_facts = self.vars, self.facts

        self.vars = {k: v.clone() for k, v in snap_vars.items()}
        self.facts = snap_facts.clone()
        self.facts.path = P.conjunction(snap_facts.path, P.negate(cv.predicate))
        else_out: list = []
        else_term = self.stmts(s.else_body, else_out)

        self._join_live_branches(s.loc, snap_vars, then_vars, then_facts,
                                then_term, else_term, cv.predicate)
        out.append(T.TIf(self._operand_of(cv, s.cond, out), then_out,
                         else_out, line=s.loc.line))
        return self._terminated_join(s, then_term, else_term)

    def _terminated_join(self, s, then_term: bool, else_term: bool) -> bool:
        """if 双分支均终止 ⇒ 其后的语句不可达（沿 stmts 上抛终止标记）。"""
        if then_term and else_term:
            self.tk.notes.append(
                f"line {s.loc.line}: both branches of the if return; "
                "code after the if is unreachable and skipped")
            return True
        return False

    def _join_live_branches(self, loc, before, then_vars, then_facts,
                            then_term, else_term, condition, static=False):
        # Only predecessors that reach the continuation participate in a join.
        # In particular, a guard followed by return dominates the continuation.
        if then_term:
            return
        if else_term:
            self.vars, self.facts = then_vars, then_facts
            return
        merge = self._merge_static_branches if static else self._merge_branches
        else_vars = self.vars
        self.vars = merge(loc, before, then_vars, else_vars, condition)
        self.facts = then_facts.intersect(self.facts)
        # Integer phi values have fresh identities, constrained on each edge.
        # This preserves guards on merged indices without leaking either value.
        for name, merged in self.vars.items():
            a, b = then_vars.get(name), else_vars.get(name)
            if (a is not None and b is not None and merged.expr is not None
                    and a.expr is not None and b.expr is not None
                    and canon(a.expr) != canon(b.expr)):
                choice = P.select(condition,
                    P.atom(Pred("==", merged.expr, a.expr)),
                    P.atom(Pred("==", merged.expr, b.expr)))
                self.facts.path = P.conjunction(self.facts.path, choice)

    @staticmethod
    def _join_value_flags(m, a, b):
        m.maybe_undefined = a.maybe_undefined or b.maybe_undefined
        if not (type(a.lit) is type(b.lit) and a.lit == b.lit):
            m.lit = None
        m.is_const = (a.is_const and b.is_const and
                      (m.lit is not None or (a.expr is not None and
                       b.expr is not None and canon(a.expr) == canon(b.expr))))
        if a.is_const and b.is_const and a.predicate is b.predicate:
            m.is_const = True
        if not m.is_const:
            m.tir = None

    def _merge_branches(self, loc, before, a, b, condition):
        out = {}
        for name in sorted(set(before) | set(a) | set(b)):
            va, vb, v0 = a.get(name), b.get(name), before.get(name)
            if va is not None and vb is not None:
                if not self._same_type(va.vtype, vb.vtype):
                    raise TilaError(
                        "TILA-TYPE-020", "branch result types differ", loc,
                        [f"    variable '{name}'",
                         f"    then: {TY.describe(va.vtype)}",
                         f"    else: {TY.describe(vb.vtype)}"])
                m = va.clone()
                self._join_value_flags(m, va, vb)
                m.predicate = P.select(condition, va.predicate, vb.predicate)
                if va.expr is not None and vb.expr is not None and \
                        canon(va.expr) == canon(vb.expr):
                    m.expr = va.expr
                else:
                    m.expr = self._forget_value(m).expr
                    if isinstance(m.expr, Sym):
                        # A phi selects already-typed values; it performs no
                        # arithmetic. Int equality is exact here and avoids an
                        # unnecessary additional BV2Int boundary in SMT.
                        self.value_types.pop(m.expr.name, None)
                m.contiguous_span = (va.contiguous_span
                                     if canon(va.contiguous_span) ==
                                     canon(vb.contiguous_span)
                                     else None) if va.contiguous_span else None
                out[name] = m
            elif va is not None or vb is not None:
                v = va or vb
                if v0 is None:
                    m = v.clone()
                    m.maybe_undefined = True
                    out[name] = m
                else:
                    if not self._same_type(v.vtype, v0.vtype):
                        raise TilaError(
                            "TILA-TYPE-020", "branch result types differ", loc,
                            [f"    variable '{name}'",
                             f"    assigned: {TY.describe(v.vtype)}",
                             f"    before:   {TY.describe(v0.vtype)}"])
                    m = v0.clone()
                    m.expr, m.predicate, m.contiguous_span = None, P.unknown(shape=shape_of(m)), None
                    out[name] = m
            else:
                out[name] = v0.clone()
        return out

    # -- StaticIf（Const 参数条件，docs/type-system.md §5.2）-----------------

    def _static_if(self, s: If, cv: VarInfo, out):
        """static-deferred if：条件纯由 Const 参数构成，装饰期不可定值。

        两分支在克隆作用域内独立检查（同 runtime-if），但合并放宽：
        Const 相关的形状差异允许（实际执行分支特化期唯一），dtype/rank/
        运行期形状差异 → static-variant（名字可存在，使用点报错）。
        TStaticIf 同时保留两分支，特化期由 Const 数值定值。
        """
        self.tk.notes.append(f"line {s.loc.line}: static-if on Const "
                             f"parameters (resolved at specialization)")
        guard = self._operand_of(cv, s.cond, out)
        snap_vars = {k: v.clone() for k, v in self.vars.items()}
        snap_facts = self.facts.clone()
        self.facts.path = P.conjunction(snap_facts.path, cv.predicate)
        then_out: list = []
        self.static_guards.append((guard, True))
        then_term = self.stmts(s.then_body, then_out)
        self.static_guards.pop()
        then_vars, then_facts = self.vars, self.facts

        self.vars = {k: v.clone() for k, v in snap_vars.items()}
        self.facts = snap_facts.clone()
        self.facts.path = P.conjunction(snap_facts.path, P.negate(cv.predicate))
        else_out: list = []
        self.static_guards.append((guard, False))
        else_term = self.stmts(s.else_body, else_out)
        self.static_guards.pop()

        self._join_live_branches(s.loc, snap_vars, then_vars, then_facts,
                                then_term, else_term, cv.predicate, static=True)
        out.append(T.TStaticIf(self._operand_of(cv, s.cond, out), then_out,
                               else_out, line=s.loc.line))
        # 特化期分支唯一：双分支均终止 ⇒ 之后不可达（同 runtime-if）。
        return self._terminated_join(s, then_term, else_term)

    def _merge_static_branches(self, loc, before, a, b, condition):
        """static-if 分支合并（放宽版）。

        - 两侧同类型（_same_type）或"特化相容"（_static_same_type：
          dtype/rank 相同且逐维 equal 或两侧均为纯 Const 符号式）→ 正常
          合并（谓词按条件选择保存、expr 仅 canon 相等保留，否则换保守
          代理符号——不携带区间事实，只支撑下游谓词直证）。
        - 两侧类型不相容 → static-variant：名字存在，使用点 TILA-TYPE-020。
        - 单侧赋值 → 同 runtime-if（maybe_undefined）；与 if 前类型不相容
          时同样 static-variant。
        """
        out = {}
        for name in sorted(set(before) | set(a) | set(b)):
            va, vb, v0 = a.get(name), b.get(name), before.get(name)
            if va is not None and vb is not None:
                if self._same_type(va.vtype, vb.vtype) or \
                        self._static_same_type(va.vtype, vb.vtype):
                    m = va.clone()     # 代表类型取 then 侧（特化期分支唯一）
                    self._join_value_flags(m, va, vb)
                    m.predicate = P.select(condition, va.predicate, vb.predicate)
                    if va.expr is not None and vb.expr is not None:
                        if canon(va.expr) == canon(vb.expr):
                            m.expr = va.expr
                        else:
                            self.lane_counter += 1
                            m.expr = Sym(f"__sif{self.lane_counter}")
                            if (is_nonneg_expr(va.expr, self.nonneg_syms) and
                                    is_nonneg_expr(vb.expr, self.nonneg_syms)):
                                self.nonneg_syms.add(m.expr.name)
                            if shape_of(m):
                                self.lane_axes[m.expr.name] = tuple(range(-len(shape_of(m)), 0))
                    else:
                        m.expr = None
                    m.contiguous_span = (va.contiguous_span
                                         if canon(va.contiguous_span) ==
                                         canon(vb.contiguous_span)
                                         else None) if va.contiguous_span \
                        else None
                    if not (type(va.lit) is type(vb.lit) and
                            va.lit == vb.lit):
                        m.lit = None       # 分支相关字面量不可作为已知值
                    out[name] = m
                else:
                    m = va.clone()
                    m.static_variant = (va.vtype, vb.vtype)
                    m.expr, m.predicate, m.contiguous_span, m.lit = \
                        None, P.unknown(shape=shape_of(m)), None, None
                    out[name] = m
            elif va is not None or vb is not None:
                v = va or vb
                if v0 is None:
                    m = v.clone()
                    m.maybe_undefined = True
                    out[name] = m
                else:
                    if self._same_type(v.vtype, v0.vtype) or \
                            self._static_same_type(v.vtype, v0.vtype):
                        m = v0.clone()
                        m.expr, m.predicate, m.contiguous_span = None, P.unknown(shape=shape_of(m)), None
                        out[name] = m
                    else:
                        m = v.clone()
                        m.static_variant = (v.vtype, v0.vtype)
                        m.expr, m.predicate, m.contiguous_span, m.lit = \
                            None, P.unknown(shape=shape_of(m)), None, None
                        out[name] = m
            else:
                out[name] = v0.clone()
        return out

    def _static_same_type(self, a, b) -> bool:
        """static-if 特化相容类型判定：dtype/元素与 rank 相同，且逐维
        equal 或两侧维均为纯 Const 符号式（特化期各自 concrete，实际
        执行的分支唯一——形状随 Const 特化而定，由后端在 trace/解释时
        以具体形状强制）。运行期符号维（N 等）不享受放宽。"""
        const_names = {c.name for c in self.tk.consts}
        if isinstance(a, TY.BlockT) and isinstance(b, TY.BlockT):
            if not self._same_type(a.elem, b.elem) or \
                    len(a.dims) != len(b.dims):
                return False
            return all(self._dim_compat(x, y, const_names)
                       for x, y in zip(a.dims, b.dims))
        if isinstance(a, TY.MaskT) and isinstance(b, TY.MaskT):
            return len(a.dims) == len(b.dims) and \
                all(self._dim_compat(x, y, const_names)
                    for x, y in zip(a.dims, b.dims))
        if isinstance(a, TY.PtrT) and isinstance(b, TY.PtrT):
            return self._same_type(a, b)
        return type(a) is type(b) and TY.describe(a) == TY.describe(b)

    @staticmethod
    def _dim_compat(x, y, const_names) -> bool:
        if equal(x, y):
            return True
        return free_syms(x) <= const_names and free_syms(y) <= const_names

    def _static_cond_syms(self, e):
        """static-deferred 条件判定。

        条件由 Const 参数与 int/bool 字面量经 + - * // % 比较、not/and/or
        构成时，返回全部自由 Const 符号名（可能为空集）；任一子式引用
        运行期标量/维符号/pid/loop 变量（soundness：那些是真运行期值）
        ⇒ None。条件的类型检查（bool 标量、拒 Mask）由调用前的 synth 完成。
        """
        if isinstance(e, Lit):
            if isinstance(e.value, (bool, int)):
                return set()
            return None
        if isinstance(e, UnaOp) and e.op == "not":
            return self._static_cond_syms(e.operand)
        if isinstance(e, Name):
            v = self.vars.get(e.id)
            if v is not None and v.is_const and self._dtype_of(v) is D.bool_:
                if type(v.lit) is bool:
                    return set()
                if v.predicate.op == "const_bool":
                    return {v.predicate.atom}
                if v.tir is not None:
                    return self._staged_operand_syms(v.tir)
            return None
        if isinstance(e, Cmp):
            if e.op in ("==", "!="):
                l, r = self._static_cond_syms(e.left), self._static_cond_syms(e.right)
                if l is not None and r is not None:
                    return l | r
            l, r = self._static_int_syms(e.left), self._static_int_syms(e.right)
            if l is None or r is None:
                return None
            return l | r
        if isinstance(e, BoolOp):
            syms: set = set()
            for v in e.values:
                s = self._static_cond_syms(v)
                if s is None:
                    return None
                syms |= s
            return syms
        return None

    def _staged_operand_syms(self, operand):
        if isinstance(operand, T.TLit):
            return set()
        if isinstance(operand, T.TName):
            return {operand.name} if operand.name in {c.name for c in self.tk.consts} else None
        children = ([operand.operand] if isinstance(operand, T.TUna) else
                    [operand.left, operand.right] if isinstance(operand, T.TBin) else [])
        if not children:
            return None
        result = set()
        for child in children:
            syms = self._staged_operand_syms(child)
            if syms is None:
                return None
            result |= syms
        return result

    def _static_int_syms(self, e):
        """int 值子式的 Const 符号集（含运行期值 ⇒ None）。

        Name 走其符号 expr 的自由符号：expr 是该值可靠的符号式（Const
        参数为 Sym、lane/pid/loop/运行期标量各有专属符号名），自由符号
        全落在 Const 参数名内才视为纯 Const 式。
        """
        if isinstance(e, Lit):
            if isinstance(e.value, int) and not isinstance(e.value, bool):
                return set()
            return None
        if isinstance(e, UnaOp) and e.op == "-":
            return self._static_int_syms(e.operand)
        if isinstance(e, Name):
            v = self.vars.get(e.id)
            if v is None or v.expr is None:
                return None
            syms = free_syms(v.expr)
            if syms <= {c.name for c in self.tk.consts}:
                return syms
            return None
        if isinstance(e, BinOp) and e.op in ("+", "-", "*", "//", "%"):
            l, r = self._static_int_syms(e.left), self._static_int_syms(e.right)
            if l is None or r is None:
                return None
            return l | r
        return None

    def _dim_eq_defer(self, a, b, loc, what: str) -> bool:
        """Shape 相等判定（type-system.md §4.3 两层）。

        语法层 equal 通过 → True；两侧自由符号非空且全部是 Const 参数名
        ⇒ 约束层延迟：登记 {"kind": "eq"} 进入 deferred，乐观返回 True
        （特化期 runtime._check_deferred 代入 Const 数值复核，失败报
        TILA-SHAPE-004）；纯常量不等（无自由符号——语法不等即定论）或
        含任何运行期符号（维 N/M、pid、loop 变量等）⇒ False，调用方
        立即报 Stage-1 错误。
        """
        if equal(a, b):
            return True
        syms = free_syms(a) | free_syms(b)
        if syms and syms <= {c.name for c in self.tk.consts}:
            self.tk.deferred.append({"kind": "eq", "loc": loc, "l": a,
                                     "r": b, "what": what})
            return True
        return False

    def _broadcast(self, s1, s2, loc, what: str = "broadcast shapes"):
        """broadcast 的 Checker 语境入口：接入 §4.3 约束层延迟。"""
        return broadcast(s1, s2, loc, defer=self._dim_eq_defer, what=what)

    def _same_type(self, a, b) -> bool:
        if isinstance(a, TY.PtrT) and isinstance(b, TY.PtrT):
            return a.element is b.element and \
                a.address_space == b.address_space and \
                a.access == b.access and a.extent == b.extent and \
                a.alignment == b.alignment and a.region_id == b.region_id
        if isinstance(a, TY.BlockT) and isinstance(b, TY.BlockT):
            if not self._same_type(a.elem, b.elem) or \
                    len(a.dims) != len(b.dims):
                return False
            return all(equal(x, y) for x, y in zip(a.dims, b.dims))
        if isinstance(a, TY.MaskT) and isinstance(b, TY.MaskT):
            return len(a.dims) == len(b.dims) and \
                all(equal(x, y) for x, y in zip(a.dims, b.dims))
        return type(a) is type(b) and TY.describe(a) == TY.describe(b)

    def _const_eval_bool(self, e):
        if isinstance(e, Lit) and isinstance(e.value, bool):
            return e.value
        if isinstance(e, UnaOp) and e.op == "not":
            value = self._const_eval_bool(e.operand)
            return None if value is None else not value
        if isinstance(e, Name):
            v = self.vars.get(e.id)
            if v is not None and v.is_const and isinstance(v.lit, bool):
                return v.lit
            return None
        if isinstance(e, Cmp):
            if e.op in ("==", "!="):
                lb, rb = self._const_eval_bool(e.left), self._const_eval_bool(e.right)
                if lb is not None and rb is not None:
                    return (lb == rb) if e.op == "==" else (lb != rb)
            l, r = self._const_eval_int(e.left), self._const_eval_int(e.right)
            if l is None or r is None:
                return None
            return {"<": l < r, "<=": l <= r, ">": l > r, ">=": l >= r,
                    "==": l == r, "!=": l != r}[e.op]
        if isinstance(e, BoolOp):
            if e.op == "and":
                unknown = False
                for operand in e.values:
                    value = self._const_eval_bool(operand)
                    if value is False:
                        return False
                    unknown |= value is None
                return None if unknown else True
            if e.op == "or":
                unknown = False
                for operand in e.values:
                    value = self._const_eval_bool(operand)
                    if value is True:
                        return True
                    unknown |= value is None
                return None if unknown else False
        return None

    def _const_eval_int(self, e):
        if isinstance(e, Lit) and isinstance(e.value, int) and \
                not isinstance(e.value, bool):
            return e.value
        if isinstance(e, Name):
            v = self.vars.get(e.id)
            if v is not None and v.is_const and isinstance(v.lit, int):
                return v.lit
        if isinstance(e, BinOp) and e.op in ("+", "-", "*", "//", "%"):
            l, r = self._const_eval_int(e.left), self._const_eval_int(e.right)
            if l is None or r is None:
                return None
            try:
                import operator
                return {"+": operator.add, "-": operator.sub,
                        "*": operator.mul, "//": operator.floordiv,
                        "%": operator.mod}[e.op](l, r)
            except ZeroDivisionError:
                return None
        return None

    def _fold_const_expr(self, e):
        """Const 表达式 → 编译期 int（type-system.md §5.1 常量折叠）。

        覆盖：int 字面量；Const 名（字面量 Const 的 .lit，或 Const 参数
        ——其数值由 _collect_params 种入 facts.num）；一元负号与
        + - * // % 上的组合（Python 整数 floor 语义）。
        运行期标量、维符号、未知 Const、float/bool、除零 ⇒ None。
        """
        if isinstance(e, Lit):
            if isinstance(e.value, int) and not isinstance(e.value, bool):
                return e.value
            return None
        if isinstance(e, Name):
            v = self.vars.get(e.id)
            if v is None:
                return None
            if v.is_const and isinstance(v.lit, int) and \
                    not isinstance(v.lit, bool):
                return v.lit
            if v.expr is not None:
                from .dims import eval_num
                syms = free_syms(v.expr)
                if syms and syms <= self.facts.num.keys():
                    try:
                        return eval_num(v.expr, self.facts.num)
                    except (ArithmeticError, TypeError):
                        return None
            return None
        if isinstance(e, UnaOp) and e.op == "-":
            n = self._fold_const_expr(e.operand)
            return None if n is None else -n
        if isinstance(e, BinOp) and e.op in ("+", "-", "*", "//", "%"):
            l = self._fold_const_expr(e.left)
            r = self._fold_const_expr(e.right)
            if l is None or r is None:
                return None
            try:
                import operator
                return {"+": operator.add, "-": operator.sub,
                        "*": operator.mul, "//": operator.floordiv,
                        "%": operator.mod}[e.op](l, r)
            except ZeroDivisionError:
                return None
        return None

    def _const_dim_expr(self, e):
        """Const 表达式 → 符号 DimExpr（折叠失败但纯 Const 符号时）。

        叶子只接受 int 字面量与 Const 名（Const 参数的 Sym / 字面量
        Const 的 Cst）；任一子式依赖运行期值 ⇒ None。未定值 Const 的
        形状走此路径（HALF * 2 → Mul(Sym, Cst)），数值留待特化期代入。
        """
        if isinstance(e, Lit):
            if isinstance(e.value, int) and not isinstance(e.value, bool):
                return Cst(e.value)
            return None
        if isinstance(e, Name):
            v = self.vars.get(e.id)
            if v is not None and v.is_const and \
                    isinstance(v.expr, (Sym, Cst)):
                return v.expr
            return None
        if isinstance(e, BinOp) and e.op in ("+", "-", "*", "//", "%"):
            l = self._const_dim_expr(e.left)
            r = self._const_dim_expr(e.right)
            if l is None or r is None:
                return None
            if e.op in ("//", "%"):
                from .dims import int_value
                if int_value(r) == 0:
                    return None      # 除零：不进入符号路径
            if e.op == "+":
                return l + r
            if e.op == "-":
                return l - r
            if e.op == "*":
                return l * r
            if e.op == "//":
                return FloorDiv(l, r)
            return Mod(l, r)
        return None

    # -- For ----------------------------------------------------------------

    def _for(self, s: For, out):
        if s.var in self.vars:
            raise TilaError("TILA-TYPE-023",
                            "loop induction variable must use a fresh name "
                            "(shadowing an existing variable is not supported)", s.loc)
        # Interval induction abstracts iteration counts/early returns. Do not
        # promote SAT witnesses in or after a loop to confirmed executions.
        self.execution_context_exact = False
        # 起点：字面量 0（既有形态）或 int 标量（运行期起点，如
        # FlashAttention 两段 KV 循环的 lo = pid * BLOCK_M）。
        start_op = None
        start_expr = Cst(0)
        if not (isinstance(s.start, Lit) and s.start.value == 0):
            if isinstance(s.start, Lit) and \
                    isinstance(s.start.value, int) and \
                    not isinstance(s.start.value, bool):
                start_op = T.TLit(s.start.value, D.i32)
                start_expr = Cst(s.start.value)
            else:
                st = self.synth(s.start, out)
                if not (isinstance(st.vtype, TY.ScalarT) and
                        st.vtype.dtype.is_int):
                    raise TilaError(
                        "TILA-TYPE-024",
                        f"loop start must be an int scalar, found "
                        f"{TY.describe(st.vtype) if st.vtype else 'a non-int value'}",
                        s.loc)
                start_op = self._operand_of(st, s.start, out)
                start_expr = st.expr
        end = self.synth(s.end, out)
        if isinstance(end.lit, int) and not isinstance(end.lit, bool):
            end.vtype = TY.ScalarT(D.i32)
            end.expr = Cst(end.lit)
        if not (isinstance(end.vtype, TY.ScalarT) and end.vtype.dtype.is_int):
            raise TilaError("TILA-TYPE-024",
                            f"loop end must be an int scalar, found "
                            f"{TY.describe(end.vtype)}", s.loc)
        step = self.synth(s.step, out)
        if step.lit is not None and isinstance(step.lit, int) and \
                not isinstance(step.lit, bool) and step.lit >= 1:
            step_op = T.TLit(step.lit, D.i32)
        elif step.is_const and isinstance(step.expr, Sym):
            step_op = T.TName(step.expr.name)
            self.tk.deferred.append({"kind": "step", "loc": s.loc,
                                     "sym": step.expr.name})
        else:
            step_v = self._fold_const_expr(s.step)   # Const 算术折叠（§5.1）
            if step_v is None or step_v < 1:
                raise TilaError(
                    "TILA-TYPE-025",
                    "loop step must be a compile-time int >= 1 "
                    "(literal or Const parameter)", s.loc)
            step_op = T.TLit(step_v, D.i32)

        loop_facts = self.facts.clone()
        outer_vars = dict(self.vars)
        self.lane_counter += 1
        loop_sym = Sym(f"__i_{s.var}_{self.lane_counter}")
        self.vars[s.var] = VarInfo(TY.ScalarT(D.i32), expr=loop_sym)
        # 归纳式：start <= i < end（sym_lo 含 / sym_hi 排他）。
        # 非负性只有起点结构非负时才登记（运行期负起点不能臆断）。
        if start_expr is not None:
            self.facts.sym_lo[loop_sym.name] = start_expr
        if end.expr is not None:
            self.facts.sym_hi[loop_sym.name] = end.expr
        if start_expr is not None and is_nonneg_expr(start_expr, self.nonneg_syms):
            self.nonneg_syms.add(loop_sym.name)

        before = dict(self.vars)        # 浅快照：未重赋值的名字保持同一对象
        # Entry facts describe initial values. Only the flat invariant domain
        # below may retain them for all iterations; other values get widened.
        pending = list(s.body)
        assigned = set()
        invariant = set(before)
        while pending:
            stmt = pending.pop()
            if isinstance(stmt, Assign):
                assigned.add(stmt.target)
                # A deliberately small inductive invariant: identity updates.
                identity = isinstance(stmt.value, Name) and stmt.value.id == stmt.target
                initial = before.get(stmt.target)
                same_literal = (isinstance(stmt.value, Lit) and initial is not None
                                and type(stmt.value.value) is type(initial.lit)
                                and stmt.value.value == initial.lit)
                if not (identity or same_literal):
                    invariant.discard(stmt.target)
            elif isinstance(stmt, If):
                pending.extend(stmt.then_body)
                pending.extend(stmt.else_body)
            elif isinstance(stmt, For):
                assigned.add(stmt.var)
                invariant.discard(stmt.var)
                pending.extend(stmt.body)
        for name in (assigned - invariant) & before.keys():
            if name != s.var:
                self.vars[name] = self._forget_value(before[name])
        body_out: list = []
        self.loop_stack.append(before)
        # return exits the whole program instance. Unless a nonempty range is
        # known, the zero-iteration path still reaches the continuation.
        body_term = self.stmts(s.body, body_out)
        self.loop_stack.pop()

        # 循环出口：loop-local 名字失效；carried（被重赋值的）保守清事实；
        # 未触碰的名字完整保留类型与事实。
        after = {}
        for name, v in self.vars.items():
            if name == s.var or name not in before:
                continue
            if name in invariant:
                after[name] = before[name]
            else:
                after[name] = self._forget_value(v)
        self.vars = after
        # A literal empty range cannot modify any outer value. Its body still
        # gets type checked, with contradictory induction bounds for accesses.
        known_empty = (isinstance(start_expr, Cst) and isinstance(end.expr, Cst)
                       and start_expr.value >= end.expr.value)
        known_nonempty = (isinstance(start_expr, Cst) and isinstance(end.expr, Cst)
                          and start_expr.value < end.expr.value)
        if known_empty:
            self.vars = outer_vars
        # A loop can execute zero times; assumptions from its body do not
        # dominate its exit. Each obligation retains its own interval snapshot.
        self.facts = loop_facts
        if start_expr is not None and end.expr is not None:
            empty = P.atom(Pred(">=", start_expr, end.expr))
            if body_term:
                # Every entered iteration returns from the program. Only the
                # zero-trip edge can reach code after this loop.
                self.facts.path = P.conjunction(self.facts.path, empty)
            for name, value in self.vars.items():
                initial = outer_vars.get(name)
                if (initial is not None and initial.expr is not None and
                        value.expr is not None and
                        canon(initial.expr) != canon(value.expr)):
                    unchanged = P.atom(Pred("==", value.expr, initial.expr))
                    self.facts.path = P.conjunction(self.facts.path,
                        P.disjunction(P.negate(empty), unchanged))
        out.append(T.TFor(s.var, self._operand_of(end, s.end, out), step_op,
                          body_out, line=s.loc.line, start=start_op))
        return body_term and known_nonempty

    def _forget_value(self, value):
        out = value.clone()
        out.expr, out.predicate, out.contiguous_span = None, P.unknown(shape=shape_of(out)), None
        out.lit, out.is_const = None, False
        out.tir = None
        dtype = self._dtype_of(out)
        if dtype is not None and dtype.is_int:
            self.lane_counter += 1
            out.expr = Sym(f"__v{self.lane_counter}")
            self.value_types[out.expr.name] = dtype.name
            if shape_of(out):
                self.lane_axes[out.expr.name] = tuple(range(-len(shape_of(out)), 0))
        return out

    # ------------------------------------------------------------------
    # 表达式合成
    # ------------------------------------------------------------------

    def synth(self, e, out) -> VarInfo:
        if isinstance(e, Lit):
            if type(e.value) is bool:
                return VarInfo(vtype=TY.ScalarT(D.bool_), lit=e.value, is_const=True,
                               predicate=P.TRUE if e.value else P.FALSE)
            return VarInfo(lit=e.value)
        if isinstance(e, Name):
            v = self.vars.get(e.id)
            if v is None:
                raise TilaError(
                    "TILA-TYPE-023",
                    f"name '{e.id}' is not defined on this path", e.loc)
            if v.static_variant is not None:
                raise TilaError(
                    "TILA-TYPE-020",
                    "branch result types differ (static-if variant: type "
                    "depends on Const specialization)", e.loc,
                    [f"    variable '{e.id}'",
                     f"    then: {TY.describe(v.static_variant[0])}",
                     f"    else: {TY.describe(v.static_variant[1])}"],
                    ["static-if 两分支给出了不同类型：对齐 dtype/rank，"
                     "或把对该名字的使用移进分支内"])
            if v.maybe_undefined:
                raise TilaError(
                    "TILA-TYPE-023",
                    f"name '{e.id}' may be undefined "
                    "(assigned in only one branch)", e.loc)
            return v.clone()
        if isinstance(e, BinOp):
            return self._binop(e, out)
        if isinstance(e, UnaOp):
            return self._unaop(e, out)
        if isinstance(e, Cmp):
            return self._cmp(e, out)
        if isinstance(e, BoolOp):
            return self._boolop(e, out)
        if isinstance(e, Call):
            return self._call(e, out)
        if isinstance(e, Expand):
            v = self.synth(e.value, out)
            if not isinstance(v.vtype, (TY.BlockT, TY.MaskT)):
                raise TilaError("TILA-SHAPE-006",
                                "[:, None] requires a block or mask value",
                                e.loc)
            dims = list(v.vtype.dims)
            dims.insert(e.axis, Cst(1))
            if isinstance(v.vtype, TY.BlockT):
                vt = TY.BlockT(v.vtype.elem, tuple(dims))
            else:
                vt = TY.MaskT(tuple(dims))
            m = v.clone()
            remap = lambda expr: self._expand_index(expr, len(v.vtype.dims), e.axis)
            if m.expr is not None:
                m.expr = remap(m.expr)
            m.predicate = P.map_atoms(v.predicate, remap, tuple(dims),
                                      ("expand", e.axis, v.vtype.dims))
            m.vtype, m.contiguous_span, m.tir = vt, None, \
                T.TExpand(vt, self._operand_of(v, e.value, out), e.axis)
            return m
        if isinstance(e, BufPtr):
            b = self.vars.get(e.buf)
            if b is None or not isinstance(b.vtype, TY.BufferT):
                raise TilaError("TILA-MEM-004",
                                f"'{e.buf}.ptr' requires a buffer parameter",
                                e.loc)
            bt = b.vtype
            if len(bt.shape) != 1:
                raise TilaError(
                    "TILA-MEM-005",
                    f"'{e.buf}.ptr' requires a rank-1 Buffer in v0",
                    e.loc,
                    [f"    buffer: {bt.describe()}",
                     f"    rank:   {len(bt.shape)}"],
                    ["使用 Buffer 坐标访问 ti.load(buf, coords) / "
                     "ti.store(buf, coords, value)；v0 不隐式 flatten"])
            self.tk.buffer_ptr_views.add(e.buf)
            extent = TY.LinearExtent(bt.shape[0])
            ptr = TY.PtrT(element=bt.elem, address_space=bt.space,
                          access=bt.access, extent=extent,
                          alignment=bt.alignment,
                          region_id=self.regions[e.buf])
            return VarInfo(vtype=ptr, expr=Cst(0),
                           tir=T.TBufPtr(vt=ptr, buffer=e.buf))
        if isinstance(e, (HTuple, HDtype)):
            raise TilaError("TILA-SYN-027",
                            "tuples/dtypes only appear in intrinsic argument "
                            "positions", e.loc)
        raise TilaError("TILA-TYPE-018", "cannot type this expression", e.loc)

    def _is_const_index(self, expr) -> bool:
        """基底是否为编译期可知（字面量/Const 参数的仿射式）。"""
        from .dims import free_syms as _fs
        if expr is None:
            return False
        const_names = {c.name for c in self.tk.consts} | \
            {n for n, v in self.vars.items() if v.is_const}
        return _fs(expr) <= const_names

    def _dtype_of(self, v: VarInfo):
        if isinstance(v.vtype, TY.ScalarT):
            return v.vtype.dtype
        if isinstance(v.vtype, TY.BlockT) and \
                isinstance(v.vtype.elem, TY.ScalarT):
            return v.vtype.elem.dtype
        return None

    def _instantiate(self, lit, dt: D.DType, loc):
        if not D.representable(lit, dt):
            raise TilaError(
                "TILA-TYPE-013",
                f"literal {lit} cannot be represented as {dt.name}", loc,
                [], [f"使用可表示的字面量，或显式 ti.cast[{dt.name}](...)"])
        return dt

    def _is_ptrish(self, vt):
        return isinstance(vt, TY.PtrT) or (isinstance(vt, TY.BlockT) and
                                           isinstance(vt.elem, TY.PtrT))

    def _is_boolean(self, v):
        return (type(v.lit) is bool or isinstance(v.vtype, TY.MaskT)
                or self._dtype_of(v) is D.bool_)

    def _boolean_operand(self, v, e, out, code="TILA-TYPE-026"):
        """ADR-012: adapt consumers without converting the public value type."""
        if not self._is_boolean(v):
            raise TilaError(code,
                            "expected Scalar[bool], Mask or Block[bool], found "
                            f"{TY.describe(v.vtype)}", e.loc)
        if type(v.lit) is bool:
            v.vtype = TY.ScalarT(D.bool_)
            v.predicate = P.TRUE if v.lit else P.FALSE
        return shape_of(v), self._operand_of(v, e, out), v.predicate

    def _binop(self, e: BinOp, out) -> VarInfo:
        l = self.synth(e.left, out)
        r = self.synth(e.right, out)

        # Boolean consumers retain each operand's predicate identity and axes.
        if (isinstance(l.vtype, TY.MaskT) or isinstance(r.vtype, TY.MaskT)
                or e.op in ("&", "|") and
                (self._is_boolean(l) or self._is_boolean(r))):
            if e.op not in ("&", "|"):
                raise TilaError("TILA-TYPE-026",
                                f"masks only support & | ~, found '{e.op}'",
                                e.loc)
            ls, lo, lp = self._boolean_operand(l, e.left, out)
            rs, ro, rp = self._boolean_operand(r, e.right, out)
            sh = self._broadcast(ls, rs, e.loc,
                                 "mask combination shapes")
            combine = P.conjunction if e.op == "&" else P.disjunction
            clauses = combine(lp, rp)
            tile = any(isinstance(v.vtype, (TY.BlockT, TY.MaskT)) for v in (l, r))
            vt = TY.MaskT(sh) if tile else TY.ScalarT(D.bool_)
            return VarInfo(vtype=vt, predicate=clauses,
                           tir=T.TBin(vt, e.op, lo, ro))

        # 指针元素 offset（+，type-system.md §8.3）
        if e.op == "+" and (self._is_ptrish(l.vtype) or
                            self._is_ptrish(r.vtype)):
            if self._is_ptrish(l.vtype):
                pv, ov, oe = l, r, e.right
            else:
                pv, ov, oe = r, l, e.left
            odt = self._dtype_of(ov)
            if odt is None or not odt.is_int:
                raise TilaError(
                    "TILA-MEM-002",
                    "pointer offset must be an integer (element offsets; "
                    "byte offsets need ti.byte_offset)", e.loc,
                    [f"    offset type: {TY.describe(ov.vtype)}"])
            pt = pv.vtype
            sh = self._broadcast(shape_of(pv), shape_of(ov), e.loc,
                                 "pointer offset shapes")
            rt = pt if (isinstance(pt, TY.PtrT) and sh == ()) else \
                (pt if isinstance(pt, TY.PtrT) else TY.BlockT(pt.elem, sh))
            if isinstance(pt, TY.PtrT) and sh != ():
                rt = TY.BlockT(pt, sh)
            m = VarInfo(vtype=rt)
            if pv.expr is not None and ov.expr is not None:
                m.expr = pv.expr + ov.expr
            m.tir = T.TPAdd(rt, self._operand_of(pv, e.left if pv is l
                                                 else e.right, out),
                            self._operand_of(ov, oe, out))
            return m

        # 数值运算（type-system.md §6）：字面量先语境化再判空
        dl, dr = self._dtype_of(l), self._dtype_of(r)
        if l.lit is not None and r.lit is None:
            if dr is None:
                raise TilaError("TILA-TYPE-027", "operands must be numeric",
                                e.loc,
                                [f"    right: {TY.describe(r.vtype)}"])
            dl = self._instantiate(l.lit, dr, e.loc)
            l.vtype = TY.ScalarT(dl)
        elif r.lit is not None and l.lit is None:
            if dl is None:
                raise TilaError("TILA-TYPE-027", "operands must be numeric",
                                e.loc,
                                [f"    left: {TY.describe(l.vtype)}"])
            dr = self._instantiate(r.lit, dl, e.loc)
            r.vtype = TY.ScalarT(dr)
        elif l.lit is not None and r.lit is not None:
            dl = dr = _lit_default_dtype(
                l.lit if isinstance(l.lit, float) or
                isinstance(r.lit, float) else l.lit)
        if dl is None or dr is None:
            raise TilaError("TILA-TYPE-027", "operands must be numeric",
                            e.loc,
                            [f"    left:  {TY.describe(l.vtype)}",
                             f"    right: {TY.describe(r.vtype)}"])
        dt = D.common_type(dl, dr)
        if dt is None:
            raise TilaError(
                "TILA-TYPE-012",
                f"cannot apply '{e.op}' to {dl.name} and {dr.name}: "
                "no implicit conversion (narrowing, signed/unsigned mix, "
                "int/float mix and f16/bf16 mix are all explicit)", e.loc,
                [f"    left:  {dl.name}", f"    right: {dr.name}"],
                [f"ti.cast[{dr.name}](...) 或 ti.cast[{dl.name}](...) "
                 "显式转换一侧"])
        if e.op == "/" and not dt.is_float:
            raise TilaError("TILA-TYPE-014",
                            "'/' is float-only; integers use '//'", e.loc)
        if e.op in ("//", "%", "<<", ">>", "&", "|", "^") and not dt.is_int:
            raise TilaError("TILA-TYPE-014", f"'{e.op}' is integer-only",
                            e.loc)
        if dt.kind == "bool":
            raise TilaError("TILA-TYPE-028",
                            "bool is not an arithmetic type (use & | ~)",
                            e.loc)
        if dt not in D.ARITH_DTYPES:      # f8e4m3fn / f8e5m2（§2 FloatStorage）
            raise TilaError(
                "TILA-TYPE-036",
                f"{dt.name} is a storage-only dtype: cast to a Float dtype "
                "before arithmetic", e.loc,
                [f"    left:  {dl.name}", f"    right: {dr.name}"],
                [f"ti.cast[ti.f32](...) 先转换到计算精度"
                 "（FP8 范式：load → cast → compute → cast → store）"])

        sh = self._broadcast(shape_of(l), shape_of(r), e.loc,
                             "binary operand shapes")
        vt = TY.ScalarT(dt) if sh == () else TY.BlockT(TY.ScalarT(dt), sh)
        m = VarInfo(vtype=vt)
        for operand in (l, r):
            if type(operand.lit) is int:
                operand.expr = Cst(operand.lit)
        if l.expr is not None and r.expr is not None and dt.is_int:
            if e.op == "+":
                m.expr = l.expr + r.expr
            elif e.op == "-":
                m.expr = l.expr - r.expr
            elif e.op == "*":
                m.expr = l.expr * r.expr
            elif e.op == "//":
                m.expr = FloorDiv(l.expr, r.expr)
            elif e.op == "%":
                m.expr = Mod(l.expr, r.expr)
            if e.op == "+":
                # contiguous(span) 传播：标量基 + 连续块 ⇒ 连续性保持
                # （refinements.md §3.2：idx = base + arange 是连续段）
                for a, b in ((l, r), (r, l)):
                    if a.contiguous_span is not None and shape_of(b) == ():
                        m.contiguous_span = a.contiguous_span
                        break
        m.tir = T.TBin(vt, e.op, self._operand_of(l, e.left, out),
                       self._operand_of(r, e.right, out))
        if dt.is_int:
            m.tir.staged = (e.op in ("+", "-", "*", "//", "%") and
                            self._is_const_index(l.expr) and
                            self._is_const_index(r.expr))
            m.tir.operand_dtype = dt
            self._guard_index_math(m)
            if m.expr is None and l.expr is not None and r.expr is not None:
                m.expr = self._integer_expr(e.op, dt, l.expr, r.expr)
        return m

    def _integer_expr(self, op, dtype, left, right=Cst(0)):
        self.lane_counter += 1
        sym = Sym(f"__bv{self.lane_counter}")
        self.value_types[sym.name] = dtype.name
        self.value_defs[sym.name] = (op, dtype.name, left, right)
        axes = {axis for name in free_syms(left) | free_syms(right)
                for axis in self.lane_axes.get(name, ())}
        if axes:
            self.lane_axes[sym.name] = tuple(sorted(axes))
        return sym

    def _guard_index_math(self, value):
        """Only retain mathematical index facts behind a launch overflow gate.

        Data-dependent arithmetic remains modular and gets a fresh value identity;
        it must be bounded by masks on the actual result, not mathematical rewrites.
        """
        if value.expr is None or value.tir.staged:
            return
        symbols = free_syms(value.expr)
        trusted = (self.tk.implicit_names |
                   {c.name for c in self.tk.consts})
        if any(s not in trusted and not s.startswith(("pid", "__lane", "__i_"))
               for s in symbols):
            value.expr = self._integer_expr("wrap", self._dtype_of(value), value.expr)
            value.contiguous_span = None
        else:
            value.tir.checked_index = True

    def _unaop(self, e: UnaOp, out):
        v = self.synth(e.operand, out)
        if e.op == "not":
            if not (isinstance(v.vtype, TY.ScalarT) and
                    v.vtype.dtype is D.bool_):
                raise TilaError("TILA-TYPE-019",
                                f"'not' requires bool scalar, found "
                                f"{TY.describe(v.vtype)}", e.loc)
            return VarInfo(vtype=v.vtype, predicate=P.negate(v.predicate),
                lit=self._const_eval_bool(e),
                is_const=self._static_cond_syms(e) is not None, tir=T.TUna(
                v.vtype, "not", self._operand_of(v, e.operand, out)))
        if e.op == "~":
            if self._is_boolean(v):
                # Preserve negation and shared identity for the general solver.
                sh, operand, predicate = self._boolean_operand(v, e.operand, out)
                vt = (TY.MaskT(sh) if isinstance(v.vtype, (TY.BlockT, TY.MaskT))
                      else TY.ScalarT(D.bool_))
                return VarInfo(vtype=vt, predicate=P.negate(predicate),
                               tir=T.TUna(vt, "~", operand))
            dt = self._dtype_of(v)
            if dt is not None and dt.is_int:
                expr = self._integer_expr("~", dt, v.expr) if v.expr is not None else None
                return VarInfo(vtype=v.vtype, expr=expr, tir=T.TUna(
                    v.vtype, "~", self._operand_of(v, e.operand, out)))
            raise TilaError("TILA-TYPE-029", "'~' requires a mask or integer",
                            e.loc)
        # '-'：字面量取负（保持语境多态）
        if v.lit is not None and v.vtype is None:
            return VarInfo(lit=-v.lit)
        dt = self._dtype_of(v)
        if dt is None or not dt.numeric:
            raise TilaError("TILA-TYPE-030",
                            "unary '-' requires a numeric operand", e.loc)
        if dt not in D.ARITH_DTYPES:      # f8e4m3fn / f8e5m2（§2 FloatStorage）
            raise TilaError(
                "TILA-TYPE-036",
                f"{dt.name} is a storage-only dtype: cast to a Float dtype "
                "before arithmetic", e.loc,
                [f"    operand: {dt.name}"],
                [f"ti.cast[ti.f32](...) 先转换到计算精度"
                 "（FP8 范式：load → cast → compute → cast → store）"])
        m = VarInfo(vtype=v.vtype)
        if v.expr is not None:
            m.expr = v.expr * Cst(-1)
        m.tir = T.TUna(v.vtype, "-",
                       self._operand_of(v, e.operand, out))
        if dt.is_int:
            m.tir.staged = self._is_const_index(v.expr)
            self._guard_index_math(m)
        return m

    @staticmethod
    def _cmp_index_expr(v: VarInfo):
        """比较参与者的符号索引式：int 字面量折为 Cst。

        使 `offs < 1000` 这类对常量的比较也产生谓词（进入 mask DAG /
        assume 事实）；float 与无符号身份的值仍返回 None。
        """
        if v.expr is not None:
            return v.expr
        if isinstance(v.lit, int) and not isinstance(v.lit, bool):
            return Cst(v.lit)
        return None

    def _cmp(self, e: Cmp, out):
        l = self.synth(e.left, out)
        r = self.synth(e.right, out)
        if self._is_boolean(l) or self._is_boolean(r):
            ls, lo, lp = self._boolean_operand(l, e.left, out)
            rs, ro, rp = self._boolean_operand(r, e.right, out)
            if e.op not in ("==", "!="):
                raise TilaError("TILA-TYPE-026", "bool comparisons support only == and !=", e.loc)
            sh = self._broadcast(ls, rs, e.loc, "bool comparison shapes")
            predicate = P.disjunction(P.conjunction(lp, rp),
                                      P.conjunction(P.negate(lp), P.negate(rp)))
            if e.op == "!=":
                predicate = P.negate(predicate)
            vt = TY.MaskT(sh) if sh else TY.ScalarT(D.bool_)
            return VarInfo(vtype=vt, predicate=predicate,
                           lit=self._const_eval_bool(e),
                           is_const=self._static_cond_syms(e) is not None,
                           tir=T.TBin(vt, e.op, lo, ro))
        dl, dr = self._dtype_of(l), self._dtype_of(r)
        if dl is not None or dr is not None:
            if l.lit is not None and r.lit is None and dr is not None:
                dl = self._instantiate(l.lit, dr, e.loc)
                l.vtype = TY.ScalarT(dl)
            elif r.lit is not None and l.lit is None and dl is not None:
                dr = self._instantiate(r.lit, dl, e.loc)
                r.vtype = TY.ScalarT(dr)
            if dl is not None and dr is not None and \
                    D.common_type(dl, dr) is None:
                raise TilaError(
                    "TILA-TYPE-012",
                    f"cannot compare {dl.name} and {dr.name} "
                    "(no implicit conversion)", e.loc,
                    [f"    left:  {dl.name}", f"    right: {dr.name}"],
                    [f"ti.cast[{dr.name}](...) 显式转换一侧"])
        sh = self._broadcast(shape_of(l), shape_of(r), e.loc,
                             "comparison shapes")
        pred = None
        le, re = self._cmp_index_expr(l), self._cmp_index_expr(r)
        if le is not None and re is not None:
            pred = Pred(e.op, le, re, frozenset({P.Origin(P.STATIC, e.loc.line, "comparison")}))
        predicate = P.atom(pred, e.loc.line, sh) if pred else P.unknown(e.loc.line, sh)
        if sh == ():
            vt = TY.ScalarT(D.bool_)
            return VarInfo(vtype=vt, predicate=predicate, tir=T.TBin(
                vt, e.op, self._operand_of(l, e.left, out),
                self._operand_of(r, e.right, out),
                staged=self._const_dim_expr(e.left) is not None and
                       self._const_dim_expr(e.right) is not None,
                operand_dtype=D.common_type(dl, dr) if dl and dr else None))
        vt = TY.MaskT(sh)
        # Comparisons retain source locations and tile shape in the DAG.
        return VarInfo(vtype=vt, predicate=predicate,
                       tir=T.TBin(vt, e.op, self._operand_of(l, e.left, out),
                                  self._operand_of(r, e.right, out),
                                  operand_dtype=D.common_type(dl, dr) if dl and dr else None))

    def _boolop(self, e: BoolOp, out):
        vals = [self.synth(v, out) for v in e.values]
        for v in vals:
            if isinstance(v.vtype, TY.MaskT):
                raise TilaError(
                    "TILA-TYPE-022",
                    f"'{e.op}' requires scalar bools; block predicates use "
                    "'&' / '|'", e.loc)
            if not (isinstance(v.vtype, TY.ScalarT) and
                    v.vtype.dtype is D.bool_):
                raise TilaError("TILA-TYPE-019",
                                f"'{e.op}' requires bool operands", e.loc)
        tir = self._operand_of(vals[0], e.values[0], out)
        for v, he in zip(vals[1:], e.values[1:]):
            tir = T.TBin(TY.ScalarT(D.bool_), e.op, tir,
                         self._operand_of(v, he, out))
        predicate = vals[0].predicate
        combine = P.conjunction if e.op == "and" else P.disjunction
        for value in vals[1:]:
            predicate = combine(predicate, value.predicate)
        return VarInfo(vtype=TY.ScalarT(D.bool_), predicate=predicate, tir=tir,
                       lit=self._const_eval_bool(e),
                       is_const=self._static_cond_syms(e) is not None)

    # ------------------------------------------------------------------
    # 内建分派（intrinsics.md §2）
    # ------------------------------------------------------------------

    def _call(self, e: Call, out) -> VarInfo:
        spec = get_intrinsic(e.func)
        if spec is None:
            raise TilaError("TILA-SYN-030",
                            f"'{e.func}' is not a tila intrinsic", e.loc)
        problem = call_shape_problem(spec, len(e.args), e.kwargs)
        if problem is not None:
            raise TilaError("TILA-SYN-036", problem, e.loc)
        handler = CHECKER_INTRINSIC_HANDLERS.get(spec.checker_handler)
        if handler is None:
            # Registry completeness tests should make this unreachable, but a
            # deterministic user diagnostic is preferable to getattr/traceback.
            raise TilaError("TILA-SYN-030",
                            f"intrinsic '{e.func}' has no checker handler",
                            e.loc)
        return handler(self, e, out)

    def _const_operand(self, e, what: str, out):
        """Const[int] 语境（arange 界、zeros 形状、range step）。

        裸 Lit / 裸 Const 名保持既有路径；组合 Const 表达式先常量折叠
        （type-system.md §5.1：HALF * 2 → 128）；不可折叠但纯 Const
        符号的表达式（未定值 Const 参数参与）保持符号路径，数值留待
        特化期（launcher / rewrite_syms）代入。
        """
        v = self.synth(e, out)
        if isinstance(e, Lit) and isinstance(e.value, int) and \
                not isinstance(e.value, bool):
            return e.value, T.TLit(e.value, D.i32), Cst(e.value)
        if isinstance(e, Name) and v.is_const:
            if v.lit is not None and isinstance(v.lit, int) and \
                    not isinstance(v.lit, bool):
                return v.lit, T.TLit(v.lit, D.i32), Cst(v.lit)
            if isinstance(v.expr, Sym):
                return self.facts.num.get(v.expr.name), \
                    T.TName(v.expr.name), v.expr
        folded = self._fold_const_expr(e)
        if folded is not None:
            return folded, T.TLit(folded, D.i32), Cst(folded)
        dim = self._const_dim_expr(e)
        if dim is not None and free_syms(dim) and v.tir is not None:
            return None, v.tir, dim
        raise TilaError(
            "TILA-CONST-001", f"{what} must be compile-time known", e.loc,
            [f"    found: {TY.describe(v.vtype)}"
             if v.vtype else "    found: runtime value"],
            ["使用 Const 参数（BLOCK: ti.Const[int]）、字面量或模块级 int 常量"])

    def _in_program_id(self, e, out):
        ax = self._axis_arg(e)
        sym = Sym(f"pid{ax}")
        self.facts.sym_lo[sym.name] = Cst(0)
        self.facts.sym_hi[sym.name] = Sym(f"__grid{ax}")
        self.nonneg_syms.add(sym.name)
        vt = TY.ScalarT(D.i32)
        return VarInfo(vtype=vt, expr=sym, tir=T.TPid(vt, ax))

    def _in_num_programs(self, e, out):
        ax = self._axis_arg(e)
        vt = TY.ScalarT(D.i32)
        return VarInfo(vtype=vt, expr=Sym(f"__grid{ax}"),
                       tir=T.TNumPrograms(vt, ax))

    def _axis_arg(self, e) -> int:
        if len(e.args) != 1 or not (isinstance(e.args[0], Lit) and
                                    isinstance(e.args[0].value, int)):
            raise TilaError("TILA-CONST-002",
                            "axis argument must be a literal int 0..2", e.loc)
        ax = e.args[0].value
        if ax not in (0, 1, 2):
            raise TilaError("TILA-CONST-002",
                            f"axis must be 0..2, found {ax}", e.loc)
        return ax

    def _in_arange(self, e, out):
        if len(e.args) != 2:
            raise TilaError("TILA-SYN-036", "arange(start, end) takes 2 args",
                            e.loc)
        if not (isinstance(e.args[0], Lit) and
                isinstance(e.args[0].value, int)):
            raise TilaError("TILA-CONST-001",
                            "arange start must be a literal int", e.loc)
        start = e.args[0].value
        end_v, end_op, end_dim = self._const_operand(e.args[1], "arange end",
                                                     out)
        dim = end_dim if start == 0 else end_dim - Cst(start)
        if end_v is not None and end_v <= start:
            raise TilaError("TILA-SHAPE-008",
                            f"arange end must exceed start "
                            f"(got {start}..{end_v})", e.loc)
        self.lane_counter += 1
        lane = Sym(f"__lane{self.lane_counter}")
        self.lane_axes[lane.name] = (-1,)
        self.facts.sym_lo[lane.name] = Cst(start)
        self.facts.sym_hi[lane.name] = end_dim
        if start >= 0:
            self.nonneg_syms.add(lane.name)
        vt = TY.BlockT(TY.ScalarT(D.i32), (dim,))
        return VarInfo(vtype=vt, expr=lane, contiguous_span=dim,
                       tir=T.TArange(vt, start, end_op))

    def _in_zeros(self, e, out):
        if len(e.args) != 2 or not isinstance(e.args[0], HTuple) or \
                not isinstance(e.args[1], HDtype):
            raise TilaError("TILA-SYN-036",
                            "zeros((d0, d1, ...), dtype) takes a shape tuple "
                            "and a dtype", e.loc)
        dims, ops = [], []
        for d in e.args[0].items:
            v, op, dim = self._const_operand(d, "zeros dimension", out)
            dims.append(dim)
            ops.append(op)
        dt = e.args[1].dtype
        vt = TY.BlockT(TY.ScalarT(dt), tuple(dims))
        return VarInfo(vtype=vt, tir=T.TZeros(vt, ops))

    def _in_reshape(self, e, out):
        """reshape(x, S2)（intrinsics.md §2.8）：Block[T, S1] → Block[T, S2]。

        维度走 Const[int] 语境（_const_operand：字面量 / Const 参数 /
        可折叠 Const 表达式）。numel(S1) ~ numel(S2)（TILA-SHAPE-005）
        分层判定（type-system.md §4.3）：
        1. 两侧维度全常量（int_value）→ Stage 1 立即数值比较；
        2. 符号乘积 canon 等价（dims.equal，如 2*BLOCK vs BLOCK*2）→
           Stage 1 直接通过，不进延迟池；
        3. 其余且自由符号全为 Const 参数名 → 延迟（kind="numel"），
           特化期代入 Const 值后数值比较；
        4. 含运行期维符号（ti.Dim）→ Stage 1 TILA-SHAPE-005：运行期
           numel 约束本版本无法验证（防御分支——目标形状里的运行期
           名字更早被 _const_operand 以 TILA-CONST-001 拒绝）。
        结果不携带事实（expr / contiguous_span 跨 reshape 失效）。
        """
        if len(e.args) != 2 or e.kwargs:
            raise TilaError("TILA-SYN-036",
                            "reshape(x, (d0, d1, ...)) takes a block and a "
                            "shape tuple", e.loc)
        v = self.synth(e.args[0], out)
        if not isinstance(v.vtype, TY.BlockT):
            raise TilaError("TILA-SHAPE-005",
                            "reshape requires a block value", e.loc,
                            [f"    found: {TY.describe(v.vtype)}"])
        shape_e = e.args[1]
        if not isinstance(shape_e, HTuple) or not shape_e.items:
            raise TilaError("TILA-SYN-036",
                            "reshape shape must be a non-empty tuple of "
                            "compile-time dims (rank >= 1)", e.loc)
        new_dims, ops = [], []
        for d in shape_e.items:
            dv, op, dim = self._const_operand(d, "reshape dimension", out)
            if dv is not None and dv < 1:
                raise TilaError("TILA-SHAPE-005",
                                "reshape dimensions must be positive", e.loc,
                                [f"    dim: {dim}"],
                                ["维度为字面量或已定值 Const；0/负维度的 "
                                 "block 无意义"])
            new_dims.append(dim)
            ops.append(op)
        old_dims = tuple(v.vtype.dims)
        new_dims = tuple(new_dims)
        self._check_reshape_numel(e.loc, old_dims, new_dims)
        vt = TY.BlockT(v.vtype.elem, new_dims)
        return VarInfo(vtype=vt, tir=T.TReshape(
            vt, self._operand_of(v, e.args[0], out), ops))

    def _check_reshape_numel(self, loc, old_dims, new_dims):
        from .dims import int_value

        def prod(dims):
            p = Cst(1)
            for d in dims:
                p = p * d
            return p

        old_p, new_p = prod(old_dims), prod(new_dims)
        ov, nv = int_value(old_p), int_value(new_p)
        if ov is not None and nv is not None:
            if ov != nv:
                raise TilaError(
                    "TILA-SHAPE-005", "reshape numel mismatch", loc,
                    [f"    input:  ({', '.join(map(str, old_dims))}) "
                     f"-> numel {ov}",
                     f"    target: ({', '.join(map(str, new_dims))}) "
                     f"-> numel {nv}"],
                    ["两个形状的元素总数必须一致：numel(S1) = numel(S2)"])
            return
        if equal(old_p, new_p):     # 符号乘积 canon 等价（N*2 = 2*N 形态）
            return
        syms = free_syms(old_p) | free_syms(new_p)
        consts = set(self.facts.num) | {c.name for c in self.tk.consts}
        if syms <= consts:
            self.tk.deferred.append({"kind": "numel", "loc": loc,
                                     "old": old_dims, "new": new_dims})
            return
        raise TilaError(
            "TILA-SHAPE-005", "reshape numel mismatch", loc,
            [f"    input:  ({', '.join(map(str, old_dims))}) "
             f"-> numel {old_p}",
             f"    target: ({', '.join(map(str, new_dims))}) "
             f"-> numel {new_p}",
             "    numel 约束依赖运行期维符号，本版本无法验证"],
            ["reshape 形状只使用字面量与 Const 参数（运行期维 ti.Dim "
             "不能参与可验证的 numel 约束）"])

    def _in_constant(self, e, out):
        from .constants import rounded_bits
        arg = e.args[0]
        if isinstance(arg, UnaOp) and arg.op == "-" and isinstance(arg.operand, Lit):
            value = arg.operand.value
            if type(value) is int or type(value) is float:
                value = -value
        elif isinstance(arg, Lit):
            value = arg.value
        else:
            raise TilaError("TILA-CONST-011", "constant source must be an exact int/float literal or captured module constant", e.loc)
        dt = e.cast_dtype
        bits = rounded_bits(value, dt, e.loc)
        vt = TY.ScalarT(dt)
        return VarInfo(vtype=vt, is_const=True, tir=T.TConstant(vt, dt, bits))

    def _in_cast(self, e, out):
        if e.cast_dtype is None or len(e.args) != 1:
            raise TilaError("TILA-SYN-036", "cast[dtype](x) takes one arg",
                            e.loc)
        v = self.synth(e.args[0], out)
        if v.vtype is None:
            v.vtype = TY.ScalarT(_lit_default_dtype(v.lit))
        dt = e.cast_dtype
        if isinstance(v.vtype, TY.BlockT):
            vt = TY.BlockT(TY.ScalarT(dt), v.vtype.dims)
        else:
            vt = TY.ScalarT(dt)
        source_dt = self._dtype_of(v)
        preserves = (source_dt and source_dt.is_int and dt.is_int and
                     D._INT_RANGE[dt][0] <= D._INT_RANGE[source_dt][0] and
                     D._INT_RANGE[source_dt][1] <= D._INT_RANGE[dt][1])
        expr = v.expr
        if expr is None and type(v.lit) is int:
            expr = Cst(v.lit)
        if not preserves:
            expr = (self._integer_expr("wrap", dt, expr)
                    if source_dt and dt.is_int and source_dt.is_int and expr is not None else None)
        return VarInfo(vtype=vt, expr=expr,
                       tir=T.TCast(vt, dt, self._operand_of(v, e.args[0], out)))

    def _in_where(self, e, out):
        if len(e.args) != 3:
            raise TilaError("TILA-SYN-036", "where(mask, a, b) takes 3 args",
                            e.loc)
        m = self.synth(e.args[0], out)
        a = self.synth(e.args[1], out)
        b = self.synth(e.args[2], out)
        self._boolean_operand(m, e.args[0], out, "TILA-TYPE-032")
        da, db = self._dtype_of(a), self._dtype_of(b)
        if a.lit is not None and b.lit is not None:
            dt0 = _lit_default_dtype(a.lit if isinstance(a.lit, float) or
                                     isinstance(b.lit, float) else a.lit)
            da = db = dt0
            a.vtype = b.vtype = TY.ScalarT(dt0)
        elif da is not None and db is not None:
            if a.lit is not None and b.lit is None:
                da = self._instantiate(a.lit, db, e.loc)
                a.vtype = TY.ScalarT(da)
            elif b.lit is not None and a.lit is None:
                db = self._instantiate(b.lit, da, e.loc)
                b.vtype = TY.ScalarT(db)
            if da is not db:
                raise TilaError(
                    "TILA-TYPE-015",
                    "where branches must have the same dtype "
                    "(no implicit promotion)", e.loc,
                    [f"    then: {da.name}", f"    else: {db.name}"],
                    ["显式 ti.cast 对齐两侧 dtype"])
        sh = self._broadcast(
            self._broadcast(shape_of(m), shape_of(a), e.loc,
                            "where operand shapes"),
            shape_of(b), e.loc, "where operand shapes")
        vt = TY.ScalarT(da) if sh == () and da else TY.BlockT(TY.ScalarT(da), sh)
        # where 分支内的内存效应（effects.md §3，v0 记录为 warning）
        for side, v in (("then", a), ("else", b)):
            if self._contains_memop(v.tir):
                self.tk.warnings.append(Warning_(
                    "TILA-EFFECT-007",
                    f"memory operation inside where '{side}' branch — "
                    "both branches are evaluated", e.loc,
                    [], ["改用相同谓词的 masked load/store，或 other= 缺省值"]))
        return VarInfo(vtype=vt, tir=T.TWhere(
            vt, self._operand_of(m, e.args[0], out),
            self._operand_of(a, e.args[1], out),
            self._operand_of(b, e.args[2], out)))

    def _contains_memop(self, node) -> bool:
        if node is None:
            return False
        if isinstance(node, (T.TLoad, T.TStore)):
            return True
        for f in ("left", "right", "operand", "a", "b", "acc", "cond",
                  "ptr", "offset"):
            if self._contains_memop(getattr(node, f, None)):
                return True
        return False

    def _in_dot(self, e, out):
        if len(e.args) != 2:
            raise TilaError("TILA-SYN-036", "dot(a, b) takes 2 args", e.loc)
        a = self.synth(e.args[0], out)
        b = self.synth(e.args[1], out)
        acc = None
        if "acc" in e.kwargs:
            acc = self.synth(e.kwargs["acc"], out)
        for v, n in ((a, "a"), (b, "b")):
            if not (isinstance(v.vtype, TY.BlockT) and
                    len(v.vtype.dims) == 2):
                raise TilaError("TILA-SHAPE-004",
                                f"dot operand '{n}' must be rank-2 block",
                                e.loc,
                                [f"    {n}: {TY.describe(v.vtype)}"])
        ad, bd = self._dtype_of(a), self._dtype_of(b)
        if ad not in D.DOT_INPUT or bd not in D.DOT_INPUT:
            raise TilaError(
                "TILA-TYPE-031", "dot operand dtype not supported", e.loc,
                [f"    a: {ad.name}", f"    b: {bd.name}",
                 f"    supported (v0): {', '.join(d.name for d in D.DOT_INPUT)}"])
        (m1, k1), (k2, n2) = a.vtype.dims, b.vtype.dims
        if not equal(k1, k2):
            s1, s2 = free_syms(k1), free_syms(k2)
            if not s1 and not s2:
                raise TilaError(          # 两个都是常量：立即判定
                    "TILA-SHAPE-004",
                    "dot inner dimensions do not match", e.loc,
                    [f"    a: {TY.describe(a.vtype)}",
                     f"    b: {TY.describe(b.vtype)}",
                     f"    required: {k1} == {k2}"],
                    ["检查 BLOCK_K / K 的 tile 划分"])
            consts = set(self.facts.num) | {c.name for c in self.tk.consts}
            if (s1 | s2) <= consts:
                self.tk.deferred.append({"kind": "eq", "loc": e.loc,
                                         "l": k1, "r": k2, "what":
                                         "dot inner dimensions"})
            else:
                raise TilaError(
                    "TILA-SHAPE-004",
                    "dot inner dimensions do not match", e.loc,
                    [f"    a: {TY.describe(a.vtype)}",
                     f"    b: {TY.describe(b.vtype)}",
                     f"    required: {k1} == {k2}"],
                    ["检查 BLOCK_K / K 的 tile 划分"])
        acc_dt = None
        if acc is not None:
            if not (isinstance(acc.vtype, TY.BlockT) and
                    len(acc.vtype.dims) == 2):
                raise TilaError("TILA-SHAPE-004",
                                "dot acc must be rank-2 block", e.loc)
            if not (equal(acc.vtype.dims[0], m1) and
                    equal(acc.vtype.dims[1], n2)):
                raise TilaError(
                    "TILA-SHAPE-004", "dot acc shape mismatch", e.loc,
                    [f"    acc: {TY.describe(acc.vtype)}",
                     f"    required: ({m1}, {n2})"])
            acc_dt = self._dtype_of(acc)
            if acc_dt not in D.DOT_ACC:
                raise TilaError("TILA-TYPE-031",
                                f"dot acc dtype {acc_dt.name} not in "
                                "DotAcc", e.loc)
        rdt = acc_dt or D.f32
        vt = TY.BlockT(TY.ScalarT(rdt), (m1, n2))
        return VarInfo(vtype=vt, tir=T.TDot(
            vt, self._operand_of(a, e.args[0], out),
            self._operand_of(b, e.args[1], out),
            self._operand_of(acc, e.kwargs["acc"], out)
            if acc is not None else None))

    def _in_sum(self, e, out):
        return self._reduce(e, "sum", out)

    def _in_max(self, e, out):
        return self._reduce(e, "max", out)

    def _reduce(self, e, op, out):
        if len(e.args) != 2:
            raise TilaError("TILA-SYN-036",
                            f"{op}(x, axis) takes 2 args", e.loc)
        x = self.synth(e.args[0], out)
        if not isinstance(x.vtype, TY.BlockT):
            raise TilaError("TILA-SHAPE-009",
                            f"{op} requires a block", e.loc)
        ax = e.args[1]
        dt = x.vtype.elem.dtype
        if dt is D.bool_:
            raise TilaError("TILA-TYPE-028", "numeric reduction requires arithmetic dtype; use .any()/.all() for bool", e.loc)
        if dt not in D.ARITH_DTYPES:
            raise TilaError("TILA-TYPE-036", "storage-only dtype: cast before reduction", e.loc)
        if not (isinstance(ax, Lit) and type(ax.value) is int):
            raise TilaError("TILA-CONST-002",
                            "reduce axis must be a literal int", e.loc)
        if ax.value not in range(len(x.vtype.dims)):
            raise TilaError("TILA-CONST-002",
                            f"axis {ax.value} out of range for rank "
                            f"{len(x.vtype.dims)}", e.loc)
        dims = tuple(d for i, d in enumerate(x.vtype.dims) if i != ax.value)
        vt = TY.BlockT(x.vtype.elem, dims) if dims else \
            TY.ScalarT(x.vtype.elem.dtype)
        return VarInfo(vtype=vt, tir=T.TReduce(vt, op,
                                               self._operand_of(x, e.args[0],
                                                                out),
                                               ax.value))

    def _in_exp(self, e, out):
        return self._exp_family(e, out, "exp")

    def _in_exp2(self, e, out):
        # intrinsics.md §2.7：exp2 与 exp 同签名（Float 域）。log2 域计算
        # 由 host 侧把 scale 乘 log2(e) 折入——FlashAttention 范式。
        return self._exp_family(e, out, "exp2")

    def _exp_family(self, e, out, op):
        if len(e.args) != 1:
            raise TilaError("TILA-SYN-036", f"{op}(x) takes 1 arg", e.loc)
        v = self.synth(e.args[0], out)
        dt = self._dtype_of(v)
        if dt is None or dt not in D.FLOAT_DTYPES:   # 不含 float_storage
            raise TilaError("TILA-TYPE-033",
                            f"{op} requires float operand, found "
                            f"{TY.describe(v.vtype)}", e.loc)
        return VarInfo(vtype=v.vtype,
                       tir=T.TUna(v.vtype, op,
                                  self._operand_of(v, e.args[0], out)))

    def _in_assume(self, e, out):
        if len(e.args) != 1:
            raise TilaError("TILA-SYN-036", "assume(pred) takes 1 arg", e.loc)
        v = self.synth(e.args[0], out)
        ok = isinstance(v.vtype, TY.MaskT) or (
            isinstance(v.vtype, TY.ScalarT) and v.vtype.dtype is D.bool_)
        if not ok:
            raise TilaError("TILA-TYPE-034",
                            "assume predicate must be bool/Mask", e.loc)
        # Preserve the current surface-language restriction; the internal DAG
        # can represent disjunction without treating its branches as facts.
        if not P.is_conjunction(v.predicate):
            raise TilaError(
                "TILA-CONST-004",
                "assume currently requires a symbolic conjunction; "
                "disjunction/negation support is deferred", e.loc,
                [], ["仅在各条件均为真时使用 ti.assume(a & b)，不能把析取拆成合取"])
        atoms = P.guaranteed(v.predicate)
        if not atoms:
            raise TilaError(
                "TILA-CONST-004",
                "assume predicate must be symbolically expressible "
                "(comparisons over index values)", e.loc,
                [], ["数据依赖的谓词无法进入证明器；如确信安全，用 unsafe_load"])
        for p in atoms:
            self.facts.add_pred(replace(p, origins=frozenset({
                P.Origin(P.USER, e.loc.line, f"assume({p.left} {p.op} {p.right})")})))
        out.append(T.TAssume(self._operand_of(v, e.args[0], out),
                             line=e.loc.line))
        return VarInfo(vtype=TY.UnitT())

    def _in_static_assert(self, e, out):
        pred = e.args[0]
        symbols = self._static_cond_syms(pred)
        if symbols is None:
            value = self.synth(pred, out)
            found = TY.describe(value.vtype) if value.vtype is not None else \
                f"literal {value.lit!r}"
            raise TilaError(
                "TILA-CONST-006",
                "static_assert needs a staged bool",
                e.loc,
                [f"found: {found} at runtime or outside the staged subset",
                 "required: Stage 1 bool or a predicate depending only on "
                 "Const[int]"],
                ["move runtime validation into ordinary control flow, or make "
                 "the predicate depend only on Const[int] parameters"],
            )
        if symbols or self.static_guards:
            value = self.synth(pred, out)
            if not (isinstance(value.vtype, TY.ScalarT) and
                    value.vtype.dtype is D.bool_):
                raise TilaError("TILA-CONST-006",
                                "static_assert needs a staged bool", e.loc)
            self.tk.deferred.append({
                "kind": "static_assert",
                "loc": e.loc,
                "pred": self._operand_of(value, pred, out),
                "symbols": tuple(sorted(symbols)),
                "guards": tuple(self.static_guards),
            })
            return VarInfo(vtype=TY.UnitT())
        cval = self._const_eval_bool(pred)
        if cval is None:
            raise TilaError("TILA-CONST-006",
                            "static_assert needs a compile-time bool", e.loc)
        if not cval:
            raise TilaError("TILA-CONST-005", "static_assert failed", e.loc,
                            ["predicate evaluated to False at Stage 1"])
        return VarInfo(vtype=TY.UnitT())

    def _in_range(self, e, out):
        raise TilaError("TILA-SYN-028",
                        "ti.range is only used as: for i in ti.range(0, end, "
                        "step)", e.loc)

    def _in_byte_offset(self, e, out):
        raise TilaError("TILA-SYN-050",
                        "ti.byte_offset is not in the v0 slice "
                        "(design: type-system.md §8.3)", e.loc)

    def _in_any(self, e, out):
        return self._mask_reduce(e, "any", out)

    def _in_all(self, e, out):
        return self._mask_reduce(e, "all", out)

    def _mask_reduce(self, e, op, out) -> VarInfo:
        """mask.any()/mask.all()（type-system.md §3.3、intrinsics.md §2.5）。

        Mask/Block[bool] → ScalarT(bool) 的 lane 归约：结果是运行期标量，不携带
        符号 expr / mask 谓词（逐 lane 值的归约无法进入证明器），
        可作 runtime-if 条件、and/or 操作数。
        """
        if len(e.args) != 1 or e.kwargs:
            raise TilaError("TILA-SYN-036",
                            f"{op}(mask) takes exactly one mask argument", e.loc)
        v = self.synth(e.args[0], out)
        if not (isinstance(v.vtype, (TY.MaskT, TY.BlockT))
                and self._is_boolean(v)):
            found = TY.describe(v.vtype) if v.vtype is not None else \
                f"literal {v.lit!r}"
            raise TilaError(
                "TILA-TYPE-019",
                f"{op}() requires a Mask or Block[bool], found {found}",
                e.loc, [f"    found: {found}"],
                ["标量 bool 直接用于 if / and / or；布尔 tile 或块谓词（如 m = offs < N）"
                 f"才有 .{op}()"])
        vt = TY.ScalarT(D.bool_)
        _, operand, _ = self._boolean_operand(v, e.args[0], out)
        return VarInfo(vtype=vt, tir=T.TUna(
            vt, op, operand))

    # -- 内存访问 ----------------------------------------------------------

    def _in_load(self, e, out):
        return self._access(e, out, is_store=False, unsafe=False)

    def _in_unsafe_load(self, e, out):
        return self._access(e, out, is_store=False, unsafe=True)

    def _in_store(self, e, out):
        return self._access(e, out, is_store=True, unsafe=False)

    def _in_unsafe_store(self, e, out):
        return self._access(e, out, is_store=True, unsafe=True)

    def _access(self, e: Call, out, is_store: bool, unsafe: bool):
        head = self.synth(e.args[0], out)
        buf_t = head.vtype if isinstance(head.vtype, TY.BufferT) else None
        ptr_t = head.vtype if self._is_ptrish(head.vtype) else None
        if buf_t is None and ptr_t is None:
            raise TilaError(
                "TILA-MEM-005",
                f"{'store' if is_store else 'load'} target must be a Buffer "
                "or pointer value", e.loc,
                [f"    found: {TY.describe(head.vtype)}"])
        kw = e.kwargs
        unknown_kw = set(kw) - ({"mask", "other"} if not is_store else
                                {"mask"})
        if unknown_kw:
            raise TilaError("TILA-SYN-036",
                            f"unknown keyword(s): {', '.join(sorted(unknown_kw))}",
                            e.loc)

        if buf_t is not None:
            return self._access_buffer(e, out, head, buf_t, is_store, unsafe)
        return self._access_ptr(e, out, head, is_store, unsafe)

    def _coords_of(self, e: Call, idx: int):
        arg = e.args[idx]
        if isinstance(arg, HTuple):
            return list(arg.items)
        return [arg]

    def _expand_index(self, expr, rank, inserted_axis):
        """Tile lane identities include their axes, so row and column views
        of the same vector cannot prove bounds for one another.
        Right-relative axes also make implicit left-padding broadcast stable.
        """
        if isinstance(expr, Cst):
            return expr
        if isinstance(expr, Sym):
            axes = self.lane_axes.get(expr.name)
            if axes is None:
                return expr
            new_axes = tuple(a - (rank + a < inserted_axis) for a in axes)
            if new_axes == axes:
                return expr
            name = expr.name.split("@")[0] + "@" + ",".join(map(str, new_axes))
            self.lane_axes[name] = new_axes
            if expr.name in self.value_types:
                self.value_types[name] = self.value_types[expr.name]
            if expr.name in self.value_defs:
                op, dtype, left, right = self.value_defs[expr.name]
                self.value_defs[name] = (op, dtype,
                    self._expand_index(left, rank, inserted_axis),
                    self._expand_index(right, rank, inserted_axis))
            for bounds in (self.facts.sym_lo, self.facts.sym_hi):
                if expr.name in bounds:
                    bounds[name] = bounds[expr.name]
            if expr.name in self.nonneg_syms:
                self.nonneg_syms.add(name)
            return Sym(name)
        return type(expr)(self._expand_index(expr.left, rank, inserted_axis),
                          self._expand_index(expr.right, rank, inserted_axis))

    def _predicate_of(self, mv: VarInfo):
        return mv.predicate

    def _check_mask(self, e, kw, expect_shape, out):
        if "mask" not in kw:
            return None, P.TRUE
        mv = self.synth(kw["mask"], out)
        shape, _, predicate = self._boolean_operand(
            mv, kw["mask"], out, "TILA-SHAPE-010")
        b = self._broadcast(shape, expect_shape, e.loc, "mask shape")
        if len(b) != len(expect_shape):
            raise TilaError("TILA-SHAPE-010", "mask shape does not match access", e.loc)
        for i, (x, y) in enumerate(zip(b, expect_shape)):
            if not self._dim_eq_defer(x, y, e.loc, "mask shape"):
                raise TilaError(
                    "TILA-SHAPE-010", "mask shape does not match access",
                    e.loc,
                    [f"    mask: {mv.vtype.describe()}",
                     f"    access shape: ({', '.join(map(str, expect_shape))})"])
        return mv, P.mapped(predicate, expect_shape,
                            ("broadcast", shape, expect_shape))

    def _snapshot_preds(self):
        return tuple(self.facts.preds.values())

    def _add_obligation(self, kind, source, axis, coord, extent, predicate,
                        loc_line):
        ob = Obligation(kind, source, axis, coord, extent, predicate,
                        loc_line, self._snapshot_preds(), self.facts.path,
                        tuple(self.facts.sym_lo.items()), tuple(self.facts.sym_hi.items()),
                        tuple(self.value_types.items()), tuple(self.value_defs.items()),
                        tuple(p.name for p in self.tk.scalars if p.dtype.is_int
                              and p.name in self.tk.explicit_scalars),
                        self.execution_context_exact)
        self.tk.obligations.append(ob)
        # Later memory operations may depend on the success/effects of this
        # access. Only the first access gets exact scalar reachability for now.
        self.execution_context_exact = False
        return ob

    def _access_buffer(self, e, out, head, bt: TY.BufferT, is_store, unsafe):
        if not isinstance(e.args[0], Name):
            raise TilaError("TILA-MEM-006",
                            "Buffer access must name the buffer directly "
                            "(tila.load(buf, coords))", e.loc)
        bname = e.args[0].id
        coords_e = self._coords_of(e, 1)
        if is_store and len(e.args) != 3:
            raise TilaError("TILA-SYN-036",
                            "store(buf, coords, value, mask=)", e.loc)
        if not is_store and len(e.args) != 2:
            raise TilaError("TILA-SYN-036", "load(buf, coords, mask=, other=)",
                            e.loc)
        if len(coords_e) != len(bt.dims):
            raise TilaError(
                "TILA-SHAPE-011", "coordinate rank does not match buffer",
                e.loc,
                [f"    buffer: {bt.describe()}",
                 f"    got {len(coords_e)} coordinate(s)"])

        if is_store and bt.access not in (TY.WRITE_ONLY, TY.READ_WRITE):
            raise TilaError(
                "TILA-MEM-001", "cannot store through a ReadOnly buffer",
                e.loc,
                [f"    buffer '{bname}': {bt.describe()}"],
                ["参数声明改为 ti.ReadWrite / ti.WriteOnly"])

        coords = [self.synth(c, out) for c in coords_e]
        shape = ()
        for cv in coords:
            if type(cv.lit) is int:
                self._instantiate(cv.lit, D.i32, e.loc)
                cv.vtype, cv.expr = TY.ScalarT(D.i32), Cst(cv.lit)
            dt = self._dtype_of(cv)
            if dt is None or not dt.is_int:
                raise TilaError("TILA-TYPE-035",
                                f"coordinate must be an integer index, found "
                                f"{TY.describe(cv.vtype)}", e.loc)
            shape = self._broadcast(shape, shape_of(cv), e.loc,
                                    "coordinate shapes") \
                if shape else shape_of(cv)
        if shape == ():
            shape = ()

        mask_v, predicate = self._check_mask(e, e.kwargs, shape, out)

        if is_store:
            val = self.synth(e.args[2], out)
            vdt = self._dtype_of(val)
            if val.lit is not None and vdt is None:
                vdt = self._instantiate(val.lit, bt.elem, e.loc)
                val.vtype = TY.ScalarT(vdt)
            if vdt is not bt.elem:
                raise TilaError(
                    "TILA-TYPE-016",
                    "store value dtype must match buffer element type "
                    "exactly (no implicit conversion)", e.loc,
                    [f"    value:  {vdt.name if vdt else '?'}",
                     f"    buffer: {bt.elem.name}"],
                    [f"ti.cast[{bt.elem.name}](value) 显式转换"])
            vshape = shape_of(val)
            if len(vshape) != len(shape) or not all(
                    self._dim_eq_defer(x, y, e.loc, "store value shape")
                    for x, y in zip(vshape, shape)):
                raise TilaError(
                    "TILA-SHAPE-012",
                    "store value shape must match the access shape exactly",
                    e.loc,
                    [f"    value:  ({', '.join(map(str, vshape)) or 'scalar'})",
                     f"    access: ({', '.join(map(str, shape)) or 'scalar'})"])
            kind = "unsafe_store" if unsafe else "store"
            self.tk.effects.append(T.TEffect("Write", self.regions[bname]))
            for i, cv in enumerate(coords):
                self._add_obligation(kind, bname, i, cv.expr
                                     if cv.expr is not None else None,
                                     bt.dims[i], predicate, e.loc.line)
            out.append(T.TStore(
                bname, [self._operand_of(cv, ce, out)
                        for cv, ce in zip(coords, coords_e)],
                None, self._operand_of(val, e.args[2], out),
                self._operand_of(mask_v, e.kwargs["mask"], out)
                if mask_v is not None else None,
                unsafe, line=e.loc.line))
            return VarInfo(vtype=TY.UnitT())

        # load
        other_op = None
        if "other" in e.kwargs:
            ov = self.synth(e.kwargs["other"], out)
            odt = self._dtype_of(ov)
            if ov.lit is not None and odt is None:
                odt = self._instantiate(ov.lit, bt.elem, e.loc)
                ov.vtype = TY.ScalarT(odt)
            if odt is not bt.elem:
                raise TilaError("TILA-TYPE-016",
                                f"other must be {bt.elem.name}", e.loc)
            # other 形状须与访问形状 broadcast-compatible（§4.3：纯 Const
            # 维不等价延迟到特化期复核）
            self._broadcast(shape_of(ov), shape, e.loc, "other shape")
            other_op = self._operand_of(ov, e.kwargs["other"], out)
        vt = TY.BlockT(TY.ScalarT(bt.elem), shape) if shape else \
            TY.ScalarT(bt.elem)
        kind = "unsafe_load" if unsafe else "load"
        self.tk.effects.append(T.TEffect("Read", self.regions[bname]))
        for i, cv in enumerate(coords):
            self._add_obligation(kind, bname, i,
                                 cv.expr if cv.expr is not None else None,
                                 bt.dims[i], predicate, e.loc.line)
        return VarInfo(vtype=vt, tir=T.TLoad(
            vt, bname,
            [self._operand_of(cv, ce, out)
             for cv, ce in zip(coords, coords_e)],
            None,
            self._operand_of(mask_v, e.kwargs["mask"], out)
            if mask_v is not None else None,
            other_op, unsafe))

    def _access_ptr(self, e, out, head, is_store, unsafe):
        pt = head.vtype
        base_pt = pt if isinstance(pt, TY.PtrT) else pt.elem
        extent = TY.extent_expr(base_pt.extent)
        region_id = base_pt.region_id
        source_name = getattr(region_id, "name", "unknown")
        shape = shape_of(head)
        mask_v, predicate = self._check_mask(e, e.kwargs, shape, out)

        if is_store:
            if len(e.args) != 2:
                raise TilaError("TILA-SYN-036",
                                "store(ptr, value, mask=)", e.loc)
            if base_pt.access not in (TY.WRITE_ONLY, TY.READ_WRITE):
                raise TilaError(
                    "TILA-MEM-001", "cannot store through a ReadOnly pointer",
                    e.loc, [f"    pointer: {TY.describe(base_pt)}"],
                    ["ReadPtr → WritePtr/RWPtr，或改用 Buffer 形式"])
            val = self.synth(e.args[1], out)
            vdt = self._dtype_of(val)
            if val.lit is not None and vdt is None:
                vdt = self._instantiate(val.lit, base_pt.elem, e.loc)
                val.vtype = TY.ScalarT(vdt)
            if vdt is not base_pt.elem:
                raise TilaError(
                    "TILA-TYPE-016",
                    "store value dtype must match pointer element type "
                    "exactly", e.loc,
                    [f"    value:   {vdt.name if vdt else '?'}",
                     f"    pointer: {base_pt.elem.name}"],
                    [f"ti.cast[{base_pt.elem.name}](value)"])
            kind = "unsafe_store" if unsafe else "store"
            self.tk.effects.append(T.TEffect("Write", region_id))
            self._add_obligation(kind, f"ptr:{source_name}", None,
                                 head.expr, extent, predicate, e.loc.line)
            out.append(T.TStore(
                None, [], self._operand_of(head, e.args[0], out),
                self._operand_of(val, e.args[1], out),
                self._operand_of(mask_v, e.kwargs["mask"], out)
                if mask_v is not None else None,
                unsafe, line=e.loc.line))
            return VarInfo(vtype=TY.UnitT())

        if len(e.args) != 1:
            raise TilaError("TILA-SYN-036", "load(ptr, mask=, other=)", e.loc)
        if base_pt.access not in (TY.READ_ONLY, TY.READ_WRITE):
            raise TilaError("TILA-MEM-001",
                            "cannot load through a WriteOnly pointer", e.loc,
                            [f"    pointer: {TY.describe(base_pt)}"])
        other_op = None
        if "other" in e.kwargs:
            ov = self.synth(e.kwargs["other"], out)
            odt = self._dtype_of(ov)
            if ov.lit is not None and odt is None:
                odt = self._instantiate(ov.lit, base_pt.elem, e.loc)
                ov.vtype = TY.ScalarT(odt)
            if odt is not base_pt.elem:
                raise TilaError("TILA-TYPE-016",
                                f"other must be {base_pt.elem.name}", e.loc)
            other_op = self._operand_of(ov, e.kwargs["other"], out)
        vt = TY.BlockT(TY.ScalarT(base_pt.elem), shape) if shape else \
            TY.ScalarT(base_pt.elem)
        kind = "unsafe_load" if unsafe else "load"
        self.tk.effects.append(T.TEffect("Read", region_id))
        self._add_obligation(kind, f"ptr:{source_name}", None,
                             head.expr, extent, predicate, e.loc.line)
        return VarInfo(vtype=vt, tir=T.TLoad(
            vt, None, [], self._operand_of(head, e.args[0], out),
            self._operand_of(mask_v, e.kwargs["mask"], out)
            if mask_v is not None else None, other_op, unsafe))

    # ------------------------------------------------------------------
    # 操作数物化
    # ------------------------------------------------------------------

    def _operand_of(self, v: VarInfo, e, out) -> T.TOperand:
        if v.tir is not None:
            return v.tir
        if isinstance(e, Name):
            return T.TName(e.id)
        if v.lit is not None:
            dt = v.vtype.dtype if isinstance(v.vtype, TY.ScalarT) else \
                _lit_default_dtype(v.lit)
            return T.TLit(v.lit, dt)
        if isinstance(e, BufPtr):
            return v.tir if v.tir is not None else T.TName(e.buf)
        raise TilaError("TILA-TYPE-018", "cannot materialize operand", e.loc)


# ADR-006: explicit symbolic handler map. Registry IDs are not interpreted via
# `_in_<name>` reflection, so renaming a method cannot silently alter dispatch.
CHECKER_INTRINSIC_HANDLERS = {
    "program_id": Checker._in_program_id,
    "num_programs": Checker._in_num_programs,
    "arange": Checker._in_arange,
    "range": Checker._in_range,
    "load": Checker._in_load,
    "store": Checker._in_store,
    "unsafe_load": Checker._in_unsafe_load,
    "unsafe_store": Checker._in_unsafe_store,
    "cast": Checker._in_cast,
    "constant": Checker._in_constant,
    "where": Checker._in_where,
    "dot": Checker._in_dot,
    "zeros": Checker._in_zeros,
    "sum": Checker._in_sum,
    "max": Checker._in_max,
    "exp": Checker._in_exp,
    "exp2": Checker._in_exp2,
    "assume": Checker._in_assume,
    "static_assert": Checker._in_static_assert,
    "byte_offset": Checker._in_byte_offset,
    "reshape": Checker._in_reshape,
    "any": Checker._in_any,
    "all": Checker._in_all,
}
