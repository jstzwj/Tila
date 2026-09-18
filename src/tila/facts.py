"""事实环境与边界证明求解器（docs/bounds-safety.md §2–§5、refinements.md §3）。

fast path = 区间抽象（仅 Const 数值参与，保证跨 launch 缓存可靠）
          + 谓词直证（mask/assume 谓词 canon 匹配）
          + cdiv 战术（grid 契约 + 整除契约的定向推理）
只有 Proven 算证明；Unknown 永不升级为 Safe（soundness 底线）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dims import (Add, Cst, DimExpr, Mul, Sym, canon, equal, int_value)

PROVEN_SAFE = "ProvenSafe"
SAFE_UNDER_CONTRACT = "SafeUnderContract"
UNKNOWN = "Unknown"
PROVEN_UNSAFE = "ProvenUnsafe"


@dataclass
class Pred:
    op: str                 # < <= ==
    left: DimExpr
    right: DimExpr

    def key(self):
        l, r, op = self.left, self.right, self.op
        if op == ">":
            l, r, op = r, l, "<"
        elif op == ">=":
            l, r, op = r, l, "<="
        if op == "==":
            kl, kr = canon(l), canon(r)
            if kr < kl:
                l, r = r, l
        return f"{op}:{canon(l)}:{canon(r)}"


@dataclass
class Facts:
    """一个程序点的符号事实集（单调增长；分支合并取交）。"""
    # 变量 → 符号索引表达式（None = 无索引意义）
    var_exprs: dict = field(default_factory=dict)
    # 符号 → (含)下界 / (不含)上界（DimExpr；默认未知）
    sym_lo: dict = field(default_factory=dict)
    sym_hi: dict = field(default_factory=dict)
    # 数值环境：仅 Const 符号（进入特化缓存键，跨 launch 不变 ⇒ 证明可靠）
    num: dict = field(default_factory=dict)
    # 谓词集（assume / 契约）
    preds: dict = field(default_factory=dict)     # key -> Pred
    # pid 符号 → (bound_expr, step_expr)：grid_ax == ceildiv(bound, step)
    # 且 0 <= pid < grid_ax（由 launcher 的 grid 契约登记）
    grid_facts: dict = field(default_factory=dict)

    def clone(self) -> "Facts":
        return Facts(dict(self.var_exprs), dict(self.sym_lo), dict(self.sym_hi),
                     dict(self.num), dict(self.preds), dict(self.grid_facts))

    def intersect(self, other: "Facts") -> "Facts":
        """分支合并：只保留两侧共同可推出的事实（保守正确）。"""
        out = self.clone()
        out.var_exprs = {k: v for k, v in out.var_exprs.items()
                         if other.var_exprs.get(k) == v}
        out.preds = {k: p for k, p in out.preds.items() if k in other.preds}
        return out

    def add_pred(self, pred: Pred):
        self.preds[pred.key()] = pred

    # -- 数值区间（仅 Const 参与）----------------------------------------

    def num_interval(self, e: DimExpr):
        lo = self._num_iv(e, True)
        hi = self._num_iv(e, False)
        return lo, hi

    def _num_iv(self, e: DimExpr, is_lo: bool):
        if isinstance(e, Cst):
            return e.value
        if isinstance(e, Sym):
            v = self.num.get(e.name)
            if v is not None:
                return v
            # 归纳符号（lane / loop / pid）的区间事实：sym_lo 为含下界、
            # sym_hi 为排他上界。仅当界表达式自身能由 Const 解出时才给值
            # （num 只含 Const 数值 ⇒ 跨 launch 缓存可靠，此规则不破坏）。
            lo_e = self.sym_lo.get(e.name)
            hi_e = self.sym_hi.get(e.name)
            if lo_e is None and hi_e is None:
                return None
            lo = self._num_iv(lo_e, True) if lo_e is not None else None
            raw_hi = self._num_iv(hi_e, False) if hi_e is not None else None
            hi = raw_hi - 1 if raw_hi is not None else None  # 排他 → 含
            if is_lo:
                return lo
            return hi
        if isinstance(e, Add):
            a, b = self._num_iv(e.left, is_lo), self._num_iv(e.right, is_lo)
            if a is None or b is None:
                return None
            return a + b
        if isinstance(e, Mul):
            lv, hv = self.num_interval(e.left)
            rv, wv = self.num_interval(e.right)
            if None in (lv, hv, rv, wv):
                return None
            cands = (lv * rv, lv * wv, hv * rv, hv * wv)
            return min(cands) if is_lo else max(cands)
        return None


def is_nonneg_expr(e: DimExpr, extra_nonneg: set[str]) -> bool:
    """结构性非负判定：pid/lane/range 变量/维符号/正常量出发的仿射式。"""
    if isinstance(e, Cst):
        return e.value >= 0
    if isinstance(e, Sym):
        return e.name in extra_nonneg
    if isinstance(e, Add):
        return is_nonneg_expr(e.left, extra_nonneg) and \
            is_nonneg_expr(e.right, extra_nonneg)
    if isinstance(e, Mul):
        return is_nonneg_expr(e.left, extra_nonneg) and \
            is_nonneg_expr(e.right, extra_nonneg)
    return False


# ---------------------------------------------------------------------------
# 义务与求解
# ---------------------------------------------------------------------------

@dataclass
class Obligation:
    kind: str               # load / store / unsafe_load / unsafe_store
    source: str             # 诊断用访问来源；不参与 effect/alias 身份分析
    axis: int | None        # Buffer 形式逐轴；Ptr 形式 None
    coord: DimExpr | None   # None = 数据依赖坐标（不可符号化）
    extent: DimExpr | None  # 数值上界；None = Ptr 为 UnknownExtent
    mask_preds: list        # DNF 子句列表（bounds-safety.md §3.2）：
                            # 子句 = Pred 合取；mask 为真 ⇒ 某一子句成立
    loc_line: int = 0
    contract: bool = False  # 证明是否依赖 launch 契约（cdiv 战术）
    preds_snapshot: list = field(default_factory=list)  # 访问点全局谓词

    def describe(self):
        axis = f"axis {self.axis}" if self.axis is not None else "flat"
        coord = self.coord if self.coord is not None else "<data-dependent>"
        extent = self.extent if self.extent is not None else \
            "<unknown extent>"
        return f"{self.kind}[{self.source}] {axis}: 0 <= {coord} < {extent}"


def _decompose_linear(e: DimExpr):
    """Add 树 → [(term, coef)]（线性项；不透明原子整体作 term）。"""
    if isinstance(e, Add):
        return _decompose_linear(e.left) + _decompose_linear(e.right)
    if isinstance(e, Mul):
        c = int_value(e.left)
        if c is not None:
            return [(e.right, c)]
        c = int_value(e.right)
        if c is not None:
            return [(e.left, c)]
    return [(e, 1)]


def evaluate_obligation(ob: Obligation, facts: Facts, nonneg_syms: set[str]) -> str:
    """四态判定（bounds-safety.md §6）。mask 谓词在该访问点视为真。

    mask 谓词为 DNF（§3.2）：mask_preds = 子句的析取，子句 = 谓词的合取。
    mask ⇒ goal 要求**每个子句**各自 ⇒ goal（析取支全部可证才算证明）；
    任一子句失败 ⇒ Unknown。无 mask（零子句）与 `~` 产生的空子句 [[]]
    都按"单个空子句"处理：仅凭全局事实证明。
    """
    if ob.kind.startswith("unsafe"):
        return PROVEN_SAFE  # 义务被显式豁免（报告仍可见）
    if ob.coord is None or ob.extent is None:
        return UNKNOWN      # 数据依赖坐标 / UnknownExtent 指针：不可符号化

    clauses = ob.mask_preds if ob.mask_preds else [[]]
    worst = PROVEN_SAFE
    for clause in clauses:
        st = _prove_clause(ob, facts, nonneg_syms, clause)
        if st == PROVEN_UNSAFE:
            return PROVEN_UNSAFE
        if st is None:
            return UNKNOWN
        if st == SAFE_UNDER_CONTRACT:
            worst = SAFE_UNDER_CONTRACT
    return worst


def _prove_clause(ob: Obligation, facts: Facts, nonneg_syms: set[str],
                  clause: list):
    """单子句证明：goal 在 facts.preds + snapshot + 子句谓词下可证？

    返回 PROVEN_SAFE / SAFE_UNDER_CONTRACT / PROVEN_UNSAFE / None（未证出）。
    判定逻辑在 _prove_clause_detail（explain 模式共享同一实现以保持一致）。
    """
    state, contract, _route = _prove_clause_detail(ob, facts, nonneg_syms,
                                                   clause)
    if contract:
        ob.contract = True    # cdiv 战术命中（含 lower 未证出的 None 分支）
    return state


def _pred_text(p: Pred) -> str:
    """谓词的可读渲染：`offs < N` 风格（DimExpr 的 str 即 canonical 形态）。"""
    return f"{p.left} {p.op} {p.right}"


def _pred_source(p: Pred, facts: Facts, ob: Obligation, clause: list) -> str:
    """谓词来源标签（就近优先：mask 子句 > 访问点快照 > 事实集）。"""
    key = p.key()
    if any(c.key() == key for c in clause):
        return "mask clause"
    if any(s.key() == key for s in ob.preds_snapshot):
        return "global predicate snapshot at access point (assume/contract)"
    return "fact set"


def _iv(x) -> str:
    """区间端点渲染：None = 无穷 / 未定。"""
    return "?" if x is None else str(x)


def _prove_clause_detail(ob: Obligation, facts: Facts, nonneg_syms: set[str],
                         clause: list):
    """_prove_clause 的可审计内核：同一判定逻辑（路线顺序、条件、短路全部
    一致），额外产出证明路线文本。

    返回 (state, contract, route)：
    - state 与 _prove_clause 完全一致（四态语义不变）；
    - contract = cdiv 战术命中（_prove_clause 据此回写 ob.contract；
      本函数只读，不改动义务/事实——explain 模式的副作用底线）；
    - route = 人类可读证明路线（surface-language.md §8 的义务证明链）。
    """
    pool = list(facts.preds.values()) + list(ob.preds_snapshot) + list(clause)

    # 下界：结构性非负（pid/lane/维/常量）或谓词直证
    lower_ok = is_nonneg_expr(ob.coord, nonneg_syms)
    lower_src = ("structural nonnegativity (pid/lane/dim/Const terms)"
                 if lower_ok else None)
    if not lower_ok:
        low_key = Pred(">=", ob.coord, Cst(0)).key()
        for p in pool:
            if p.key() == low_key:
                lower_ok = True
                lower_src = (f"predicate `{_pred_text(p)}` "
                             f"({_pred_source(p, facts, ob, clause)})")
                break

    # 上界三条路
    # (a) 谓词直证：clause ⇒ coord < bound
    goal = Pred("<", ob.coord, ob.extent).key()
    for p in pool:
        if p.key() == goal:
            route = (f"direct predicate `{_pred_text(p)}` "
                     f"({_pred_source(p, facts, ob, clause)}) proves the "
                     f"upper bound")
            if lower_src is not None:
                route += f"; lower bound by {lower_src}"
            return PROVEN_SAFE, False, route

    # (b) 数值区间：expr 仅含 Const（进入缓存键，可靠）
    lo, hi = facts.num_interval(ob.coord)
    b_lo, b_hi = facts.num_interval(ob.extent)
    if hi is not None and b_lo is not None and hi < b_lo:
        if lower_ok:
            return (PROVEN_SAFE, False,
                    f"numeric interval: coord {ob.coord} ∈ "
                    f"[{_iv(lo)}, {_iv(hi)}], extent {ob.extent} ∈ "
                    f"[{_iv(b_lo)}, {_iv(b_hi)}] ⇒ {_iv(hi)} < {_iv(b_lo)}; "
                    f"lower bound by {lower_src}")
        return (None, False,
                f"numeric interval proves the upper bound ({ob.coord} ≤ "
                f"{_iv(hi)} < {_iv(b_lo)}) but the lower bound "
                f"0 <= {ob.coord} is unproven (not structurally nonneg, no "
                f"`{ob.coord} >= 0` predicate in scope)")

    # ProvenUnsafe：Const 域可证整体越界
    if lo is not None and b_hi is not None and lo >= b_hi:
        return (PROVEN_UNSAFE, False,
                f"provably out of bounds: coord lower bound {lo} >= bound "
                f"upper bound {b_hi}")
    if lo is not None and b_lo is not None and lo >= b_lo and hi is not None and hi >= b_lo and lo == hi:
        return (PROVEN_UNSAFE, False,
                f"provably out of bounds: constant coord {lo} >= bound "
                f"lower bound {b_lo}")
    if hi is not None and hi < 0:
        return (PROVEN_UNSAFE, False,
                f"provably negative coordinate: upper bound {hi} < 0")

    # (c) cdiv 战术：coord = pid*STEP + lane，grid == ceildiv(bound, STEP)，
    #     契约 bound % STEP == 0，且 pid/lane 界内 ⇒ 上界成立（依赖契约）
    terms = _decompose_linear(ob.coord)
    # (c0) 精确维 grid：coord 单独是 pid，grid == bound（step 1）⇒ 界内
    if len(terms) == 1 and isinstance(terms[0][0], Sym) and terms[0][1] == 1:
        grid = facts.grid_facts.get(terms[0][0].name)
        if grid is not None and equal(grid[1], Cst(1)) and \
                equal(grid[0], ob.extent) and lower_ok:
            return (PROVEN_SAFE, False,
                    f"exact-dim grid fact: {terms[0][0].name} spans "
                    f"[0, {grid[0]}) with step 1 (registered at launch); "
                    f"lower bound by {lower_src}")
    if len(terms) == 2:
        for (t1, c1), (t2, c2) in ((terms[0], terms[1]), (terms[1], terms[0])):
            if c1 != 1 or c2 != 1 or not isinstance(t2, Sym):
                continue
            lane_sym = t2
            step = facts.sym_hi.get(lane_sym.name)
            if step is None:
                continue
            pid_name = _match_pid_times_step(t1, step)
            if pid_name is None:
                continue
            grid = facts.grid_facts.get(pid_name)
            if grid is None:
                continue
            g_bound, g_step = grid
            if not equal(g_step, step) or not equal(g_bound, ob.extent):
                continue
            # 契约：bound % step == 0（整除谓词须在该子句上下文中可得：
            # facts / snapshot / 子句谓词）
            div_key = Pred("==", _mod(ob.extent, step), Cst(0)).key()
            div_pred = next((p for p in pool if p.key() == div_key), None)
            if div_pred is not None:
                route = (f"cdiv tactic: {ob.coord} = {pid_name}*{step} + "
                         f"{lane_sym.name}; grid axis = "
                         f"ceildiv({g_bound}, {step}) ⇒ {pid_name} < "
                         f"{g_bound}; divisibility contract "
                         f"`{_pred_text(div_pred)}` "
                         f"({_pred_source(div_pred, facts, ob, clause)}) — "
                         f"depends on launch contract")
                if not lower_ok:
                    route += (f"; lower bound 0 <= {ob.coord} unproven "
                              f"(not structurally nonneg, no "
                              f"`{ob.coord} >= 0` predicate in scope)")
                return (SAFE_UNDER_CONTRACT if lower_ok else None), True, route

    # 全部路线未命中：说明证明缺口（explain 的 per-clause 失败清单）
    gap = (f"no upper-bound predicate `{ob.coord} < {ob.extent}` in scope "
           f"(fact set: {len(facts.preds)} pred(s); access-point snapshot: "
           f"{len(ob.preds_snapshot)}; mask clause: {len(clause)}); numeric "
           f"interval not decisive (coord ∈ [{_iv(lo)}, {_iv(hi)}], bound ∈ "
           f"[{_iv(b_lo)}, {_iv(b_hi)}]); no matching grid fact (grid facts "
           f"are launch-time contracts — not available here)")
    if not lower_ok:
        gap += (f"; lower bound 0 <= {ob.coord} also unproven (not "
                f"structurally nonneg, no `{ob.coord} >= 0` predicate)")
    return None, False, gap


def explain_obligation(ob: Obligation, facts: Facts,
                       nonneg_syms: set[str]) -> str:
    """explain 模式的义务评估（surface-language.md §8 --explain）：
    与 evaluate_obligation 相同的四态判定 + 逐子句证明路线，只读——
    不置 ob.contract、不改 facts。

    状态聚合与 evaluate_obligation 逐分支一致：unsafe 义务显式豁免；
    数据依赖坐标 / UnknownExtent 指针为 Unknown；mask 谓词 DNF 下每个子句
    各给一行路线（析取支全部可证才算证明，首个未证出/可证越界的子句
    决定状态——其余子句仍列出以便审计）。
    """
    if ob.kind.startswith("unsafe"):
        return (f"state: {PROVEN_SAFE}\n"
                "proof: explicitly waived (unsafe access) — the obligation "
                "is recorded but exempt from proof")
    if ob.coord is None or ob.extent is None:
        why = ("coordinate is data-dependent (no symbolic index expression)"
               if ob.coord is None else
               "pointer parameter has no Region — bound unknown")
        return f"state: {UNKNOWN}\nproof: {why}; not provable statically"

    clauses = ob.mask_preds if ob.mask_preds else [[]]
    lines = []
    decided = None        # 首个 PROVEN_UNSAFE / None 子句（评估器即刻定格）
    saw_contract = False
    for i, clause in enumerate(clauses, 1):
        state, _contract, route = _prove_clause_detail(ob, facts, nonneg_syms,
                                                       clause)
        label = (f"clause {i}/{len(clauses)}" if ob.mask_preds
                 else "no mask — global facts only")
        lines.append(f"proof: {label}: {route}")
        if decided is None:
            if state == PROVEN_UNSAFE:
                decided = PROVEN_UNSAFE
            elif state is None:
                decided = UNKNOWN
        if state == SAFE_UNDER_CONTRACT:
            saw_contract = True
    if decided is not None:
        final = decided
    elif saw_contract:
        final = SAFE_UNDER_CONTRACT
    else:
        final = PROVEN_SAFE
    return f"state: {final}\n" + "\n".join(lines)


def _match_pid_times_step(term, step) -> str | None:
    """term 是否为 Mul(pid_sym, step) 形态；是则返回 pid 符号名。"""
    if not isinstance(term, Mul):
        return None
    if isinstance(term.right, Sym) and equal(term.left, step):
        return term.right.name
    if isinstance(term.left, Sym) and equal(term.right, step):
        return term.left.name
    return None


def _mod(a, b):
    from .dims import Mod
    return Mod(a, b)
