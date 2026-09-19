"""Runtime：@ti.jit、特化（Stage 2）、launch 契约、双后端（Triton / interp）。

launch 协议（surface-language.md §6）：
    kernel[grid](tensors..., scalars..., CONST=v, ...)
    kernel.launch_auto(tensors..., scalars..., CONST=v, ...)
grid 元素为 int / ti.cdiv(a, b) / callable(meta)；launch_auto 的 grid
由 launch analysis 推导（_derive_grid：义务的 pid*STEP+lane 规范模式）。
"""

from __future__ import annotations

from . import numeric

import ast
import dataclasses
import os
import sys

import numpy as np

from . import dtypes as D
from . import frontend
from . import interp as interp_mod
from . import lowering
from .checker import check_kernel
from .dims import (Cst, DimExpr, Mul, Sym, equal, free_syms, int_value,
                   rewrite_syms)
from .errors import Loc, TilaError, TilaLaunchContractError
from .facts import (Facts, Pred, _decompose_linear, evaluate_obligation,
                    explain_obligation, PROVEN_SAFE, PROVEN_UNSAFE,
                    SAFE_UNDER_CONTRACT, UNKNOWN)
from .intrinsics import INTRINSIC_REGISTRY_SEMANTIC_REVISION
from . import types as TY
from . import tir as T

try:
    import torch
    torch.Tensor        # 功能校验：残留的空目录会以命名空间包形式导入成功
except (ImportError, AttributeError):
    torch = None

_TORCH_DT = {
    "float16": D.f16, "float32": D.f32, "float64": D.f64,
    "bfloat16": D.bf16, "int8": D.i8, "int16": D.i16, "int32": D.i32,
    "int64": D.i64, "uint8": D.u8, "bool": D.bool_,
}


def _eval_const_operand(node, consts: dict):
    """Evaluate the typed staged-bool TIR subset used by static_assert."""
    if isinstance(node, T.TName):
        if node.name not in consts:
            raise ValueError(f"unresolved Const symbol {node.name!r}")
        return consts[node.name]
    if isinstance(node, T.TLit):
        return node.value
    if isinstance(node, T.TUna):
        value = _eval_const_operand(node.operand, consts)
        if node.op == "-":
            return -value
        if node.op == "not":
            return not value
        raise ValueError(f"unsupported staged unary operator {node.op!r}")
    if isinstance(node, T.TBin):
        left = _eval_const_operand(node.left, consts)
        if node.op == "and":
            return bool(left) and bool(_eval_const_operand(node.right, consts))
        if node.op == "or":
            return bool(left) or bool(_eval_const_operand(node.right, consts))
        right = _eval_const_operand(node.right, consts)
        import operator
        operations = {
            "+": operator.add, "-": operator.sub, "*": operator.mul,
            "//": operator.floordiv, "%": operator.mod,
            "<": operator.lt, "<=": operator.le, ">": operator.gt,
            ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
        }
        operation = operations.get(node.op)
        if operation is None:
            raise ValueError(f"unsupported staged binary operator {node.op!r}")
        return operation(left, right)
    raise TypeError(f"unsupported staged operand {type(node).__name__}")


def _render_const_operand(node) -> str:
    if isinstance(node, T.TName):
        return node.name
    if isinstance(node, T.TLit):
        return repr(node.value)
    if isinstance(node, T.TUna):
        return f"({node.op} {_render_const_operand(node.operand)})"
    if isinstance(node, T.TBin):
        return (f"({_render_const_operand(node.left)} {node.op} "
                f"{_render_const_operand(node.right)})")
    return repr(node)
_NP_DT = {np.dtype(dt.np_dtype).name: dt for dt in
          (D.bool_, D.i8, D.i16, D.i32, D.i64, D.u8, D.u16, D.u32, D.u64,
           D.f16, D.f32, D.f64)}
try:  # bf16：ml_dtypes 可用时 numpy bfloat16 数组可绑定（interp 同源）
    import ml_dtypes
    _NP_DT[np.dtype(ml_dtypes.bfloat16).name] = D.bf16
except ImportError:
    pass


class _Cdiv:
    """ti.cdiv(a, b) grid 标记：携带 (a, b) 以便识别 cdiv 模式登记 pid 事实。"""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def value(self, meta: dict) -> int:
        a = self.a(meta) if callable(self.a) else self.a
        b = self.b(meta) if callable(self.b) else self.b
        if type(a) is not int or type(b) is not int or a < 0 or b <= 0:
            raise TilaLaunchContractError(
                "TILA-TYPE-104", "cdiv grid requires exact integers a >= 0 and b > 0")
        return -(-a // b)


def cdiv(a, b):
    return _Cdiv(a, b)


def _jit(fn):
    if not callable(fn):
        raise TilaError("TILA-SYN-000", "@ti.jit decorates a function")
    return JITFunction(fn)


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        hir = frontend.compile_stage1(fn)
        self.tk: T.TKernel = check_kernel(hir)     # Stage 1（装饰期）
        self.assume_launch = None                  # (preds, ast) 由装饰器注入
        self._kern_cache: dict = {}
        self.last_report: str = ""
        self.last_proof_results: tuple = ()

    # ------------------------------------------------------------------

    def __getitem__(self, grid):
        if grid is None:    # launch_auto 通道：grid 延迟到绑定后推导
            return _Launcher(self, None)
        if not isinstance(grid, tuple):
            grid = (grid,)
        if not 1 <= len(grid) <= 3:
            raise TilaLaunchContractError("TILA-TYPE-104", "grid must have one to three axes")
        return _Launcher(self, grid)

    def launch_auto(self, *args, **kwargs):
        """grid 由 launch analysis 推导（surface-language.md §6）：绑定
        张量/Const 后按义务的 pid*STEP+lane / 标量 pid 模式推导每轴。"""
        return self[None](*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise TilaError(
            "TILA-SYN-060",
            "kernels launch with an explicit grid or an auto-derived one: "
            "kernel[grid](args...) / kernel.launch_auto(args...)",
            fixes=[f"如 {self.__name__}[(ti.cdiv(n, BLOCK),)](...) 或 "
                   f"{self.__name__}.launch_auto(...)"])

    # -- CLI / explain ------------------------------------------------------

    def materialize(self, consts: dict | None = None):
        """无 launch 的特化（check/build）：解 Const、复查延迟约束与义务。"""
        cenv = self._resolve_consts(consts)
        self._check_consts(cenv)
        self._check_deferred(cenv)
        numeric_pending = numeric.validate(self.tk, cenv)
        # check/build 无 launch 数值：以符号 grid 契约代位（纯 pid 坐标 +
        # 维符号上界 ⇒ grid_ax == bound, step 1）。launch 期以数值复核
        # （精确维登记不匹配 ⇒ 无事实 ⇒ 义务失败拒绝执行），契约由
        # launcher 强制，此推导 sound。
        self._evaluate_obligations(cenv, grid_facts=self._symbolic_grid_facts(),
                                   extra_preds=[], launch_checked=False)
        if numeric_pending:
            self.last_report += "\n  integer launch contracts pending: " + "; ".join(dict.fromkeys(numeric_pending))
        src = lowering.Lowering(self.tk, _debug()).kernel_source()
        return src, self.tk.dump()

    def _symbolic_grid_facts(self):
        gf = {}
        for ob in self.tk.obligations:
            if ob.coord is None or ob.extent is None:
                continue
            if isinstance(ob.coord, Sym) and ob.coord.name.startswith("pid"):
                ax = ob.coord.name[3:]
                if ax.isdigit():
                    gf.setdefault(ob.coord.name, (ob.extent, Cst(1)))
        return gf

    def report(self) -> str:
        tk = self.tk
        out = [f"kernel @{tk.name}",
               f"  effects: " + ", ".join(
                   effect.describe() for effect in tk.effects)
               or "  effects: (none)"]
        aliases = tk.runtime_aliases or tk.aliases
        out.append("  aliases: " + (", ".join(
            alias.describe() for alias in aliases) if aliases else "(none)"))
        out.append(f"  obligations: {len(tk.obligations)}")
        for ob in tk.obligations:
            out.append(f"    {ob.describe()}")
        for w in tk.warnings:
            out.append("  " + w.render().replace("\n", "\n  "))
        for n in tk.notes:
            out.append(f"  note: {n}")
        for var, span in tk.hints:
            out.append(f"  hint: max_contiguous({var}, {span})")
        return "\n".join(out)

    def explain(self, consts: dict | None = None) -> str:
        """--explain 审计输出（surface-language.md §8）：类型环境、事实集、
        义务证明链、发射的 hint、效应汇总 + warnings/notes——全部可审计。

        只读：不 run lowering、不触发 strict 义务 error（Unknown 只展示，
        不 raise）、不改动 tk / last_report。Const 解析同 materialize
        （缺值 raise TILA-CONST-007，精化违反 raise TILA-CONST-003）。

        当前边界：
        - grid 事实是 launch 期由 launcher / launch analysis 登记的契约，
          explain 的 Facts 里 grid_facts 为空——依赖 cdiv 战术（SafeUnder-
          Contract）的义务此处显示 Unknown 并注明，launch 时重新评估；
        - 每条义务自带访问点的 predicates/path/interval 快照，保留 assume
          来源；检查、launch 和 explain 共享不可变 ProofResult。
        """
        tk = self.tk
        cenv = self._resolve_consts(consts)
        self._check_consts(cenv)     # 只读的精化校验（与 materialize 同源）

        facts = Facts()              # 与 _evaluate_obligations 特化期同构
        facts.num = dict(cenv)
        facts.grid_facts = {}        # launch 期契约：explain 不可见（见上）
        facts.sym_hi = dict(tk.sym_hi)
        facts.sym_lo = dict(tk.sym_lo)
        from .solver import ProofSession
        proof_session = ProofSession()

        L = [f"kernel @{tk.name} — explain (surface-language.md §8)"]
        ctext = ", ".join(f"{k}={v}" for k, v in sorted(cenv.items()))
        L.append(f"  consts: {ctext if ctext else '(none)'}")
        L.append("  note: grid facts are launch-time contracts and are not "
                 "registered here; obligations relying on the cdiv tactic "
                 "may prove (SafeUnderContract) at launch")
        pending = numeric.validate(tk, cenv)
        if pending:
            L.append("integer launch contracts (must pass before execution):")
            L.extend("  " + msg for msg in dict.fromkeys(pending))

        L.append("types:")
        if tk.types:
            for n, t in tk.types.items():
                L.append(f"  {n} : {TY.describe(t)}")
        else:
            L.append("  (none)")

        L.append("facts:")
        syms = sorted(set(tk.sym_lo) | set(tk.sym_hi))
        for s in syms:
            lo = tk.sym_lo.get(s)
            hi = tk.sym_hi.get(s)
            lo_t = str(lo) if lo is not None else "-inf"
            hi_t = str(hi) if hi is not None else "+inf"
            L.append(f"  {s} ∈ [{lo_t}, {hi_t})")
        for k, v in sorted(cenv.items()):
            L.append(f"  {k} = {v} (Const)")
        if tk.nonneg_syms:
            L.append(f"  nonneg symbols: {', '.join(sorted(tk.nonneg_syms))}")
        for region, alignment in tk.runtime_alignments.items():
            L.append(f"  aligned {region}: {alignment} bytes (launch verified)")
        if not syms and not cenv and not tk.nonneg_syms and \
                not tk.runtime_alignments:
            L.append("  (none)")

        L.append("hints:")
        if tk.hints:
            for var, span in tk.hints:
                L.append(f"  max_contiguous({var}, {span})")
                for origin in sorted(tk.hint_origins.get(var, ())):
                    L.append(f"    dependency: {origin.kind} at line {origin.line}: {origin.detail}")
        else:
            L.append("  (none)")

        L.append("effects:")
        if tk.effects:
            for effect in tk.effects:
                L.append(f"  {effect.describe()}")
        else:
            L.append("  (none)")

        L.append("aliases:")
        aliases = tk.runtime_aliases or tk.aliases
        if aliases:
            scope = "runtime launch binding" if tk.runtime_aliases else \
                "static conservative defaults"
            L.append(f"  scope: {scope}")
            for alias in aliases:
                L.append(f"  {alias.describe()}")
        else:
            L.append("  (none)")

        L.append("obligations:")
        if tk.obligations:
            for ob in tk.obligations:
                L.append(f"  - {ob.describe()}")
                for line in explain_obligation(ob, facts,
                                               tk.nonneg_syms, proof_session).splitlines():
                    L.append(f"      {line}")
        else:
            L.append("  (none)")

        L.append("warnings:")
        if tk.warnings:
            for w in tk.warnings:
                L.append("  " + w.render().replace("\n", "\n  "))
        else:
            L.append("  (none)")

        L.append("notes:")
        if tk.notes:
            for n in tk.notes:
                L.append(f"  {n}")
        else:
            L.append("  (none)")

        if tk.deferred:
            L.append("deferred constraints:")
            for d in tk.deferred:
                if d["kind"] == "eq":
                    L.append(f"  eq: {d['what']}: {d['l']} == {d['r']}")
                elif d["kind"] == "step":
                    L.append(f"  step: loop step {d['sym']} >= 1")
                elif d["kind"] == "numel":
                    old = ", ".join(map(str, d["old"]))
                    new = ", ".join(map(str, d["new"]))
                    L.append(f"  numel: reshape ({old}) -> ({new}) "
                             f"numel equal")
                else:               # 防御：未知 kind 原样列出
                    L.append(f"  {d['kind']}: {d.get('what', '')}")
        return "\n".join(L)

    # -- Stage 2 内部 -------------------------------------------------------

    def _resolve_consts(self, supplied: dict | None) -> dict[str, int]:
        supplied = {} if supplied is None else dict(supplied)
        declared = {const.name for const in self.tk.consts}
        unknown = sorted(set(supplied) - declared)
        if unknown:
            raise TilaError(
                "TILA-CONST-010",
                f"unknown Const override(s): {', '.join(unknown)}",
                details=[f"declared Const parameters: "
                         f"{', '.join(sorted(declared)) or '(none)'}"],
                fixes=["remove the unknown --const/override name, or declare "
                       "a matching Const[int] parameter"],
            )
        resolved = {}
        for const in self.tk.consts:
            if const.name in supplied:
                value = supplied[const.name]
            elif const.default is not None:
                value = const.default
            else:
                raise TilaError(
                    "TILA-CONST-007",
                    f"Const parameter '{const.name}' has no value",
                    details=["required at specialization: exact Python int"],
                    fixes=[f"pass {const.name}=<int>, --const "
                           f"{const.name}=<int>, or add an integer default"],
                )
            if type(value) is not int:
                raise TilaError(
                    "TILA-CONST-008",
                    f"Const '{const.name}' must be an exact Python int",
                    details=[f"found: {type(value).__name__} value {value!r}",
                             "required: type(value) is int "
                             "(bool and coercible objects are rejected)"],
                    fixes=[f"pass {const.name}=<integer> without implicit "
                           "conversion"],
                )
            resolved[const.name] = value
        return resolved

    def _check_consts(self, cenv: dict):
        for c in self.tk.consts:
            v = cenv.get(c.name)
            if v is None:
                raise TilaError("TILA-CONST-007",
                                f"Const '{c.name}' unresolved at "
                                "specialization")
            if type(v) is not int:
                raise TilaError(
                    "TILA-CONST-008",
                    f"Const '{c.name}' must be an exact Python int",
                    details=[f"found: {type(v).__name__} value {v!r}"])
            for r in c.refinements:
                if not r.check(v):
                    raise TilaError(
                        "TILA-CONST-003", "Const refinement violated",
                        details=[f"    {c.name} = {v} violates {r.text}"],
                        fixes=[f"传满足 {r.text} 的值，或放宽声明"])

    def _check_deferred(self, cenv: dict):
        for d in self.tk.deferred:
            if d["kind"] == "eq":
                l = rewrite_syms(d["l"], {k: Cst(v) for k, v in cenv.items()})
                r = rewrite_syms(d["r"], {k: Cst(v) for k, v in cenv.items()})
                if not equal(l, r):
                    raise TilaError(
                        "TILA-SHAPE-004",
                        f"{d['what']} do not match at specialization",
                        d["loc"],
                        [f"    {d['l']} != {d['r']}",
                         f"    with consts {cenv}"])
            elif d["kind"] == "step":
                v = cenv.get(d["sym"])
                if v is None or v < 1:
                    raise TilaError("TILA-TYPE-025",
                                    f"loop step ({d['sym']}={v}) must be >= 1",
                                    d["loc"])
            elif d["kind"] == "numel":
                # reshape numel 约束（intrinsics.md §2.8）：Const 值代入
                # 两个维度列表后数值比较乘积。
                from .dims import int_value
                mapping = {k: Cst(v) for k, v in cenv.items()}

                def _numel(shape):
                    prod = 1
                    for dim in shape:
                        iv = int_value(rewrite_syms(dim, mapping))
                        if iv is None:
                            raise TilaError(   # 防御：延迟约束不应含运行期符号
                                "TILA-SHAPE-005",
                                "reshape numel check hit a non-Const dim "
                                "(deferred constraint must be Const-only)",
                                d["loc"],
                                [f"    dim: {dim}",
                                 f"    with consts {cenv}"])
                        prod *= iv
                    return prod

                old_n = _numel(d["old"])
                new_n = _numel(d["new"])
                if old_n != new_n:
                    raise TilaError(
                        "TILA-SHAPE-005", "reshape numel mismatch",
                        d["loc"],
                        [f"    input:  ({', '.join(map(str, d['old']))}) "
                         f"-> numel {old_n}",
                         f"    target: ({', '.join(map(str, d['new']))}) "
                         f"-> numel {new_n}",
                         f"    with consts {cenv}"])
            elif d["kind"] == "static_assert":
                try:
                    value = _eval_const_operand(d["pred"], cenv)
                except (ArithmeticError, TypeError, ValueError) as exc:
                    raise TilaError(
                        "TILA-CONST-009",
                        "static_assert Const expression could not be evaluated",
                        d["loc"],
                        [f"predicate: {_render_const_operand(d['pred'])}",
                         f"with consts: {cenv}",
                         f"reason: {type(exc).__name__}: {exc}"],
                    ) from None
                if type(value) is not bool:
                    raise TilaError(
                        "TILA-CONST-006",
                        "static_assert specialization result is not bool",
                        d["loc"],
                        [f"predicate: {_render_const_operand(d['pred'])}",
                         f"result: {value!r}"],
                    )
                if not value:
                    raise TilaError(
                        "TILA-CONST-005",
                        "static_assert failed at specialization",
                        d["loc"],
                        [f"predicate: {_render_const_operand(d['pred'])}",
                         f"with consts: {cenv}"],
                        ["choose Const values satisfying the assertion, or "
                         "correct/remove the assertion"],
                    )

    def _evaluate_obligations(self, cenv: dict, grid_facts: dict,
                              extra_preds: list, launch_checked=True, launch_bindings=()):
        from .solver import ProofSession
        proof_session = ProofSession()
        from dataclasses import replace
        from . import predicates as P
        facts = Facts()
        facts.num = dict(cenv)
        facts.grid_facts = grid_facts
        facts.grid_checked = launch_checked
        facts.launch_bindings = launch_bindings
        facts.sym_hi = dict(self.tk.sym_hi)
        facts.sym_lo = dict(self.tk.sym_lo)
        for p in extra_preds:
            if launch_checked:
                facts.add_pred(replace(p, origins=frozenset({
                    P.Origin(P.CHECKED, detail="validated assume_launch predicate")})))
        results = []
        warn_diags = []
        for ob in self.tk.obligations:
            nonneg = self.tk.nonneg_syms | {name for name, value in cenv.items() if value >= 0}
            result = evaluate_obligation(ob, facts, nonneg, proof_session)
            state = result.verdict
            results.append((ob, result))
            if state == PROVEN_UNSAFE:
                raise TilaError(     # 无条件 error（任何模式，bounds-safety §6）
                    "TILA-BOUNDS-003", "out-of-bounds access is provable",
                    Loc(ob.loc_line), [f"    {ob.describe()}"],
                    ["修正坐标、补 mask，或 unsafe_load/unsafe_store"])
            if state == UNKNOWN and not (not launch_checked and result.pending_contracts):
                warn = self._raise_unknown(ob, result)
                if warn is not None:
                    warn_diags.append(warn)
        self.last_report = "\n".join(
            f"  {ob.describe()} -> {st.summary}" +
            ("  (pending launch contract)" if st.pending_contracts else "")
            for ob, st in results)
        self.last_proof_results = tuple(results)
        if warn_diags:
            self.last_report += "\n" + "\n".join(warn_diags)
        return results

    def _raise_unknown(self, ob, result=None):
        """Unknown 义务（TILA-BOUNDS-001/002）的严格度分派。

        strict（默认）：raise。TILA_SAFETY=warn（refinements.md §5.3 /
        bounds-safety.md §6）：降级为 stderr warning（复用诊断渲染，首行
        error→warning），记录进 last_report，kernel 继续编译/运行。
        ProvenUnsafe（TILA-BOUNDS-003）不经此路径——任何模式都不降级。
        """
        err = self._unknown_error(ob)
        if result is not None:
            err.details.append(result.render())
        if os.environ.get("TILA_SAFETY", "strict") != "warn":
            raise err
        lines = err.render().split("\n")
        lines[0] = lines[0].replace("error[", "warning[", 1)
        text = "\n".join(lines)
        print(text, file=sys.stderr)
        return text

    def _unknown_error(self, ob) -> TilaError:
        # Scan DAG atoms for wrong-dimension diagnostics, without DNF expansion.
        # Atom presence here is diagnostic only and never establishes a proof.
        from .dims import canon
        wrong = None
        goal_hit = False
        if ob.coord is not None and ob.extent is not None:
            goal_key = Pred("<", ob.coord, ob.extent).key()
            from .predicates import atoms
            for p in atoms(ob.mask):
                if p.key() == goal_key:
                    goal_hit = True
                elif p.op == "<" and \
                        canon(p.left) == canon(ob.coord) and \
                        canon(p.right) != canon(ob.extent):
                    wrong = p
        if wrong is not None and not goal_hit:
            return TilaError(
                "TILA-BOUNDS-002",
                f"cannot prove axis {ob.axis} in bounds: mask compares the "
                "wrong dimension", Loc(ob.loc_line),
                [f"    required: 0 <= {ob.coord} < {ob.extent}",
                 f"    mask provides: {wrong.left} {wrong.op} {wrong.right}"],
                [f"改 mask 为 {ob.coord} < {ob.extent}（或补齐该轴谓词）"])
        fixes = ["补 mask：mask = offs < N（逐轴），或 @ti.assume_launch 声明"
                 "整除契约，或 ti.unsafe_load/unsafe_store 显式豁免"]
        if ob.coord is None:
            fixes = ["数据依赖坐标无法静态证明：ti.assume(...) 注入事实，或 "
                     "ti.unsafe_load/unsafe_store 显式豁免"]
        if ob.extent is None:
            fixes = ["裸指针为 UnknownExtent：改用 Buffer 形式访问，或 "
                     "ti.Ptr[dt, access, extent, alignment] 声明元素数量"]
        return TilaError(
            "TILA-BOUNDS-001",
            "possible out-of-bounds access (cannot prove safety)",
            Loc(ob.loc_line),
            [f"    {ob.describe()}",
             "    证明失败：缺少上界谓词或契约"],
            fixes)


def _debug() -> bool:
    return os.environ.get("TILA_DEBUG", "") == "1"


def _triton_cache_key(jf, consts: dict):
    """Semantic cache key; registry changes invalidate compiled kernels."""
    return (INTRINSIC_REGISTRY_SEMANTIC_REVISION, id(jf),
            tuple(sorted(consts.items())), _debug())


class _Launcher:
    def __init__(self, jf: JITFunction, grid):
        self.jf = jf
        self.grid = grid

    def __call__(self, *args, **kwargs):
        jf = self.jf
        tk = jf.tk
        dim_vals: dict[str, int] = {}
        stride_vals: dict[str, int] = {}
        scalar_vals: dict[str, int | float] = {}
        tensors: dict[str, object] = {}
        consts: dict[str, int] = {}

        # ---- 位置实参 → 声明序参数（buffer | ptr | 显式标量）
        # param_order 由 checker 记录源码声明序；隐式符号（维/步长/Ptr
        # extent）与 Const 不占位置实参。空表（旧构造的 TKernel）回退到
        # buffers-then-explicit-scalars。
        if tk.param_order:
            order = list(tk.param_order)
        else:
            order = ([("buffer", b.name) for b in tk.buffers] +
                     [("scalar", s.name) for s in tk.scalars
                      if not _is_implicit(tk, s.name)])
        consumable = [e for e in order
                      if e[0] in ("buffer", "ptr") or
                      (e[0] == "scalar" and not _is_implicit(tk, e[1]))]
        if len(args) > len(consumable):
            raise TilaLaunchContractError(
                "TILA-TYPE-101", f"too many arguments: expected "
                f"{len(consumable)}, got {len(args)}")
        pos = 0
        for kind, name in consumable:
            if pos >= len(args):
                if kind == "buffer":
                    raise TilaLaunchContractError(
                        "TILA-TYPE-101",
                        f"missing tensor argument for buffer '{name}'")
                if kind == "ptr":
                    raise TilaLaunchContractError(
                        "TILA-TYPE-101",
                        f"missing tensor argument for pointer parameter "
                        f"'{name}'")
                break        # 标量缺失 → 下方 kwargs 形态检查报错
            if kind in ("buffer", "ptr"):
                tensors[name] = args[pos]     # ptr 参数同走张量绑定
            else:
                scalar_vals[name] = args[pos]
            pos += 1

        # ---- Const 参数（kwargs / 默认）
        for c in tk.consts:
            if c.name in kwargs:
                v = kwargs.pop(c.name)
                if type(v) is not int:
                    raise TilaLaunchContractError(
                        "TILA-CONST-008",
                        f"Const '{c.name}' must be an exact Python int",
                        [f"found: {type(v).__name__} value {v!r}",
                         "required: type(value) is int"])
                consts[c.name] = v
            elif c.default is not None:
                consts[c.name] = c.default
            else:
                raise TilaLaunchContractError(
                    "TILA-CONST-007", f"Const '{c.name}' required")

        # ---- 显式标量的 kwargs 形态
        explicit = [p for p in tk.scalars if not _is_implicit(tk, p.name)]
        for s in explicit:
            if s.name in kwargs:
                if s.name in scalar_vals:
                    raise TilaLaunchContractError(
                        "TILA-TYPE-101",
                        f"scalar '{s.name}' passed twice (positional + kw)")
                scalar_vals[s.name] = kwargs.pop(s.name)
            elif s.name not in scalar_vals:
                raise TilaLaunchContractError(
                    "TILA-TYPE-101", f"missing scalar argument '{s.name}'")
            v = scalar_vals[s.name]
            if s.dtype.is_int and not (isinstance(v, int) and
                                       not isinstance(v, bool)):
                raise TilaLaunchContractError(
                    "TILA-TYPE-101",
                    f"scalar '{s.name}' expects int, got {v!r}")
            if s.dtype.is_float and not isinstance(v, (int, float)):
                raise TilaLaunchContractError(
                    "TILA-TYPE-101",
                    f"scalar '{s.name}' expects float, got {v!r}")

        unknown_kw = set(kwargs)
        if unknown_kw:
            raise TilaLaunchContractError(
                "TILA-TYPE-101",
                f"unknown keyword argument(s): {', '.join(sorted(unknown_kw))}")

        # ---- 张量绑定：dtype/shape/strides/data_ptr（surface-language.md §6）
        tk.runtime_alignments = {}
        for b in tk.buffers:
            t = tensors[b.name]
            dt, shape, strides, ptr = _tensor_info(t)
            _check_index_metadata(shape, strides)
            if dt is not b.vtype.elem:
                raise TilaLaunchContractError(
                    "TILA-TYPE-101",
                    f"buffer '{b.name}' dtype mismatch",
                    [f"    declared: {b.vtype.elem.name}",
                     f"    got:      {dt.name}"])
            if len(shape) != len(b.vtype.dims):
                raise TilaLaunchContractError(
                    "TILA-TYPE-102", f"buffer '{b.name}' rank mismatch",
                    [f"    declared rank: {len(b.vtype.dims)}",
                     f"    got rank:      {len(shape)}"])
            for i, de in enumerate(b.vtype.dims):
                _bind_dim(dim_vals, de, shape[i], b.name, i)
            for i in range(len(shape)):
                stride_vals[f"{b.name}_stride{i}"] = strides[i]
            if b.name in tk.buffer_ptr_views and strides[0] != 1:
                raise TilaLaunchContractError(
                    "TILA-MEM-005",
                    f"'{b.name}.ptr' requires stride[0] == 1 in v0",
                    [f"    buffer shape:   {shape}",
                     f"    element strides: {strides}"],
                    ["传入 stride-1 的一维视图，或使用 Buffer 坐标访问"])
            if b.vtype.aligned is not None and ptr % b.vtype.aligned != 0:
                raise TilaLaunchContractError(
                    "TILA-MEM-003",
                    f"buffer '{b.name}' violates Aligned[{b.vtype.aligned}]",
                    [f"    data_ptr() = {ptr}"],
                    ["确保张量按声明对齐（torch 分配通常 256B 对齐），"
                     "或去掉 aligned 声明"])
            if b.vtype.aligned is not None:
                tk.runtime_alignments[b.region_id] = b.vtype.aligned

        # ---- 裸指针参数绑定：dtype / contiguous / Extent / Alignment
        for p in tk.ptr_params:
            t = tensors[p.name]
            dt, shape, strides, ptr = _tensor_info(t)
            _check_index_metadata(shape, strides)
            if dt is not p.vtype.elem:
                raise TilaLaunchContractError(
                    "TILA-TYPE-101",
                    f"pointer parameter '{p.name}' dtype mismatch",
                    [f"    declared: {p.vtype.elem.name}",
                     f"    got:      {dt.name}"])
            if not _is_contiguous(shape, strides):
                raise TilaLaunchContractError(
                    "TILA-TYPE-102",
                    f"pointer parameter '{p.name}' requires contiguous storage",
                    [f"    shape:   {shape}", f"    strides: {strides}"],
                    ["传入 contiguous tensor/array，或改用 Buffer 坐标访问"])
            available = 1
            for size in shape:
                available *= size
            _bind_ptr_extent(dim_vals, scalar_vals, consts,
                             TY.extent_expr(p.vtype.extent),
                             available, p.name)
            if p.vtype.aligned is not None and \
                    ptr % p.vtype.aligned != 0:
                raise TilaLaunchContractError(
                    "TILA-MEM-003",
                    f"pointer parameter '{p.name}' violates "
                    f"Aligned[{p.vtype.aligned}]",
                    [f"    data_ptr() = {ptr}"],
                    ["确保张量按声明对齐，或去掉 aligned 声明"])
            if p.vtype.aligned is not None:
                tk.runtime_alignments[p.vtype.region_id] = p.vtype.aligned

        # alias 是独立分析维度：不改写 RegionId，也不参与 bounds 数值式。
        tk.runtime_aliases = _runtime_alias_facts(tk, tensors)

        # ---- 标量精化（launch 契约）+ 显式标量与维符号绑定的一致性
        for s in tk.scalars:
            if s.name not in scalar_vals:
                if s.name in dim_vals:
                    scalar_vals[s.name] = dim_vals[s.name]
                    continue
                if s.name in stride_vals:
                    scalar_vals[s.name] = stride_vals[s.name]
                    continue
                owner = _extent_owner(tk, s.name)
                if owner is not None:
                    raise TilaLaunchContractError(
                        "TILA-TYPE-101",
                        f"extent symbol '{s.name}' of pointer parameter "
                        f"'{owner}' is not bound",
                        [f"    Ptr extent 需要在 launch 时取得数值"],
                        [f"用维符号 {s.name} 声明某个伴随 Buffer 的 shape"
                         f"（如 b: ti.Buffer[..., ({s.name},)]），"
                         f"或把它声明为显式标量参数"])
                raise TilaLaunchContractError(
                    "TILA-TYPE-101",
                    f"scalar '{s.name}' unbound (dim/stride 绑定失败)")
            elif s.name in dim_vals and scalar_vals[s.name] != dim_vals[s.name]:
                raise TilaLaunchContractError(
                    "TILA-TYPE-102",
                    f"scalar '{s.name}' conflicts with its shape binding",
                    [f"    passed: {scalar_vals[s.name]}",
                     f"    shape:   {dim_vals[s.name]}"])
            v = scalar_vals[s.name]
            # 维/步长等 int 标量维持 int 语境；float 标量不得截断
            # （ti.f32 | ti.Positive 的 0.125 不应被 int() 成 0）。
            cv = int(v) if getattr(s.dtype, "is_int", True) else v
            for r in s.refined:
                if not r.check(cv):
                    raise TilaLaunchContractError(
                        "TILA-TYPE-103",
                        f"scalar '{s.name}' violates {r.text}",
                        [f"    value: {v}"])

        # ---- Const 精化
        jf._check_consts(consts)

        # Every implicit/explicit scalar crosses the same typed ABI boundary.
        for s in tk.scalars:
            if s.dtype.is_int:
                v = scalar_vals[s.name]
                lo, hi = numeric.limits(s.dtype)
                if not lo <= v <= hi:
                    raise TilaLaunchContractError(
                        "TILA-NUM-001", f"scalar '{s.name}' does not fit {s.dtype.name}")

        # ---- grid 解析 + cdiv 模式登记（bounds-safety.md §5）
        meta = dict(dim_vals)
        meta.update(stride_vals)
        meta.update(scalar_vals)
        meta.update(consts)
        grid_facts = {}
        if self.grid is None:
            # launch_auto：绑定完成后由义务推导 grid（事实直接来自推导，
            # 比显式 grid 的数值回配更精确）
            grid, grid_facts = _derive_grid(tk, dim_vals, consts)
        else:
            grid_ints = []
            for ax, g in enumerate(self.grid[:3]):
                if isinstance(g, _Cdiv):
                    gi = g.value(meta)
                    bound_sym, step_sym = _match_cdiv(g, dim_vals, consts)
                    if bound_sym is not None:
                        grid_facts[f"pid{ax}"] = (
                            Sym(bound_sym),
                            Sym(step_sym) if isinstance(step_sym, str)
                            else Cst(step_sym))
                else:
                    gi = g(meta) if callable(g) else g
                    # 精确维 grid：grid[ax] == 某维数值 ⇒ pid_ax < 该维（事实登记）
                    for dname, dval in dim_vals.items():
                        if dval == gi:
                            grid_facts[f"pid{ax}"] = (Sym(dname), Cst(1))
                            break
                if type(gi) is not int or not 0 <= gi <= (1 << 31) - 1:
                    raise TilaLaunchContractError(
                        "TILA-TYPE-104", f"grid[{ax}] must be an exact int in [0, 2**31-1], got {gi}")
                grid_ints.append(int(gi))
            grid = tuple(grid_ints)

        # ---- assume_launch 契约（每次启动检查 → 通过后作为事实）
        extra_preds = []
        if jf.assume_launch is not None:
            preds, ast_expr = jf.assume_launch
            if not eval(compile(ast_expr, "<assume_launch>", "eval"), {},
                        dict(meta)):
                raise TilaLaunchContractError(
                    "TILA-BOUNDS-010",
                    f"assume_launch contract failed for {jf.__name__}",
                    [f"    predicate over {sorted(dim_vals)} / "
                     f"{sorted(consts)}"],
                    ["契约保护的是无 mask 完整 tile 访问——调整 shape 或 "
                     "BLOCK，或去掉该契约并补 mask"])
            extra_preds.extend(preds)

        # ---- Stage 2：延迟约束 + 义务四态（strict/warn 由 TILA_SAFETY 决定）
        jf._check_deferred(consts)
        if any(type(g) is not int or not 0 <= g <= (1 << 31) - 1 for g in grid):
            raise TilaLaunchContractError("TILA-TYPE-104", "auto grid must fit nonnegative i32")
        numeric.validate(tk, consts, scalar_vals, grid + (1,) * (3 - len(grid)))
        jf._evaluate_obligations(consts, grid_facts, extra_preds,
            launch_bindings=tuple(sorted(meta.items())) +
                (("grid", tuple(grid)),))

        # ---- 执行：torch+cuda+triton → GPU；否则 interp
        self._execute(tensors, scalar_vals, stride_vals, consts, grid)

    def _execute(self, tensors, scalar_vals, stride_vals, consts, grid):
        jf, tk = self.jf, self.jf.tk
        use_triton = False
        if torch is not None:
            t0 = next(iter(tensors.values()), None)
            if isinstance(t0, torch.Tensor) and t0.is_cuda:
                use_triton = True
        if os.environ.get("TILA_INTERP") == "1":
            use_triton = False
        if use_triton:
            try:
                import triton
            except ImportError:
                raise TilaError(
                    "TILA-TARGET-004",
                    "tensor is on CUDA but triton is not installed")
            key = _triton_cache_key(jf, consts)
            if key not in jf._kern_cache:
                src = lowering.Lowering(tk, _debug()).kernel_source()
                # Triton retrieves Python source with inspect, including helpers.
                import linecache
                filename = f"<tila:{tk.name}:{id(jf)}:{_debug()}>"
                linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
                ns: dict = {"__name__": "tila_generated"}
                exec(compile(src, filename, "exec"), ns)
                jf._kern_cache[key] = ns[tk.name]
            kern = jf._kern_cache[key]
            # 实参序 = 签名序：buffer 张量 → ptr 参数张量 → 标量 → Const
            args = [tensors[b.name] for b in tk.buffers]
            args += [tensors[p.name] for p in tk.ptr_params]
            args += [scalar_vals[s.name] for s in tk.scalars]
            args += [consts[c.name] for c in tk.consts]
            kern[grid](*args, num_warps=4)
            return
        # interp：numpy 视图（torch CPU 张量共享内存）；ptr 参数以自身名
        # 注册为 interp buffer（指针值 = ("ptr", name, offset)）。
        np_bufs = {}
        for param in list(tk.buffers) + list(tk.ptr_params):
            t = tensors[param.name]
            if torch is not None and isinstance(t, torch.Tensor):
                if t.is_cuda:
                    raise TilaError(
                        "TILA-TARGET-005",
                        "CUDA tensor without triton backend——用 CPU 张量/"
                        "numpy 走 interpreter，或安装 triton")
                np_bufs[param.name] = t.detach().numpy()
            else:
                np_bufs[param.name] = np.asarray(t)
        interp_mod.run_kernel(tk, np_bufs, scalar_vals, consts, grid,
                              debug=_debug())


def _is_implicit(tk: T.TKernel, name: str) -> bool:
    """标量是否为隐式符号（不能由用户显式传参）——用户声明的标量除外。"""
    return name in tk.implicit_names and name not in tk.explicit_scalars


def _extent_owner(tk: T.TKernel, name: str) -> str | None:
    """name 是否出现在某裸指针参数的 extent 中 → 返回该参数名。"""
    for p in tk.ptr_params:
        extent = TY.extent_expr(p.vtype.extent)
        if extent is not None and name in free_syms(extent):
            return p.name
    return None


def _bind_ptr_extent(dim_vals: dict, scalar_vals: dict, consts: dict,
                     extent: DimExpr | None, available: int, pname: str):
    """绑定/验证裸 Ptr 的元素 extent；unknown 不制造 bounds 事实。"""
    if extent is None:
        return
    if isinstance(extent, Sym) and extent.name not in consts and \
            extent.name not in dim_vals and extent.name not in scalar_vals:
        # ADR-001：未绑定的单一维 extent 从实际连续存储元素数特化。
        dim_vals[extent.name] = available
    mapping = {k: Cst(v) for k, v in
               {**consts, **dim_vals, **scalar_vals}.items()
               if isinstance(v, int) and not isinstance(v, bool)}
    value = int_value(rewrite_syms(extent, mapping))
    if value is None:
        missing = sorted(free_syms(extent) - set(mapping))
        raise TilaLaunchContractError(
            "TILA-TYPE-101",
            f"extent of pointer parameter '{pname}' is not fully bound",
            [f"    extent: {extent}",
             f"    unbound symbols: {', '.join(missing) or '<expression>'}"],
            ["用 Buffer shape、Const 或单一 ti.Dim extent 提供绑定"])
    if value < 0:
        raise TilaLaunchContractError(
            "TILA-TYPE-102",
            f"pointer parameter '{pname}' has negative extent {value}")
    if value > available:
        raise TilaLaunchContractError(
            "TILA-TYPE-102",
            f"pointer parameter '{pname}' extent exceeds available storage",
            [f"    declared extent: {value} elements",
             f"    available:       {available} elements"])


def _bind_dim(dim_vals: dict, de: DimExpr, value: int, bname: str, axis: int):
    if isinstance(de, Sym):
        if de.name in dim_vals and dim_vals[de.name] != value:
            raise TilaLaunchContractError(
                "TILA-TYPE-102",
                f"symbol '{de.name}' bound to conflicting shapes",
                [f"    {bname}.shape[{axis}] = {value}, "
                 f"previously {dim_vals[de.name]}"])
        dim_vals[de.name] = value
        return
    if isinstance(de, Cst):
        if de.value != value:
            raise TilaLaunchContractError(
                "TILA-TYPE-102",
                f"buffer '{bname}' axis {axis} has fixed shape {de.value}",
                [f"    got: {value}"])
        return
    raise TilaLaunchContractError(
        "TILA-TYPE-102",
        f"buffer '{bname}' axis {axis}: unsupported shape expression {de}")


def _check_index_metadata(shape, strides):
    if any(not 0 <= n <= (1 << 31) - 1 for n in shape):
        raise TilaLaunchContractError("TILA-NUM-001", "shape dimensions must fit nonnegative i32")
    if any(not -(1 << 31) <= s <= (1 << 31) - 1 for s in strides):
        raise TilaLaunchContractError("TILA-NUM-001", "element strides must fit i32")
    if sum(max(0, n - 1) * abs(s) for n, s in zip(shape, strides)) > (1 << 63) - 1:
        raise TilaLaunchContractError("TILA-NUM-001", "linear element offset must fit i64")


def _tensor_info(t):
    """dtype/shape/strides(元素)/data_ptr，torch 与 numpy 双协议。"""
    if torch is not None and isinstance(t, torch.Tensor):
        dt = _TORCH_DT.get(str(t.dtype).removeprefix("torch."))
        strides = tuple(t.stride())
        return dt, tuple(t.shape), strides, t.data_ptr()
    arr = np.asarray(t)
    dt = _NP_DT.get(str(arr.dtype))
    strides = tuple(s // arr.itemsize for s in arr.strides)
    return dt, arr.shape, strides, arr.ctypes.data


def _is_contiguous(shape, strides) -> bool:
    """C-order contiguous；size 0/1 的轴不约束 stride。"""
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size > 1 and stride != expected:
            return False
        expected *= max(size, 1)
    return True


def _storage_interval(t):
    """保守的字节区间；仅用于 launch alias 关系提升。"""
    if torch is not None and isinstance(t, torch.Tensor):
        if t.numel() == 0:
            return t.data_ptr(), t.data_ptr()
        if any(s < 0 for s in t.stride()):
            return None
        span = 1 + sum((size - 1) * stride
                       for size, stride in zip(t.shape, t.stride())
                       if size)
        start = t.data_ptr()
        return start, start + span * t.element_size()
    arr = np.asarray(t)
    start = int(arr.ctypes.data)
    if arr.size == 0:
        return start, start
    spans = [(size - 1) * stride
             for size, stride in zip(arr.shape, arr.strides)]
    lo = start + sum(min(0, span) for span in spans)
    hi = start + sum(max(0, span) for span in spans) + arr.itemsize
    return lo, hi


def _runtime_alias_facts(tk: T.TKernel, tensors: dict) -> list[TY.AliasFact]:
    memory_params = [(b.name, b.region_id) for b in tk.buffers] + \
        [(p.name, p.vtype.region_id) for p in tk.ptr_params]
    facts = []
    for i, (left_name, left_region) in enumerate(memory_params):
        for right_name, right_region in memory_params[i + 1:]:
            left, right = tensors[left_name], tensors[right_name]
            li, ri = _storage_interval(left), _storage_interval(right)
            if left is right:
                relation = TY.AliasRelation.MUST_ALIAS
                reason = "same launch object"
            elif li is not None and ri is not None and \
                    (li[1] <= ri[0] or ri[1] <= li[0]):
                relation = TY.AliasRelation.NO_ALIAS
                reason = "disjoint runtime storage intervals"
            else:
                relation = TY.AliasRelation.MAY_ALIAS
                reason = "runtime storage may overlap"
            facts.append(TY.AliasFact(left_region, right_region,
                                      relation, reason))
    return facts


def _match_cdiv(g: _Cdiv, dim_vals: dict, consts: dict):
    """把 cdiv(a, b) 的数值匹配回 (维符号, Const 符号或常量)。"""
    a = g.a
    b = g.b
    bound = None
    for name, v in dim_vals.items():
        if v == a:
            bound = name
            break
    if bound is None:
        return None, None
    for name, v in consts.items():
        if v == b:
            return bound, name
    return bound, b


# ---------------------------------------------------------------------------
# launch analysis：launch_auto 的 grid 推导（surface-language.md §6）
# ---------------------------------------------------------------------------

def _walk_pids(node, axes: set):
    """递归收集 TPid 轴：语句体/条件/循环/坐标/表达式操作数的全部嵌套
    位置（TPid 经 VarInfo tir 值可藏在任意 TExpr 操作数里）。"""
    if isinstance(node, T.TPid):
        axes.add(node.axis)
    if not hasattr(node, "__dataclass_fields__"):
        return
    for f in dataclasses.fields(node):
        v = getattr(node, f.name)
        if isinstance(v, (T.TOperand, T.TStmt)):
            _walk_pids(v, axes)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, (T.TOperand, T.TStmt)):
                    _walk_pids(x, axes)


def _used_pid_axes(tk: T.TKernel) -> set:
    """kernel 实际使用的 pid 轴集合（推导的需求侧）。"""
    axes: set = set()
    for st in tk.body:
        _walk_pids(st, axes)
    return axes


def _axis_step(coord: DimExpr, pid_name: str):
    """coord 是否呈现 pid 轴的规范访问模式；是则返回 step 表达式。

    - blocked：pid*STEP + lane（Add 线性项里的 Mul 原子配对 step）；
    - 标量坐标：coord 恰为 pid 本身（STEP=1，行/列 per-program 模式）。
    """
    terms = _decompose_linear(coord)
    if len(terms) == 1 and terms[0][1] == 1 and \
            isinstance(terms[0][0], Sym) and terms[0][0].name == pid_name:
        return Cst(1)
    for t, _c in terms:
        if not isinstance(t, Mul):
            continue
        if isinstance(t.right, Sym) and t.right.name == pid_name:
            return t.left
        if isinstance(t.left, Sym) and t.left.name == pid_name:
            return t.right
    return None


def _step_value(step: DimExpr, consts: dict):
    """step 的数值：Const 符号代入后常量折叠（Cst / Sym(BLOCK) / 组合式）。
    非数值或 < 1（步长无意义）返回 None。"""
    if consts:
        step = rewrite_syms(step, {k: Cst(v) for k, v in consts.items()})
    v = int_value(step)
    return v if v is not None and v >= 1 else None


def _derive_axis(tk: T.TKernel, ax: int, dim_vals: dict, consts: dict):
    """从义务推导 pid_ax 的 (bound Sym, step DimExpr, grid int)。

    义务已编码规范模式（checker 产出 coord 符号式）：匹配
    pid_ax*STEP+lane / 纯 pid_ax 的义务给出 bound（须为已绑定数值的
    运行期维符号）与 step（须可由 Const 求值）。多个义务不一致 →
    推导失败；无任何匹配 → 失败（pid 只用于非访问计算等情形）。
    """
    pid_name = f"pid{ax}"
    found = None
    for ob in tk.obligations:
        if ob.coord is None or ob.extent is None:
            continue
        step = _axis_step(ob.coord, pid_name)
        if step is None:
            continue
        bound = ob.extent
        if not (isinstance(bound, Sym) and bound.name in dim_vals):
            continue        # 上界须是 shape 绑定出数值的维符号
        step_val = _step_value(step, consts)
        if step_val is None:
            continue        # step 须可数值求值（Const 符号 / 常量）
        cand = (bound, step, -(-dim_vals[bound.name] // step_val))
        if found is not None and not (equal(found[0], cand[0]) and
                                      equal(found[1], cand[1])):
            raise TilaLaunchContractError(
                "TILA-TYPE-105",
                f"cannot auto-derive grid axis {ax} for '{tk.name}': "
                "obligations disagree on (bound, step)",
                [f"    {found[0]} / {found[1]}  vs  {cand[0]} / {cand[1]}"],
                [f"传显式 grid：{tk.name}[(ti.cdiv(n, BLOCK),)](...)"])
        found = cand
    if found is None:
        raise TilaLaunchContractError(
            "TILA-TYPE-105",
            f"cannot auto-derive grid axis {ax} for '{tk.name}': pid{ax} "
            "never reaches a memory access in the canonical "
            "pid*STEP+lane / scalar-pid pattern",
            [f"    obligations: {len(tk.obligations)}，无一匹配轴 {ax}"],
            [f"改用显式 grid：{tk.name}[(ti.cdiv(n, BLOCK),)](...)"])
    return found


def _derive_grid(tk: T.TKernel, dim_vals: dict, consts: dict):
    """launch_auto 的 grid 推导：每个使用中的轴按义务取
    ceildiv(bound, step)，未用轴补 1；秩 = 最高使用轴 + 1（最少 1）。

    返回 (grid ints, grid_facts)：事实直接取自推导出的 (bound, step)
    对——无需像显式 grid 那样数值回配（严格更优，bounds-safety.md §5）。
    """
    used = _used_pid_axes(tk)
    if not used:
        return (1,), {}
    grid = []
    grid_facts = {}
    for ax in range(max(used) + 1):
        bound, step, gi = _derive_axis(tk, ax, dim_vals, consts)
        grid.append(gi)
        if ax in used:
            grid_facts[f"pid{ax}"] = (bound, step)
    return tuple(grid), grid_facts


def assume_launch(pred: str):
    """@ti.assume_launch("N % BLOCK == 0")：launch 契约（bounds-safety §5.2）。"""
    def deco(fn):
        if not isinstance(fn, JITFunction):
            raise TilaError(
                "TILA-SYN-061",
                "@ti.assume_launch must be applied above @ti.jit")
        tree = ast.parse(pred, mode="eval")
        preds = []

        def walk(node):
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                for v in node.values:
                    walk(v)
                return
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                op = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
                      ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}[
                    type(node.ops[0])]
                l = _expr(node.left, fn.tk)
                r = _expr(node.comparators[0], fn.tk)
                preds.append(Pred(op, l, r))
                return
            raise TilaError("TILA-SYN-062",
                            f"unsupported assume_launch predicate: {pred!r}",
                            fixes=["支持单比较与 and 合取，操作数为维/Const 符号与整数字面量"])

        walk(tree.body)
        fn.assume_launch = (preds, ast.parse(pred, mode="eval"))
        return fn
    return deco


def _expr(node, tk: T.TKernel):
    names = {s.name for s in tk.scalars} | {c.name for c in tk.consts}
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Cst(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise TilaError("TILA-SYN-063",
                            f"assume_launch 名字 '{node.id}' 不是 kernel 的"
                            "标量/Const 参数")
        return Sym(node.id)
    if isinstance(node, ast.BinOp):
        from .dims import Add, Mod, Mul, Sub
        l = _expr(node.left, tk)
        r = _expr(node.right, tk)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Mod):
            return Mod(l, r)
    raise TilaError("TILA-SYN-063", "unsupported assume_launch expression")
