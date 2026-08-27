"""launch analysis（docs/type-checker.md §5 步骤 4，docs/triton-lowering.md §7）。

typing 之后的独立 phase：在 typed TIR 上做语义模式识别——
  轴数   = kernel 内实际使用的 program_id 最大轴 + 1
  每轴   = (维, tiling constexpr)：维来自该轴 mask 谓词（offs < N）覆盖的
           符号维（静态维直接代入字面量）；tiling constexpr 是在 `pid * CE`
           中被引用的那个，每轴必须唯一确定。
推导不出 → E17。这是语义模式识别而非类型检查，故独立成 phase——未来
autotune、多种 grid 策略、persistent kernel 都挂这一层。
"""

from __future__ import annotations

from typing import Dict, List

from .. import tir
from ..diagnostics import Loc, err
from ..types import Const, Symbol


def _as_constexpr_operand(op_by_id, oid: str):
    """操作数是 constexpr 名或整数字面量 → 返回 str|int；否则 None。"""
    src = op_by_id.get(oid)
    if isinstance(src, tir.TConstParamRef):
        return src.name
    if isinstance(src, tir.TConstInt):
        return src.value
    return None


def _unwrap_expand_dim(op_by_id, oid: str) -> str:
    """跟随 expand_dim 链回到被展开的 tile（mask 谓词常作用于展开后的 rows2/cols2）。"""
    seen = set()
    while oid not in seen and isinstance(op_by_id.get(oid), tir.TExpandDim):
        seen.add(oid)
        oid = op_by_id[oid].tile
    return oid


def _flatten(ops) -> list:
    """拍平嵌套循环体（v0.4-kloop §6）：tiling 模式识别在指令全集上进行，
    与指令所在循环层级无关。"""
    out = []
    for op in ops:
        out.append(op)
        if isinstance(op, tir.TFor):
            out.extend(_flatten(op.body))
    return out


def plan_launch(ops: List[tir.TOp], env, loc: Loc) -> tir.LaunchPlan:
    ops = _flatten(ops)
    op_by_id: Dict[str, tir.TOp] = {op.id: op for op in ops if op.id}

    pid_by_axis: Dict[int, tir.TProgramId] = {}
    for op in ops:
        if isinstance(op, tir.TProgramId):
            pid_by_axis.setdefault(op.axis, op)

    axes = []
    for axis in sorted(pid_by_axis):
        pid = pid_by_axis[axis]

        # 候选 tiling：mul(pid, CE)，CE 为 constexpr 名或字面量
        mul_cands = []
        for op in ops:
            if isinstance(op, tir.TArith) and op.op == "*":
                if op.lhs == pid.id:
                    ce = _as_constexpr_operand(op_by_id, op.rhs)
                elif op.rhs == pid.id:
                    ce = _as_constexpr_operand(op_by_id, op.lhs)
                else:
                    continue
                if ce is not None:
                    mul_cands.append((op, ce))

        # 候选 tile：add(mul 结果, arange)
        tile_cands = []
        for m, ce in mul_cands:
            for op in ops:
                if isinstance(op, tir.TArith) and op.op == "+":
                    if op.lhs == m.id:
                        other = op.rhs
                    elif op.rhs == m.id:
                        other = op.lhs
                    else:
                        continue
                    src = op_by_id.get(other)
                    if isinstance(src, tir.TArange):
                        tile_cands.append((op, src, ce))

        if not tile_cands:
            raise err(loc, "E17",
                      f"cannot derive the launch grid: no `pid * CE + tila.arange(...)` "
                      f"tiling pattern found for program_id({axis})",
                      "the launcher grid rule is cdiv(dim, tiling) per axis; the tiling "
                      "idiom must be recognizable in the kernel body")

        ces = {ce for _, _, ce in tile_cands}
        if len(ces) > 1:
            names = ", ".join(str(c) for c in sorted(ces, key=str))
            raise err(loc, "E17",
                      f"cannot derive the launch grid: multiple constexpr values tile "
                      f"axis {axis} ({names}); exactly one tiling constexpr per axis")

        # 该轴的维：直接作用于 tiled offset 的 `offs < dim` 谓词
        # （谓词可经 expand_dim 展开——rows2 < M 作用于 rows 的展开）
        tiled_ids = {_unwrap_expand_dim(op_by_id, t.id) for t, _, _ in tile_cands}
        tiled_ids |= {t.id for t, _, _ in tile_cands}
        dims = set()
        for op in ops:
            if isinstance(op, tir.TCmp) and op.op in ("<", "<="):
                if _unwrap_expand_dim(op_by_id, op.lhs) not in tiled_ids:
                    continue
                src = op_by_id.get(op.rhs)
                if isinstance(src, tir.TSymRef):
                    dims.add(src.name)
                elif isinstance(src, tir.TConstInt):
                    dims.add(src.value)
        if not dims:
            raise err(loc, "E17",
                      f"cannot derive the launch grid: no `offs < dim` bound predicate "
                      f"found for the axis-{axis} tile",
                      "the grid dimension comes from the mask predicate covering the tile")
        if len(dims) > 1:
            ds = ", ".join(str(d) for d in sorted(dims, key=str))
            raise err(loc, "E17",
                      f"cannot derive the launch grid: axis {axis} is bounded by "
                      f"multiple dimensions ({ds}); the mapping must be unique")

        axes.append(tir.GridAxis(dims.pop(), ces.pop()))

    # shape 契约三件套（semantic-model.md §7 的运行期断言来源）
    rank_asserts, static_dim_asserts, sym_dim_asserts = [], [], []
    stride_bindings = []
    for name, bty in env.buffers.items():
        rank_asserts.append((name, len(bty.shape)))
        for i, d in enumerate(bty.shape):
            if isinstance(d, Const):
                static_dim_asserts.append((name, i, d.value))
            else:
                sym_dim_asserts.append((name, i, d.name))
        from ..types.memory import Strided
        if isinstance(bty.mem, Strided):
            for i, s in enumerate(bty.mem.strides):
                stride_bindings.append(
                    (name, i, s.value if isinstance(s, Const) else s.name))

    return tir.LaunchPlan(
        axes=tuple(axes),
        rank_asserts=tuple(rank_asserts),
        static_dim_asserts=tuple(static_dim_asserts),
        sym_dim_asserts=tuple(sym_dim_asserts),
        stride_bindings=tuple(stride_bindings),
    )
