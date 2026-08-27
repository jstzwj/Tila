"""分布推导与 join 检查（docs/type-checker.md §3，E05；v0.6a DistExpr 重构）。

checker 规则层：strict_join / read_join / join_distributions / marginal /
grow_to_shape / infer_coordinate_dist。纯分布代数在 types/dist.py
（normalize_dist / equiv_dist / lift）与 types/join.py（axis_segments /
is_proper_broadcast_projection）。join 判定全是结构性的——shape 符号证明
（constraints 层）永不参与分布判定。

Owner / Reader 纪律（评审 §52）：任何运算先判断"谁拥有结果元素"——
  strict_join  同形双侧都是 owner：equiv → 任一侧；否则 E05
  read_join    一侧是结果 shape 的（真）广播投影：它是 reader，结果 = 全形侧
  join_distributions  以上两者的分派 + 逐轴段联合（双广播侧各持有非平凡轴：
  掩码/坐标模式——评审 §32 的 infer_coordinate_dist 同款机制）
"""

from __future__ import annotations

from typing import Optional

from ..diagnostics import Loc, err
from ..types.dist import (
    Alpha,
    Identity,
    Lift,
    Mma,
    NODIST,
    NoDist,
    Product,
    Seed,
    Slice,
    TileDist,
    dist_str,
    equiv_dist,
    lift,
    normalize_dist,
)
from ..types.join import axis_segments, is_proper_broadcast_projection

_BROADCAST_HINT = (
    "result ownership must be single: both operands hold every element of the "
    "result shape (strict join), or one is a size-1 broadcast reader of the "
    "other (read join)"
)


def require_equiv(d1: TileDist, d2: TileDist, loc: Loc, what: str) -> None:
    if not equiv_dist(d1, d2):
        raise err(
            loc, "E05", f"distributions are not equivalent in {what}",
            f"lhs: {dist_str(normalize_dist(d1))}",
            f"rhs: {dist_str(normalize_dist(d2))}",
            _BROADCAST_HINT,
        )


def strict_join(d1: TileDist, d2: TileDist, loc: Loc, what: str) -> TileDist:
    """L5：同形双侧 owner 的 strict join——equiv → d1，否则 E05。

    E05 的定义（评审 §51）：strict-join failure between two result-owning
    distributions；不再是泛化的 "layout mismatch"。

    Seed 单位元（L7 语义的 join 层落地，v0.6b）：任一侧是种子态 → 结果 =
    另一侧（常量分布 ⊕ L ≡ L——与 `+=` 物化同源；maximum 的种子读并入，
    docs/v0.6-attention.md §2.4）。
    """
    d1, d2 = normalize_dist(d1), normalize_dist(d2)
    if isinstance(d1, Seed):
        return d2
    if isinstance(d2, Seed):
        return d1
    if not equiv_dist(d1, d2):
        raise err(
            loc, "E05",
            f"strict-join failure between two result-owning distributions in {what}",
            f"lhs: {dist_str(d1)}",
            f"rhs: {dist_str(d2)}",
            _BROADCAST_HINT,
        )
    return d1


def read_join(owner: TileDist, o_shape, reader: TileDist, r_shape,
              loc: Loc, what: str) -> Optional[TileDist]:
    """L6：one-sided read join——reader 是 owner 的真广播投影 → owner。

    reader 为 NoDist（标量面）时恒成立（标量不携带分布，评审 §25）。
    不适用时返回 None（调用方落入 join_distributions 的逐轴路径），
    不在本函数内拒绝——"不在 ReadJoin 里自动合成 owner"由此成立。
    """
    if isinstance(reader, NoDist):
        return normalize_dist(owner)
    if r_shape is not None and is_proper_broadcast_projection(r_shape, o_shape):
        return normalize_dist(owner)
    return None


def join_distributions(d1: TileDist, s1, d2: TileDist, s2,
                       loc: Loc, what: str) -> TileDist:
    """二元运算的分布推导（R14/R-BT 的 v0.6a 形式）。

    1. 同形双侧 → strict_join（E05 主战场不动）；
    2. 恰一侧是另一侧的真广播投影（R-BT one-sided）→ 全形侧获胜（读侧免检）；
    3. 其余（双广播侧）→ 逐轴取非平凡侧的分布段构成 n-ary Product（掩码 /
       坐标模式；两侧各持有自己的非平凡轴）。size-1 轴不产生段（其轴本身
       无所有权；expand_dim 的 Lift 在 axis_segments 中剥离）。
    """
    if s1 == s2:
        return strict_join(d1, d2, loc, what)
    o = read_join(d1, s1, d2, s2, loc, what)
    if o is not None:
        return o
    o = read_join(d2, s2, d1, s1, loc, what)
    if o is not None:
        return o
    p1, p2 = axis_segments(d1), axis_segments(d2)
    n1 = sum(1 for d in s1 if getattr(d, "value", None) != 1)
    n2 = sum(1 for d in s2 if getattr(d, "value", None) != 1)
    if len(p1) != n1 or len(p2) != n2:
        raise err(
            loc, "E05", f"cannot derive per-axis distributions for join in {what}",
            f"lhs: {dist_str(normalize_dist(d1))} covers {len(p1)} axis group(s), "
            f"shape has {n1} non-trivial axis(es)",
            f"rhs: {dist_str(normalize_dist(d2))} covers {len(p2)} axis group(s), "
            f"shape has {n2} non-trivial axis(es)",
            "per-axis join requires axis-decomposable distributions "
            "(seeds or products of seeds)",
        )
    it1, it2 = iter(p1), iter(p2)
    segs = []
    for dd1, dd2 in zip(s1, s2):
        one1, one2 = getattr(dd1, "value", None) == 1, getattr(dd2, "value", None) == 1
        if one1 and one2:
            continue
        if one2:
            segs.append(next(it1))
        elif one1:
            segs.append(next(it2))
        else:
            a, b = next(it1), next(it2)
            if normalize_dist(a) != normalize_dist(b):
                raise err(
                    loc, "E05", f"axis distributions are not equivalent in {what}",
                    f"lhs axis: {dist_str(normalize_dist(a))}",
                    f"rhs axis: {dist_str(normalize_dist(b))}",
                )
            segs.append(a)
    if not segs:  # 全 size-1 的退化情形
        return Identity(())
    return Product(tuple(segs))


def infer_coordinate_dist(coord_tys, loc: Loc, what: str):
    """坐标分布的联合推导（评审 §32/§33）：逐坐标 shape ⊗ + 分布逐轴联合。

    坐标 tile 各自携带自己的分布（expand_dim 的 Lift）；返回 (坐标广播 shape,
    联合分布 = n-ary Product / 单段)。完全不看 MemoryLayout——两套系统正交。
    坐标位是内存边界位：StateRead（Alpha）出现在坐标 → E20 AccumRead
    （逃逸；与 store 实参同纪律）。"""
    for (_, cl, cloc) in coord_tys:
        if isinstance(cl, Alpha):
            raise err(cloc, "E20",
                      "an accumulator may be read only inside pure value "
                      "expressions; addressing coordinates are a memory "
                      "boundary (escape is not allowed)",
                      subcode="AccumRead")
    shape = layout = None
    for (cs, cl, cloc) in coord_tys:
        if shape is None:
            shape, layout = cs, cl
        else:
            from . import broadcast as bcheck

            new_shape = bcheck.require_broadcast(shape, cs, cloc, what)
            layout = join_distributions(layout, shape, cl, cs, cloc, what)
            shape = new_shape
    return shape, normalize_dist(layout)


def grow_to_shape(l: TileDist, s_from, s_to, loc: Loc, what: str) -> TileDist:
    """把分布 l 从 s_from 广播到更大的 s_to（R-PT 的 cond 撑大值侧时使用）：
    每个 s_from 平凡而 s_to 非平凡的轴包一层 Lift（无所有权轴）。结果恒为
    正规形式（同轴双 Lift 在 normalize_dist 中幂等去重）。"""
    out = normalize_dist(l)
    for axis, (df, dt) in enumerate(zip(s_from, s_to)):
        if getattr(df, "value", None) == 1 and getattr(dt, "value", None) != 1:
            out = lift(out, axis)
    return normalize_dist(out)


def marginal(l: TileDist, s, k: int, loc: Loc, what: str) -> TileDist:
    """R20 边缘化（v0.5-reduce §2.1 + v0.6-attention §2.1 分支六）：归约 =
    分布的边缘化——被归约轴积分掉，幸存轴保留自身分布。构造规则而非化简律：

      0. Lift(D, axis=k)      → D（被归约轴恰是 Lift 的无所有权轴）
      1. Product(L₀, L₁)      → 幸存分量（Product 只在双非平凡轴时构造）
      2. Σ[k] ≡ 1（归约平凡轴） → L 原样（无所有权轴对分布代数不可见；
                                  Lift 情形由 0 先行处理）
      3. 唯一非平凡轴被归约      → Identity(())（结果全平凡；与
                                 join_distributions 的全 size-1 退化同款）
      4. Seed(Σ)              → Seed(Σ∖k)（全同值的边缘分布仍是全同值——
                                 Seed 种子对轴删除封闭）
      6. Mma（原子种子类）      → Slice(Mma, k)——边缘 = 亲代切片（provenance；
                                 oracle 第三轮 H3-S：Triton 的 tt.reduce 结果
                                 encoding 即 #ttg.slice<{dim=k, parent=…}>）
      其余（未来的原子 term）   → E21 ReduceLayout（v0.6 分支六后可达布局
                                 全部有定义——E21 对合法输入成为空位）。
    """
    n = normalize_dist(l)
    if isinstance(n, Seed):
        return Seed(tuple(d for i, d in enumerate(s) if i != k))
    if isinstance(n, Lift) and n.axis == k:
        return n.of
    nontriv = [i for i, d in enumerate(s) if getattr(d, "value", None) != 1]
    if not nontriv:
        return Identity(())
    parts = axis_segments(n)
    if isinstance(n, Product) and len(parts) == 2 and len(nontriv) == 2:
        return n.components[1] if k == 0 else n.components[0]
    if len(parts) == 1 and len(nontriv) == 1:
        return n if k != nontriv[0] else Identity(())
    if isinstance(n, Mma):
        return Slice(n, k)
    raise err(
        loc, "E21",
        f"semantic marginalization is not defined for this distribution in {what}",
        f"operand dist: {dist_str(n)} covers {len(parts)} axis group(s), "
        f"shape has {len(nontriv)} non-trivial axis(es)",
        "Mma inputs are covered by branch 6 since v0.6 (Slice of the parent); "
        "this error is reachable only for future atomic dist terms",
        subcode="ReduceLayout",
    )