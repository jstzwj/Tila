"""layout 推导与 equiv 检查（docs/type-checker.md §3，E05；v0.2 预览 R14 的 Product 推导）。"""

from __future__ import annotations

from ..diagnostics import Loc, TilaError, err
from ..types.layout import (
    Identity,
    LayoutTerm,
    ProductL,
    Zeros,
    equiv,
    layout_str,
    normalize,
)


def require_equiv(l1: LayoutTerm, l2: LayoutTerm, loc: Loc, what: str) -> None:
    if not equiv(l1, l2):
        raise err(
            loc, "E05", f"layouts are not equivalent in {what}",
            f"lhs: {layout_str(normalize(l1))}",
            f"rhs: {layout_str(normalize(l2))}",
            "v0.1 has no layout conversion; both operands must descend from the "
            "same distribution seed",
        )


def _parts(l: LayoutTerm) -> list:
    """非擦除正规项的逐轴分解：Product 递归展开；其余项（identity 种子等）是单段。"""
    l = normalize(l)
    if isinstance(l, ProductL):
        return _parts(l.lhs) + _parts(l.rhs)
    return [l]


def _is_size1(dim) -> bool:
    v = getattr(dim, "value", None)
    return v == 1


def broadcast_layout(l1: LayoutTerm, s1, l2: LayoutTerm, s2,
                     loc: Loc, what: str) -> LayoutTerm:
    """R14（v0.2 预览）的 layout 推导：逐轴取非平凡侧的分布。

    - 某轴只有一方非平凡（长度 >1）→ 取该方分布段；
    - 两方都非平凡 → 该轴要求两侧分布段等价（否则 E05），取其一；
    - size-1 轴分布平凡，不产生段（layout 代数对 size-1 轴不可见）。
    结果 = 各段按轴序左折叠的 Product；恰一段时就是该段本身。
    """
    p1, p2 = _parts(l1), _parts(l2)
    n1 = sum(1 for d in s1 if not _is_size1(d))
    n2 = sum(1 for d in s2 if not _is_size1(d))
    if len(p1) != n1 or len(p2) != n2:
        raise err(
            loc, "E05", f"cannot derive per-axis layouts for broadcast in {what}",
            f"lhs: {layout_str(normalize(l1))} covers {len(p1)} axis group(s), "
            f"shape has {n1} non-trivial axis(es)",
            f"rhs: {layout_str(normalize(l2))} covers {len(p2)} axis group(s), "
            f"shape has {n2} non-trivial axis(es)",
            "broadcasting requires per-axis decomposable layouts "
            "(seeds or products of seeds)",
        )
    it1, it2 = iter(p1), iter(p2)
    segs = []
    for d1, d2 in zip(s1, s2):
        one1, one2 = _is_size1(d1), _is_size1(d2)
        if one1 and one2:
            continue
        if one2:
            segs.append(next(it1))
        elif one1:
            segs.append(next(it2))
        else:
            a, b = next(it1), next(it2)
            if normalize(a) != normalize(b):
                raise err(
                    loc, "E05", f"axis layouts are not equivalent in {what}",
                    f"lhs axis: {layout_str(normalize(a))}",
                    f"rhs axis: {layout_str(normalize(b))}",
                )
            segs.append(a)
    if not segs:  # 全 size-1 的退化情形
        return Identity(())
    out = segs[0]
    for seg in segs[1:]:
        out = ProductL(out, seg)
    return out


def marginal(l: LayoutTerm, s, k: int, loc: Loc, what: str) -> LayoutTerm:
    """R20 边缘化（v0.5-reduce §2.1）：归约 = 分布的边缘化——被归约轴积分掉，
    幸存轴保留自身分布。构造规则而非化简律（无新 term）：

      1. Product(L₀, L₁)          → 幸存因子（Product 只在双非平凡轴时构造）
      2. Σ[k] ≡ 1（归约平凡轴）   → L 原样（layout 代数对 size-1 轴不可见）
      3. 唯一非平凡轴被归约       → Identity(())（结果全平凡；与 broadcast_layout
                                   的全 size-1 退化同款）
      4. Zeros(Σ)                 → Zeros(Σ∖k)（全同值的边缘分布仍是全同值——
                                   Zeros 种子对轴删除封闭）
      其余（Mma 等不可按轴分解项）→ E21 ReduceLayout：**语义层**边缘化未定义
      （不声称 backend 做不到——v0.6 Mma 归约若落地是 Convertible 新规则）。
    """
    n = normalize(l)
    nontriv = [i for i, d in enumerate(s) if getattr(d, "value", None) != 1]
    if isinstance(n, Zeros):
        return Zeros(tuple(d for i, d in enumerate(s) if i != k))
    if not nontriv:
        return Identity(())
    parts = _parts(n)
    if len(parts) == 2 and len(nontriv) == 2:
        return n.rhs if k == 0 else n.lhs
    if len(parts) == 1 and len(nontriv) == 1:
        return n if k != nontriv[0] else Identity(())
    raise err(
        loc, "E21",
        f"semantic marginalization is not defined for this layout in {what}",
        f"operand layout: {layout_str(n)} covers {len(parts)} axis group(s), "
        f"shape has {len(nontriv)} non-trivial axis(es)",
        "the layout is not Product-compatible (a dot result carries an MMA "
        "distribution whose row/column axes are not independent); attention "
        "(softmax over dot results) is the v0.6 topic — reduce loaded or "
        "computed tiles in the Product family",
        subcode="ReduceLayout",
    )
