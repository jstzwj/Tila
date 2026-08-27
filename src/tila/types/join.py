"""join 纯结构助手（评审 §49/§55：types/join.py）。

与 types/dist.py（term + normalize/equiv）分离：join 层只做 shape 驱动的
结构判定与逐轴段分解——strict_join / read_join 的规则实现（含 E05 诊断）
在 checker/layout.py；此处只有可独立测试的纯谓词。

约束：本模块（以及 types/dist.py）绝不依赖 shape-symbol proof（types/shape.py
与将来的 constraints.py 只处理标量/shape 符号，不看到 Mma/Product/Slice）——
join 判定全是结构性的，无 SMT。
"""

from __future__ import annotations

from .dist import DistExpr, Identity, Lift, Mma, Product, Seed, Slice, normalize_dist


def is_proper_broadcast_projection(sb, sf) -> bool:
    """Σ_b ⊰ Σ_f（R-BT 的 one-sided 判定，docs/v0.6-attention.md §2.2）：
    逐轴 s_b 平凡或与 s_f 相等，且至少一轴 s_b 平凡而 s_f 非平凡（真广播，
    排除同形——同形双侧仍走 equiv）。"""
    if len(sb) != len(sf):
        return False
    proper = False
    for db, df in zip(sb, sf):
        v = getattr(db, "value", None)
        if v == 1:
            proper = proper or getattr(df, "value", None) != 1
        elif db != df:
            return False
    return proper


def axis_segments(d: DistExpr) -> list:
    """非平凡轴分布段（按轴序）：Product 递归展开；Lift 剥离无所有权轴
    （rank ≤ 2 下第 axis 轴外的唯一轴承托 of）；Seed 是单段（v0.6a 重构
    修复：与旧 `_parts` 的 Zeros 行为对齐——种子 tile 参与逐轴联合时以
    单段入场，其不可观测分布在 Product 段中原样保留）；其余项（identity
    种子、Mma、Slice）是单段。段本身不再含 Lift/Product。"""
    d = normalize_dist(d)
    if isinstance(d, Product):
        out = []
        for c in d.components:
            out.extend(axis_segments(c))
        return out
    if isinstance(d, Lift):
        return axis_segments(d.of)
    if isinstance(d, (Identity, Mma, Slice, Seed)):
        return [d]
    raise AssertionError(f"no axis decomposition for dist term {d!r}")