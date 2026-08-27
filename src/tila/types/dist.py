"""DistExpr：logical tile element → computation ownership（docs/type-system.md §3）。

与 MemoryLayout（logical coordinate → memory address，types/memory.py）正交：
Tile 有 DistExpr、无 MemoryLayout；Buffer 有 MemoryLayout、无 DistExpr。

最终冻结的分类（评审 D，v0.6a 重构）——

    DistExpr ::= Identity(shape)
               | Product(DistExpr[rank])
               | Mma(M, N, K)
               | Lift(DistExpr, axis)
               | Slice(DistExpr, axis)

    ── 角色 ──
    Identity  base      arange（唯一种子构造点）
    Product   composite 坐标广播 / 逐轴段结合（n-ary，非二元树；D2 拍平）
    Mma       atomic    dot（不可分解的计算分布原子）
    Lift      derived   expand_dim / size-1 轴广播（R-PT 值侧撑大）
    Slice     derived   reduce（亲代切片 = provenance，不声称与独立低秩分布等价）

NoDist（评审 §25）：scalar/literal/constexpr 不携带分布——read_join(owner,
NoDist) = owner。语义上是 ScalarType 的分布面（类型层不物化字段）。

Seed 不是 DistExpr（评审 §38）：它是累加器种子状态在类型层的载体（zeros
值需要类型），join 单位元语义由 checker 的累加状态机实现（v0.6b 拆出完整
AccumulatorType{state = Seed(value) | Materialized(DistExpr)} 之前，Seed(Σ)
驻留 Tile 的 dist 字段）。一律不在析构律中化简。

normalize_dist 只做便宜、确定性的 rewrite（评审 §16 D1–D6），不做 speculative
rewrite：Slice(Mma,…) ≢ Identity、Lift 不裂回、Product 不合并回 Mma。

律（评审 §50 冻结的 7 条；v0.1 的擦除律 L1–L4 与写入包装 term 一并退役）：
  L1  结构等价        D ≡ normalize_dist(D)
  L2  Product 等价    逐项等价
  L3  Lift provenance Lift(A,k) ≡ Lift(B,k) ⇔ A ≡ B（同轴）
  L4  Slice provenance Slice(A,k) ≡ Slice(B,k) ⇔ A ≡ B（同轴）
  L5  strict join     同形双侧 owner：equiv → 任一侧；否则 E05
  L6  read join       一侧为 broadcast projection：结果 = 全形 owner 侧
  L7  marginalization marginal(Product, k) = 幸存因子；marginal(Mma, k) = Slice
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from . import shape


@dataclass(frozen=True)
class Identity:
    """arange 种子：元素 0..n-1 的规范分布。"""

    shape: tuple


@dataclass(frozen=True)
class Product:
    """独立坐标分布的笛卡尔复合（v0.2 起；Tila 语义分布积）。

    n-ary（评审 §7）：components 是逐轴分布段；不嵌套 Product（normalize_dist
    D2 拍平）。第 i 个 component 对应第 i 个逻辑轴。
    """

    components: tuple


@dataclass(frozen=True)
class Mma:
    """dot 种子（v0.2 matmul fragment）：MMA 指令规定的分布（原子）。"""

    m: int
    n: int
    k: int


@dataclass(frozen=True)
class Lift:
    """维度提升 / size-1 轴（评审 §13-§14、§31，v0.6a 更名自 Broadcast）。

    Lift(D, axis)：第 axis 轴不携带独立分布（expand_dim 的新增 size-1 轴，
    或 R-PT 值侧被谓词撑大的广播轴），D 的分布覆盖其余轴。不是 shape
    broadcasting 的替身——shape 广播由 types/shape.py 负责，Lift 只记录
    分布的"无所有权轴"。rank ≤ 2 下第 axis 轴外的唯一轴承托 D。
    """

    of: "DistExpr"
    axis: int


@dataclass(frozen=True)
class Slice:
    """原子布局沿轴 k 的边缘（v0.6a，docs/v0.6-attention.md §2.1 分支六）。

    rank-1 provenance：幸存轴继承其在亲代复合分布中的位置——Mma 的行/列
    分布不是独立轴分布的积，边缘分布没有独立构造的等价物，只能以后源方式
    陈述。equiv 仅同亲代结构相等；≢ Identity。能力边界：可作为普通 tile
    分布被读取（归约/经广播被读取/where·elem 操作数），不是可与任意同形
    tile 做 arithmetic join 的独立分布族。
    """

    parent: "DistExpr"
    axis: int


@dataclass(frozen=True)
class NoDist:
    """标量面：scalar/literal/constexpr 不携带分布（评审 §25）。

    ScalarType 在类型层没有 dist 字段——NoDist 是 join API 上的哨兵：
    read_join(owner, NODIST) = owner。
    """


NODIST = NoDist()


@dataclass(frozen=True)
class Seed:
    """distinguished 种子态（v0.4 zeros 的继承者，v0.6a 更名，非 DistExpr）。

    全同元素 tile 的分布不可观测——Seed 记录"分布未定"，按定义排除在
    DistExpr 等价世界之外（两个同构 Seed 相等是唯一成立情形），却以 join
    单位元身份参与累加合并（checker 状态机：Seed(Σ) + Tile[D] → D）。
    checker 的累加器检查以状态机表述（Seed → Materialized(L) → same L）；
    本 term 是该状态在类型层的载体（zeros 值需要类型）。v0.6b 起状态机
    携带种子值（Seed(value=V)，值属于指令 TZeros/TFull、状态属于类型态——
    `type-system.md` §3.2/§5 R19）。
    """

    shape: tuple


@dataclass(frozen=True)
class Alpha:
    """StateRead 布局元变量（v0.6b，`docs/v0.6-attention.md` §2.6）。

    累加器体内读的占位分布（第一遍 typing 用）：读的静态类型 =
    Tile[dt, Σ, α_acc]；同形 join 记约束、透明位不记；循环收尾 α := L_exit
    后统一复检（实现 = 两遍 typing：第一遍宽松求出口类型与约束解，第二遍
    以真实类型重推）。Alpha 不进入 DistExpr 等价世界——normalize 恒等、
    join 层对它的处理由 checker 的宽松路径承担。
    """

    name: str


DistExpr = Union[Identity, Product, Mma, Lift, Slice, NoDist]
TileDist = Union[DistExpr, Seed, Alpha]


def normalize_dist(d: TileDist) -> TileDist:
    """律 D1–D6（评审 §16）：便宜、确定性、无 speculative rewrite。"""
    if isinstance(d, (Identity, Mma, NoDist, Seed, Alpha)):
        return d
    if isinstance(d, Slice):  # D4 亲代已 normalize
        return Slice(normalize_dist(d.parent), d.axis)
    if isinstance(d, Lift):  # D5 + 同轴幂等（v0.2 L6 继承）
        inner = normalize_dist(d.of)
        if isinstance(inner, Lift) and inner.axis == d.axis:
            return inner
        return Lift(inner, d.axis)
    if isinstance(d, Product):  # D2 拍平 + D3 逐分量
        comps = []
        for c in d.components:
            c = normalize_dist(c)
            if isinstance(c, Product):
                comps.extend(c.components)
            else:
                comps.append(c)
        return Product(tuple(comps))
    raise AssertionError(f"unknown dist term {d!r}")


def equiv_dist(d1: TileDist, d2: TileDist) -> bool:
    return normalize_dist(d1) == normalize_dist(d2)


def lift(d: DistExpr, axis: int) -> Lift:
    """构造 Lift（checker 的 expand_dim / R-PT 撑大的公共入口）。"""
    return Lift(normalize_dist(d), axis)


# ---------------------------------------------------------------------------
# 打印（canonical dump 用；名字编号由 printer 的 DistNamer 管理）
# ---------------------------------------------------------------------------


def dist_str(d: TileDist, name_of=None) -> str:
    """渲染 dist term；name_of: DistExpr → str（已命名项渲染为名字）。"""
    if name_of is not None:
        n = name_of(d)
        if n is not None:
            return n
    if isinstance(d, Identity):
        inner = ", ".join(shape.dim_str(dim) for dim in d.shape)
        return f"identity({inner})"
    if isinstance(d, Product):
        inner = ",".join(dist_str(c, name_of) for c in d.components)
        return f"product({inner})"
    if isinstance(d, Mma):
        return f"mma({d.m},{d.n},{d.k})"
    if isinstance(d, Lift):
        return f"lift({dist_str(d.of, name_of)}, {d.axis})"
    if isinstance(d, Slice):
        return f"slice({dist_str(d.parent, name_of)}, {d.axis})"
    if isinstance(d, NoDist):
        return "no_dist"
    if isinstance(d, Seed):
        inner = ", ".join(shape.dim_str(dim) for dim in d.shape)
        return f"seed({inner})"
    if isinstance(d, Alpha):
        return f"α({d.name})"
    raise AssertionError(f"unknown dist term {d!r}")