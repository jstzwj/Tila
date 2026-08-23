"""Layout 代数（docs/type-system.md §3，编译器内部——用户不可见）。

DistLayout term 是等价/来源描述符（provenance / equivalence descriptor），
不是 GPU 具体分布的描述。策略：追踪等价类，不追踪具体映射。

化简律（normalize 到不动点，自底向上重写）：
  (L1) load(M, L)       ≡ L      数据搬运不改变"谁持有哪个元素"
  (L2) bcast_scalar(L)  ≡ L      标量广播不改变分布
  (L3) cast(L)          ≡ L      dtype 转换不改变分布
  (L4) join(L, L)       ≡ L      等价分布逐元素运算，结果分布不变
  (L5) Product 同余 + 单位元律  （v0.2；此处已实现以保持签名通用）

等价 equiv = normalize 后结构相等。v0.1 事实：唯一种子构造点是 arange，
L1–L4 全部是擦除性的，正规形式恒为 Identity(n)——等价判定退化为种子相等。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from . import shape
from .shape import Const, Symbol  # noqa: F401  (re-export convenience)


@dataclass(frozen=True)
class RowMajor:
    """MemoryLayout：隐式默认（v0.1 唯一取值）。不物化 stride 表——
    lowering 从运行期 shape 计算行主序步长；launcher 断言张量实际连续。"""

    def __str__(self) -> str:
        return "row_major"


@dataclass(frozen=True)
class Strided:
    """MemoryLayout：显式声明的逐轴步长（v0.3，docs/v0.3-strides.md §2.1）。

    strides 每项是 Const(n) | Symbol(name)，长度 = rank。M 不参与等价判定：
    它是 per-Buffer 的事实，两个 Buffer 类型从不互相 unify。
    """

    strides: tuple

    def __str__(self) -> str:
        inner = ", ".join(shape.dim_str(d) for d in self.strides)
        return f"strided({inner})"


ROW_MAJOR = RowMajor()
MemoryLayout = Union[RowMajor, Strided]


def strides_of(mem: MemoryLayout, shape) -> tuple:
    """Address Function 的逐轴系数（docs/v0.3-strides.md §1.3，唯一事实源）。

        L(i₀,…,i_{r−1}) = Σᵢ iᵢ · strides_of(mem, shape)[i]

    Strided → 声明值；RowMajor → 由 shape 推导（rank ≤ 2：stride(i) = d_{i+1}、
    末轴 1，无符号乘积）。rank ≥ 3 的默认行主序需要 d·d 乘积（DimExpr 阶段二），
    随 rank 解禁——NotImplementedError。lowering 按系数发射文本；
    interpreter 以逻辑索引实现同一地址函数（等价性由 stride 契约保证，
    docs/v0.3-strides.md §5）。
    """
    if isinstance(mem, Strided):
        return tuple(mem.strides)
    dims = []
    for i in range(len(shape)):
        tail = shape[i + 1:]
        if not tail:
            dims.append(Const(1))
        elif len(tail) == 1:
            dims.append(tail[0])
        else:
            raise NotImplementedError(
                "default row-major strides for rank >= 3 are deferred "
                "with the rank limit (v0.3-strides §9)")
    return tuple(dims)


@dataclass(frozen=True)
class Identity:
    """arange 种子：元素 0..n-1 的规范分布。"""

    shape: tuple


@dataclass(frozen=True)
class Mma:
    """dot 种子（v0.2 matmul fragment）：MMA 指令规定的分布。

    第一个不由 arange 推导的 layout term——由后端（MMA 指令形态）决定而非
    由来源决定；结构相等即等价，与 identity/product 族互不相等。
    """

    m: int
    n: int
    k: int


@dataclass(frozen=True)
class Zeros:
    """distinguished 种子态（v0.4，docs/v0.4-kloop.md §2.1）。

    全同元素 tile 的分布不可观测——Zeros 记录"分布未定"（ZeroSeed），不是
    一种具体 DistLayout：按定义排除在 equiv 之外（两个"未定"相等是唯一
    成立情形），却以 join 单位元身份参与合并（L7 = 种子物化规则）。
    checker 的累加器检查以状态机表述（Seed → Materialized(L) → same L）；
    本 term 是该状态在类型层的载体（zeros 值需要类型）。
    """

    shape: tuple


L7_NOTE = "L7（v0.4）：join(Zeros(Σ), L) ≡ L——常量分布是 join 的单位元"


@dataclass(frozen=True)
class LoadL:
    """经 load 产生（mem 是来源 MemoryLayout，仅作来源记录，不参与等价）。"""

    mem: MemoryLayout
    idx: "LayoutTerm"


@dataclass(frozen=True)
class BcastScalarL:
    of: "LayoutTerm"


@dataclass(frozen=True)
class CastL:
    of: "LayoutTerm"


@dataclass(frozen=True)
class JoinL:
    lhs: "LayoutTerm"
    rhs: "LayoutTerm"


@dataclass(frozen=True)
class ProductL:
    """独立坐标分布的笛卡尔复合（v0.2 起；Tila 语义分布积）。"""

    lhs: "LayoutTerm"
    rhs: "LayoutTerm"


@dataclass(frozen=True)
class BroadcastL:
    """size-1 轴广播（v0.2 起）。"""

    of: "LayoutTerm"
    axis: int


LayoutTerm = Union[Identity, Mma, Zeros, LoadL, BcastScalarL, CastL, JoinL,
                   ProductL, BroadcastL]


def normalize(l: LayoutTerm) -> LayoutTerm:
    """应用律 L1–L7 到不动点（项构造已保证一层化简即达不动点）。"""
    if isinstance(l, (Identity, Mma, Zeros)):
        return l
    if isinstance(l, LoadL):  # L1
        return normalize(l.idx)
    if isinstance(l, BcastScalarL):  # L2
        return normalize(l.of)
    if isinstance(l, CastL):  # L3
        return normalize(l.of)
    if isinstance(l, JoinL):
        ln, rn = normalize(l.lhs), normalize(l.rhs)
        if isinstance(ln, Zeros):
            return rn          # L7：种子物化（v0.4-kloop §2.1）
        if isinstance(rn, Zeros):
            return ln          # L7 对称
        return ln              # L4
    if isinstance(l, ProductL):  # L5 同余
        return ProductL(normalize(l.lhs), normalize(l.rhs))
    if isinstance(l, BroadcastL):  # L6 幂等（v0.2）
        inner = normalize(l.of)
        if isinstance(inner, BroadcastL) and inner.axis == l.axis:
            return inner
        return BroadcastL(inner, l.axis)
    raise AssertionError(f"unknown layout term {l!r}")


def equiv(l1: LayoutTerm, l2: LayoutTerm) -> bool:
    return normalize(l1) == normalize(l2)


# ---------------------------------------------------------------------------
# 打印（canonical dump 用；名字编号由 printer 的 LayoutNamer 管理）
# ---------------------------------------------------------------------------


def layout_str(l: LayoutTerm, name_of=None) -> str:
    """渲染 layout term；name_of: LayoutTerm → str（已命名项渲染为名字）。"""
    if name_of is not None:
        n = name_of(l)
        if n is not None:
            return n
    if isinstance(l, Identity):
        inner = ", ".join(shape.dim_str(d) for d in l.shape)
        return f"identity({inner})"
    if isinstance(l, Mma):
        return f"mma({l.m},{l.n},{l.k})"
    if isinstance(l, Zeros):
        inner = ", ".join(shape.dim_str(d) for d in l.shape)
        return f"zeros({inner})"
    if isinstance(l, LoadL):
        return f"load({l.mem}, {layout_str(l.idx, name_of)})"
    if isinstance(l, BcastScalarL):
        return f"bcast_scalar({layout_str(l.of, name_of)})"
    if isinstance(l, CastL):
        return f"cast({layout_str(l.of, name_of)})"
    if isinstance(l, JoinL):
        return f"join({layout_str(l.lhs, name_of)}, {layout_str(l.rhs, name_of)})"
    if isinstance(l, ProductL):
        return f"product({layout_str(l.lhs, name_of)},{layout_str(l.rhs, name_of)})"
    if isinstance(l, BroadcastL):
        return f"broadcast({layout_str(l.of, name_of)}, {l.axis})"
    raise AssertionError(f"unknown layout term {l!r}")
