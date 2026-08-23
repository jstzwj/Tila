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
    """MemoryLayout：v0.1 唯一取值（Triton 指针语义）。"""

    def __str__(self) -> str:
        return "row_major"


ROW_MAJOR = RowMajor()
MemoryLayout = RowMajor


@dataclass(frozen=True)
class Identity:
    """arange 种子：元素 0..n-1 的规范分布。"""

    shape: tuple


@dataclass(frozen=True)
class Mma:
    """dot 种子（v0.2 matmul fragment）：MMA 指令规定的分布。

    第一个不由 arange 推导的 layout term——由后端（MMA 指令形态）决定而非
    由来源决定；结构相等即等价，与 identity/product 族互不等价。
    """

    m: int
    n: int
    k: int


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


LayoutTerm = Union[Identity, Mma, LoadL, BcastScalarL, CastL, JoinL, ProductL, BroadcastL]


def normalize(l: LayoutTerm) -> LayoutTerm:
    """应用律 L1–L5 到不动点（项构造已保证一层化简即达不动点）。"""
    if isinstance(l, (Identity, Mma)):
        return l
    if isinstance(l, LoadL):  # L1
        return normalize(l.idx)
    if isinstance(l, BcastScalarL):  # L2
        return normalize(l.of)
    if isinstance(l, CastL):  # L3
        return normalize(l.of)
    if isinstance(l, JoinL):  # L4
        return normalize(l.lhs)
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
