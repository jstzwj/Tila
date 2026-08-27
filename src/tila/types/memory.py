"""MemoryLayout：logical coordinate → memory address（docs/type-system.md §3.1）。

与 DistExpr（logical tile element → computation ownership，types/dist.py）
正交：Tile 有 DistExpr、无 MemoryLayout；Buffer 有 MemoryLayout、无 DistExpr。
MemoryLayout 只回答 (i₀,…,iₙ) 对应内存哪里——offset = Σᵢ iᵢ·strideᵢ。
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