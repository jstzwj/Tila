"""类型分层（docs/type-system.md §1/§6）。

评审 D（v0.6a DistExpr 重构）：Type ::= ScalarType(dtype)
       | TileType(dtype, shape, dist)
       | BufferType(dtype, shape, mem)
       | AddressType(dtype, shape, dist)
       | UnitType

两个正交的语义层（docs/type-system.md §3）：
  MemoryLayout   logical coordinate → memory address（Buffer 持有）
  DistExpr       logical tile element → computation ownership（Tile/Address 持有）
Tile 没有 MemoryLayout；Buffer 没有 DistExpr。
"""

from __future__ import annotations

from dataclasses import dataclass

from .dist import TileDist, dist_str
from .memory import MemoryLayout, Strided
from .shape import Shape, dim_str, shape_str


@dataclass(frozen=True)
class TilaType:
    pass


@dataclass(frozen=True)
class ScalarType(TilaType):
    dtype: str  # v0.1 值域只有 i32 / f32


@dataclass(frozen=True)
class TileType(TilaType):
    dtype: str
    shape: Shape
    dist: TileDist


@dataclass(frozen=True)
class BufferType(TilaType):
    dtype: str
    shape: Shape
    mem: MemoryLayout


@dataclass(frozen=True)
class AddressType(TilaType):
    dtype: str
    shape: Shape
    dist: TileDist


@dataclass(frozen=True)
class UnitType(TilaType):
    pass


UNIT = UnitType()


def type_str(t: TilaType, name_of=None) -> str:
    """类型打印（canonical dump 口径）。

    Scalar(i32) / Tile<i32, (128,), L0> / Buffer<f32, (N,)> /
    Address<f32, (128,), L0> / ()。name_of 提供 dist 名字注册表；
    不提供时 dist 打印正规形式（诊断口径）。
    """
    if isinstance(t, ScalarType):
        return f"Scalar({t.dtype})"
    if isinstance(t, TileType):
        return f"Tile<{t.dtype}, {shape_str(t.shape)}, {dist_str(t.dist, name_of)}>"
    if isinstance(t, BufferType):
        if isinstance(t.mem, Strided):
            inner = ",".join(dim_str(d) for d in t.mem.strides)
            return f"Buffer<{t.dtype}, {shape_str(t.shape)}, strides=({inner})>"
        return f"Buffer<{t.dtype}, {shape_str(t.shape)}>"
    if isinstance(t, AddressType):
        return f"Address<{t.dtype}, {shape_str(t.shape)}, {dist_str(t.dist, name_of)}>"
    if isinstance(t, UnitType):
        return "()"
    raise AssertionError(f"unknown type {t!r}")