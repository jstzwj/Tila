"""Tila 类型对象（docs/type-system.md §1）。

同一组类既是用户注解求值的产物（spec），也是 checker 的类型表示：
注解在装饰期求值成这些对象，特化期把 Const 符号代入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import lcm

from . import dtypes as D
from .dims import Cst, DimExpr, int_value

# ---------------------------------------------------------------------------
# 访问能力与地址空间（type-system.md §8–§9）
# ---------------------------------------------------------------------------

class Access(Enum):
    READ_ONLY = "ReadOnly"
    WRITE_ONLY = "WriteOnly"
    READ_WRITE = "ReadWrite"

    def __str__(self):
        return self.value


class AddressSpace(Enum):
    GLOBAL = "Global"
    SHARED = "Shared"
    LOCAL = "Local"

    def __str__(self):
        return self.value


READ_ONLY = Access.READ_ONLY
WRITE_ONLY = Access.WRITE_ONLY
READ_WRITE = Access.READ_WRITE
GLOBAL = AddressSpace.GLOBAL
SHARED = AddressSpace.SHARED
LOCAL = AddressSpace.LOCAL

# 命名空间别名（ti.ReadOnly 等）
ReadOnly = READ_ONLY
WriteOnly = WRITE_ONLY
ReadWrite = READ_WRITE
Global = GLOBAL
Shared_ = SHARED
Local_ = LOCAL


# ---------------------------------------------------------------------------
# 内存身份与别名关系（ADR-002）
# ---------------------------------------------------------------------------

class RegionId:
    """编译器生成的内存身份；不属于任何公共类型参数。"""


@dataclass(frozen=True)
class ParamRegion(RegionId):
    index: int
    name: str

    def __str__(self):
        return f"ParamRegion[{self.index}:{self.name}]"


@dataclass(frozen=True)
class BufferRegion(RegionId):
    index: int
    name: str

    def __str__(self):
        return f"BufferRegion[{self.index}:{self.name}]"


@dataclass(frozen=True)
class InternalRegion(RegionId):
    allocation_id: int

    def __str__(self):
        return f"InternalRegion[{self.allocation_id}]"


@dataclass(frozen=True)
class UnknownRegion(RegionId):
    def __str__(self):
        return "UnknownRegion"


UNKNOWN_REGION = UnknownRegion()


class AliasRelation(str, Enum):
    MUST_ALIAS = "MustAlias"
    MAY_ALIAS = "MayAlias"
    NO_ALIAS = "NoAlias"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class AliasFact:
    left: RegionId
    right: RegionId
    relation: AliasRelation
    reason: str

    def describe(self):
        return f"{self.relation}[{self.left}, {self.right}] ({self.reason})"


def default_alias_relation(left: RegionId,
                           right: RegionId) -> AliasRelation:
    """无额外契约时的保守 alias lattice。"""
    if left == right and not isinstance(left, UnknownRegion):
        return AliasRelation.MUST_ALIAS
    if isinstance(left, InternalRegion) and isinstance(right, InternalRegion):
        return AliasRelation.NO_ALIAS
    return AliasRelation.MAY_ALIAS


@dataclass(frozen=True)
class Alignment:
    bytes: int

    @property
    def text(self):
        return f"Aligned[{self.bytes}]"

    def __str__(self):
        return self.text


@dataclass(frozen=True)
class UnknownAlignment:
    def __str__(self):
        return "UnknownAlignment"


UNKNOWN_ALIGNMENT = UnknownAlignment()


class Extent:
    """Bounds 数值域；与 RegionId 身份域完全正交（ADR-002）。"""


@dataclass(frozen=True)
class LinearExtent(Extent):
    expr: DimExpr

    def __str__(self):
        return str(self.expr)


@dataclass(frozen=True)
class UnknownExtent(Extent):
    def __str__(self):
        return "UnknownExtent"


UNKNOWN_EXTENT = UnknownExtent()


@dataclass(frozen=True)
class BoundStrides:
    values: tuple[DimExpr, ...]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


@dataclass(frozen=True)
class UnboundStrides:
    def __str__(self):
        return "UnboundStrides"


UNBOUND_STRIDES = UnboundStrides()


class _AlignedMeta:
    """唯一公共构造：ti.Aligned[16]。"""
    def __getitem__(self, k):
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or \
                k & (k - 1):
            raise TypeError(
                "Aligned[k]: k must be a positive power-of-two int (bytes)")
        return Alignment(k)

    def __call__(self, *args, **kwargs):
        raise TypeError("Aligned uses subscript syntax: Aligned[k]")


Aligned = _AlignedMeta()


def _access_name(a: Access) -> str:
    if not isinstance(a, Access):
        raise TypeError(f"invalid internal access capability: {a!r}")
    return a.value


# ---------------------------------------------------------------------------
# 精化谓词（docs/refinements.md §2）
# ---------------------------------------------------------------------------

class Refinement:
    def check(self, value: int) -> bool:
        raise NotImplementedError

    @property
    def text(self) -> str:
        raise NotImplementedError


class PowerOfTwo(Refinement):
    def check(self, value):
        return value > 0 and (value & (value - 1)) == 0

    @property
    def text(self):
        return "PowerOfTwo"


class Positive(Refinement):
    def check(self, value):
        return value > 0

    @property
    def text(self):
        return "Positive"


class NonNegative(Refinement):
    def check(self, value):
        return value >= 0

    @property
    def text(self):
        return "NonNegative"


@dataclass(frozen=True)
class RangeRefinement(Refinement):
    lo: int
    hi: int

    def check(self, value):
        return self.lo <= value <= self.hi

    @property
    def text(self):
        return f"Range[{self.lo}, {self.hi}]"


@dataclass(frozen=True)
class MultipleOfRefinement(Refinement):
    k: int

    def check(self, value):
        return value % self.k == 0

    @property
    def text(self):
        return f"MultipleOf[{self.k}]"


class _RangeMeta:
    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Range requires two bounds: Range[lo, hi]")
        lo, hi = item
        if any(isinstance(v, bool) or not isinstance(v, int)
               for v in (lo, hi)):
            raise TypeError("Range[lo, hi]: bounds must be int literals")
        if lo > hi:
            raise TypeError("Range[lo, hi]: lo must be <= hi")
        return RangeRefinement(lo, hi)

    def __call__(self, *args, **kwargs):
        raise TypeError("Range uses subscript syntax: Range[lo, hi]")


class _MultipleOfMeta:
    def __getitem__(self, k):
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise TypeError("MultipleOf[k]: k must be a positive int")
        return MultipleOfRefinement(k)

    def __call__(self, *args, **kwargs):
        raise TypeError("MultipleOf uses subscript syntax: MultipleOf[k]")


Range = _RangeMeta()
MultipleOf = _MultipleOfMeta()


def _normalize_refinements(items, *, integer_only: bool = True) -> tuple:
    """校验、稳定去重，并拒绝可静态判定的矛盾合取。"""
    out = []
    for item in items:
        if isinstance(item, type) and issubclass(item, Refinement):
            item = item()
        if not isinstance(item, Refinement):
            raise TypeError(f"expected a refinement, got {item!r}")
        if not integer_only and isinstance(
                item, (MultipleOfRefinement, PowerOfTwo)):
            raise TypeError(f"{item.text} only applies to integer values")
        if not any(type(existing) is type(item) and
                   existing.text == item.text for existing in out):
            out.append(item)

    lo, hi = None, None
    multiples = []
    power_of_two = False
    for item in out:
        if isinstance(item, RangeRefinement):
            lo = item.lo if lo is None else max(lo, item.lo)
            hi = item.hi if hi is None else min(hi, item.hi)
        elif isinstance(item, Positive):
            lo = 1 if lo is None else max(lo, 1)
        elif isinstance(item, NonNegative):
            lo = 0 if lo is None else max(lo, 0)
        elif isinstance(item, MultipleOfRefinement):
            multiples.append(item.k)
        elif isinstance(item, PowerOfTwo):
            power_of_two = True
            lo = 1 if lo is None else max(lo, 1)
    if lo is not None and hi is not None and lo > hi:
        raise TypeError("refinement conjunction has an empty range")
    multiple = lcm(*multiples) if multiples else 1
    if power_of_two and multiple & (multiple - 1):
        raise TypeError("PowerOfTwo cannot satisfy this MultipleOf constraint")
    if hi is not None:
        lower = lo if lo is not None else -(abs(hi) + multiple)
        if power_of_two:
            candidate = 1
            while candidate < max(1, lower, multiple):
                candidate <<= 1
            if candidate > hi:
                raise TypeError("refinement conjunction has no valid integer")
        else:
            candidate = -((-lower) // multiple) * multiple
            if candidate > hi:
                raise TypeError("refinement conjunction has no valid integer")
    return tuple(out)


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

@dataclass
class ScalarT:
    dtype: D.DType

    def describe(self):
        return self.dtype.name


@dataclass
class RefinedScalar:
    """标量参数 + 值精化（launch 契约）：N: ti.i32 | ti.Positive。"""
    dtype: D.DType
    refinements: tuple

    def describe(self):
        return self.dtype.name + "".join(f" | {r.text}" for r in self.refinements)


@dataclass
class ConstT:
    """编译期参数（type-system.md §5）。装饰期只有名字+精化+默认值，
    特化期代入数值。"""
    name: str
    refinements: tuple = ()
    default: int | bool | None = None
    dtype: D.DType = field(default_factory=lambda: D.i32)

    @property
    def value_kind(self):
        return "Bool" if self.dtype is D.bool_ else "Int"

    def describe(self):
        base = f"Const[{self.value_kind.lower()}{'' if not self.refinements else ', '}"
        base += ", ".join(r.text for r in self.refinements) + "]"
        return base


@dataclass
class PtrT:
    element: D.DType
    address_space: AddressSpace = GLOBAL
    access: Access = READ_WRITE
    extent: Extent = field(default_factory=lambda: UNKNOWN_EXTENT)
    alignment: Alignment | UnknownAlignment = field(
        default_factory=lambda: UNKNOWN_ALIGNMENT)
    # 编译器赋值的内存来源标识；不是公共 Ptr 参数（ADR-002）。
    region_id: RegionId = field(default_factory=lambda: UNKNOWN_REGION)

    def __post_init__(self):
        if not isinstance(self.address_space, AddressSpace):
            raise TypeError("PtrT.address_space must be AddressSpace")
        if not isinstance(self.access, Access):
            raise TypeError("PtrT.access must be Access")
        if not isinstance(self.extent, Extent):
            raise TypeError("PtrT.extent must be Extent")
        if not isinstance(self.alignment, (Alignment, UnknownAlignment)):
            raise TypeError(
                "PtrT.alignment must be Alignment or UnknownAlignment")

    # 旧消费方的只读兼容别名；规范字段名与 ADR-001 保持一致。
    @property
    def elem(self):
        return self.element

    @property
    def space(self):
        return self.address_space

    @property
    def aligned(self):
        return alignment_bytes(self.alignment)

    def describe(self):
        extent = str(self.extent)
        alignment = str(self.alignment)
        return "Ptr[" + ", ".join((
            self.element.name, str(self.address_space), _access_name(self.access),
            extent, alignment,
        )) + "]"


@dataclass
class BlockT:
    elem: object            # ScalarT 或 PtrT
    dims: tuple             # tuple[DimExpr, ...]

    def describe(self):
        ed = self.elem.describe()
        shape = "(" + ", ".join(str(d) for d in self.dims) + ")"
        return f"Block[{ed}, {shape}]"


@dataclass
class MaskT:
    dims: tuple

    def describe(self):
        return "Mask[(" + ", ".join(str(d) for d in self.dims) + ")]"


@dataclass
class BufferT:
    element: D.DType
    shape: tuple            # tuple[DimExpr, ...]
    access: Access = READ_WRITE
    alignment: Alignment | UnknownAlignment = field(
        default_factory=lambda: UNKNOWN_ALIGNMENT)
    address_space: AddressSpace = GLOBAL
    strides: BoundStrides | UnboundStrides = field(
        default_factory=lambda: UNBOUND_STRIDES)

    def __post_init__(self):
        if not isinstance(self.address_space, AddressSpace):
            raise TypeError("BufferT.address_space must be AddressSpace")
        if not isinstance(self.access, Access):
            raise TypeError("BufferT.access must be Access")
        if not isinstance(self.alignment, (Alignment, UnknownAlignment)):
            raise TypeError(
                "BufferT.alignment must be Alignment or UnknownAlignment")
        if not isinstance(self.strides, (BoundStrides, UnboundStrides)):
            raise TypeError("BufferT.strides must be BoundStrides or UnboundStrides")

    # 旧消费方的只读兼容别名；规范字段名与 ADR-003 保持一致。
    @property
    def elem(self):
        return self.element

    @property
    def dims(self):
        return self.shape

    @property
    def aligned(self):
        return alignment_bytes(self.alignment)

    @property
    def space(self):
        return self.address_space

    def describe(self):
        inner = ", ".join(str(d) for d in self.shape)
        if len(self.shape) == 1:
            inner += ","
        shape = f"({inner})"
        alignment = str(self.alignment)
        return (f"Buffer[{self.element.name}, {shape}, "
                f"{_access_name(self.access)}, {alignment}]")


@dataclass
class UnitT:
    def describe(self):
        return "Unit"


UNIT = UnitT()


def describe(t) -> str:
    if isinstance(t, (ScalarT, RefinedScalar, ConstT, PtrT, BlockT, MaskT,
                      BufferT, UnitT)):
        return t.describe()
    if isinstance(t, D.DType):
        return t.name
    return repr(t)


# ---------------------------------------------------------------------------
# 用户注解构造器（在注解位置求值：ti.Buffer[ti.f32, (N,), ti.ReadOnly]）
# ---------------------------------------------------------------------------

def _coerce_shape(shape) -> tuple:
    if not isinstance(shape, tuple):
        shape = (shape,)
    out = []
    for d in shape:
        if isinstance(d, bool):
            raise TypeError("bool is not a dimension")
        if isinstance(d, int):
            from .dims import Cst
            out.append(Cst(d))
        elif isinstance(d, DimExpr):
            out.append(d)
        else:
            raise TypeError(f"bad dimension: {d!r} (expected ti.Dim or int)")
    return tuple(out)


class _BufferMeta:
    """ADR-003: Buffer[element, shape, access?, alignment?]。"""

    def __getitem__(self, item):
        if not isinstance(item, tuple) or not (2 <= len(item) <= 4):
            raise TypeError(
                "Buffer[T, Shape], Buffer[T, Shape, Access], or "
                "Buffer[T, Shape, Access, Alignment]")
        elem, shape = item[0], item[1]
        if not isinstance(elem, D.DType):
            raise TypeError(f"Buffer element type must be a dtype, got {elem!r}")
        access = item[2] if len(item) >= 3 else READ_WRITE
        aligned = item[3] if len(item) >= 4 else None
        if access not in (READ_ONLY, WRITE_ONLY, READ_WRITE):
            raise TypeError(f"bad access capability: {access!r}")
        return BufferT(element=elem, shape=_coerce_shape(shape),
                       access=access,
                       alignment=_coerce_alignment(aligned),
                       address_space=GLOBAL, strides=UNBOUND_STRIDES)


def _coerce_extent(extent) -> Extent:
    """公共 Extent → 结构化 LinearExtent/UnknownExtent。"""
    if extent is None:
        return UNKNOWN_EXTENT
    if isinstance(extent, bool):
        raise TypeError("Ptr extent must be a non-negative int or ti.Dim expression")
    if isinstance(extent, int):
        if extent < 0:
            raise TypeError("Ptr extent must be non-negative (elements)")
        return LinearExtent(Cst(extent))
    if isinstance(extent, DimExpr):
        value = int_value(extent)
        if value is not None and value < 0:
            raise TypeError("Ptr extent must be non-negative (elements)")
        return LinearExtent(extent)
    raise TypeError("Ptr extent must be a non-negative int or ti.Dim expression")


def _coerce_alignment(aligned) -> Alignment | UnknownAlignment:
    if aligned is None:
        return UNKNOWN_ALIGNMENT
    if isinstance(aligned, Alignment):
        return aligned
    if isinstance(aligned, bool) or not isinstance(aligned, int) or \
            aligned <= 0 or aligned & (aligned - 1):
        raise TypeError("Ptr alignment must be a positive power of two (bytes)")
    return Alignment(aligned)


def extent_expr(extent: Extent) -> DimExpr | None:
    if isinstance(extent, LinearExtent):
        return extent.expr
    if isinstance(extent, UnknownExtent):
        return None
    raise TypeError(f"invalid Extent: {extent!r}")


def alignment_bytes(alignment: Alignment | UnknownAlignment) -> int | None:
    if isinstance(alignment, Alignment):
        return alignment.bytes
    if isinstance(alignment, UnknownAlignment):
        return None
    raise TypeError(f"invalid Alignment: {alignment!r}")


def _ptr_type(elem, space, access, extent, aligned) -> PtrT:
    if not isinstance(elem, D.DType):
        raise TypeError(f"Ptr element type must be a dtype, got {elem!r}")
    if space not in (GLOBAL, SHARED, LOCAL):
        raise TypeError(f"bad address space: {space!r}")
    if space != GLOBAL:
        raise TypeError(f"Ptr address space {space} is not supported in v0")
    if access not in (READ_ONLY, WRITE_ONLY, READ_WRITE):
        raise TypeError(f"bad access capability: {access!r}")
    return PtrT(element=elem, address_space=space, access=access,
                extent=_coerce_extent(extent),
                alignment=_coerce_alignment(aligned))


class _PtrMeta:
    """ADR-001 Ptr syntax: compact 1–4 items or canonical five items."""

    def __getitem__(self, item):
        args = item if isinstance(item, tuple) else (item,)
        if not 1 <= len(args) <= 5:
            raise TypeError(
                "Ptr[T], Ptr[T, Access, Extent?, Alignment?], or "
                "Ptr[T, AddressSpace, Access, Extent, Alignment]")
        elem = args[0]
        if len(args) == 5:
            _, space, access, extent, aligned = args
        else:
            space = GLOBAL
            access = args[1] if len(args) >= 2 else READ_WRITE
            extent = args[2] if len(args) >= 3 else None
            aligned = args[3] if len(args) >= 4 else None
        return _ptr_type(elem, space, access, extent, aligned)


class _ShortPtrMeta:
    def __init__(self, access):
        self.access = access

    def __getitem__(self, item):
        args = item if isinstance(item, tuple) else (item,)
        if not 1 <= len(args) <= 3:
            raise TypeError(
                "ReadPtr/WritePtr/RWPtr[T, Extent?, Alignment?]")
        elem = args[0]
        extent = args[1] if len(args) >= 2 else None
        aligned = args[2] if len(args) >= 3 else None
        return _ptr_type(elem, GLOBAL, self.access, extent, aligned)


class _ConstMeta:
    """ti.Const[int] / ti.Const[int, ti.PowerOfTwo]（装饰期由 frontend 定名）。"""

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if item and item[0] is bool:
            if len(item) != 1:
                raise TypeError("Const[bool] does not accept refinements")
            return ("const_bool", ())
        if not item or item[0] is not int or any(x is int for x in item[1:]):
            raise TypeError("Const v0 syntax is Const[int, Refinement, ...]")
        return ("const", _normalize_refinements(item[1:], integer_only=True))


Buffer = _BufferMeta()
Ptr = _PtrMeta()
ReadPtr = _ShortPtrMeta(READ_ONLY)
WritePtr = _ShortPtrMeta(WRITE_ONLY)
RWPtr = _ShortPtrMeta(READ_WRITE)
Const = _ConstMeta()
