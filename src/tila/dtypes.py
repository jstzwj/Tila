"""DType 域：dtype 宇宙、能力类、widening 格与字面量可表示性
（docs/type-system.md §2、§6）。"""

from __future__ import annotations

import numpy as np


class DType:
    __slots__ = ("name", "kind", "bits", "np_dtype", "tl_name")

    def __init__(self, name: str, kind: str, bits: int, np_dtype, tl_name: str):
        self.name = name
        self.kind = kind          # bool | int | uint | float | float_storage
        self.bits = bits
        self.np_dtype = np_dtype
        self.tl_name = tl_name

    def __repr__(self):
        return self.name

    # 标量精化注解糖：ti.i32 | ti.Positive（type-system.md §6.4）
    def __or__(self, ref):
        from .types import RefinedScalar, _normalize_refinements
        items = tuple(ref) if isinstance(ref, (tuple, list)) else (ref,)
        if not self.numeric:
            raise TypeError("refinements require a numeric scalar dtype")
        refs = _normalize_refinements(items, integer_only=self.is_int)
        return RefinedScalar(self, refs)

    @property
    def is_float(self):
        return self.kind in ("float", "float_storage")

    @property
    def is_int(self):
        return self.kind in ("int", "uint")

    @property
    def numeric(self):
        return self.kind != "bool"


bool_ = DType("bool", "bool", 1, np.bool_, "tl.int1")
i8 = DType("i8", "int", 8, np.int8, "tl.int8")
i16 = DType("i16", "int", 16, np.int16, "tl.int16")
i32 = DType("i32", "int", 32, np.int32, "tl.int32")
i64 = DType("i64", "int", 64, np.int64, "tl.int64")
u8 = DType("u8", "uint", 8, np.uint8, "tl.uint8")
u16 = DType("u16", "uint", 16, np.uint16, "tl.uint16")
u32 = DType("u32", "uint", 32, np.uint32, "tl.uint32")
u64 = DType("u64", "uint", 64, np.uint64, "tl.uint64")
f16 = DType("f16", "float", 16, np.float16, "tl.float16")
bf16 = DType("bf16", "float", 16, None, "tl.bfloat16")   # interp 需 ml_dtypes
f32 = DType("f32", "float", 32, np.float32, "tl.float32")
f64 = DType("f64", "float", 64, np.float64, "tl.float64")
f8e4m3fn = DType("f8e4m3fn", "float_storage", 8, None, "tl.float8e4nv")
f8e5m2 = DType("f8e5m2", "float_storage", 8, None, "tl.float8e5")

ALL = {d.name: d for d in (bool_, i8, i16, i32, i64, u8, u16, u32, u64,
                           f16, bf16, f32, f64, f8e4m3fn, f8e5m2)}

# 能力类（type-system.md §2）
FLOAT_DTYPES = (f16, bf16, f32, f64)
INT_DTYPES = (i8, i16, i32, i64)
UINT_DTYPES = (u8, u16, u32, u64)
ARITH_DTYPES = FLOAT_DTYPES + INT_DTYPES + UINT_DTYPES
DOT_INPUT = (f16,)                       # MVP 冻结（roadmap Phase 0）
DOT_ACC = (f32, f16)

_INT_RANGE = {
    i8: (-(1 << 7), (1 << 7) - 1), i16: (-(1 << 15), (1 << 15) - 1),
    i32: (-(1 << 31), (1 << 31) - 1), i64: (-(1 << 63), (1 << 63) - 1),
    u8: (0, (1 << 8) - 1), u16: (0, (1 << 16) - 1),
    u32: (0, (1 << 32) - 1), u64: (0, (1 << 64) - 1),
}

# 安全 widening 边（type-system.md §6.1）。隐式转换只允许沿边走。
_WIDEN_EDGES = {
    i8: (i16,), i16: (i32,), i32: (i64,),
    u8: (u16,), u16: (u32,), u32: (u64,),
    f16: (f32,), f32: (f64,), bf16: (f32,),
}


def can_widen(a: DType, b: DType) -> bool:
    """a 是否可以安全 widening 到 b（含经中间边）。"""
    if a is b:
        return True
    seen = {a}
    frontier = [a]
    while frontier:
        nxt = []
        for d in frontier:
            for t in _WIDEN_EDGES.get(d, ()):
                if t is b:
                    return True
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    return False


def common_type(a: DType, b: DType):
    """binop 推导规则（type-system.md §6.1）：相等取一；单向 widening 取宽者；
    否则 None（调用方报 TILA-TYPE-012）。"""
    if a is b:
        return a
    if can_widen(a, b):
        return b
    if can_widen(b, a):
        return a
    return None


def _bf16_trunc(value) -> float:
    """bf16 = f32 高 16 位：纯 numpy 截断往返，不依赖 ml_dtypes。"""
    f = np.float32(value)
    bits = f.view(np.uint32) & np.uint32(0xFFFF0000)
    return float(bits.view(np.float32))


def representable(value, dt: DType) -> bool:
    """字面量语境实例化的可表示性（type-system.md §6.3）。

    int：精确范围；float：bits>=32 接受任意十进制字面量（标准行为），
    f16/bf16 要求往返精确（不允许静默舍入）；bf16 用 f32 高 16 位截断
    判定（无需 ml_dtypes）。
    """
    if isinstance(value, bool):
        return dt is bool_
    if isinstance(value, int):
        if dt.kind in ("int", "uint"):
            lo, hi = _INT_RANGE[dt]
            return lo <= value <= hi
        if dt.is_float:
            # int 字面量进入 float 语境（type-system.md §6.3）：精确可表示
            if dt is f64:
                return True
            if dt is f32:
                return float(np.float32(value)) == value
            if dt is f16:
                return float(np.float16(value)) == value
            if dt is bf16:
                return _bf16_trunc(value) == float(np.float32(value))
        return False
    if isinstance(value, float):
        if dt is f64 or dt is f32:
            return True
        if dt is f16:
            return float(np.float16(value)) == value
        if dt is bf16:
            return _bf16_trunc(value) == float(np.float32(value))
        return False
    return False


def zero_value(dt: DType):
    if dt.kind == "bool":
        return False
    if dt.kind in ("int", "uint"):
        return 0
    return 0.0
