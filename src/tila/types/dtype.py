"""dtype 域与能力表（docs/type-system.md §1.1）。

universe 与 Triton triton.language 对齐（17 种）；运算语义刻意收紧：
能力表是 typing rule 的前提，违反 → E16。内部一律使用短名（f32 / i32 / fp8e4m3fn …）；
表面语言同时接受短名（grammar 口径）与 Triton 长名（全部示例使用的口径，如
tila.float32 / tila.float8e4m3fn），两者解析到同一内部 dtype。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# 内部短名（唯一权威；打印、TIR、错误信息一律用它）
BOOL = "bool"
INT_DTYPES: Tuple[str, ...] = ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")
FLOAT_DTYPES: Tuple[str, ...] = ("f16", "bf16", "f32", "f64")
FP8_DTYPES: Tuple[str, ...] = ("fp8e4m3", "fp8e5m2", "fp8e4m3fn", "fp8e4m3b15")

DTYPES: Tuple[str, ...] = (BOOL,) + INT_DTYPES + FLOAT_DTYPES + FP8_DTYPES

# 表面名字空间：短名 + Triton 长名 → 内部短名
_SURFACE_ALIASES: Dict[str, str] = {
    "bool": "bool",
    "float32": "f32",
    "float64": "f64",
    "float16": "f16",
    "bfloat16": "bf16",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "float8e4m3": "fp8e4m3",
    "float8e5m2": "fp8e5m2",
    "float8e4m3fn": "fp8e4m3fn",
    "float8e4m3b15": "fp8e4m3b15",
}
# 表面可写名字全集（tila.<name> 之后）：短名 + 长名
SURFACE_DTYPE_NAMES: Dict[str, str] = dict(_SURFACE_ALIASES)
for _d in DTYPES:
    SURFACE_DTYPE_NAMES.setdefault(_d, _d)

STORAGE_ONLY: FrozenSet[str] = frozenset(FP8_DTYPES)  # 只能 load/store/cast


def is_int(dt: str) -> bool:
    return dt in INT_DTYPES


def is_float(dt: str) -> bool:
    """浮点类别（字面量 R12 匹配口径）：FP8 也是浮点存储格式，配 FLOAT 字面量。"""
    return dt in FLOAT_DTYPES or dt in FP8_DTYPES


def is_bool(dt: str) -> bool:
    return dt == BOOL


def is_storage_only(dt: str) -> bool:
    return dt in STORAGE_ONLY


def can_be_tensor_element(dt: str) -> bool:
    """能力表：bool 不作 Tensor/Buffer 元素（mask 与比较结果除外）。"""
    return dt != BOOL


# 运算能力集（docs/type-system.md §1.1；op ∉ 集合 ⇒ E16，统一原则）
ARITH_OPS = ("+", "-", "*", "/")
CMP_OPS = ("<", "<=", ">", ">=", "==", "!=")
LOGIC_OPS = ("&", "|")

_ARITH_FOR: Dict[str, FrozenSet[str]] = {}
for _d in DTYPES:
    if _d in INT_DTYPES:
        _ARITH_FOR[_d] = frozenset({"+", "-", "*"})
    elif _d in FLOAT_DTYPES:
        _ARITH_FOR[_d] = frozenset({"+", "-", "*", "/"})
    else:  # bool 与 FP8：无算术
        _ARITH_FOR[_d] = frozenset()

_CMP_FOR: Dict[str, FrozenSet[str]] = {}
for _d in DTYPES:
    if _d == BOOL:
        _CMP_FOR[_d] = frozenset({"==", "!="})
    elif _d in INT_DTYPES or _d in FLOAT_DTYPES:
        _CMP_FOR[_d] = frozenset(CMP_OPS)
    else:  # FP8：无比较
        _CMP_FOR[_d] = frozenset()


def arith_ops(dt: str) -> FrozenSet[str]:
    return _ARITH_FOR[dt]


def cmp_ops(dt: str) -> FrozenSet[str]:
    return _CMP_FOR[dt]


def logic_ok(dt: str) -> bool:
    return dt == BOOL


# 一元数学能力表（v0.5，docs/v0.5-reduce.md §2.2）：op → 允许的 dtype 集合。
# abs 覆盖 int+float 双域，使 UNARY 成为真实矩阵而非全员 float-only；
# log2（v0.6b，评审焦点 #3）与 exp2 同款 FLOAT 域——UNARY 完全数据驱动，
# 无 per-op 逻辑。能力表是 infer/lowering/interp 唯一事实源。
UNARY_OPS: Tuple[str, ...] = ("exp", "exp2", "sqrt", "abs", "log2")
_UNARY_FOR: Dict[str, FrozenSet[str]] = {
    "exp": frozenset(FLOAT_DTYPES),
    "exp2": frozenset(FLOAT_DTYPES),
    "log2": frozenset(FLOAT_DTYPES),
    "sqrt": frozenset(FLOAT_DTYPES),
    "abs": frozenset(INT_DTYPES + FLOAT_DTYPES),
}


def unary_ok(dt: str, op: str) -> bool:
    return dt in _UNARY_FOR.get(op, frozenset())


# cast 矩阵全开：数值 dtype 之间任意互转（含 bool 与 FP8 两个端点），语义 ≡ Triton .to
def cast_allowed(src: str, dst: str) -> bool:
    return src in DTYPES and dst in DTYPES


# Triton dtype 映射（docs/triton-lowering.md §6）
TRITON_DTYPE: Dict[str, str] = {
    "bool": "tl.int1",
    "i8": "tl.int8",
    "i16": "tl.int16",
    "i32": "tl.int32",
    "i64": "tl.int64",
    "u8": "tl.uint8",
    "u16": "tl.uint16",
    "u32": "tl.uint32",
    "u64": "tl.uint64",
    "f16": "tl.float16",
    "bf16": "tl.bfloat16",
    "f32": "tl.float32",
    "f64": "tl.float64",
    "fp8e4m3": "tl.float8e4m3",
    "fp8e5m2": "tl.float8e5m2",
    "fp8e4m3fn": "tl.float8e4m3fn",
    "fp8e4m3b15": "tl.float8e4m3b15",
}

# 表面 tila.<长名> 反查（诊断建议用）：内部短名 → Triton 长名
TILA_SURFACE_NAME: Dict[str, str] = {v: k for k, v in _SURFACE_ALIASES.items()}
