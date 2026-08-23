"""二元运算分派矩阵（docs/type-checker.md §4.1）——唯一权威来源。

同 dtype 规则：(lhs_dtype, rhs_dtype) → 结果 dtype；表外组合 → E02。
能力门（矩阵命中之后、规则前提的一部分）：查 types/dtype.py 的能力表——
op ∉ 该 dtype 类别的运算集 ⇒ E16（统一原则，不逐条枚举）。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..types import dtype as dt

# 全表 = 各 dtype 与自身的配对（v0.1 无任何混 dtype 组合；混合 → E02）
ARITH_RULES: Dict[Tuple[str, str], str] = {
    (d, d): d for d in dt.DTYPES
}


def arith_result(lhs_dtype: str, rhs_dtype: str) -> Optional[str]:
    return ARITH_RULES.get((lhs_dtype, rhs_dtype))


def capability_arith(d: str, op: str) -> bool:
    return op in dt.arith_ops(d)


def capability_cmp(d: str, op: str) -> bool:
    return op in dt.cmp_ops(d)


def capability_logic(d: str) -> bool:
    return dt.logic_ok(d)


def storage_only_hint(d: str) -> str:
    return (f"fp8 dtypes are storage-only: no arithmetic, no comparisons; "
            f"hints: tila.cast(x, tila.float16) + tila.cast(y, tila.float16) "
            f"(element dtype {d})")
