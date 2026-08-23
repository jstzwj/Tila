"""INTRINSICS 静态表（docs/language-spec.md §2）。

内建 = compiler intrinsic，不是 Python 函数：frontend 对 `tila.<name>` 做
intrinsic resolution，查本表；表外属性 → E13。本模块同时承载 `tila` 命名空间的
全部合法表面名字（内建、dtype、jit/constexpr/Tensor 注解名）。
"""

from __future__ import annotations

INTRINSICS = ("program_id", "arange", "load", "store", "cast", "expand_dim", "dot",
              "range", "zeros")

# 注解 / 装饰器名（非 intrinsic，出现在特定语法位置）
MARKER_NAMES = ("jit", "constexpr", "Tensor")

# kwarg 白名单（docs/language-spec.md §4）
KWARG_NAMES = ("mask", "other")
