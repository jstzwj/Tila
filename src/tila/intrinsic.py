"""INTRINSICS 静态表（docs/language-spec.md §2）。

内建 = compiler intrinsic，不是 Python 函数：frontend 对 `tila.<name>` 做
intrinsic resolution，查本表；表外属性 → E13。本模块同时承载 `tila` 命名空间的
全部合法表面名字（内建、dtype、jit/constexpr/Tensor 注解名、常量）。
"""

from __future__ import annotations

# dataflow 内建（值指令）+ program-context query（只读观测，不参与 launch 推导；
# 分类见 language-spec §5 与 v0.5-reduce §1.4）
INTRINSICS = ("program_id", "arange", "load", "store", "cast", "expand_dim", "dot",
              "range", "zeros", "full", "maximum", "sum", "max", "exp", "exp2",
              "sqrt", "abs", "log2", "where", "num_programs", "launch_assert")

# 注解 / 装饰器名（非 intrinsic，出现在特定语法位置）
MARKER_NAMES = ("jit", "constexpr", "Tensor")

# 裸常量名（非调用位置使用；v0.5：neg_inf = max 的归约零元 / 掩码 max 哨兵）
CONSTANT_NAMES = ("neg_inf",)

# kwarg 白名单（docs/language-spec.md §4；各内建再自查自己的白名单）
KWARG_NAMES = ("mask", "other", "axis")
