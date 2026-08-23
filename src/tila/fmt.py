"""确定性文本格式化工具。"""

from __future__ import annotations

import struct


def f32_round(v: float) -> float:
    """先按 f32 舍入再 repr，避免 double 表示歧义（docs/triton-lowering.md §5）。"""
    return struct.unpack("f", struct.pack("f", v))[0]


def fmt_float(v: float) -> str:
    return repr(f32_round(v))
