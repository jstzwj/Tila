"""后端校验层（docs/triton-lowering.md §8）：B01–Bxx 与 lowering 分离。

compiler correctness（Tila 自己的检查）与 backend compatibility（Triton 接受
产物）由此分开；B 码失败发生在 lowering 之前或验收期，lowering total 不被破坏。
"""

from __future__ import annotations

from typing import List

from ...diagnostics import BackendError
from ... import tir
from ...types import numel, TileType, AddressType

TRITON_MAX_TENSOR_NUMEL = 2 ** 20

# fp8 目标架构下限（Triton 编译期报告；Tila 透传，B01 记录提示）
_FP8_ARCH_HINT = (
    "fp8 dtype availability depends on the target architecture "
    "(e.g. fp8e4m3fn needs SM89+); the Triton compiler reports the exact requirement"
)


def validate(kernel: tir.TKernel) -> List[BackendError]:
    """静态后端校验（不依赖 triton 安装）。"""
    errors: List[BackendError] = []
    seen = set()
    for op in kernel.ops:
        ty = op.tila_type
        if not isinstance(ty, (TileType, AddressType)):
            continue
        if ty in seen:
            continue
        seen.add(ty)
        n = numel(ty.shape)
        if n is not None and n > TRITON_MAX_TENSOR_NUMEL:
            errors.append(BackendError(
                "B02",
                f"tile has {n} elements, exceeding TRITON_MAX_TENSOR_NUMEL "
                f"({TRITON_MAX_TENSOR_NUMEL})",
            ))
    for p in kernel.params:
        if p.kind == "buffer" and p.tila_type.dtype.startswith("fp8"):
            errors.append(BackendError(
                "B01",
                f"buffer '{p.name}' uses {p.tila_type.dtype}; {_FP8_ARCH_HINT}",
            ))
    return errors


def validate_with_triton(source: str) -> List[BackendError]:
    """以 exec + 触发一次 Triton 编译来校验产物（无 GPU 环境则到导入为止）。"""
    try:
        import triton  # noqa: F401
    except ImportError:
        return [BackendError("B00", "triton is not installed; backend validation "
                                    "limited to static checks")]
    errors: List[BackendError] = []
    try:
        namespace: dict = {}
        exec(compile(source, "<tila-lowered>", "exec"), namespace)
    except Exception as e:  # noqa: BLE001 - 任何 exec 失败都是后端兼容性问题
        errors.append(BackendError("B03", f"generated module failed to exec: {e}"))
    return errors
