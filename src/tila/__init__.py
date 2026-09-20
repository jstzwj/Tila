"""tila —— 静态强类型 GPU kernel DSL，编译到 Triton。

规范用法（docs/surface-language.md）：

    import tila as ti

    N = ti.Dim("N")

    @ti.jit
    def add_kernel(
        x:   ti.Buffer[ti.f32, (N,), ti.ReadOnly],
        out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
        BLOCK: ti.Const[int, ti.PowerOfTwo] = 128,
    ):
        pid = ti.program_id(0)
        offs = pid * BLOCK + ti.arange(0, BLOCK)
        mask = offs < N
        a = ti.load(x, offs, mask=mask)
        ti.store(out, offs, a, mask=mask)

    add_kernel[(ti.cdiv(n, 128),)](x, out, BLOCK=128)
"""

from .runtime import JITFunction, assume_launch, cdiv, _jit as jit
from .host import host_int
from .atomic import MemoryOrder, MemoryScope, Relaxed, GPU

# 维与类型构造
from .dims import Dim
from . import types as _ty
from .intrinsics import PUBLIC_INTRINSIC_NAMES

# dtypes
from .dtypes import (bool_ as bool, i8, i16, i32, i64, u8, u16, u32, u64,
                     f8e4m3fn, f8e5m2, f16, bf16, f32, f64)

# 注解命名空间
from .types import (Buffer, Ptr, ReadPtr, WritePtr, RWPtr, Const,
                    ReadOnly, WriteOnly, ReadWrite, Global, Shared_,
                    Local_, PowerOfTwo, Positive, NonNegative, Range,
                    MultipleOf, Aligned)

Shared = Shared_
Local = Local_

__version__ = "0.3.0.dev0"

__all__ = [
    "MemoryOrder", "MemoryScope", "Relaxed", "GPU",
    "jit", "assume_launch", "cdiv", "host_int", "Dim",
    "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
    "f8e4m3fn", "f8e5m2", "f16", "bf16", "f32", "f64",
    "Buffer", "Ptr", "ReadPtr", "WritePtr", "RWPtr", "Const",
    "ReadOnly", "WriteOnly", "ReadWrite", "Global", "Shared", "Local",
    "PowerOfTwo", "Positive", "NonNegative", "Range", "MultipleOf", "Aligned",
]

# ---------------------------------------------------------------------------
# 内建占位符：kernel 体内 tila.* 名字由 frontend 静态解析（这里的这些
# 对象在 Python 层不可调用——调用它们一定是绕过了 @ti.jit）。
# ---------------------------------------------------------------------------


class _Intrinsic:
    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        raise TypeError(
            f"ti.{self.name} is a kernel intrinsic: it can only be used "
            "inside a @ti.jit function (parsed statically, not executed)")

    def __getitem__(self, dt):
        raise TypeError(f"ti.{self.name}[dt](x) only works inside @ti.jit")


def _make_intrinsics():
    return {name: _Intrinsic(name) for name in PUBLIC_INTRINSIC_NAMES}


_intr = _make_intrinsics()
globals().update(_intr)
__all__ += list(PUBLIC_INTRINSIC_NAMES)
