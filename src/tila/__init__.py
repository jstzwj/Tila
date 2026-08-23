"""Tila：strongly-typed、shape-safe、layout-aware 的 tile-based GPU 编程 DSL。

`tila` 名字空间既是编译器包入口，也承载 Python 层 placeholder
（api.py：直接调用内建会得到明确的 RuntimeError，永远不会被真正执行）。
"""

from . import api  # noqa: F401
from .api import (  # noqa: F401
    Tensor,
    arange,
    bfloat16,
    bool,
    cast,
    constexpr,
    float16,
    float32,
    float64,
    float8e4m3,
    float8e4m3b15,
    float8e4m3fn,
    float8e5m2,
    int8,
    int16,
    int32,
    int64,
    jit,
    load,
    program_id,
    store,
    uint8,
    uint16,
    uint32,
    uint64,
)
from .diagnostics import BackendError, Loc, Note, TilaError  # noqa: F401
from .runtime import TilaKernel, jit  # noqa: F401
from .driver import CompileResult, compile_kernel  # noqa: F401

__version__ = "0.1.0"
