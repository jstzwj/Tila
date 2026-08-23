"""`tila` 命名空间的 Python 层 placeholder（docs/language-spec.md §2）。

这些对象永远不会被真正执行：`.tila` 源码由 Tila 编译器编译（ast.parse → 子集
校验 → checker → lowering），`import tila` 只是语法必需。把本模块导入到真实
`tila` 包命名空间后，直接用 Python 跑 .tila 文件会在调用处得到清晰的
RuntimeError，而不是静默的错误行为。
"""

from __future__ import annotations

from typing import Any


class _Placeholder:
    """不可调用的占位基类：调用即报"只能在 @tila.jit 内由编译器处理"。"""

    _name = "?"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"tila.{self._name} can only be used inside @tila.jit (compiled by the Tila compiler)"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<tila placeholder: tila.{self._name}>"


def _make(name: str) -> _Placeholder:
    return type(f"_{name}Placeholder", (_Placeholder,), {"_name": name})()


class _TensorMeta:
    """`tila.Tensor[dt, dims]` 注解 placeholder：接受任意下标、不求值。"""

    def __getitem__(self, item):
        return ("Tensor", item)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<tila annotation: Tensor>"


Tensor = _TensorMeta()

# jit 由 runtime 提供（@tila.jit 是可执行的运行时入口，不再是纯 placeholder）
from .runtime import TilaKernel, jit  # noqa: E402, F401

constexpr = _make("constexpr")

program_id = _make("program_id")
arange = _make("arange")
load = _make("load")
store = _make("store")
cast = _make("cast")
expand_dim = _make("expand_dim")
dot = _make("dot")

# dtype 引用（17 种，长名与短名各一套 placeholder）
bool = _make("bool")
int8, int16, int32, int64 = _make("int8"), _make("int16"), _make("int32"), _make("int64")
uint8, uint16, uint32, uint64 = _make("uint8"), _make("uint16"), _make("uint32"), _make("uint64")
float16, bfloat16, float32, float64 = (
    _make("float16"),
    _make("bfloat16"),
    _make("float32"),
    _make("float64"),
)
float8e4m3, float8e5m2, float8e4m3fn, float8e4m3b15 = (
    _make("float8e4m3"),
    _make("float8e5m2"),
    _make("float8e4m3fn"),
    _make("float8e4m3b15"),
)

f16, bf16, f32, f64 = _make("f16"), _make("bf16"), _make("f32"), _make("f64")
i8, i16, i32, i64 = _make("i8"), _make("i16"), _make("i32"), _make("i64")
u8, u16, u32, u64 = _make("u8"), _make("u16"), _make("u32"), _make("u64")
fp8e4m3, fp8e5m2 = _make("fp8e4m3"), _make("fp8e5m2")
fp8e4m3fn, fp8e4m3b15 = _make("fp8e4m3fn"), _make("fp8e4m3b15")
