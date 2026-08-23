"""@tila.jit 运行时：像 @triton.jit 一样在 Python 里定义并直接调用 Tila kernel。

用法（写在真实 .py 文件中；符号维出现在注解里时需要
`from __future__ import annotations`——注解按 PEP 563 延迟求值）：

    from __future__ import annotations
    import tila

    @tila.jit
    def add(a: tila.Tensor[tila.float32, N],
            b: tila.Tensor[tila.float32, N],
            c: tila.Tensor[tila.float32, N],
            BLOCK: tila.constexpr = 128):
        pid = tila.program_id(0)
        offs = pid * BLOCK + tila.arange(0, BLOCK)
        mask = offs < N
        x = tila.load(a + offs, mask=mask)
        y = tila.load(b + offs, mask=mask)
        z = x + y
        tila.store(c + offs, z, mask=mask)

    add(a, b, c, BLOCK=64)   # torch.Tensor(cuda) → GPU；np.ndarray → CPU 解释器

机制：inspect.getsource 抓取函数源码 → ast.parse → **同一套编译管线**
（子集校验 / checker E 码 / launch analysis / total lowering），按
(源码, constexpr overrides) 缓存编译产物；每个 override 组合一次完整编译
（TIR 是特化产物、跨值不复用，semantic-model.md §6）。grid 与 N 等符号维
由 launch plan 自动推导，无需调用方书写。
"""

from __future__ import annotations

import ast as pyast
import importlib.util
import inspect
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class _Compiled:
    """一个 (源码, overrides) 组合的编译产物；launcher 惰性加载并复用。"""

    kernel: Any                      # tir.TKernel
    triton_source: str
    tir_dump: str
    _launcher: Any = field(default=None, repr=False)

    def gpu_launcher(self):
        """@triton.jit 需要真实源文件：落盘临时 .py 再 import（一次性，之后缓存）。"""
        if self._launcher is None:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "tila_jit_gen.py"
                path.write_text(self.triton_source, encoding="utf-8", newline="\n")
                spec = importlib.util.spec_from_file_location(path.stem, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            # JITFunction 已缓存源码，临时目录删除后 launcher 仍可调用
            self._launcher = getattr(mod, f"{self.kernel.name}_launch")
        return self._launcher


class TilaKernel:
    """@tila.jit 装饰后的 kernel：直接调用即编译 + 启动。"""

    def __init__(self, fn):
        try:
            raw_lines, start_line = inspect.getsourcelines(fn)
        except (OSError, TypeError) as e:
            raise RuntimeError(
                "@tila.jit kernels must be defined in a real Python source file "
                "(the runtime compiles them from source via inspect.getsource)"
            ) from e
        lines = [l.rstrip("\n") for l in raw_lines]
        indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
        # 诊断位置校准：AST 相对（dedent 后）→ 文件绝对行列
        self._loc_line_offset = start_line - 1
        self._loc_col_offset = indent
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "<kernel>")
        self.__doc__ = getattr(fn, "__doc__", None)
        self._source = textwrap.dedent("\n".join(lines) + "\n")
        self._cache: Dict[Tuple[Tuple[str, int], ...], _Compiled] = {}
        self._meta: Optional[List[Tuple[str, str]]] = None  # (参数名, kind)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self._cache)
        return f"<tila kernel {self.__name__} ({n} specialization(s))>"

    # ------------------------------------------------------------------
    # 编译（与 CLI 完全同一套管线）
    # ------------------------------------------------------------------

    def _kernel_def(self):
        from . import frontend

        tree = pyast.parse(self._source)  # 含 @tila.jit 装饰器行
        for node in pyast.walk(tree):
            if hasattr(node, "lineno"):
                node.lineno += self._loc_line_offset
                node.col_offset += self._loc_col_offset
        module = pyast.Module(
            body=[
                pyast.Import(names=[pyast.alias(name="tila", asname=None)]),
                tree.body[0],
            ],
            type_ignores=[],
        )
        return frontend.convert(module)  # 复用全部子集校验与 intrinsic resolution

    def _meta_params(self) -> List[Tuple[str, str]]:
        if self._meta is None:
            from .ast import nodes as t

            kd = self._kernel_def()
            kinds = []
            for p in kd.params:
                if p.ann is None:
                    kinds.append((p.name, "scalar"))
                elif isinstance(p.ann, t.ConstexprAnn):
                    kinds.append((p.name, "constexpr"))
                else:
                    kinds.append((p.name, "buffer"))
            self._meta = kinds
        return self._meta

    def _entry_for(self, overrides: Dict[str, int]) -> _Compiled:
        key = tuple(sorted(overrides.items()))
        if key not in self._cache:
            from . import backend, tir
            from .checker import check_kernel_verbose

            kernel, _explain = check_kernel_verbose(self._kernel_def(), overrides)
            self._cache[key] = _Compiled(
                kernel=kernel,
                triton_source=backend.triton.emit(kernel),
                tir_dump=tir.dump(kernel),
            )
        return self._cache[key]

    # ------------------------------------------------------------------
    # 调用：torch.Tensor(cuda) → GPU；np.ndarray → reference interpreter
    # ------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        meta = self._meta_params()

        overrides: Dict[str, int] = {}
        for k, v in kwargs.items():
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"constexpr argument '{k}' must be an int, got {type(v).__name__}")
            overrides[k] = v

        positional = [(name, kind) for name, kind in meta if kind != "constexpr"]
        if len(args) != len(positional):
            want = ", ".join(name for name, _ in positional)
            raise TypeError(
                f"{self.__name__}() takes {len(positional)} positional argument(s) "
                f"({want}) plus constexpr keyword arguments, got {len(args)}")
        tensors = {name: val for (name, kind), val in zip(positional, args)
                   if kind == "buffer"}
        scalars = {name: val for (name, kind), val in zip(positional, args)
                   if kind == "scalar"}

        entry = self._entry_for(overrides)  # 编译期校验 overrides（E13/E15…）

        kinds_seen = {type(v).__module__.split(".")[0] for v in tensors.values()}
        if kinds_seen <= {"numpy"}:
            from .interp import run_kernel

            run_kernel(entry.kernel, tensors, scalars, overrides or None)
            return None
        if kinds_seen <= {"torch"}:
            import torch

            for name, v in tensors.items():
                if not v.is_cuda:
                    raise TypeError(
                        f"buffer '{name}' is a CPU torch tensor; move it with .cuda() "
                        f"for GPU launch, or pass numpy arrays to run the CPU "
                        f"reference interpreter")
            launcher = entry.gpu_launcher()
            call_args = [tensors[name] for name, kind in meta if kind == "buffer"] + \
                        [scalars[name] for name, kind in meta if kind == "scalar"]
            launcher(*call_args, **overrides)
            return None
        raise TypeError(
            f"mixed buffer types {sorted(kinds_seen)}; use torch CUDA tensors "
            f"(GPU) or numpy arrays (interpreter) consistently")

    # 便捷访问 ----------------------------------------------------------

    @property
    def cache_info(self) -> Dict[str, int]:
        """当前缓存的关键字：override 组合 → 编译次数（用于诊断/测试）。"""
        return {str(dict(k)): 1 for k in self._cache}


def jit(fn):
    """kernel marker + 运行时入口：装饰后直接调用即编译并启动。"""
    if not callable(fn):
        raise TypeError("use @tila.jit directly (without parentheses)")
    return TilaKernel(fn)
