"""compile_kernel 与 CLI（docs/triton-lowering.md §8）。

流水线：parse → subset validation → tila_ast → type check → TIR →
launch analysis → [backend validation] → Triton lowering。
编译是廉价纯函数；每个 constexpr override 组合一次完整编译
（TIR 是特化产物、跨值不复用，semantic-model.md §6）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import backend, checker, frontend, tir
from .diagnostics import BackendError, TilaError


@dataclass
class CompileResult:
    kernel: tir.TKernel
    triton_source: str
    tir_dump: str
    source: str
    overrides: Dict[str, int]
    backend_errors: List[BackendError] = field(default_factory=list)
    explain: list = field(default_factory=list)


def compile_kernel(source: str,
                   overrides: Optional[Dict[str, int]] = None,
                   *,
                   validate_backend: bool = False) -> CompileResult:
    """Tila 源码 → (TKernel, Triton 源码, TIR dump)；失败抛 TilaError。"""
    overrides = dict(overrides or {})
    pyast = frontend.parse(source)
    kernel_def = frontend.convert(pyast)
    kernel, explain = checker.check_kernel_verbose(kernel_def, overrides)

    result = CompileResult(
        kernel=kernel,
        triton_source=backend.triton.emit(kernel),
        tir_dump=tir.dump(kernel),
        source=source,
        overrides=overrides,
        explain=explain,
    )
    if validate_backend:
        result.backend_errors = backend.triton.validator.validate(kernel)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_overrides(pairs: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--constexpr expects NAME=VALUE, got '{item}'")
        name, _, value = item.partition("=")
        try:
            out[name.strip()] = int(value)
        except ValueError:
            raise SystemExit(f"--constexpr value must be an integer, got '{value}'")
    return out


def _render_explain(res: CompileResult) -> str:
    lines = []
    src_lines = res.source.splitlines()
    for entry in res.explain:
        loc = entry["loc"]
        text = src_lines[loc.line - 1].strip() if loc.line - 1 < len(src_lines) else "?"
        text = text.split("#")[0].rstrip()  # 源码行尾注释不进判定日志
        lines.append(f"line {loc.line}  {text}")
        lines.append(f"    dtype : {entry['dtype']}")
        lines.append(f"    shape : {entry['shape']}")
        lines.append(f"    layout: {entry['layout']}    ok")
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    # LF + 单尾换行（Windows 下须显式 newline，否则 CRLF 击穿逐字节比对）
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tila", description="Tila compiler: a strongly-typed frontend to Triton")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="compile a .tila kernel to Triton source")
    p_build.add_argument("source", type=Path)
    p_build.add_argument("-o", "--out", type=Path, default=Path("build"))
    p_build.add_argument("--constexpr", action="append", default=[], metavar="NAME=VALUE")
    p_build.add_argument("--explain", action="store_true")
    p_build.add_argument("--verify", action="store_true", help="run backend validation")

    p_run = sub.add_parser("run", help="compile and execute (GPU if triton present, "
                                       "else reference interpreter)")
    p_run.add_argument("source", type=Path)
    p_run.add_argument("--constexpr", action="append", default=[], metavar="NAME=VALUE")

    p_dump = sub.add_parser("dump", help="print the canonical TIR dump")
    p_dump.add_argument("source", type=Path)
    p_dump.add_argument("--constexpr", action="append", default=[], metavar="NAME=VALUE")

    args = parser.parse_args(argv)
    overrides = _parse_overrides(args.constexpr)
    source = args.source.read_text(encoding="utf-8")

    try:
        res = compile_kernel(source, overrides, validate_backend=getattr(args, "verify", False))
    except TilaError as e:
        print(e.render(), file=sys.stderr)
        return 1

    if args.command == "dump":
        sys.stdout.write(res.tir_dump)
        return 0

    if args.command == "build":
        if args.verify and res.backend_errors:
            for be in res.backend_errors:
                print(be.render(), file=sys.stderr)
            return 2
        out_dir: Path = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / f"{res.kernel.name}.triton.py", res.triton_source)
        _write(out_dir / f"{res.kernel.name}.tir.txt", res.tir_dump)
        print(f"wrote {out_dir / (res.kernel.name + '.triton.py')}")
        print(f"wrote {out_dir / (res.kernel.name + '.tir.txt')}")
        if args.explain and res.explain:
            print()
            print(_render_explain(res))
        return 0

    # run
    print(res.triton_source)
    try:
        import triton  # noqa: F401
        have_triton = True
    except ImportError:
        have_triton = False
    if not have_triton:
        print("# triton not installed — reference interpreter smoke only")
        _interp_smoke(res)
    else:
        launcher = getattr(_load_generated_module(res), f"{res.kernel.name}_launch")
        _gpu_smoke(res, launcher)
    return 0


def _load_generated_module(res: CompileResult):
    """@triton.jit 需要真实源文件（inspect.getsourcelines）：落盘临时 .py 再 import。"""
    import importlib.util
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"tila_{res.kernel.name}_gen.py"
        path.write_text(res.triton_source, encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod  # JITFunction 已缓存源码，临时目录删除后仍可调用


def _interp_smoke(res: CompileResult) -> None:
    import numpy as np

    from .interp import run_kernel

    rng = np.random.default_rng(0)
    n = 257
    buffers = {}
    for p in res.kernel.params:
        if p.kind == "buffer":
            buffers[p.name] = rng.standard_normal(
                _smoke_shape(p.tila_type.shape, n)).astype(np.float32)
    try:
        run_kernel(res.kernel, buffers, {}, res.overrides or None)
        print(f"# interpreter: ran grid over N={n}, {len(buffers)} buffer(s) — OK")
    except Exception as e:  # noqa: BLE001
        print(f"# interpreter smoke failed: {e}")


def _smoke_shape(shape, n):
    """按注解构造 smoke 形状：静态维取注解值，符号维取 n。"""
    return tuple(d.value if hasattr(d, "value") else n for d in shape)


_TORCH_DTYPE = {
    "f32": "float32", "f16": "float16", "bf16": "bfloat16", "f64": "float64",
    "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
    "u8": "uint8", "u16": "int16", "u32": "int32", "u64": "int64",
    "bool": "bool",
    "fp8e4m3": "float8_e4m3fn", "fp8e5m2": "float8_e5m2",
    "fp8e4m3fn": "float8_e4m3fn", "fp8e4m3b15": "float8_e4m3fn",
}


def _gpu_smoke(res: CompileResult, launcher) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            print("# torch present but CUDA unavailable — skipping GPU launch")
            return
        n = 257
        args = []
        for p in res.kernel.params:
            if p.kind == "buffer":
                shape = _smoke_shape(p.tila_type.shape, n)
                t = torch.randn(*shape, device="cuda").to(
                    getattr(torch, _TORCH_DTYPE.get(p.tila_type.dtype, "float32")))
                args.append(t)
            elif p.kind == "scalar":
                args.append(2)
            # sym 维不传：launcher 从 buffer 形状自行读取并断言契约
        launcher(*args, **(res.overrides or {}))
        torch.cuda.synchronize()
        print(f"# GPU launch OK ({len(args)} args, sym dims = {n})")
    except Exception as e:  # noqa: BLE001
        print(f"# GPU smoke failed: {e}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
