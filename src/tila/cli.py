"""CLI：python -m tila check|build|run|dump|explain（surface-language.md §8）。

用法：
    python -m tila check examples/add_kernel.py [--const BLOCK=64]
    python -m tila check examples/add_kernel.py --explain   # 审计输出（§8）
    python -m tila explain examples/add_kernel.py [--const ...]
    python -m tila build examples/add_kernel.py -o build/ [--const ...]
    python -m tila run   examples/add_kernel.py        # 执行文件自身
    python -m tila dump  examples/add_kernel.py        # 打印 canonical TIR
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from .errors import TilaError, TilaLaunchContractError


def _configure_utf8_stdio():
    """让 CLI 的 Unicode 诊断在 Windows 及重定向场景下稳定输出。

    Windows 上 Python 可能从宿主进程继承 cp1252 等窄编码；Tila 的诊断
    包含中文、数学符号和状态符号，直接 print 会因此抛
    UnicodeEncodeError。TextIOWrapper.reconfigure 自 Python 3.7 起可用，
    但测试捕获器或嵌入式宿主提供的流未必支持它，因此按能力探测。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            # 已关闭、不可重配或由宿主接管的流保持原样；CLI 的调用方仍可
            # 通过 PYTHONIOENCODING/宿主配置决定编码。
            pass


def _load(path: str):
    spec = importlib.util.spec_from_file_location("_tila_user_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # @ti.jit 装饰即 Stage 1
    return mod


def _jit_functions(mod):
    from .runtime import JITFunction
    return {n: v for n, v in vars(mod).items()
            if isinstance(v, JITFunction)}


def _consts(argv, declarations=()):
    kinds = {c.name: c.value_kind for c in declarations}
    out = {}
    for c in argv or []:
        name, _, v = c.partition("=")
        name = name.strip()
        if not name or not _ or not v.strip():
            raise TilaError(
                "TILA-CONST-008",
                f"invalid --const value {c!r}",
                details=["required form: --const NAME=<base-10 integer>"],
                fixes=["for example: --const BLOCK=128"],
            )
        try:
            if kinds.get(name) == "Bool":
                if v.strip() not in ("true", "false"):
                    raise TilaError("TILA-CONST-008",
                                    f"Const '{name}' requires CLI true or false")
                out[name] = v.strip() == "true"
                continue
            out[name] = int(v.strip(), 10)
        except ValueError:
            raise TilaError(
                "TILA-CONST-008",
                f"Const '{name}' must be an exact base-10 integer",
                details=[f"found CLI text: {v.strip()!r}"],
                fixes=[f"pass --const {name}=<integer>"],
            ) from None
    return out


def main(argv=None):
    _configure_utf8_stdio()
    ap = argparse.ArgumentParser(prog="tila")
    ap.add_argument("command", choices=["check", "build", "run", "dump",
                                        "explain"])
    ap.add_argument("path")
    ap.add_argument("-o", "--out", default="build")
    ap.add_argument("--const", action="append", metavar="NAME=V")
    ap.add_argument("--explain", action="store_true",
                    help="check 通过后附加 --explain 审计输出"
                         "（类型环境/事实集/义务证明链/hint/效应，§8）")
    ap.add_argument("--show-query", action="store_true",
                    help="explain: append raw SMT-LIB replay queries; omitted from stable audit output")
    ap.add_argument("--show-witness", action="store_true", help="explain: show solver-selected witness bindings")
    ap.add_argument("--show-cache", action="store_true", help="explain: show per-call cache telemetry")
    ap.add_argument("--show-effects", action="store_true", help="explain: show versioned per-access effect details")
    ap.add_argument("--safety", choices=["strict", "warn"], default="strict",
                    help="bounds 义务严格度：strict（默认）Unknown → error；"
                         "warn → warning 后继续（refinements.md §5.3；"
                         "TILA-BOUNDS-003 任何模式都 error）")
    args = ap.parse_args(argv)

    os.environ["TILA_SAFETY"] = args.safety   # runtime 义务求值共享同一开关

    try:
        mod = _load(args.path)
        kerns = _jit_functions(mod)
        if not kerns:
            print(f"{args.path}: no @ti.jit kernels found")
            return 1
        parsed_consts = {name: _consts(args.const, k.tk.consts) for name, k in kerns.items()}
        if args.command == "run":
            fn = getattr(mod, "main", None)
            if fn is None:
                print(f"{args.path}: no main() to run（run = 执行文件自身）")
                return 1
            fn()
            return 0
        ok = True
        for name, k in kerns.items():
            consts = parsed_consts[name]
            if args.command == "dump":
                print(k.tk.dump(), end="")
                continue
            if args.command == "explain":
                # explain：不 materialize（Unknown 义务只展示、不 raise），
                # 直接给出 §8 审计输出
                print(k.explain(consts, show_query=args.show_query,
                                show_witness=args.show_witness, show_cache=args.show_cache,
                                show_effects=args.show_effects))
                continue
            src, tir = k.materialize(consts)
            if args.command == "check":
                label = consts if consts else "defaults"
                print(f"✓ {name}: Stage 1 + Stage 2 (consts={label}) 通过")
                print(k.report())
                if args.explain:
                    print(k.explain(consts, show_query=args.show_query,
                                    show_witness=args.show_witness, show_cache=args.show_cache,
                                    show_effects=args.show_effects))
                continue
            if args.command == "build":
                os.makedirs(args.out, exist_ok=True)
                base = os.path.splitext(os.path.basename(args.path))[0]
                with open(os.path.join(args.out, f"{name}.triton.py"),
                          "w", encoding="utf-8", newline="\n") as f:
                    f.write(src)
                with open(os.path.join(args.out, f"{name}.tir.txt"),
                          "w", encoding="utf-8", newline="\n") as f:
                    f.write(tir)
                print(f"✓ {name}: wrote {args.out}/{name}.triton.py, "
                      f"{name}.tir.txt")
        return 0 if ok else 1
    except TilaError as e:
        print(e.render(), file=sys.stderr)
        if args.show_witness and e.proof_result is not None and e.proof_result.candidate_counterexample:
            print("witness bindings (model-specific):", file=sys.stderr)
            for name, value in sorted(e.proof_result.candidate_counterexample):
                print(f"  {name}: {value}", file=sys.stderr)
        if args.show_cache and e.proof_result is not None:
            print("proof cache telemetry: " + e.proof_result.cache_status, file=sys.stderr)
        if args.show_query and e.proof_result is not None and e.proof_result.query:
            print("SMT-LIB replay query:", file=sys.stderr)
            print(e.proof_result.query.rstrip(), file=sys.stderr)
        return 2
    except TilaLaunchContractError as e:
        print(e, file=sys.stderr)
        return 3
    except Exception as e:
        if os.environ.get("TILA_DEBUG", "") == "1":
            raise
        err = TilaError(
            "TILA-INTERNAL-001",
            "unexpected internal failure",
            details=[f"{type(e).__name__}: {e}"],
            fixes=["set TILA_DEBUG=1 to obtain a traceback, then report the "
                   "smallest reproducer"],
        )
        print(err.render(), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
