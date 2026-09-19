"""Pinned Triton 3.6 preflight boundary and replayable failure diagnostics.

Compilation/loading is separated from execution. Runtime/device execution
errors are deliberately outside this boundary.
"""
import json
import os
from pathlib import Path
import tempfile
import traceback

from .errors import Loc, TilaError


def mapped_line(exc, source, source_map):
    """Triton AST locations are relative to exc.src, not the generated module."""
    if isinstance(exc, SyntaxError):
        return source_map.get(exc.lineno, 0)
    excerpt = getattr(exc, "src", None)
    node = getattr(exc, "node", None)
    if not excerpt or not getattr(node, "lineno", None):
        return 0
    lines, fragment = source.splitlines(), excerpt.splitlines()
    matches = [i for i in range(len(lines)) if lines[i:i + len(fragment)] == fragment]
    if len(matches) != 1:
        return 0  # Helpers/ambiguous excerpts must not acquire invented locations.
    return source_map.get(matches[0] + node.lineno, 0)


def failure(exc, *, phase, source, source_map, context, resource=False):
    code = "TILA-TARGET-011" if resource else "TILA-TARGET-010"
    title = "compiled kernel exceeds target resources" if resource else "backend compilation or loading failed"
    line = mapped_line(exc, source, source_map)
    details = [f"backend phase: {phase}"]
    if resource:
        details.append(f"resource: {exc.name}; required: {exc.required}; limit: {exc.limit}")
    original_lines = context.get("tila_source", "").splitlines()
    loc = Loc(line, 0, original_lines[line - 1] if 0 < line <= len(original_lines) else "")
    err = TilaError(code, title, loc, details,
                    fixes=["检查源位置与编译参数；资源超限时减小 tile 或调整 num_warps；保留 backend_artifact 供重放"])
    err.backend_artifact = None
    try:
        root = os.environ.get("TILA_BACKEND_ARTIFACTS")
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="tila-backend-", dir=root))
        (folder / "kernel.py").write_text(source)
        payload = dict(context, schema="tila.backend-failure.v1", phase=phase,
                       code=code, source_map=source_map, tila_line=line,
                       exception_type=type(exc).__name__, original_reason=str(exc),
                       traceback="".join(traceback.format_exception(exc)))
        if resource:
            payload["resource"] = {"name": exc.name, "required": exc.required, "limit": exc.limit}
        (folder / "failure.json").write_text(json.dumps(payload, indent=2))
        err.backend_artifact = str(folder)
    except OSError as artifact_error:
        err.artifact_error = str(artifact_error)  # Do not mask the backend failure.
    return err


def preflight(kernel, args, grid, num_warps, *, source, source_map, context):
    """Compile and load without launching. Private adapter pinned to Triton 3.6."""
    from triton.runtime import driver
    from triton.runtime.errors import OutOfResources
    phase = "compile"
    try:
        compiled = kernel.warmup(*args, grid=grid, num_warps=num_warps)
        phase = "resources"
        device = driver.active.get_current_device()
        props = driver.active.utils.get_device_properties(device)
        shared = compiled.metadata.shared
        limit = props["max_shared_mem"]
        if shared > limit:
            raise OutOfResources(shared, limit, "shared memory")
        phase = "load"
        compiled._init_handles()
        # Triton skips repeated handle initialization after a previous failed
        # thread check; repeat this hard check even on a cached compiled object.
        threads = compiled.metadata.num_warps * 32
        if threads > compiled.n_max_threads:
            raise OutOfResources(threads, compiled.n_max_threads, "threads")
        return {"shared_bytes": shared, "shared_limit_bytes": limit,
                "threads": threads, "max_threads": compiled.n_max_threads,
                "registers_per_thread": compiled.n_regs, "spill_count": compiled.n_spills,
                "performance_only": ["registers_per_thread", "spill_count"]}
    except Exception as exc:
        raise failure(exc, phase=phase, source=source, source_map=source_map, context=context,
                      resource=isinstance(exc, OutOfResources)) from exc
