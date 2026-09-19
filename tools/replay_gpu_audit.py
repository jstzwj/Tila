"""Replay an extracted CI artifact's preserved source with its locked environment."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="directory containing report.json and source/")
    parser.add_argument("--case", help="pytest -k expression")
    args = parser.parse_args()
    source = args.run.resolve() / "source"
    if not (source / "ci/gpu/uv.lock").is_file():
        parser.error("artifact must include source/ci/gpu/uv.lock")
    result = subprocess.run(["uv", "sync", "--project", "ci/gpu", "--frozen", "--python", "3.11.9"], cwd=source)
    if result.returncode:
        return result.returncode
    run = Path(tempfile.mkdtemp(prefix="replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-"), dir=args.run.resolve()))
    (run / "kernels").mkdir()
    (run / "tmp").mkdir()
    env = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONHASHSEED="0",
               TILA_INTERP="0", TRITON_INTERPRET="0", TILA_SAFETY="strict", TILA_DEBUG="0",
               PYTEST_ADDOPTS="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               TILA_GPU_KERNELS=str(run / "kernels"), TMPDIR=str(run / "tmp"),
               TILA_BACKEND_ARTIFACTS=str(run / "backend"))
    argv = [str(source / "ci/gpu/.venv/bin/python"), "-m", "pytest",
            "tests/gpu_semantics.py", "tests/gpu_examples.py", "tests/gpu_launch.py", "tests/gpu_backend.py", "tests/gpu_capabilities.py", "-v", "--tb=long", "--showlocals",
            f"--junitxml={run / 'results.xml'}"]
    if args.case:
        argv += ["-k", args.case]
    # Replay is diagnostic. Only gpu_audit.py can mark baseline acceptance.
    return subprocess.run(argv, cwd=source, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
