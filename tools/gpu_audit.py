"""M2-08 fixed-environment gate. No CUDA or version mismatch is a failure.

Run: PYTHONPATH=src python tools/gpu_audit.py
Artifacts are append-only, including failures and generated kernel sources.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    "python": "3.11.9", "torch": "2.10.0+cu128", "triton": "3.6.0",
    "cuda": "12.8", "numpy": "1.24.3", "ml_dtypes": "0.5.4",
    "gpu": "NVIDIA GeForce RTX 3090", "capability": [8, 6], "driver": "595.84",
}
EXPECTED_TESTS = 70


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def environment():
    import torch
    import triton
    import numpy
    import ml_dtypes
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runner required; CPU success is not GPU evidence")
    # All physical devices are recorded separately; select the driver for the
    # active runtime via CUDA's device UUID, including CUDA_VISIBLE_DEVICES.
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    driver = command(["nvidia-smi", f"--id={uuid}", "--query-gpu=driver_version", "--format=csv,noheader"])
    return dict(python=platform.python_version(), torch=torch.__version__,
                triton=triton.__version__, cuda=torch.version.cuda,
                numpy=numpy.__version__, ml_dtypes=ml_dtypes.__version__,
                gpu=props.name, capability=list(torch.cuda.get_device_capability(0)),
                driver=driver)


def mismatches(actual):
    return {key: {"expected": value, "actual": actual.get(key)}
            for key, value in BASELINE.items() if actual.get(key) != value}


def test_summary(path):
    cases = ET.parse(path).findall(".//testcase")
    return {"tests": len(cases), **{tag: sum(case.find(tag) is not None for case in cases)
                                  for tag in ("failure", "error", "skipped")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/m2-gpu-audit")
    parser.add_argument("--exploratory", action="store_true", help="allow other versions; never a baseline acceptance")
    parser.add_argument("--case", help="pytest -k selection; partial runs are never full acceptance")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=args.output.resolve()))
    report = {"schema": "tila.gpu-audit.v1", "seed": 208009,
              "baseline": BASELINE, "status": "failed", "artifact_dir": str(run),
              "exploratory": args.exploratory, "selection": args.case,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "platform": platform.platform()}
    try:
        report["revision"] = command(["git", "rev-parse", "HEAD"])
        report["worktree"] = command(["git", "status", "--porcelain"])
        (run / "tracked.patch").write_text(command(["git", "diff", "HEAD", "--", "."]))
        # Preserve tracked + untracked Python sources, configs and docs, so a
        # dirty working tree can be replayed without guessing its old contents.
        import shutil
        paths = command(["git", "ls-files", "--cached", "--others", "--exclude-standard"]).splitlines()
        for name in paths:
            source = ROOT / name
            if source.is_file() and not source.resolve().is_relative_to(args.output.resolve()):
                target = run / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        report["environment"] = environment()
        report["mismatches"] = mismatches(report["environment"])
        if report["mismatches"] and not args.exploratory:
            raise RuntimeError(f"ADR-009 environment mismatch: {report['mismatches']}")
        (run / "kernels").mkdir()
        (run / "tmp").mkdir()
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "src"), PYTHONHASHSEED="0",
                   PYTEST_ADDOPTS="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   TILA_INTERP="0", TRITON_INTERPRET="0", TILA_SAFETY="strict", TILA_DEBUG="0",
                   TILA_GPU_KERNELS=str(run / "kernels"), TMPDIR=str(run / "tmp"))
        replay_env = {key: env[key] for key in (
            "PYTHONPATH", "PYTHONHASHSEED", "PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "TILA_INTERP", "TRITON_INTERPRET", "TILA_SAFETY", "TILA_DEBUG", "TILA_GPU_KERNELS", "TMPDIR")}
        report["execution_env"] = replay_env
        from importlib.metadata import distributions
        report["packages"] = {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]}
        argv = [sys.executable, "-m", "pytest", "tests/gpu_semantics.py", "-v", "-s",
                "--tb=long", "--showlocals", f"--junitxml={run / 'results.xml'}"]
        if args.case:
            argv += ["-k", args.case]
        report["command"] = argv
        # replay.json + preserved source + exact node IDs in XML/log permit
        # single-case replay; deterministic suites require no random minimizer.
        (run / "replay.json").write_text(json.dumps({"argv": argv, "env": replay_env}, indent=2))
        print(f"GPU audit artifacts: {run}", flush=True)
        with (run / "pytest.log").open("w") as log:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = proc.wait()
        summary = test_summary(run / "results.xml")
        report["summary"] = summary
        if summary["skipped"] or summary["failure"] or summary["error"] or (not args.case and summary["tests"] != EXPECTED_TESTS):
            code = code or 1
        report["exit_code"] = code
        report["status"] = "passed" if code == 0 else "failed"
        report["accepted"] = code == 0 and not args.exploratory and not args.case and not report["mismatches"]
        return code
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["accepted"] = False
        print(report["error"], file=sys.stderr)
        return 1
    finally:
        (run / "report.json").write_text(json.dumps(report, indent=2))
        print(f"Report: {run / 'report.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
