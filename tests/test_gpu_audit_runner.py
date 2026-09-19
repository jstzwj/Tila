"""The GPU gate must never turn missing hardware/skips into acceptance."""
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("gpu_audit", Path(__file__).resolve().parents[1] / "tools/gpu_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_environment_mismatch_is_explicit():
    assert audit.mismatches(audit.BASELINE) == {}
    changed = {**audit.BASELINE, "triton": "other"}
    assert audit.mismatches(changed) == {"triton": {"expected": "3.6.0", "actual": "other"}}


def test_no_gpu_saves_failure_report(tmp_path, monkeypatch):
    def unavailable():
        raise RuntimeError("CUDA runner required")
    monkeypatch.setattr(audit, "environment", unavailable)
    monkeypatch.setattr(audit, "command", lambda args: "")
    monkeypatch.setattr("sys.argv", ["gpu_audit.py", "--output", str(tmp_path)])
    assert audit.main() == 1
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert report["status"] == "failed" and report["accepted"] is False
    assert "CUDA runner required" in report["error"]


def test_skipped_results_are_visible(tmp_path):
    path = tmp_path / "results.xml"
    path.write_text('<testsuites><testsuite><testcase/><testcase><skipped/></testcase><testcase><failure/></testcase></testsuite></testsuites>')
    assert audit.test_summary(path) == {"tests": 3, "skipped": 1, "failure": 1, "error": 0}
