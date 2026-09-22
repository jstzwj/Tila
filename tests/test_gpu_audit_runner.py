"""The GPU gate must never turn missing hardware/skips into acceptance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


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


@pytest.mark.parametrize('modern', (False, True))
def test_replay_includes_correctness_suites_and_preserves_legacy_artifacts(tmp_path, monkeypatch, modern):
    spec = importlib.util.spec_from_file_location('replay_gpu_audit',
        Path(__file__).resolve().parents[1] / 'tools/replay_gpu_audit.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    source = tmp_path / 'source'
    (source / 'ci/gpu').mkdir(parents=True)
    (source / 'ci/gpu/uv.lock').write_text('fixture')
    (source / 'tests').mkdir()
    if modern:
        for name in ('gpu_soundness.py', 'gpu_c2.py'):
            (source / 'tests' / name).write_text('')
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(replay.subprocess, 'run', run)
    monkeypatch.setattr('sys.argv', ['replay_gpu_audit.py', str(tmp_path), '--case', 'c2'])
    assert replay.main() == 0
    command, options = calls[-1]
    for name in ('tests/gpu_soundness.py', 'tests/gpu_c2.py'):
        assert (name in command) is modern
    assert command[-2:] == ['-k', 'c2']
    assert options['cwd'] == source
    assert Path(options['env']['TILA_C2_FAILURE_DIR']).parent.is_relative_to(tmp_path)
