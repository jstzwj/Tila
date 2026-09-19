"""Real compiler/resource negatives; forbidden launch is observed explicitly."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import triton
import tila as ti
from tila.errors import TilaError
from tila.lowering import Lowering
from test_m3_launch import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


def replay(folder):
    spec = importlib.util.spec_from_file_location("replay_backend", Path(__file__).parents[1] / "tools/replay_backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replay(folder)


def forbid_launch(monkeypatch):
    original = triton.runtime.jit.JITFunction.run
    def run(self, *args, **kwargs):
        assert kwargs.get("warmup") is True, "kernel execution was attempted"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(triton.runtime.jit.JITFunction, "run", run)


def test_real_compiler_failure_mapping_and_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("TILA_BACKEND_ARTIFACTS", str(tmp_path))
    original = Lowering.kernel_source
    def broken(self):
        # Controlled fault at an existing mapped load: exercise real Triton
        # compilation errors without broadening the public language subset.
        return original(self).replace("tl.load(", "tl.m3_missing_load(")
    monkeypatch.setattr(Lowering, "kernel_source", broken)
    forbid_launch(monkeypatch)
    kernel = ti.jit(copy.fn)
    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(TilaError) as caught:
        kernel[(1,)](x)
    err = caught.value
    assert err.code == "TILA-TARGET-010"
    assert err.loc.line == kernel.tk.body[1].line
    assert "ti.load" in err.loc.src_line
    assert err.__cause__ is not None and torch.count_nonzero(x).item() == 0
    payload = json.loads((Path(err.backend_artifact) / "failure.json").read_text())
    assert "m3_missing_load" in payload["original_reason"]
    with pytest.raises(TilaError) as replayed:
        replay(err.backend_artifact)
    assert replayed.value.code == err.code and replayed.value.loc.line == err.loc.line


def test_real_shared_memory_overflow_never_launches(tmp_path, monkeypatch):
    from matmul import matmul_kernel
    monkeypatch.setenv("TILA_BACKEND_ARTIFACTS", str(tmp_path))
    forbid_launch(monkeypatch)
    kernel = ti.jit(matmul_kernel.fn)
    a = torch.zeros((256, 256), dtype=torch.float16, device="cuda")
    out = torch.full((256, 256), -7, dtype=torch.float32, device="cuda")
    # Static TIR is admissible; compiled shared memory exceeds SM86 capacity.
    for _ in range(2):  # Failed compilation/handle cache must not bypass checks.
        with pytest.raises(TilaError) as caught:
            kernel[(1, 1)](a, a, out, BM=256, BN=256, BK=128)
        assert caught.value.code == "TILA-TARGET-011"
        assert "shared memory" in str(caught.value)
        assert kernel.last_backend_resources is None
        assert torch.all(out == -7).item()
    with pytest.raises(TilaError) as replayed:
        replay(caught.value.backend_artifact)
    assert replayed.value.code == "TILA-TARGET-011"
    assert replayed.value.details == caught.value.details


def test_resource_report_on_success_and_cache_hit():
    kernel = ti.jit(copy.fn)
    x = torch.zeros(8, dtype=torch.int32, device="cuda")
    for _ in range(2):
        kernel[(1,)](x)
        report = kernel.last_backend_resources
        assert report["shared_bytes"] <= report["shared_limit_bytes"]
        assert report["registers_per_thread"] > 0
        assert report["threads"] <= report["max_threads"]
    assert torch.all(x == 2).item()
    kernel[(0,)](x)
    assert kernel.last_backend_resources is None
