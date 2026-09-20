"""Source mapping and stable diagnostics do not need Triton or a GPU."""
import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tila.backend import failure, mapped_line
from tila.lowering import Lowering
from tila import tir as T
from test_m3_launch import copy


def test_source_map_preserves_text_and_definitions():
    emitter = Lowering(copy.tk)
    source = emitter.kernel_source()
    mapping = dict(emitter.source_map)
    assert source == emitter.kernel_source() and mapping == emitter.source_map
    for text, statement in zip(("i =", "v =", "tl.store("), copy.tk.body):
        matches = [i for i, line in enumerate(source.splitlines(), 1) if text in line]
        assert matches and all(mapping[i] == statement.line for i in matches)
    assert all(source.splitlines()[i - 1].startswith("    ") for i in mapping)


def test_fragment_relative_line_mapping():
    source = "import x\n@decorator\ndef f():\n    broken()\n"
    exc = SimpleNamespace(src="def f():\n    broken()\n", node=ast.parse("x\nx").body[1])
    assert mapped_line(exc, source, {4: 17}) == 17
    exc.src = "def helper():\n    broken()\n"
    assert mapped_line(exc, source, {4: 17}) == 0


def test_nested_source_map_keeps_inner_statement_locations():
    branch = T.TIf(T.TLit(True, None), [T.TAssume(T.TLit(True, None), line=21)],
                   [T.TAssume(T.TLit(True, None), line=23)], line=20)
    loop = T.TFor("j", T.TLit(2, None), T.TLit(1, None), [branch], line=19)
    emitter = Lowering(replace(copy.tk, body=[loop]), debug_asserts=True)
    source = emitter.kernel_source().splitlines()
    assert [emitter.source_map[i] for i, text in enumerate(source, 1) if "tl.device_assert(" in text] == [21, 23]
    assert [emitter.source_map[i] for i, text in enumerate(source, 1) if text.strip() == "else:"] == [20]


def test_located_diagnostic_golden(tmp_path, monkeypatch):
    monkeypatch.setenv("TILA_BACKEND_ARTIFACTS", str(tmp_path))
    exc = RuntimeError("backend-specific formatting")
    exc.src = "def f():\n    broken()\n"
    exc.node = ast.parse(exc.src).body[0].body[0]
    err = failure(exc, phase="compile", source="import x\n" + exc.src, source_map={3: 2},
                  context={"tila_source": "def f():\n    ti.load(x, i)\n"})
    assert err.render() + "\n" == (Path(__file__).parent / "golden/backend_located.txt").read_text()


@pytest.mark.parametrize("resource", [False, True])
def test_backend_diagnostic_golden_and_raw_artifact(tmp_path, monkeypatch, resource):
    monkeypatch.setenv("TILA_BACKEND_ARTIFACTS", str(tmp_path))
    exc = RuntimeError("unstable backend detail /tmp/compiler-123")
    exc.name, exc.required, exc.limit = "shared memory", 2048, 1024
    err = failure(exc, phase="resources" if resource else "compile", source="generated\n",
                  source_map={}, context={"tila_source": "original\n"}, resource=resource)
    golden = Path(__file__).parent / "golden" / ("backend_resource.txt" if resource else "backend_compile.txt")
    assert err.render() + "\n" == golden.read_text()
    payload = json.loads((Path(err.backend_artifact) / "failure.json").read_text())
    assert payload["original_reason"] == str(exc)
    assert payload["source_map"] == {} and payload["tila_source"] == "original\n"
    if resource:
        assert payload["resource"] == {"name": "shared memory", "required": 2048, "limit": 1024}
    assert (Path(err.backend_artifact) / "kernel.py").read_text() == "generated\n"


def test_artifact_failure_does_not_mask_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setenv("TILA_BACKEND_ARTIFACTS", str(tmp_path / "unusable"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("denied")))
    err = failure(RuntimeError("backend"), phase="load", source="", source_map={}, context={})
    assert err.code == "TILA-TARGET-010" and err.backend_artifact is None
    assert "denied" in err.artifact_error
