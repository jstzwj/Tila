"""Canonical five-example TIR, Triton source and audit explain snapshots."""
import importlib
from pathlib import Path
import sys

import pytest
import tila as ti

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
EXAMPLES = ("add_kernel", "softmax", "matmul", "self_attention", "fused_attention")


def snapshots(name):
    module = importlib.import_module(name)
    existing = getattr(module, name if name == "add_kernel" else name + "_kernel")
    # Fresh TIR avoids aliases and facts recorded by unrelated launch tests.
    kernel = ti.jit(existing.fn)
    source, tir = kernel.materialize({})
    return {"triton.py": source, "tir.txt": tir.rstrip() + "\n",
            "explain.txt": kernel.explain({}).rstrip() + "\n"}


@pytest.mark.parametrize("name", EXAMPLES)
@pytest.mark.parametrize("kind", ["tir.txt", "triton.py", "explain.txt"])
def test_official_example_golden(name, kind, monkeypatch):
    monkeypatch.setenv("TILA_DEBUG", "0")
    monkeypatch.setenv("TILA_SAFETY", "strict")
    actual = snapshots(name)[kind]
    expected = (ROOT / "tests/golden/examples" / f"{name}.{kind}").read_text()
    assert actual == expected
