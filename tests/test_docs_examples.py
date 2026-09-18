"""Markdown Python 示例分类与 smoke tests。

标签必须紧邻代码块：
  <!-- tila-example: current; mode=exec|syntax -->
  <!-- tila-example: diagnostic -->
  <!-- tila-example: future; milestone=M1..M6 -->
  <!-- tila-example: generated -->
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = (ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")))
MARKER = re.compile(
    r"^<!-- tila-example: (current|diagnostic|future|generated)"
    r"(?:; mode=(exec|syntax))?(?:; milestone=(M[1-6]))? -->$"
)


@dataclass(frozen=True)
class Example:
    path: Path
    line: int
    kind: str
    mode: str | None
    milestone: str | None
    source: str

    @property
    def id(self):
        return f"{self.path.relative_to(ROOT)}:{self.line}:{self.kind}"


def _examples():
    found = []
    for path in MARKDOWN:
        lines = path.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            if lines[i] != "```python":
                i += 1
                continue
            assert i > 0, f"{path}:1: Python fence has no tila-example marker"
            marker = MARKER.fullmatch(lines[i - 1])
            assert marker is not None, (
                f"{path}:{i + 1}: every Python fence needs an immediately "
                "preceding tila-example marker"
            )
            try:
                end = lines.index("```", i + 1)
            except ValueError:
                raise AssertionError(f"{path}:{i + 1}: unclosed Python fence")
            found.append(Example(path, i + 1, marker.group(1), marker.group(2),
                                 marker.group(3),
                                 "\n".join(lines[i + 1:end]) + "\n"))
            i = end + 1
    return found


EXAMPLES = _examples()


def test_python_examples_are_explicitly_classified():
    assert EXAMPLES
    kinds = {example.kind for example in EXAMPLES}
    assert kinds == {"current", "diagnostic", "future", "generated"}
    for example in EXAMPLES:
        if example.kind == "current":
            assert example.mode in {"exec", "syntax"}, example.id
            assert example.milestone is None, example.id
        elif example.kind == "future":
            assert example.mode is None, example.id
            assert example.milestone is not None, example.id
        else:
            assert example.mode is None and example.milestone is None, example.id


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.id)
def test_all_markdown_python_blocks_are_valid_python_syntax(example):
    compile(example.source, f"<{example.id}>", "exec")


EXECUTABLE = [e for e in EXAMPLES if e.kind == "current" and e.mode == "exec"]


@pytest.mark.parametrize("example", EXECUTABLE, ids=lambda e: e.id)
def test_current_executable_examples_reach_real_stage1(example, tmp_path):
    """真实运行文档代码；@ti.jit 会执行 frontend + checker Stage 1。"""
    source = tmp_path / "documented_example.py"
    source.write_text(example.source, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(source)], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")


def test_there_are_real_executable_and_future_examples():
    assert len(EXECUTABLE) >= 2
    assert any(e.kind == "future" and e.milestone == "M4" for e in EXAMPLES)
    assert any(e.kind == "future" and e.milestone == "M5" for e in EXAMPLES)
