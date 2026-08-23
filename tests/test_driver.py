"""driver/CLI 测试：build/dump 命令、--constexpr、--explain、产物 LF 行尾。"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-m", "tila", *args],
        capture_output=True, text=True, cwd=str(ROOT), env=env)


def test_cli_build_writes_lf_artifacts(tmp_path):
    out = tmp_path / "out"
    r = _run_cli("build", "examples/add.tila", "-o", str(out))
    assert r.returncode == 0, r.stderr
    py = (out / "add.triton.py").read_bytes()
    tir_txt = (out / "add.tir.txt").read_bytes()
    assert b"\r" not in py and b"\r" not in tir_txt
    golden = (ROOT / "tests/golden/add.triton.py").read_bytes()
    assert py == golden


def test_cli_dump_stdout(tmp_path, capsys):
    r = _run_cli("dump", "examples/add.tila")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("func @add(")
    assert "%z = add %x %y : Tile<f32, (128,), L0> [#z]" in r.stdout


def test_cli_constexpr_override(tmp_path):
    r = _run_cli("dump", "examples/add.tila", "--constexpr", "BLOCK=64")
    assert r.returncode == 0, r.stderr
    assert "Tile<i32, (64,), L0>" in r.stdout
    assert "arange 0 BLOCK" in r.stdout  # 发射保留名字（模型 B）


def test_cli_build_explain(tmp_path):
    out = tmp_path / "out"
    r = _run_cli("build", "examples/add.tila", "-o", str(out), "--explain")
    assert r.returncode == 0, r.stderr
    assert "dtype : f32 == f32" in r.stdout
    assert "layout:" in r.stdout
    assert "identity(128) ~ identity(128)" in r.stdout


def test_cli_invalid_program_exits_nonzero(tmp_path):
    bad = tmp_path / "bad.tila"
    bad.write_text("import tila\n\n\n@tila.jit\ndef k(a):\n    z = x + y\n",
                   encoding="utf-8")
    r = _run_cli("build", str(bad), "-o", str(tmp_path / "o"))
    assert r.returncode == 1
    assert "E01" in r.stderr


def test_compile_kernel_rejects_before_any_output():
    from tila.driver import compile_kernel
    from tila.diagnostics import TilaError

    try:
        compile_kernel("import tila\n\n\n@tila.jit\ndef k():\n    x = 1\n    x = 2\n")
        raised = False
    except TilaError:
        raised = True
    assert raised  # E14：绝不产出任何 Triton 代码


def test_backend_validator_static_checks():
    from tila.backend.triton import validator
    from tila.driver import compile_kernel

    res = compile_kernel(open(ROOT / "examples/add.tila", encoding="utf-8").read(),
                         validate_backend=True)
    assert res.backend_errors == []

    res8 = compile_kernel(open(ROOT / "examples/fp8_add.tila", encoding="utf-8").read(),
                          validate_backend=True)
    codes = {e.code for e in res8.backend_errors}
    assert "B01" in codes  # fp8 架构可用性提示（B 码与 E 码分开）
