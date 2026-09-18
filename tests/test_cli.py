"""CLI：check / build / dump 冒烟（surface-language.md §8）。"""

import os
import subprocess
import sys

import pytest

from tila.cli import main

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_check_add_kernel(capsys):
    rc = main(["check", os.path.join(EXAMPLES, "add_kernel.py")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "add_kernel" in out
    assert "obligations" in out


def test_build_add_kernel(tmp_path, capsys):
    rc = main(["build", os.path.join(EXAMPLES, "add_kernel.py"),
               "-o", str(tmp_path)])
    assert rc == 0
    src_file = tmp_path / "add_kernel.triton.py"
    tir_file = tmp_path / "add_kernel.tir.txt"
    assert src_file.exists() and tir_file.exists()
    assert "@triton.jit" in src_file.read_text(encoding="utf-8")


def test_dump_add_kernel(capsys):
    rc = main(["dump", os.path.join(EXAMPLES, "add_kernel.py")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "kernel @add_kernel" in out
    assert "obligation" in out


def test_check_reports_stage2_bounds_error(capsys, tmp_path):
    bad = tmp_path / "bad_kernel.py"
    bad.write_text(
        "import tila as ti\n"
        "N = ti.Dim('N')\n\n"
        "@ti.jit\n"
        "def bad(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite], "
        "BLOCK: ti.Const[int] = 64):\n"
        "    pid = ti.program_id(0)\n"
        "    offs = pid * BLOCK + ti.arange(0, BLOCK)\n"
        "    v = ti.load(x, offs)\n"          # 无 mask 尾块
        "    ti.store(x, offs, v)\n",
        encoding="utf-8")
    rc = main(["check", str(bad)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "TILA-BOUNDS-001" in err


@pytest.mark.parametrize("args", [
    ["--help"],
    ["check", "examples/add_kernel.py"],
    ["explain", "examples/add_kernel.py"],
])
def test_cli_reconfigures_narrow_windows_encoding(args):
    """模拟宿主继承 cp1252：CLI 自行切到 UTF-8，不因诊断字符崩溃。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    proc = subprocess.run(
        [sys.executable, "-m", "tila", *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    proc.stdout.decode("utf-8")
    proc.stderr.decode("utf-8")
