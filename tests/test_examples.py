"""官方 CPU examples 必须是可直接运行的跨平台 smoke tests。"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "add_kernel.py",
    "softmax.py",
    "matmul.py",
    "self_attention.py",
    "fused_attention.py",
)


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs_with_narrow_windows_encoding(name):
    # 示例的用户可见摘要刻意保持 ASCII；即使宿主给出 cp1252，也应正常
    # 结束。kernel 诊断的 Unicode 输出由 CLI 自己配置为 UTF-8。
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("cp1252", errors="replace")
    assert b" OK" in proc.stdout
