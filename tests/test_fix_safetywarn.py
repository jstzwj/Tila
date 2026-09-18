"""--safety=warn / TILA_SAFETY：Unknown bounds 义务的降级路径
（refinements.md §5.3、bounds-safety.md §6）。

- strict（默认）：Unknown → error TILA-BOUNDS-001/002；
- warn：Unknown → stderr warning（诊断复用，首行 error→warning），kernel
  继续编译/运行，warning 也记入 JITFunction.last_report；
- ProvenUnsafe（TILA-BOUNDS-003）：任何模式都无条件 error。
"""

import numpy as np
import pytest

import tila as ti
from tila.cli import main
from tila.errors import TilaError

N = ti.Dim("N")


@ti.jit
def tail_kernel(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
                BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
    pid = ti.program_id(0)
    offs = pid * BLOCK + ti.arange(0, BLOCK)
    v = ti.load(x, offs)                      # 无 mask 尾块 → Unknown
    ti.store(x, offs, v, mask=offs < N)


@ti.jit
def provably_oob(x: ti.Buffer[ti.f32, (128,), ti.ReadWrite],
                 BLOCK: ti.Const[int, ti.PowerOfTwo]):
    ti.store(x, BLOCK * BLOCK, 1.0)           # BLOCK=64 ⇒ 4096 ≥ 128


UNMASKED_SRC = '''"""无 mask 尾块 kernel（check --safety=warn 冒烟）。"""
import tila as ti

N = ti.Dim("N")


@ti.jit
def tail(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
         BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
    pid = ti.program_id(0)
    offs = pid * BLOCK + ti.arange(0, BLOCK)
    v = ti.load(x, offs)
    ti.store(x, offs, v, mask=offs < N)
'''


# (a) strict 默认：回归护栏——无 mask 尾块硬拒绝
def test_strict_default_rejects_unmasked_tail(monkeypatch):
    monkeypatch.delenv("TILA_SAFETY", raising=False)
    x = np.ones(100, dtype=np.float32)
    with pytest.raises(TilaError) as ei:
        tail_kernel[(ti.cdiv(100, 64),)](x, BLOCK=64)
    assert ei.value.code == "TILA-BOUNDS-001"


# (b) warn：同一 launch 在解释器上运行，stderr 出 warning，last_report 记录
def test_warn_mode_downgrades_and_runs(monkeypatch, capsys):
    monkeypatch.setenv("TILA_SAFETY", "warn")
    x = np.ones(100, dtype=np.float32)
    tail_kernel[(ti.cdiv(100, 64),)](x, BLOCK=64)      # 不抛
    err = capsys.readouterr().err
    assert "warning[TILA-BOUNDS-001]" in err
    assert "possible out-of-bounds access" in err      # 完整诊断被复用
    assert "TILA-BOUNDS-001" in tail_kernel.last_report
    assert np.all(x == 1.0)                            # kernel 确实运行了


# (c) ProvenUnsafe：warn 模式下也必须 raise（无条件 error）
def test_proven_unsafe_raises_in_warn_mode(monkeypatch):
    monkeypatch.setenv("TILA_SAFETY", "warn")
    x = np.ones(128, dtype=np.float32)
    with pytest.raises(TilaError) as ei:
        provably_oob[(1,)](x, BLOCK=64)
    assert ei.value.code == "TILA-BOUNDS-003"


def test_proven_unsafe_raises_in_strict_mode(monkeypatch):
    monkeypatch.delenv("TILA_SAFETY", raising=False)
    x = np.ones(128, dtype=np.float32)
    with pytest.raises(TilaError) as ei:
        provably_oob[(1,)](x, BLOCK=64)
    assert ei.value.code == "TILA-BOUNDS-003"


# (d) CLI：check --safety=warn 走 materialize 同一求值路径，rc 0 + stderr warning
def test_cli_check_safety_warn(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TILA_SAFETY", raising=False)
    f = tmp_path / "unmasked_tail.py"
    f.write_text(UNMASKED_SRC, encoding="utf-8")
    rc = main(["check", str(f), "--safety=warn"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "warning[TILA-BOUNDS-001]" in captured.err
    assert "Stage 2" in captured.out                    # 检查本身通过


def test_cli_check_strict_default(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TILA_SAFETY", raising=False)
    f = tmp_path / "unmasked_tail.py"
    f.write_text(UNMASKED_SRC, encoding="utf-8")
    rc = main(["check", str(f)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "error[TILA-BOUNDS-001]" in captured.err
