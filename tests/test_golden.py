"""黄金测试：生成产物与 tests/golden/ 逐字节一致（含空行/缩进/换行符）。

add 的黄金文件内容取自规范文档（type-checker.md §8 / triton-lowering.md §2），
是 v0.1 的正式验收物；fp8_add / saxpy / masked_add 为经人工核对的快照。
"""

from pathlib import Path

import pytest

from tila.driver import compile_kernel

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
GOLDEN = Path(__file__).resolve().parent / "golden"

# add 的黄金内容取自规范文档；fp8_add / saxpy / masked_add / batched_add /
# matmul 为经与文档走查人工核对后的快照
PROGRAMS = ["add", "fp8_add", "saxpy", "masked_add", "batched_add", "matmul"]


def _compile(name: str, overrides=None):
    source = (EXAMPLES / f"{name}.tila").read_text(encoding="utf-8")
    return compile_kernel(source, overrides)


@pytest.mark.parametrize("name", PROGRAMS)
def test_triton_source_byte_exact(name):
    res = _compile(name)
    golden = (GOLDEN / f"{name}.triton.py").read_bytes()
    assert res.triton_source.encode() == golden, f"{name}: Triton 源码与黄金不一致"


@pytest.mark.parametrize("name", PROGRAMS)
def test_tir_dump_byte_exact(name):
    res = _compile(name)
    golden = (GOLDEN / f"{name}.tir.txt").read_bytes()
    assert res.tir_dump.encode() == golden, f"{name}: TIR dump 与黄金不一致"


@pytest.mark.parametrize("name", PROGRAMS)
def test_golden_files_are_lf(name):
    for f in (GOLDEN / f"{name}.triton.py", GOLDEN / f"{name}.tir.txt"):
        raw = f.read_bytes()
        assert b"\r" not in raw, f"{f.name} 含 CR"
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), \
            f"{f.name} 必须恰好一个尾换行"


def test_add_golden_matches_docs_block():
    """add 黄金 TIR 与规范文档 §8 走查块逐行对应（防黄金文件被无意改动）。"""
    lines = (GOLDEN / "add.tir.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("func @add(a: Buffer<f32, (N,)>")
    assert lines[1] == "L0 = identity(128)"
    assert lines[2] == ""
    assert lines[3] == "%pid = program_id 0 : Scalar(i32) [#pid]"
    assert lines[-1] == "return"
    assert sum(1 for l in lines if l.startswith("store ")) == 1


def test_constexpr_specialization_changes_shapes_only():
    """BLOCK=64 特化：类型里的 shape 取特化值，发射仍保留名字（模型 B）。"""
    res = _compile("add", {"BLOCK": 64})
    assert "Tile<i32, (64,), L0>" in res.tir_dump
    assert "%r0 = arange 0 BLOCK : Tile<i32, (64,), L0>" in res.tir_dump
    assert "tl.arange(0, BLOCK)" in res.triton_source
    # TIR 是特化产物、跨值不复用：默认编译与 64 特化的 dump 不同
    assert res.tir_dump != _compile("add").tir_dump
