"""@tila.jit 运行时测试：像 @triton.jit 一样定义并直接调用。

kernel 必须定义在真实 .py 文件中（inspect.getsource），因此测试通过
tmp_path 落盘 + importlib 加载用户模块。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tila import TilaKernel, jit
from tila.diagnostics import TilaError

ADD_MODULE = '''\
from __future__ import annotations

import tila


@tila.jit
def add(
    a: tila.Tensor[tila.float32, N],
    b: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    y = tila.load(b, (offs,), mask=mask)
    z = x + y
    tila.store(c, (offs,), z, mask=mask)
'''


def _load_module(tmp_path: Path, source: str, name: str = "user_kernels"):
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_jit_returns_runtime_kernel(tmp_path):
    mod = _load_module(tmp_path, ADD_MODULE)
    assert isinstance(mod.add, TilaKernel)
    assert mod.add.__name__ == "add"


def test_numpy_call_runs_interpreter(tmp_path):
    mod = _load_module(tmp_path, ADD_MODULE)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500).astype(np.float32)
    b = rng.standard_normal(500).astype(np.float32)
    c = np.zeros(500, dtype=np.float32)
    mod.add(a, b, c)
    np.testing.assert_allclose(c, a + b, rtol=1e-6)


def test_constexpr_specialization_and_cache(tmp_path):
    mod = _load_module(tmp_path, ADD_MODULE)
    assert len(mod.add._cache) == 0
    rng = np.random.default_rng(1)
    a = rng.standard_normal(300).astype(np.float32)
    for block, expect_new in ((None, True), (64, True), (64, False)):
        c = np.zeros(300, dtype=np.float32)
        before = len(mod.add._cache)
        mod.add(a, a, c) if block is None else mod.add(a, a, c, BLOCK=block)
        np.testing.assert_allclose(c, a + a, rtol=1e-6)
        assert len(mod.add._cache) == before + (1 if expect_new else 0)
    # 不同特化的 TIR 不同（shape 取特化值），发射保留名字
    entries = list(mod.add._cache.values())
    assert "(128,)" in entries[0].tir_dump and "(64,)" in entries[1].tir_dump
    assert "tl.arange(0, BLOCK)" in entries[1].triton_source


def test_type_errors_are_tila_errors(tmp_path):
    """kernel 里的语言错误在直接调用时同样以 E 码拒绝（同一套 checker）。"""
    bad = ADD_MODULE.replace("z = x + y", "z = x + y\n    w = z + a")
    mod = _load_module(tmp_path, bad)
    a = np.zeros(8, dtype=np.float32)
    with pytest.raises(TilaError) as ei:
        mod.add(a, a, a.copy())
    assert ei.value.code == "E07"  # Buffer 出现在非 R7 位置


def test_diagnostics_report_file_accurate_locations(tmp_path):
    """@tila.jit 路径的诊断行号校准到文件绝对行列（inspect 源 + dedent 偏移）。"""
    # z = r + s 在文件第 9 行，BinOp 表达式起点（r）在第 8 列
    src = (
        "from __future__ import annotations\n"
        "import tila\n"
        "\n"
        "\n"
        "@tila.jit\n"
        "def bad(a: tila.Tensor[tila.float32, 8]):\n"
        "    r = tila.arange(0, 64)\n"
        "    s = tila.arange(0, 128)\n"
        "    z = r + s\n"
        "    tila.store(a, (r,), z)\n"
    )
    mod = _load_module(tmp_path, src, name="loc_kernels")
    with pytest.raises(TilaError) as ei:
        mod.bad(np.zeros(8, dtype=np.float32))
    assert ei.value.code == "E03"
    assert ei.value.loc.line == 9   # z = r + s
    assert ei.value.loc.col == 8     # 表达式起点（r）


def test_unknown_constexpr_kwarg_rejected(tmp_path):
    mod = _load_module(tmp_path, ADD_MODULE)
    a = np.zeros(8, dtype=np.float32)
    with pytest.raises(TilaError) as ei:
        mod.add(a, a, a.copy(), NOPE=4)
    assert ei.value.code == "E13"


def test_arity_mismatch_clear_error(tmp_path):
    mod = _load_module(tmp_path, ADD_MODULE)
    a = np.zeros(8, dtype=np.float32)
    with pytest.raises(TypeError, match="positional"):
        mod.add(a, a)


def test_static_dim_module_needs_no_future_import(tmp_path):
    # 全静态维 → 注解不含符号名，def 期无需 PEP 563；body 的界谓词用字面量
    src = ADD_MODULE.replace("from __future__ import annotations\n", "") \
        .replace("tila.float32, N", "tila.float32, 512") \
        .replace("mask = offs < N", "mask = offs < 512")
    mod = _load_module(tmp_path, src, name="static_kernels")
    a = np.arange(512, dtype=np.float32)
    c = np.zeros(512, dtype=np.float32)
    mod.add(a, a, c)
    np.testing.assert_allclose(c, a + a, rtol=1e-6)


def test_jit_requires_callable():
    with pytest.raises(TypeError, match="without parentheses"):
        jit("not a function")


def test_2d_kernel_via_runtime(tmp_path):
    src = '''\
from __future__ import annotations

import tila


@tila.jit
def batched_add(
    a: tila.Tensor[tila.float32, M, N],
    b: tila.Tensor[tila.float32, M, N],
    c: tila.Tensor[tila.float32, M, N],
    BM: tila.constexpr = 64,
    BN: tila.constexpr = 128,
):
    pid_m = tila.program_id(0)
    pid_n = tila.program_id(1)
    rows = pid_m * BM + tila.arange(0, BM)
    cols = pid_n * BN + tila.arange(0, BN)
    rows2 = tila.expand_dim(rows, 1)
    cols2 = tila.expand_dim(cols, 0)
    mask = (rows2 < M) & (cols2 < N)
    x = tila.load(a, (rows2, cols2), mask=mask)
    y = tila.load(b, (rows2, cols2), mask=mask)
    z = x + y
    tila.store(c, (rows2, cols2), z, mask=mask)
'''
    mod = _load_module(tmp_path, src, name="k2d")
    rng = np.random.default_rng(2)
    a = rng.standard_normal((70, 140)).astype(np.float32)
    b = rng.standard_normal((70, 140)).astype(np.float32)
    c = np.zeros((70, 140), dtype=np.float32)
    mod.batched_add(a, b, c, BM=8, BN=16)
    np.testing.assert_allclose(c, a + b, rtol=1e-6)


# ---------------------------------------------------------------------------
# GPU 路径（需要 triton + CUDA）
# ---------------------------------------------------------------------------

gpu = pytest.mark.gpu
try:
    import torch

    has_gpu = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    has_gpu = False


@gpu
@pytest.mark.skipif(not has_gpu, reason="CUDA 不可用")
def test_gpu_call_matches_torch(tmp_path):
    mod = _load_module(tmp_path, ADD_MODULE, name="gpu_kernels")
    a = torch.randn(1000, device="cuda")
    b = torch.randn(1000, device="cuda")
    c = torch.zeros_like(a)
    mod.add(a, b, c, BLOCK=64)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)


@gpu
@pytest.mark.skipif(not has_gpu, reason="CUDA 不可用")
def test_cpu_torch_tensor_gets_clear_error(tmp_path):
    mod = _load_module(tmp_path, ADD_MODULE, name="gpu_kernels2")
    a = torch.randn(16)
    with pytest.raises(TypeError, match=r"\.cuda\(\)"):
        mod.add(a, a, torch.zeros_like(a))
