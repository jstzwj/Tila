"""GPU 集成测试（marker `gpu`；development-plan §10 Stage 1 里程碑验收）。

同一 TIR 双路径：reference interpreter（CPU/NumPy）与 Triton lowering（GPU）
结果必须一致；add 同时与 torch.add 对拍。无 triton / 无 CUDA / 后端 C 编译器
不可用（Windows 需 MSVC 环境与 CC=cl）时逐项跳过。
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent

try:
    import torch

    HAS_TORCH = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_TRITON, reason="triton 未安装"),
    pytest.mark.skipif(not HAS_TORCH, reason="torch/CUDA 不可用"),
]

from tila.driver import compile_kernel  # noqa: E402
from tila.interp import run_kernel  # noqa: E402


def _load_generated(res):
    """@triton.jit 需要真实文件：写入临时 .py 并 import。"""
    with tempfile.TemporaryDirectory() as td:
        mod_path = Path(td) / "tila_gen_kernel.py"
        mod_path.write_text(res.triton_source, encoding="utf-8", newline="\n")
        spec = importlib.util.spec_from_file_location("tila_gen_kernel", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 返回闭包捕获的 launcher（模块删了也能用：JITFunction 已编译缓存）
        return mod, getattr(mod, f"{res.kernel.name}_launch")


def _launch(res, launcher, tensors, **constexpr):
    try:
        launcher(*tensors, **constexpr)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        if "compiler" in str(e).lower() or "cc" in str(e).lower() or \
                "cuda_utils" in str(e):
            pytest.skip(f"triton 后端 C 编译器不可用（Windows 需 MSVC 环境 + CC=cl）: {e}")
        raise


@pytest.mark.parametrize("N", [1, 127, 128, 129, 1000])
@pytest.mark.parametrize("BLOCK", [32, 128])
def test_gpu_add_matches_torch(N, BLOCK):
    src = (ROOT / "examples/add.tila").read_text(encoding="utf-8")
    res = compile_kernel(src, {"BLOCK": BLOCK})
    _, launcher = _load_generated(res)
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.zeros_like(a)
    _launch(res, launcher, (a, b, c), BLOCK=BLOCK)
    torch.testing.assert_close(c, a + b)


def test_gpu_vs_interpreter_same_tir():
    """同一 TIR 双路径（development-plan §8 的验收图）：interpreter == GPU。"""
    src = (ROOT / "examples/masked_add.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)

    rng = np.random.default_rng(7)
    n = 300
    a_np = rng.standard_normal(n).astype(np.float32)
    b_np = rng.standard_normal(n).astype(np.float32)

    # 路径 1：reference interpreter
    c_interp = np.zeros(n, dtype=np.float32)
    run_kernel(res.kernel, {"a": a_np, "b": b_np, "c": c_interp})

    # 路径 2：Triton lowering → GPU
    a_t = torch.from_numpy(a_np).cuda()
    b_t = torch.from_numpy(b_np).cuda()
    c_t = torch.zeros_like(a_t)
    _launch(res, launcher, (a_t, b_t, c_t))

    np.testing.assert_allclose(c_interp, c_t.cpu().numpy(), rtol=1e-6, atol=1e-7)


def test_gpu_saxpy():
    src = (ROOT / "examples/saxpy.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    n = 513
    alpha = 7
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.zeros_like(x)
    _launch(res, launcher, (x, y, out), alpha=alpha)
    torch.testing.assert_close(out, alpha * x + y)


def test_gpu_batched_add_2d():
    """v0.2 二维预览：2D grid + expand_dim + size-1 广播在 GPU 上与 torch 对拍。"""
    src = (ROOT / "examples/batched_add.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 100, 300  # 非整除 BM/BN，验证 mask 谓词
    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    b = torch.randn(m, n, device="cuda", dtype=torch.float32)
    c = torch.zeros_like(a)
    _launch(res, launcher, (a, b, c))
    torch.testing.assert_close(c, a + b)


def test_gpu_batched_add_vs_interpreter():
    """2D 的同一 TIR 双路径：interpreter（CPU）== Triton（GPU）。"""
    src = (ROOT / "examples/batched_add.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)

    rng = np.random.default_rng(13)
    m, n = 70, 140
    a_np = rng.standard_normal((m, n)).astype(np.float32)
    b_np = rng.standard_normal((m, n)).astype(np.float32)

    c_interp = np.zeros((m, n), dtype=np.float32)
    run_kernel(res.kernel, {"a": a_np, "b": b_np, "c": c_interp})

    a_t = torch.from_numpy(a_np).cuda()
    b_t = torch.from_numpy(b_np).cuda()
    c_t = torch.zeros_like(a_t)
    _launch(res, launcher, (a_t, b_t, c_t))

    np.testing.assert_allclose(c_interp, c_t.cpu().numpy(), rtol=1e-6, atol=1e-7)


def test_gpu_matmul_fragment():
    """v0.2 matmul fragment：dot/MMA 布局在 GPU 上与 torch.matmul 对拍。

    K <= BK=64 且非 2 的幂（K=50），验证 masked other=0.0 的包络。
    """
    src = (ROOT / "examples/matmul.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n, k = 100, 140, 50
    a = (torch.randn(m, k, device="cuda") * 0.5).half()
    b = (torch.randn(k, n, device="cuda") * 0.5).half()
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a, b, c))
    want = a.float() @ b.float()
    torch.testing.assert_close(c, want, rtol=1e-2, atol=1e-2)


def test_gpu_matmul_vs_interpreter():
    src = (ROOT / "examples/matmul.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    rng = np.random.default_rng(17)
    m, n, k = 70, 130, 64
    a_np = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
    b_np = (rng.standard_normal((k, n)) * 0.5).astype(np.float16)
    c_interp = np.zeros((m, n), dtype=np.float32)
    run_kernel(res.kernel, {"a": a_np, "b": b_np, "c": c_interp})
    a_t = torch.from_numpy(a_np).cuda()
    b_t = torch.from_numpy(b_np).cuda()
    c_t = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a_t, b_t, c_t))
    np.testing.assert_allclose(c_interp, c_t.cpu().numpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    HAS_TORCH and torch.cuda.get_device_capability() < (8, 9),
    reason="fp8e4m3fn 需要 SM89+")
def test_gpu_fp8_add():
    src = (ROOT / "examples/fp8_add.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    n = 256
    a = torch.randn(n, device="cuda", dtype=torch.float16).clamp(-2, 2)
    b = torch.randn(n, device="cuda", dtype=torch.float16).clamp(-2, 2)
    a8 = a.to(torch.float8_e4m3fn)
    b8 = b.to(torch.float8_e4m3fn)
    c8 = torch.zeros(n, device="cuda", dtype=torch.float8_e4m3fn)
    _launch(res, launcher, (a8, b8, c8))
    expected = (a8.to(torch.float16) + b8.to(torch.float16)).to(torch.float8_e4m3fn)
    torch.testing.assert_close(c8.float(), expected.float())
