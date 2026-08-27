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


@pytest.mark.parametrize("kind", ["transposed", "padded"])
def test_gpu_matmul_noncontiguous_b(kind):
    """v0.3 strides 的 GPU 验收主场：非连续 b（转置视图 / padded 存储）
    在声明式 strides 下与 torch.matmul 对拍——v0.2 平面寻址做不到。"""
    src = (ROOT / "examples/matmul.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n, k = 66, 90, 50
    if kind == "transposed":
        b = (torch.randn(n, k, device="cuda") * 0.5).half().t()  # (k,n) strides (1,n)
        assert not b.is_contiguous()
    else:
        base = (torch.randn(k, 128, device="cuda") * 0.5).half()
        b = base[:, :n]                                          # strides (128,1)
    a = (torch.randn(m, k, device="cuda") * 0.5).half()
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a, b, c))
    want = a.float() @ b.float()
    torch.testing.assert_close(c, want, rtol=1e-2, atol=1e-2)


def test_gpu_rowmajor_buffer_rejects_noncontiguous():
    """RowMajor 默认声明的 buffer 收到非连续张量 → launcher 断言失败（正确拒绝，
    而非静默读错元素）。"""
    src = (ROOT / "examples/batched_add.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 64, 128
    a = torch.randn(m, n, device="cuda")
    b = torch.randn(m, n, device="cuda")
    c_bad = torch.randn(n, m, device="cuda").t()  # 非连续
    with pytest.raises(AssertionError):
        _launch(res, launcher, (a, b, c_bad))


@pytest.mark.parametrize("transposed", [False, True])
def test_gpu_matmul_storage_offset_views(transposed):
    """storage offset ABI（v0.3-strides §1.3）：base(b) = 逻辑张量原点
    （torch data_ptr 含 storage_offset）——带非零偏移的视图（连续/转置）必须
    照样正确（评审 §17 采纳的覆盖项）。"""
    src = (ROOT / "examples/matmul.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, k, n, off = 66, 48, 40, 64
    a = (torch.randn(m, k, device="cuda") * 0.5).half()
    big = (torch.randn(off + n * k, device="cuda") * 0.5).half()
    seg = big[off:]                                  # storage offset = 64
    b = seg.view(n, k).t() if transposed else seg.view(k, n)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a, b, c))
    want = a.float() @ b.float()
    torch.testing.assert_close(c, want, rtol=1e-2, atol=1e-2)


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


# ---------------------------------------------------------------------------
# v0.4 K 循环与累加器（docs/v0.4-kloop.md §8 的 GPU 差分主场）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [0, 1, 50, 64, 65, 512])
def test_gpu_matmul_loop_arbitrary_k(K):
    # K=0：零迭代恒等式（C = A·B = 0；空维张量的 stride 契约空真）
    """全 K matmul（K-loop 版）：GPU vs torch.matmul。K 非 BK 倍数由
    mask + other=0.0 补零——正确性包络不受 K <= BK 限制（v0.4 验收主场）。"""
    src = (ROOT / "examples/matmul_loop.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 100, 140
    a = (torch.randn(m, K, device="cuda") * 0.5).half()
    b = (torch.randn(K, n, device="cuda") * 0.5).half()
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a, b, c))
    want = a.float() @ b.float()
    torch.testing.assert_close(c, want, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kind", ["transposed", "padded"])
def test_gpu_matmul_loop_noncontiguous_b(kind):
    """K-loop × 非连续 b：循环不改内存语义（v0.3 strides 正交性回归）。"""
    src = (ROOT / "examples/matmul_loop.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n, k = 66, 90, 130
    if kind == "transposed":
        b = (torch.randn(n, k, device="cuda") * 0.5).half().t()
        assert not b.is_contiguous()
    else:
        base = (torch.randn(k, 128, device="cuda") * 0.5).half()
        b = base[:, :n]
    a = (torch.randn(m, k, device="cuda") * 0.5).half()
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a, b, c))
    want = a.float() @ b.float()
    torch.testing.assert_close(c, want, rtol=1e-2, atol=1e-2)


def test_gpu_matmul_loop_vs_interpreter():
    """同一 TIR 双路径：K-loop 的 interpreter == GPU。"""
    src = (ROOT / "examples/matmul_loop.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    rng = np.random.default_rng(19)
    m, n, k = 70, 130, 200
    a_np = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
    b_np = (rng.standard_normal((k, n)) * 0.5).astype(np.float16)
    c_interp = np.zeros((m, n), dtype=np.float32)
    run_kernel(res.kernel, {"a": a_np, "b": b_np, "c": c_interp})
    a_t = torch.from_numpy(a_np).cuda()
    b_t = torch.from_numpy(b_np).cuda()
    c_t = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a_t, b_t, c_t))
    np.testing.assert_allclose(c_interp, c_t.cpu().numpy(), rtol=1e-2, atol=1e-2)


def test_gpu_two_loops_same_accumulator():
    """两个顺序 for 累加同一 acc（equiv 分支）在 GPU 上的差分。"""
    src = """import tila


@tila.jit
def k2(a: tila.Tensor[tila.float16, M, K], b: tila.Tensor[tila.float16, K, N, (sb0, sb1)],
       c: tila.Tensor[tila.float32, M, N], BKH: tila.constexpr = 64):
    pid_m = tila.program_id(0)
    pid_n = tila.program_id(1)
    rm2 = tila.expand_dim(pid_m * 64 + tila.arange(0, 64), 1)
    rn2 = tila.expand_dim(pid_n * 128 + tila.arange(0, 128), 0)
    rk2 = tila.expand_dim(tila.arange(0, BKH), 0)
    rk3 = tila.expand_dim(tila.arange(0, BKH), 1)
    c_m = (rm2 < M) & (rn2 < N)
    acc = tila.zeros((64, 128), tila.float32)
    for k0 in tila.range(0, K, BKH):
        rka = rk2 + k0
        x = tila.load(a, (rm2, rka), mask=(rm2 < M) & (rka < K), other=0.0)
        y = tila.load(b, (rk3 + k0, rn2), mask=((rk3 + k0) < K) & (rn2 < N), other=0.0)
        acc += tila.dot(x, y)
    for k1 in tila.range(0, K, BKH):
        rkb = rk2 + k1
        x2 = tila.load(a, (rm2, rkb), mask=(rm2 < M) & (rkb < K), other=0.0)
        y2 = tila.load(b, (rk3 + k1, rn2), mask=((rk3 + k1) < K) & (rn2 < N), other=0.0)
        acc += tila.dot(x2, y2)
    tila.store(c, (rm2, rn2), acc, mask=c_m)
"""
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, k, n = 66, 130, 90
    a = (torch.randn(m, k, device="cuda") * 0.5).half()
    b = (torch.randn(k, n, device="cuda") * 0.5).half()
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a, b, c))
    want = 2.0 * (a.float() @ b.float())
    torch.testing.assert_close(c, want, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# v0.5 归约与逐元素内建（docs/v0.5-reduce.md）：softmax 验收主场 + 归约/一元/
# where/neg_inf/num_programs 的 GPU 差分
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,BN", [
    (1, 1, 128), (63, 127, 128), (64, 128, 128), (65, 129, 256),
    (100, 140, 256), (130, 300, 512),
])
def test_gpu_softmax_vs_torch(M, N, BN):
    """v0.5 验收主场：softmax vs torch.softmax（f16 进出 / f32 内部）。
    数据含 f16 最低值（−65504，−∞ 哨兵不被污染）、全负行、大正值（max-减法
    防 exp 溢出）。"""
    src = (ROOT / "examples/softmax.tila").read_text(encoding="utf-8")
    res = compile_kernel(src, {"BN": BN})
    _, launcher = _load_generated(res)
    rng = np.random.default_rng(M + N)
    a_np = rng.standard_normal((M, N)).astype(np.float16)
    a_np[0, 0] = -65504.0
    a_np[M // 2] = -60000.0
    a_np[0, min(1, N - 1)] = 30000.0
    a = torch.from_numpy(a_np).cuda()
    o = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    _launch(res, launcher, (a, o), BN=BN)
    want = torch.softmax(a.float(), dim=-1).half()
    torch.testing.assert_close(o, want, rtol=2e-2, atol=2e-3)


def test_gpu_softmax_vs_interpreter():
    """同一 TIR 双路径：softmax 的 interpreter == GPU。"""
    src = (ROOT / "examples/softmax.tila").read_text(encoding="utf-8")
    res = compile_kernel(src, {"BN": 256})
    _, launcher = _load_generated(res)
    rng = np.random.default_rng(23)
    m, n = 100, 140
    a_np = rng.standard_normal((m, n)).astype(np.float16)
    a_np[0, 0] = -65504.0
    o_interp = np.zeros((m, n), dtype=np.float16)
    run_kernel(res.kernel, {"a": a_np, "o": o_interp}, {}, {"BN": 256})
    a = torch.from_numpy(a_np).cuda()
    o = torch.zeros(m, n, device="cuda", dtype=torch.float16)
    _launch(res, launcher, (a, o), BN=256)
    np.testing.assert_allclose(
        o_interp.astype(np.float32), o.cpu().numpy().astype(np.float32),
        rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("op", ["sum", "max"])
def test_gpu_reduce_f32_vs_torch(op):
    src = """import tila


@tila.jit
def red(a: tila.Tensor[tila.float32, M, N], o: tila.Tensor[tila.float32, M, N],
        BM: tila.constexpr = 64, BN: tila.constexpr = 128):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
    x = tila.load(a, (rm2, rn2), mask=m2)
    s = tila.{OP}(x, axis=1)
    s2 = tila.expand_dim(s, 1)
    y = x - s2
    tila.store(o, (rm2, rn2), y, mask=m2)
""".replace("{OP}", op)
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 100, 128
    a = torch.randn(m, n, device="cuda", dtype=torch.float32)
    o = torch.zeros_like(a)
    _launch(res, launcher, (a, o))
    fn = torch.sum if op == "sum" else torch.amax
    want = a - fn(a, dim=1, keepdim=True)
    torch.testing.assert_close(o, want, rtol=1e-5, atol=1e-6)


def test_gpu_reduce_narrow_dtypes_dtype_preserved():
    """f16 / i32 归约：结果 dtype 恢复（tl.max 的内部提升由 cast 抵消；
    int 求和 dtype 显式）——GPU 上与 torch 数值对拍。"""
    src16 = """import tila


@tila.jit
def r16(a: tila.Tensor[tila.float16, M, N], o: tila.Tensor[tila.float16, M],
        BM: tila.constexpr = 64, BN: tila.constexpr = 128):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
    x = tila.load(a, (rm2, rn2), mask=m2)
    mx = tila.max(x, axis=1)
    rm = pid * BM + tila.arange(0, BM)
    tila.store(o, (rm,), mx, mask=rm < M)
"""
    res = compile_kernel(src16)
    _, launcher = _load_generated(res)
    m, n = 100, 128
    a = (torch.randn(m, n, device="cuda") * 100).half()
    o = torch.zeros(m, device="cuda", dtype=torch.float16)
    _launch(res, launcher, (a, o))
    assert o.dtype == torch.float16
    torch.testing.assert_close(o.float(), a.float().max(dim=1).values.float())

    src32 = src16.replace("float16", "int32").replace(
        "mx = tila.max(x, axis=1)", "mx = tila.sum(x, axis=1)")
    res = compile_kernel(src32)
    _, launcher = _load_generated(res)
    ai = torch.randint(-1000, 1000, (m, n), device="cuda", dtype=torch.int32)
    oi = torch.zeros(m, device="cuda", dtype=torch.int32)
    _launch(res, launcher, (ai, oi))
    assert oi.dtype == torch.int32
    torch.testing.assert_close(oi, ai.sum(dim=1, dtype=torch.int32))


@pytest.mark.parametrize("op,torch_fn", [
    ("exp", torch.exp), ("exp2", torch.exp2), ("sqrt", torch.sqrt),
    ("abs", torch.abs),
])
def test_gpu_elem_vs_torch(op, torch_fn):
    src = """import tila


@tila.jit
def el(a: tila.Tensor[tila.float32, M, N], o: tila.Tensor[tila.float32, M, N],
       BM: tila.constexpr = 64, BN: tila.constexpr = 128):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
    x = tila.load(a, (rm2, rn2), mask=m2)
    y = tila.{OP}(x)
    tila.store(o, (rm2, rn2), y, mask=m2)
""".replace("{OP}", op)
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 100, 128
    a = torch.abs(torch.randn(m, n, device="cuda"))  # sqrt 域非负
    o = torch.zeros_like(a)
    _launch(res, launcher, (a, o))
    torch.testing.assert_close(o, torch_fn(a), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("M,N,BN", [(100, 300, 64), (64, 128, 128), (1, 1, 128)])
def test_gpu_rowsum_loop(M, N, BN):
    """归约 × K-loop 累加（acc += tila.sum(e, 1)）：GPU vs torch.sum。"""
    src = """import tila


@tila.jit
def rowsum(a: tila.Tensor[tila.float32, M, N], o: tila.Tensor[tila.float32, M],
           BM: tila.constexpr = 64, BN: tila.constexpr = 128):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rm = pid * BM + tila.arange(0, BM)
    acc = tila.zeros((BM,), tila.float32)
    for k0 in tila.range(0, N, BN):
        cols2 = tila.expand_dim(k0 + tila.arange(0, BN), 0)
        m2 = (rm2 < M) & (cols2 < N)
        t = tila.load(a, (rm2, cols2), mask=m2)
        e = tila.where(m2, t, 0.0)
        acc += tila.sum(e, 1)
    tila.store(o, (rm,), acc, mask=rm < M)
"""
    res = compile_kernel(src, {"BN": BN})
    _, launcher = _load_generated(res)
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    o = torch.zeros(M, device="cuda", dtype=torch.float32)
    _launch(res, launcher, (a, o), BN=BN)
    torch.testing.assert_close(o, a.sum(dim=1), rtol=2e-4, atol=1e-4)


def test_gpu_num_programs_observational():
    """num_programs(0) 读 grid 维（cdiv(M,BM)）——只读观测，不改变 launch。"""
    src = """import tila


@tila.jit
def npk(a: tila.Tensor[tila.float32, M, N], o: tila.Tensor[tila.float32, M, N],
        BM: tila.constexpr = 64, BN: tila.constexpr = 128):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
    x = tila.load(a, (rm2, rn2), mask=m2)
    nblk = tila.num_programs(0)
    tila.store(o, (rm2, rn2), x + tila.cast(nblk, tila.float32), mask=m2)
"""
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 130, 128
    a = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    o = torch.zeros_like(a)
    _launch(res, launcher, (a, o))
    assert torch.all(o == 3.0)  # cdiv(130,64) = 3


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_gpu_elem_narrow_float_f32_compute(dt):
    """f16/bf16 数学一元：Triton 3.7.1 只接受 fp32/fp64——lowering 经
    cast→f32 计算→cast 回实现；GPU 与 torch（f32 计算 + 一次性舍入）对拍。"""
    src = """import tila


@tila.jit
def eln(a: tila.Tensor[DT, M, N], o: tila.Tensor[DT, M, N],
        BM: tila.constexpr = 64, BN: tila.constexpr = 128):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
    x = tila.load(a, (rm2, rn2), mask=m2)
    tila.store(o, (rm2, rn2), tila.sqrt(x), mask=m2)
"""
    name = "float16" if dt == torch.float16 else "bfloat16"
    src = src.replace("DT", f"tila.{name}")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 64, 128
    a = torch.abs(torch.randn(m, n, device="cuda")).to(dt)
    o = torch.zeros_like(a)
    _launch(res, launcher, (a, o))
    want = torch.sqrt(a.float()).to(dt)
    torch.testing.assert_close(o.float(), want.float(), rtol=1e-2, atol=1e-3)


def test_gpu_exp_f16_roundtrip():
    src = """import tila


@tila.jit
def ef(a: tila.Tensor[tila.float16, M, N], o: tila.Tensor[tila.float16, M, N],
       BM: tila.constexpr = 64, BN: tl_dtype = 128):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
    x = tila.load(a, (rm2, rn2), mask=m2)
    tila.store(o, (rm2, rn2), tila.exp(x), mask=m2)
"""
    src = src.replace("tl_dtype", "tila.constexpr")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n = 64, 128
    a = (torch.randn(m, n, device="cuda") * 2).half()
    o = torch.zeros_like(a)
    _launch(res, launcher, (a, o))
    want = torch.exp(a.float()).half()
    torch.testing.assert_close(o.float(), want.float(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# v0.6a attention fragment（docs/v0.6-attention.md §8.4）：分支六/R-PT/R-BT
# 的 GPU 验收——softmax(QK^T)·V vs torch、双路径互拍
# ---------------------------------------------------------------------------


def _attn_ref(q, k, v):
    return (torch.softmax(q.float() @ k.float().transpose(0, 1), dim=-1)
            @ v.float())


@pytest.mark.parametrize("M,N,D,BN,BD", [
    (1, 1, 64, 64, 64), (63, 63, 64, 64, 64), (64, 64, 64, 64, 64),
    (64, 127, 64, 128, 64), (100, 127, 64, 128, 64), (128, 128, 64, 128, 64),
    (33, 64, 16, 64, 64), (64, 33, 33, 64, 64), (200, 100, 64, 128, 64),
])
def test_gpu_attention_vs_torch(M, N, D, BN, BD):
    """attention fragment vs torch 参考（f16 进出 / f32 内部）。数据含大
    logits（±3e4：exp 安全由构造保证）、全负行、f16 最低值元素。
    注：BN=BD=128 的 dot 组合需 64KB 共享内存（超本机 TITAN Xp 的 48KB
    硬件限）——block 组合受目标 SM 共享内存约束（与手写 Triton 同款），
    典型 flash 配置 BD=64 不受限。"""
    src = (ROOT / "examples/attention.tila").read_text(encoding="utf-8")
    res = compile_kernel(src, {"BN": BN, "BD": BD})
    _, launcher = _load_generated(res)
    rng = np.random.default_rng(M + N * 3 + D)
    q_np = ((rng.standard_normal((M, D)) + 0.3) * 200.0).astype(np.float16)
    k_np = ((rng.standard_normal((N, D)) - 0.2) * 200.0).astype(np.float16)
    q_np[0, 0] = -65504.0
    k_np[N // 2] = -60000.0
    v_np = rng.standard_normal((N, D)).astype(np.float16)
    q = torch.from_numpy(q_np).cuda()
    k = torch.from_numpy(k_np).cuda()
    v = torch.from_numpy(v_np).cuda()
    o = torch.zeros(M, D, device="cuda", dtype=torch.float16)
    _launch(res, launcher, (q, k, v, o), BN=BN, BD=BD)
    want = _attn_ref(q, k, v).half()
    torch.testing.assert_close(o.float(), want.float(), rtol=2e-2, atol=2e-3)


def test_gpu_attention_vs_interpreter():
    """同一 TIR 双路径：attention 的 interpreter == GPU（翻译语义验收）。"""
    src = (ROOT / "examples/attention.tila").read_text(encoding="utf-8")
    res = compile_kernel(src, {"BN": 128, "BD": 64})
    _, launcher = _load_generated(res)
    rng = np.random.default_rng(77)
    m, n, d = 100, 127, 64
    q_np = (rng.standard_normal((m, d)) * 1.5).astype(np.float16)
    k_np = (rng.standard_normal((n, d)) * 3.0).astype(np.float16)
    v_np = rng.standard_normal((n, d)).astype(np.float16)
    o_interp = np.zeros((m, d), dtype=np.float16)
    run_kernel(res.kernel, {"q": q_np, "k": k_np, "v": v_np, "o": o_interp},
               {}, {"BN": 128, "BD": 64})
    q = torch.from_numpy(q_np).cuda()
    k = torch.from_numpy(k_np).cuda()
    v = torch.from_numpy(v_np).cuda()
    o = torch.zeros(m, d, device="cuda", dtype=torch.float16)
    _launch(res, launcher, (q, k, v, o), BN=128, BD=64)
    np.testing.assert_allclose(
        o_interp.astype(np.float32), o.cpu().numpy().astype(np.float32),
        rtol=1e-2, atol=2e-3)


def test_gpu_attention_double_softmax_block_stays_mma():
    """布局轨迹的 GPU 侧健全性：softmax 块两个 dot + 归约/where/除法全链
    编译执行（H3-S/H3-C 的运行时对应物——值正确性）。"""
    src = (ROOT / "examples/attention.tila").read_text(encoding="utf-8")
    res = compile_kernel(src)
    _, launcher = _load_generated(res)
    m, n, d = 64, 64, 64
    q = torch.randn(m, d, device="cuda").half()
    k = torch.randn(n, d, device="cuda").half()
    v = torch.randn(n, d, device="cuda").half()
    o = torch.zeros(m, d, device="cuda", dtype=torch.float16)
    _launch(res, launcher, (q, k, v, o))
    want = _attn_ref(q, k, v).half()
    torch.testing.assert_close(o.float(), want.float(), rtol=2e-2, atol=2e-3)
