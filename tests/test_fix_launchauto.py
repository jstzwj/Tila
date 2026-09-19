"""launch_auto：grid 由 launch analysis 推导（surface-language.md §6）。

推导规则（runtime._derive_grid）：
- 使用中的 pid 轴按义务的规范模式取 grid = ceildiv(bound, step)：
  blocked（pid*STEP+lane）与标量坐标（纯 pid，STEP=1）两种形态；
- 推导出的 (bound, step) 直接登记为 grid 事实（无需数值回配），
  供 cdiv 战术 / 精确维战术证明义务；
- 未用轴 = 1；秩 = 最高使用轴 + 1（最少 1）；
- 无匹配义务 / 多义务不一致 → TILA-TYPE-105（要求显式 grid）。
"""

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError, TilaLaunchContractError

from test_runtime_interp import matmul_kernel

N = ti.Dim("N")
M = ti.Dim("M")
Nn = ti.Dim("Nn")


@ti.jit
def add_kernel(
    x:   ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    y:   ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    BLOCK: ti.Const[int, ti.PowerOfTwo] = 128,
):
    pid = ti.program_id(0)
    offs = pid * BLOCK + ti.arange(0, BLOCK)
    m = offs < N
    a = ti.load(x, offs, mask=m)
    b = ti.load(y, offs, mask=m)
    ti.store(out, offs, a + b, mask=m)


@ti.jit
def grid1_probe(
    out:   ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    BLOCK: ti.Const[int, ti.PowerOfTwo] = 64,
):
    pid = ti.program_id(0)
    offs = pid * BLOCK + ti.arange(0, BLOCK)
    m = offs < N
    v = ti.cast[ti.f32](ti.num_programs(0)) + ti.zeros((BLOCK,), ti.f32)
    ti.store(out, offs, v, mask=m)


class TestLaunchAuto1D:
    def test_add_canonical(self):
        """(a) 1D 规范 add：结果正确且推导 grid = ceildiv(N, BLOCK)。"""
        n = 1000
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        add_kernel.launch_auto(x, y, out, BLOCK=64)
        assert np.allclose(out, x + y)
        # strict 模式下义务全体可证（无 TILA-BOUNDS-001 即已证明；
        # 报告再确认四态里没有 Unknown）
        assert all(result.verdict == "ProvenSafe"
                   for _, result in add_kernel.last_proof_results)
        assert "Unknown" not in add_kernel.last_report

    def test_derived_grid_exact_value(self):
        n = 1000
        out = np.zeros(n, dtype=np.float32)
        grid1_probe.launch_auto(out, BLOCK=64)
        assert out[0] == -(-1000 // 64)          # = 16
        assert np.all(out[:n] == out[0])         # 每个 program 看到同一 grid

    def test_getitem_none_is_the_auto_channel(self):
        """kernel[None] 是 launch_auto 的内部等价通道。"""
        n = 128
        x = np.ones(n, dtype=np.float32)
        y = np.ones(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        add_kernel[None](x, y, out, BLOCK=128)
        assert np.allclose(out, x + y)

    def test_unused_pid_grid_is_one(self):
        n = 100
        out = np.zeros(n, dtype=np.float32)
        # 无 pid 使用的 kernel：秩最少 1、grid[0] = 1

        @ti.jit
        def nopid(out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                  BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            ti.store(out, offs, ti.zeros((BLOCK,), ti.f32) + 1.0, mask=m)

        nopid.launch_auto(out, BLOCK=64)
        assert np.all(out[:64] == 1.0)           # 单 program 覆盖首块
        assert out[64:].sum() == 0.0

    def test_syn060_error_mentions_launch_auto(self):
        x = np.zeros(4, dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            add_kernel(x, x, x)
        assert ei.value.code == "TILA-SYN-060"
        assert "launch_auto" in str(ei.value)


class TestLaunchAutoContract:
    def test_unmasked_full_tile_under_assume_launch(self):
        """(b) 无 mask 完整 tile + assume_launch：推导 grid 事实喂给
        cdiv 战术 → SafeUnderContract（而非 TILA-BOUNDS-001）。"""

        @ti.assume_launch("N % BLOCK == 0")
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            v = ti.load(x, offs)
            ti.store(x, offs, v)

        x = np.arange(128, dtype=np.float32) + 1.0
        ref = x.copy()
        k.launch_auto(x, BLOCK=64)
        assert np.array_equal(x, ref)
        assert "SafeUnderContract" in k.last_report
        assert "Unknown" not in k.last_report


class TestLaunchAuto2D:
    def test_matmul_per_axis(self):
        """(c) 模块级 matmul_kernel：逐轴推导 grid = (cdiv(M,BM),
        cdiv(Nn,BN))，数值对齐 numpy（非整除维）。"""
        m, k, n = 33, 100, 17
        rng = np.random.default_rng(m * 1000 + n + k)
        a = rng.standard_normal((m, k)).astype(np.float16)
        b = rng.standard_normal((k, n)).astype(np.float16)
        c = np.zeros((m, n), dtype=np.float32)
        matmul_kernel.launch_auto(a, b, c, BM=16, BN=8, BK=32)
        ref = a.astype(np.float32) @ b.astype(np.float32)
        assert np.allclose(c, ref, atol=2e-2)

    def test_derived_2d_grid_exact_values(self):
        @ti.jit
        def grid2_probe(out: ti.Buffer[ti.f32, (M, Nn), ti.WriteOnly],
                        BM: ti.Const[int, ti.PowerOfTwo] = 16,
                        BN: ti.Const[int, ti.PowerOfTwo] = 16):
            pid_m = ti.program_id(0)
            pid_n = ti.program_id(1)
            rm = pid_m * BM + ti.arange(0, BM)
            rn = pid_n * BN + ti.arange(0, BN)
            m = (rm < M)[:, None] & (rn < Nn)[None, :]
            v = ti.cast[ti.f32](ti.num_programs(0) * 100 +
                                ti.num_programs(1))
            ti.store(out, (rm[:, None], rn[None, :]),
                     v + ti.zeros((BM, BN), ti.f32), mask=m)

        m, n = 33, 17
        out = np.zeros((m, n), dtype=np.float32)
        grid2_probe.launch_auto(out, BM=16, BN=16)
        assert np.all(out == 100 * 3 + 2)         # grid = (3, 2)


class TestLaunchAutoScalarRow:
    def test_softmax_scalar_pid(self):
        """(d) 标量行坐标（softmax：axis0 坐标恰为 pid）→ STEP=1，
        grid = M。"""

        @ti.jit
        def softmax(x: ti.Buffer[ti.f16, (M, Nn), ti.ReadOnly],
                    out: ti.Buffer[ti.f16, (M, Nn), ti.WriteOnly],
                    BN: ti.Const[int, ti.PowerOfTwo] = 256):
            pid = ti.program_id(0)                 # 行号（标量坐标）
            offs = ti.arange(0, BN)
            m = offs < Nn
            v = ti.load(x, (pid, offs), mask=m, other=-65504.0)
            vf = ti.cast[ti.f32](v)
            mx = ti.max(vf, 0)
            e = ti.exp(vf - mx)
            s = ti.sum(e, 0)
            r = e / s
            ti.store(out, (pid, offs), ti.cast[ti.f16](r), mask=m)

        m, n = 8, 200
        rng = np.random.default_rng(5)
        x = rng.standard_normal((m, n)).astype(np.float16)
        out = np.zeros_like(x)
        softmax.launch_auto(x, out, BN=256)
        ref = np.exp(x.astype(np.float32) -
                     x.astype(np.float32).max(1, keepdims=True))
        ref /= ref.sum(1, keepdims=True)
        assert np.allclose(out.astype(np.float32), ref, atol=2e-3)

    def test_scalar_row_grid_exact_value(self):
        @ti.jit
        def row_probe(out: ti.Buffer[ti.f32, (M, Nn), ti.WriteOnly],
                      BN: ti.Const[int, ti.PowerOfTwo] = 256):
            pid = ti.program_id(0)                 # 标量行坐标
            offs = ti.arange(0, BN)
            m = offs < Nn
            v = ti.cast[ti.f32](ti.num_programs(0)) + ti.zeros((BN,), ti.f32)
            ti.store(out, (pid, offs), v, mask=m)

        m, n = 8, 200
        out = np.zeros((m, n), dtype=np.float32)
        row_probe.launch_auto(out, BN=256)
        assert np.all(out == m)                    # grid = (M,) = (8,)


class TestLaunchAutoNegative:
    def _kernel(self):
        @ti.jit
        def k(out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = ti.arange(0, BLOCK)
            m = offs < N
            z = ti.zeros((BLOCK,), ti.f32)
            ti.store(out, offs, ti.cast[ti.f32](pid) + z, mask=m)
        return k

    def test_pid_only_in_non_access_computation(self):
        """(e) pid 只进入存储值、不进入任何访问坐标 → 无匹配义务，
        launch_auto 报清晰错误。"""
        k = self._kernel()
        out = np.zeros(100, dtype=np.float32)
        with pytest.raises(TilaLaunchContractError) as ei:
            k.launch_auto(out, BLOCK=64)
        assert ei.value.code == "TILA-TYPE-105"
        assert "axis 0" in str(ei.value)
        assert "grid" in str(ei.value)

    def test_same_kernel_explicit_grid_still_works(self):
        k = self._kernel()
        out = np.zeros(128, dtype=np.float32)
        k[(2,)](out, BLOCK=64)
        # offs 不含 pid：两个 program 都写 0..63，后者（pid=1）覆盖
        assert np.all(out[:64] == 1.0)
        assert np.all(out[64:] == 0.0)            # 64..127 从未被触碰

    def test_conflicting_obligations_fail_derivation(self):
        """同轴义务 (bound, step) 不一致 → 推导失败。"""

        @ti.jit
        def k(x: ti.Buffer[ti.f32, (M,), ti.ReadWrite],
              y: ti.Buffer[ti.f32, (Nn,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            mx = offs < M
            my = offs < Nn
            vx = ti.load(x, offs, mask=mx)
            ti.store(x, offs, vx, mask=mx)
            vy = ti.load(y, offs, mask=my)
            ti.store(y, offs, vy, mask=my)

        x = np.ones(128, dtype=np.float32)
        y = np.ones(256, dtype=np.float32)
        with pytest.raises(TilaLaunchContractError) as ei:
            k.launch_auto(x, y, BLOCK=64)
        assert ei.value.code == "TILA-TYPE-105"
        assert "disagree" in str(ei.value)
