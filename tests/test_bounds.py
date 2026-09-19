"""边界安全矩阵：义务四态、mask 证明、契约、逃逸舱（bounds-safety.md）。"""

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError, TilaLaunchContractError

N = ti.Dim("N")
M = ti.Dim("M")
NN = ti.Dim("NN")


def _run1d(kern, n, block, *arrays):
    kern[(ti.cdiv(n, block),)](*arrays, BLOCK=block)


class TestUnmaskedTail:
    def _kernel(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            v = ti.load(x, offs)          # 无 mask 尾块
            ti.store(x, offs, v)
        return k

    def test_rejected_at_launch(self):
        k = self._kernel()
        x = np.ones(100, dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            _run1d(k, 100, 64, x)
        assert ei.value.code == "TILA-BOUNDS-001"
        assert "offs" in str(ei.value) or "pid" in str(ei.value)

    def test_divisible_shape_still_needs_mask_or_contract(self):
        """N 可整除但无契约：符号证明仍不通过（正确行为，§5.1）。"""
        k = self._kernel()
        x = np.ones(128, dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            _run1d(k, 128, 64, x)
        assert ei.value.code == "TILA-BOUNDS-001"


class TestMaskedAccess:
    @staticmethod
    def _kernel():
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m)
            ti.store(x, offs, v, mask=m)
        return k

    def test_proven_and_runs(self):
        k = self._kernel()
        for n in (1, 63, 64, 65, 1000):
            x = np.arange(n, dtype=np.float32) * 2.0
            ref = x.copy()
            _run1d(k, n, 64, x)
            assert np.array_equal(x, ref)

    def test_obligations_all_proven(self):
        k = self._kernel()
        x = np.ones(64, dtype=np.float32)
        _run1d(k, 64, 64, x)
        # 无异常即所有义务 ProvenSafe


class TestWrongDimensionMask:
    def test_axis1_mask_compares_wrong_dim(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (M, NN), ti.ReadWrite],
              BM: ti.Const[int, ti.PowerOfTwo] = 32,
              BN: ti.Const[int, ti.PowerOfTwo] = 32):
            pid_m = ti.program_id(0)
            pid_n = ti.program_id(1)
            rm = pid_m * BM + ti.arange(0, BM)
            rn = pid_n * BN + ti.arange(0, BN)
            # 行 mask 用了 NN（错维）、列 mask 用了 M（错维）
            v = ti.load(x, (rm[:, None], rn[None, :]),
                        mask=(rm < NN)[:, None] & (rn < M)[None, :])
            ti.store(x, (rm[:, None], rn[None, :]), v,
                     mask=(rm < NN)[:, None] & (rn < M)[None, :])

        x = np.ones((64, 64), dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            k[(2, 2)](x, BM=32, BN=32)
        assert ei.value.code == "TILA-BOUNDS-002"
        assert "wrong dimension" in str(ei.value)


class TestLaunchContract:
    def test_assume_launch_unmasked_full_tile(self):
        @ti.assume_launch("N % BLOCK == 0")
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            v = ti.load(x, offs)        # cdiv 战术 + 整除契约 ⇒ SafeUnderContract
            ti.store(x, offs, v)

        x = np.arange(128, dtype=np.float32) + 1.0
        ref = x.copy()
        _run1d(k, 128, 64, x)
        assert np.array_equal(x, ref)

    def test_contract_violation_raises_at_launch(self):
        @ti.assume_launch("N % BLOCK == 0")
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            v = ti.load(x, offs)
            ti.store(x, offs, v)

        x = np.ones(100, dtype=np.float32)   # 100 % 64 != 0
        with pytest.raises(TilaLaunchContractError) as ei:
            _run1d(k, 100, 64, x)
        assert ei.value.code == "TILA-BOUNDS-010"


class TestEscapeHatches:
    def test_unsafe_load_bypasses_proof(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            v = ti.unsafe_load(x, offs)
            ti.store(x, offs, v, mask=offs < N)

        x = np.ones(100, dtype=np.float32)
        with pytest.raises(IndexError, match="bounds check failed"):
            _run1d(k, 100, 64, x)  # 豁免证明不等于让实际越界读取合法化
        assert any(o.kind == "unsafe_load" for o in k.tk.obligations)

    def test_assume_enables_gather(self):
        @ti.jit
        def gather(idx_buf: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
                   data: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                   out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                   BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            idx = ti.load(idx_buf, offs, mask=m, other=0)
            ti.assume((idx >= 0) & (idx < N))
            v = ti.load(data, idx)
            ti.store(out, offs, v, mask=m)

        n = 100
        rng = np.random.default_rng(3)
        idx = rng.integers(0, n, size=n).astype(np.int32)
        data = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        _run1d(gather, n, 64, idx, data, out)
        assert np.allclose(out, data[idx])

    def test_gather_without_assume_rejected(self):
        @ti.jit
        def gather(idx_buf: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
                   data: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                   out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                   BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            idx = ti.load(idx_buf, offs, mask=m, other=0)
            v = ti.load(data, idx)
            ti.store(out, offs, v, mask=m)

        n = 8
        idx = np.zeros(n, dtype=np.int32)
        data = np.ones(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            _run1d(gather, n, 64, idx, data, out)
        assert ei.value.code == "TILA-BOUNDS-001"


class TestConstRefinement:
    def test_power_of_two_violated_at_spec(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m)
            ti.store(x, offs, v, mask=m)

        x = np.ones(64, dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            _run1d(k, 64, 100, x)      # 100 不是 2 的幂
        assert ei.value.code == "TILA-CONST-003"


class TestScalarContracts:
    def test_positive_refinement_violation(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              n: ti.i32 | ti.Positive,
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < n
            v = ti.load(x, offs, mask=m)
            ti.store(x, offs, v, mask=m)

        x = np.ones(16, dtype=np.float32)
        with pytest.raises(TilaLaunchContractError) as ei:
            k[(1,)](x, 0)
        assert ei.value.code == "TILA-TYPE-103"
