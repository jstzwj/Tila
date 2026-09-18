"""Runtime + interpreter：数值正确性、launch 绑定、契约矩阵。"""

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError, TilaLaunchContractError

N = ti.Dim("N")
M = ti.Dim("M")
Nn = ti.Dim("Nn")
K = ti.Dim("K")


@ti.jit
def matmul_kernel(
    a: ti.Buffer[ti.f16, (M, K), ti.ReadOnly],
    b: ti.Buffer[ti.f16, (K, Nn), ti.ReadOnly],
    c: ti.Buffer[ti.f32, (M, Nn), ti.WriteOnly],
    BM: ti.Const[int, ti.PowerOfTwo] = 64,
    BN: ti.Const[int, ti.PowerOfTwo] = 64,
    BK: ti.Const[int] = 32,
):
    pid_m = ti.program_id(0)
    pid_n = ti.program_id(1)
    rm = pid_m * BM + ti.arange(0, BM)
    rn = pid_n * BN + ti.arange(0, BN)
    rk = ti.arange(0, BK)
    mm_mask = rm < M
    mn_mask = rn < Nn
    acc = ti.zeros((BM, BN), ti.f32)
    for k0 in ti.range(0, K, BK):
        k_mask = (k0 + rk) < K
        at = ti.load(a, (rm[:, None], (k0 + rk)[None, :]),
                     mask=mm_mask[:, None] & k_mask[None, :])
        bt = ti.load(b, ((k0 + rk)[:, None], rn[None, :]),
                     mask=k_mask[:, None] & mn_mask[None, :])
        acc = ti.dot(at, bt, acc=acc)
    ti.store(c, (rm[:, None], rn[None, :]), acc,
             mask=mm_mask[:, None] & mn_mask[None, :])


class TestVectorOps:
    def _add(self):
        @ti.jit
        def add(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                y: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                BLOCK: ti.Const[int, ti.PowerOfTwo] = 128):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            b = ti.load(y, offs, mask=m)
            ti.store(out, offs, a + b, mask=m)
        return add

    @pytest.mark.parametrize("n", [1, 31, 128, 129, 1000, 4096])
    def test_add_matches_numpy(self, n):
        add = self._add()
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        add[(ti.cdiv(n, 128),)](x, y, out, BLOCK=128)
        assert np.allclose(out, x + y)

    def test_saxpy_scalar_param(self):
        @ti.jit
        def saxpy(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                  y: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                  alpha: ti.f32,
                  BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m)
            ti.store(y, offs, v * alpha, mask=m)

        n = 200
        rng = np.random.default_rng(7)
        x = rng.standard_normal(n).astype(np.float32)
        y = np.zeros(n, dtype=np.float32)
        saxpy[(ti.cdiv(n, 64),)](x, y, 2.5, BLOCK=64)
        assert np.allclose(y, x * 2.5)

    def test_explicit_dim_scalar_checked_against_tensor(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              N: ti.i32,
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m)
            ti.store(x, offs, v, mask=m)

        x = np.ones(64, dtype=np.float32)
        # 显式传入与张量绑定一致的 N：通过
        k[(1,)](x, 64)
        # 不一致：拒绝
        with pytest.raises(TilaLaunchContractError):
            k[(1,)](x, 32)


class TestMatmul:
    @pytest.mark.parametrize("m,n,k", [
        (64, 64, 32), (128, 64, 96), (33, 17, 100),   # 非整除 M/N/K
        (1, 1, 1), (65, 65, 33),
    ])
    def test_matmul_matches_numpy(self, m, n, k):
        rng = np.random.default_rng(m * 1000 + n + k)
        a = rng.standard_normal((m, k)).astype(np.float16)
        b = rng.standard_normal((k, n)).astype(np.float16)
        c = np.zeros((m, n), dtype=np.float32)
        matmul_kernel[(ti.cdiv(m, 64), ti.cdiv(n, 64))](
            a, b, c, BM=64, BN=64, BK=32)
        ref = a.astype(np.float32) @ b.astype(np.float32)
        assert np.allclose(c, ref, atol=2e-2)

    def test_transposed_b_operand_strides(self):
        """步长绑定：非连续 b（转置）按实际 stride 寻址。"""
        rng = np.random.default_rng(11)
        m, k, n = 64, 96, 64
        a = rng.standard_normal((m, k)).astype(np.float16)
        b_t = rng.standard_normal((n, k)).astype(np.float16)
        b = b_t.T                                  # (k, n) 视图，非连续
        c = np.zeros((m, n), dtype=np.float32)
        matmul_kernel[(1, 1)](a, b, c, BM=64, BN=64, BK=32)
        ref = a.astype(np.float32) @ b.astype(np.float32)
        assert np.allclose(c, ref, atol=2e-2)


class TestSoftmaxReduceWhere:
    def test_softmax_rowwise(self):
        @ti.jit
        def softmax(x: ti.Buffer[ti.f16, (M, Nn), ti.ReadOnly],
                    out: ti.Buffer[ti.f16, (M, Nn), ti.WriteOnly],
                    BN: ti.Const[int, ti.PowerOfTwo] = 256):
            pid = ti.program_id(0)                 # 行号（标量坐标）
            offs = ti.arange(0, BN)
            m = offs < Nn
            # max 归约的零元：masked lane 取 f16 最低值 -65504
            # （neg_inf 模式——否则全负行的 max 会被 other=0 污染）
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
        softmax[(m,)](x, out, BN=256)
        ref = np.exp(x.astype(np.float32) -
                     x.astype(np.float32).max(1, keepdims=True))
        ref /= ref.sum(1, keepdims=True)
        assert np.allclose(out.astype(np.float32), ref, atol=2e-3)


class TestWhereAndEffects:
    def test_where_semantics(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m)
            w = ti.where(m, v, 1.0)
            ti.store(out, offs, w, mask=m)

        n = 100
        x = np.arange(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        k[(ti.cdiv(n, 64),)](x, out, BLOCK=64)
        assert np.array_equal(out, x)

    def test_load_in_where_warns(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              y: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            b = ti.load(y, offs, mask=m)
            w = ti.where(m, a, b)     # 无效应版（都先 load）——这里不告警
            ti.store(out, offs, w, mask=m)

        # 真正的告警场景：where 分支里内联 load
        @ti.jit
        def k2(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
               y: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
               out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
               BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            w = ti.where(m, ti.load(x, offs, mask=m), ti.load(y, offs, mask=m))
            ti.store(out, offs, w, mask=m)

        codes = [w.code for w in k2.tk.warnings]
        assert "TILA-EFFECT-007" in codes


class TestPtrForm:
    def test_ptr_form_roundtrip(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            p = x.ptr                        # Buffer → Ptr（region = N）
            q = p + offs                     # 元素 offset
            v = ti.load(q, mask=m)
            ti.store(q, v * 2.0, mask=m)

        n = 100
        x = np.arange(n, dtype=np.float32)
        k[(ti.cdiv(n, 64),)](x, BLOCK=64)
        assert np.array_equal(x, np.arange(n, dtype=np.float32) * 2.0)

    def test_store_through_readptr_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                  BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
                pid = ti.program_id(0)
                offs = pid * BLOCK + ti.arange(0, BLOCK)
                m = offs < N
                p = x.ptr
                q = p + offs
                ti.store(q, 1.0, mask=m)
        assert ei.value.code == "TILA-MEM-001"


class TestLaunchChecks:
    def test_dtype_mismatch(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            offs = ti.arange(0, BLOCK)
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=offs < N)

        x = np.zeros(64, dtype=np.float64)     # 声明 f32，传入 f64
        with pytest.raises(TilaLaunchContractError) as ei:
            k[(1,)](x)
        assert ei.value.code == "TILA-TYPE-101"

    def test_shape_symbol_conflict(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (M, Nn), ti.WriteOnly],
              y: ti.Buffer[ti.f32, (M, Nn), ti.WriteOnly],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 8):
            rm = ti.arange(0, BLOCK)
            rn = ti.arange(0, BLOCK)
            ti.store(x, (rm[:, None], rn[None, :]),
                     ti.zeros((BLOCK, BLOCK), ti.f32),
                     mask=(rm < M)[:, None] & (rn < Nn)[None, :])
            ti.store(y, (rm[:, None], rn[None, :]),
                     ti.zeros((BLOCK, BLOCK), ti.f32),
                     mask=(rm < M)[:, None] & (rn < Nn)[None, :])

        x = np.zeros((4, 8), dtype=np.float32)   # M 绑定 4
        y = np.zeros((8, 4), dtype=np.float32)   # M 绑定 8 —— 同符号冲突
        with pytest.raises(TilaLaunchContractError) as ei:
            k[(1, 1)](x, y)
        assert ei.value.code == "TILA-TYPE-102"

    def test_alignment_declaration(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite, 16],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m)
            ti.store(x, offs, v, mask=m)

        buf = np.zeros(130, dtype=np.float32)
        x = buf[1:]                     # data_ptr 偏移 4B → 不满足 Aligned[16]
        with pytest.raises(TilaLaunchContractError) as ei:
            k[(ti.cdiv(len(x), 64),)](x, BLOCK=64)
        assert ei.value.code == "TILA-MEM-003"

    def test_kernel_call_without_grid_hint(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            pass
        with pytest.raises(TilaError) as ei:
            k(None)
        assert ei.value.code == "TILA-SYN-060"
