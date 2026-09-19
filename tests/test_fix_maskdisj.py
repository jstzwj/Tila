"""Mask 组合语义回归；M2-02 已迁移到共享 DAG，无 DNF 展开。

快速路径仍保守处理一般否定；当前 assume 公开子集仍限符号合取。
"""

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError

N = ti.Dim("N")


def _run1d(kern, n, block, *arrays):
    kern[(ti.cdiv(n, block),)](*arrays, BLOCK=block)


class TestMaskDisjunction:
    def test_disjunction_of_proving_masks_runs(self):
        """(a) 两个析取支都独立证明 offs < N ⇒ 通过，且 interp 结果正确。"""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m1 = offs < N
            m2 = offs < N
            v = ti.load(x, offs, mask=m1 | m2)
            ti.store(x, offs, v, mask=m1 | m2)

        for n in (1, 63, 64, 65, 100):
            x = np.arange(n, dtype=np.float32) * 2.0
            ref = x.copy()
            _run1d(k, n, 64, x)
            assert np.array_equal(x, ref)

    def test_one_unprovable_disjunct_rejected(self):
        """(b) 仅一支可证（offs < 1000 不是目标谓词）⇒ TILA-BOUNDS-001。"""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m1 = offs < N
            v = ti.load(x, offs, mask=m1 | (offs < 1000))
            ti.store(x, offs, v, mask=m1)

        x = np.ones(100, dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            _run1d(k, 100, 64, x)
        assert ei.value.code == "TILA-BOUNDS-001"


class TestMaskConjunctionRegression:
    def test_conjunctive_masks_still_prove(self):
        """(c) `&` 组合回归：合取 mask 仍可证（镜像既有 & 用例）。"""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m1 = offs < N
            m2 = offs < N
            v = ti.load(x, offs, mask=m1 & m2)
            ti.store(x, offs, v, mask=m1 & m2)

        for n in (1, 63, 100):
            x = np.arange(n, dtype=np.float32) + 1.0
            ref = x.copy()
            _run1d(k, n, 64, x)
            assert np.array_equal(x, ref)

    def test_conjunction_with_unprovable_extra_still_proves(self):
        """合取子句携带额外（非目标）谓词时目标谓词仍在场 ⇒ 可证。"""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            v = ti.load(x, offs, mask=(offs < N) & (offs < 1000))
            ti.store(x, offs, v, mask=offs < N)

        x = np.arange(100, dtype=np.float32) + 1.0
        ref = x.copy()
        _run1d(k, 100, 64, x)
        assert np.array_equal(x, ref)


class TestMaskNegationConservative:
    def test_negated_mask_proved_by_default_smt(self):
        """M2-03: ~(offs >= N) proves the upper bound via Z3."""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            v = ti.load(x, offs, mask=~(offs >= N))
            ti.store(x, offs, v, mask=~(offs >= N))

        x = np.ones(100, dtype=np.float32)
        _run1d(k, 100, 64, x)
        assert np.array_equal(x, np.ones(100, dtype=np.float32))

    def test_negated_mask_with_contract_route_proves(self):
        """(d') ~ 保守性不影响其他证明路径：整除契约下无 mask 访问仍可证。"""
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
        _run1d(k, 128, 64, x)
        assert np.array_equal(x, ref)


class TestAssumeDisjunctionRejected:
    def test_assume_over_disjunction_rejected(self):
        """assume 的当前公开子集不接受析取。"""
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(idx_buf: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
                  data: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                  BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
                pid = ti.program_id(0)
                offs = pid * BLOCK + ti.arange(0, BLOCK)
                m = offs < N
                idx = ti.load(idx_buf, offs, mask=m, other=0)
                ti.assume((idx >= 0) | (idx < N))  # 析取：不可全局假定
                v = ti.load(data, idx)
                ti.store(data, offs, v, mask=m)
        assert ei.value.code == "TILA-CONST-004"

    def test_assume_negated_mask_rejected(self):
        """assume(~m) 仍在当前公开子集之外。"""
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
                  BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
                pid = ti.program_id(0)
                offs = pid * BLOCK + ti.arange(0, BLOCK)
                m = offs < N
                ti.assume(~m)
                v = ti.load(x, offs, mask=m)
                ti.store(x, offs, v, mask=m)
        assert ei.value.code == "TILA-CONST-004"
