"""A9 修复验证：bf16 字面量可表示性 + numpy bf16 绑定（roadmap Phase 0 冻结）。

- representable(·, bf16) 用 f32 高 16 位截断往返判定，不依赖 ml_dtypes；
- runtime._NP_DT 在 ml_dtypes 可用时登记 bfloat16，numpy bf16 数组可绑定；
- widening 格不变（bf16→f32 仍是唯一 bf16 加宽边）。
"""

import numpy as np
import pytest
import ml_dtypes

import tila as ti
from tila import dtypes as D
from tila.errors import TilaError

N = ti.Dim("N")


class TestRepresentableBf16:
    """(c) dtypes 单元级断言。"""

    def test_representable_int_literals(self):
        assert D.representable(0, D.bf16)
        assert D.representable(1, D.bf16)
        assert D.representable(256, D.bf16)

    def test_representable_float_literals(self):
        assert D.representable(1.0, D.bf16)
        assert D.representable(0.5, D.bf16)

    def test_not_representable(self):
        assert not D.representable(0.1, D.bf16)

    def test_other_float_dtypes_unchanged(self):
        # f64/f32/f16 行为不受影响
        assert D.representable(0.1, D.f32)
        assert D.representable(0.1, D.f64)
        assert not D.representable(0.1, D.f16)
        assert D.representable(0, D.f32)
        assert D.representable(0.5, D.f16)

    def test_widening_unchanged(self):
        assert D.can_widen(D.bf16, D.f32)
        assert D.can_widen(D.bf16, D.f64)
        assert not D.can_widen(D.bf16, D.f16)
        assert not D.can_widen(D.f32, D.bf16)


class TestLoadOtherLiteralBf16:
    """(a)/(b) Stage 1（装饰期）行为。

    `other` 必须以字面量形式出现在 kernel 源码里（闭包变量对 kernel
    不可见），故各用例显式内联字面量。
    """

    def test_other_zero_ok(self):
        @ti.jit
        def k(x: ti.Buffer[ti.bf16, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m, other=0)   # A9 repro
            ti.store(x, offs, v, mask=m)

    def test_other_half_ok(self):
        @ti.jit
        def k(x: ti.Buffer[ti.bf16, (N,), ti.ReadWrite],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(x, offs, mask=m, other=0.5)
            ti.store(x, offs, v, mask=m)

    def test_other_tenth_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.bf16, (N,), ti.ReadWrite],
                  BLOCK: ti.Const[int] = 64):
                offs = ti.arange(0, BLOCK)
                m = offs < N
                v = ti.load(x, offs, mask=m, other=0.1)
                ti.store(x, offs, v, mask=m)
        assert ei.value.code == "TILA-TYPE-013"


class TestNumpyBf16Binding:
    """(issue 2) _NP_DT 登记 bfloat16。"""

    def test_np_dt_entry(self):
        from tila.runtime import _NP_DT
        assert _NP_DT[np.dtype(ml_dtypes.bfloat16).name] is D.bf16


class TestBf16InterpAdd:
    """(d) 解释器端到端：bf16 masked add 对照 f32 参考。"""

    def test_bf16_add_end_to_end(self):
        bf16 = ml_dtypes.bfloat16

        @ti.jit
        def add(x: ti.Buffer[ti.bf16, (N,), ti.ReadOnly],
                y: ti.Buffer[ti.bf16, (N,), ti.ReadOnly],
                out: ti.Buffer[ti.bf16, (N,), ti.WriteOnly],
                BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            a = ti.load(x, offs, mask=m)
            b = ti.load(y, offs, mask=m)
            ti.store(out, offs, a + b, mask=m)

        rng = np.random.default_rng(0)
        n = 100                      # 非 BLOCK 整数倍 → mask 生效
        x = (rng.standard_normal(n) * 4.0).astype(np.float32).astype(bf16)
        y = (rng.standard_normal(n) * 4.0).astype(np.float32).astype(bf16)
        out = np.zeros(n, dtype=bf16)

        add[(ti.cdiv(n, 64),)](x, y, out)

        # f32 参考（两 bf16 之和在 f32 中精确）再舍回 bf16 = 正确舍入结果
        ref = (x.astype(np.float32) + y.astype(np.float32)).astype(bf16)
        assert np.array_equal(out.astype(np.float32), ref.astype(np.float32))
