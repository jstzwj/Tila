"""A3 修复验证：mask.any()/mask.all() 内建（type-system.md §3.3、
intrinsics.md §2.5）。

- frontend：m.any()/m.all() 方法形态重写为普通调用；别名形态 ti.any(m)、
  带参形态 m.any(1, 2) 仍拒绝；
- checker：单参数 Mask → ScalarT(bool)（lane 归约，无符号 expr/谓词）；
  非 Mask 操作数 → TILA-TYPE-019；
- lowering：tl.sum(tl.cast(...)) 归约表达式（tl.sum 无 axis 归约到标量）；
- interp：np.any/np.all，repro 模式在 runtime-if 上端到端跑通。
"""

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError

N = ti.Dim("N")


def _any_kernel():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          BLOCK: ti.Const[int] = 64):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        if m.any():                       # A3 repro：此前 TILA-SYN-036
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
    return k


def _all_kernel():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          BLOCK: ti.Const[int] = 32):
        offs = ti.arange(0, BLOCK)
        m = offs < 32
        if m.all():                       # BLOCK=32：谓词全真
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=offs < N)
    return k


def _all_gate_kernel():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          BLOCK: ti.Const[int] = 32):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        if m.all():
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
    return k


class TestStage1AndInterp:
    """(a) repro 通过 stage 1 并在 numpy 解释器上运行。"""

    def test_any_stage1_typechecks(self):
        k = _any_kernel()                 # 装饰期检查通过（不抛错即通过）
        assert k.tk is not None

    def test_any_true_stores(self):
        k = _any_kernel()
        x = np.full(8, 5.0, dtype=np.float32)     # N=8 < BLOCK：any() 为真
        k[(1,)](x)
        assert (x == 0).all()                     # store 发生

    def test_any_false_skips_store(self):
        k = _any_kernel()
        x = np.full(0, 5.0, dtype=np.float32)     # N=0：any() 为假
        k[(1,)](x)                                # 正常运行，不触碰内存
        assert x.shape == (0,)

    def test_any_result_is_scalar_bool_in_boolop(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              BLOCK: ti.Const[int] = 32):
            offs = ti.arange(0, BLOCK)
            m = offs < N
            if m.any() and m.all():       # 标量 bool 可进 and/or
                ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)

        x = np.full(32, 5.0, dtype=np.float32)    # N=32：any 且 all 均真
        k[(1,)](x)
        assert (x == 0).all()


class TestAll:
    """(b) m.all() 同样可用（store 由 all() 门控）。"""

    def test_all_true_stores(self):
        k = _all_kernel()
        x = np.full(64, 5.0, dtype=np.float32)    # N=64, BLOCK=32：offs<32 全真
        k[(1,)](x)
        assert (x[:32] == 0).all()                # 门开 → store 发生
        assert (x[32:] == 5.0).all()              # 访问只覆盖前 32

    def test_all_false_skips_store(self):
        k = _all_gate_kernel()
        x = np.full(16, 7.0, dtype=np.float32)    # N=16 < BLOCK=32：非全真
        k[(1,)](x)
        assert (x == 7.0).all()                   # 门关 → store 未执行


class TestLowering:
    """(c) materialize() 产出 tl.sum(tl.cast(...)) 归约表达式。"""

    def test_any_lowering(self):
        src, _dump = _any_kernel().materialize()
        assert "tl.sum(tl.cast(" in src
        assert "> 0" in src
        assert "tl.arange(0, BLOCK)" in src       # 操作数是 mask 表达式

    def test_all_lowering(self):
        src, _dump = _all_kernel().materialize()
        assert "tl.sum(tl.cast(~" in src          # ~mask 求和 == 0
        assert "== 0" in src


class TestNegative:
    """(d) 拒绝路径保持不变。"""

    def test_method_with_args_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                  BLOCK: ti.Const[int] = 64):
                offs = ti.arange(0, BLOCK)
                m = offs < N
                if m.any(1, 2):           # 方法形态仅限零参
                    ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
        assert ei.value.code == "TILA-SYN-036"

    def test_alias_call_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                  BLOCK: ti.Const[int] = 64):
                offs = ti.arange(0, BLOCK)
                m = offs < N
                if ti.any(m):             # 文档只定义方法形态 m.any()
                    ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
        assert ei.value.code == "TILA-SYN-030"

    def test_float_block_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                  BLOCK: ti.Const[int] = 64):
                offs = ti.arange(0, BLOCK)
                v = ti.zeros((BLOCK,), ti.f32)
                if v.any():               # float 块不是 Mask
                    ti.store(x, offs, v, mask=offs < N)
        assert ei.value.code == "TILA-TYPE-019"

    def test_scalar_bool_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
                  BLOCK: ti.Const[int] = 64):
                offs = ti.arange(0, BLOCK)
                flag = N > 0              # 标量 bool：应直接用于 if
                if flag.any():
                    ti.store(x, offs, ti.zeros((BLOCK,), ti.f32),
                             mask=offs < N)
        assert ei.value.code == "TILA-TYPE-019"
