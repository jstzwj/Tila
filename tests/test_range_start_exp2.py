"""运行期起点 range（intrinsics.md §2.1 扩展）与 exp2 内建的验证。

参考 Triton 官方 fused-attention tutorial 的结构：
  - 两段 KV 循环 lo/hi 由 pid 推导（运行期起点）；
  - exp2 域计算（FlashAttention 范式）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

import tila as ti
from tila import tir as T
from tila.dims import Cst
from tila.errors import TilaError

N = ti.Dim("N")


# ---------------------------------------------------------------------------
# 运行期起点 range
# ---------------------------------------------------------------------------

def test_runtime_start_runs_and_lowers():
    """pid 推导的起点：TIR 携带 start，lowering 生成 range(start, ...)，
    interp 按起点迭代。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          BLOCK: ti.Const[int, ti.PowerOfTwo] = 32):
        pid = ti.program_id(0)
        offs = pid * BLOCK + ti.arange(0, BLOCK)
        m = offs < N
        # 从本块起点开始步进 16：2 次迭代，起点为运行期值；
        # 存块形状须与访问精确一致 → 标量广播成块
        for i in ti.range(pid * BLOCK, pid * BLOCK + BLOCK, 16):
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) +
                     ti.cast[ti.f32](i), mask=m)

    fors = [s for s in k.tk.body if isinstance(s, T.TFor)]
    assert fors[0].start is not None

    x = np.zeros(64, dtype=np.float32)
    k[(2,)](x, BLOCK=32)
    # 块 0 迭代 i=0,16（后写覆盖）；块 1 迭代 i=32,48
    assert x[0] == 16.0 and x[31] == 16.0
    assert x[32] == 48.0 and x[63] == 48.0

    src, _ = k.materialize({"BLOCK": 32})
    assert "range((pid * BLOCK), " in src


def test_zero_start_has_no_start_operand():
    """字面量 0 起点：既有形态完全不变（TFor.start 为 None）。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
        for i in ti.range(0, N, 8):
            pass

    fors = [s for s in k.tk.body if isinstance(s, T.TFor)]
    assert fors[0].start is None
    src, _ = k.materialize()
    assert "range(0, " in src


def test_non_int_start_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              s: ti.f32):
            for i in ti.range(s, N, 8):
                pass
    assert ei.value.code == "TILA-TYPE-024"


def test_runtime_start_bounds_still_provable():
    """运行期起点循环内的掩码访问：义务仍由 mask 谓词证明。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
          BLOCK: ti.Const[int, ti.PowerOfTwo] = 32):
        pid = ti.program_id(0)
        base = pid * BLOCK
        offs = ti.arange(0, 16)
        for i in ti.range(base, base + BLOCK, 16):
            rv = i + offs
            v = ti.load(x, rv, mask=rv < N)
            ti.store(x, rv, v, mask=rv < N)

    # 全义务可证（无 TILA-BOUNDS-001）即通过；load/store 往返保持原值
    x = np.ones(64, dtype=np.float32)
    k[(2,)](x, BLOCK=32)
    assert np.allclose(x, 1.0)


# ---------------------------------------------------------------------------
# exp2
# ---------------------------------------------------------------------------

def test_exp2_interp_and_lowering():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
        offs = ti.arange(0, 16)
        v = ti.zeros((16,), ti.f32) + 2.0
        m = offs < N
        ti.store(x, offs, ti.exp2(v), mask=m)

    x = np.zeros(16, dtype=np.float32)
    k[(1,)](x)
    assert np.allclose(x, 4.0)

    src, _ = k.materialize()
    assert "tl.math.exp2(" in src


def test_exp2_int_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            offs = ti.arange(0, 16)
            ti.store(x, offs, ti.exp2(offs), mask=offs < N)
    assert ei.value.code == "TILA-TYPE-033"


def test_exp2_shape_and_dtype_preserved():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
        offs = ti.arange(0, 16)
        m = offs < N
        v = ti.zeros((16,), ti.f32) - 1.0
        y = ti.exp2(v)          # f32 (16,) → 同构
        ti.store(x, offs, y, mask=m)

    assert k.tk.types["y"].dims == (Cst(16),)
    x = np.zeros(16, dtype=np.float32)
    k[(1,)](x)
    assert np.allclose(x, 0.5)
