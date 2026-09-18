"""A5 修复验证：Const 算术在 checker 内常量折叠（docs/type-system.md §5.1）。

HALF * 2 之类的 Const 表达式必须折叠为编译期 int（arange 界、zeros
形状、ti.range 步长）；运行期值仍以 TILA-CONST-001 拒绝；未定值的
Const 参数保持符号路径（数值留待特化期）。
"""

import numpy as np
import pytest

import tila as ti
from tila import tir as T
from tila.dims import Cst
from tila.errors import TilaError

N = ti.Dim("N")
SIZE = 32          # 模块级 int 常量：frontend 直接脱糖为 Lit


# (a) 文档示例：ti.arange(0, HALF * 2) 折叠为 128 --------------------------

def test_arange_const_expr_folded():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          HALF: ti.Const[int] = 64):
        offs = ti.arange(0, HALF * 2)
        m = offs < N
        ti.store(x, offs, ti.zeros((128,), ti.f32), mask=m)

    vt = k.tk.types["offs"]
    assert vt.dims == (Cst(128),)            # Block 形状折叠为 (128,)
    assert vt.describe() == "Block[i32, (128)]"
    src, dump = k.materialize()
    assert "tl.arange(0, 128)" in src        # lowering 输出整数界
    assert "arange(0, 128)" in dump          # TIR dump 同样折叠


# (b) zeros 形状维度折叠 ----------------------------------------------------

def test_zeros_shape_const_expr_folded():
    @ti.jit
    def k(HALF: ti.Const[int] = 64):
        v = ti.zeros((HALF * 2, 4), ti.f32)

    assert k.tk.types["v"].dims == (Cst(128), Cst(4))
    src, _ = k.materialize()
    assert "tl.zeros((128, 4), tl.float32)" in src


# (c) ti.range 步长折叠 -----------------------------------------------------

def test_range_step_const_expr_folded():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], K: ti.i32,
          HALF: ti.Const[int] = 64):
        offs = ti.arange(0, HALF * 2)
        acc = ti.zeros((HALF * 2,), ti.f32)
        for i in ti.range(0, K, HALF // 32):
            acc = acc + 1.0
        m = offs < N
        ti.store(x, offs, acc, mask=m)

    fors = [s for s in k.tk.body if isinstance(s, T.TFor)]
    assert len(fors) == 1
    assert isinstance(fors[0].step, T.TLit) and fors[0].step.value == 2


def test_range_step_literal_and_const_name_unchanged():
    """回归控制：字面量步长与裸 Const 名步长不得因折叠改动。"""

    @ti.jit
    def k(K: ti.i32, BK: ti.Const[int] = 8):
        for i in ti.range(0, K, 2):
            pass
        for j in ti.range(0, K, BK):
            pass

    fors = [s for s in k.tk.body if isinstance(s, T.TFor)]
    assert isinstance(fors[0].step, T.TLit) and fors[0].step.value == 2
    assert isinstance(fors[1].step, T.TName) and fors[1].step.name == "BK"
    assert any(d.get("kind") == "step" and d.get("sym") == "BK"
               for d in k.tk.deferred)


# (d) 负控制：运行期值仍拒绝；未知 Const 保持符号 --------------------------

def test_arange_runtime_scalar_still_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(n: ti.i32):
            offs = ti.arange(0, n)
    assert ei.value.code == "TILA-CONST-001"


def test_arange_const_times_runtime_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], n: ti.i32,
              HALF: ti.Const[int] = 64):
            offs = ti.arange(0, HALF * n)
    assert ei.value.code == "TILA-CONST-001"


def test_arange_symbolic_const_expr_without_default():
    """无默认值的 Const：stage 1 符号通过，形状携带符号乘积。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          HALF: ti.Const[int]):
        offs = ti.arange(0, HALF * 2)

    d0 = k.tk.types["offs"].dims[0]
    assert "HALF" in str(d0) and str(d0) == "2*HALF"
    src, dump = k.materialize({"HALF": 128})    # 特化期可代入
    assert "tl.arange(0, (HALF * 2))" in src


# 模块级 int 常量同样折叠（frontend 已脱糖为 Lit） -------------------------

def test_module_level_const_arithmetic_folded():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
        offs = ti.arange(0, SIZE * 4)
        m = offs < N
        ti.store(x, offs, ti.zeros((SIZE * 4,), ti.f32), mask=m)

    assert k.tk.types["offs"].dims == (Cst(128),)
    src, _ = k.materialize()
    assert "tl.arange(0, 128)" in src


# (e) 数值运行：折叠路径在解释器上执行 ------------------------------------

def test_numeric_run_masked_store():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          HALF: ti.Const[int] = 64):
        offs = ti.arange(0, HALF * 2)          # 128 lanes
        m = offs < N                           # N = 100：尾部 lane 屏蔽
        ti.store(x, offs, ti.cast[ti.f32](offs), mask=m)

    x = np.full(100, -1.0, dtype=np.float32)
    k[(1,)](x)
    # 若 mask 丢失，lane 100..127 会被裁剪到 99 并覆盖 x[99]（=127.0）
    assert list(x) == [float(i) for i in range(100)]
