"""裸 return（提前退出）端到端：frontend / checker / tir / lowering / interp。

docs/surface-language.md §2.1 允许表提及 return；v0 语义：
- 裸 `return` 合法 = 提前退出当前 program instance；
- `return <value>` 拒绝（TILA-SYN-021：Triton kernel 不向 host 返回值，
  建议写入输出 buffer）；
- checker 跳过 return 之后的不可达语句（死代码中的类型错误不报告，
  记 note）；if 双分支均 return ⇒ if 之后不可达；for 体内的 return
  终止该次迭代的剩余语句，但循环不终止外围块（保守可达性）。
"""

import re

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError

N = ti.Dim("N")


def _early_exit_kernel():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], c: ti.i32,
          BLOCK: ti.Const[int] = 64):
        pid = ti.program_id(0)
        offs = pid * BLOCK + ti.arange(0, BLOCK)
        m = offs < N
        if c > 0:
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 1.0, mask=m)
            return                      # 裸 return：第二个 store 永不执行
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 2.0, mask=m)
    return k


# (a) interp 数值语义：c=1 走 then 分支存 1.0 后提前退出；c=0 存 2.0。

def test_interp_early_exit_branch():
    k = _early_exit_kernel()
    n = 128
    out = np.full(n, -1.0, dtype=np.float32)
    k[(ti.cdiv(n, 64),)](out, 1)
    np.testing.assert_array_equal(out, np.ones(n, dtype=np.float32))


def test_interp_fallthrough_branch():
    k = _early_exit_kernel()
    n = 128
    out = np.full(n, -1.0, dtype=np.float32)
    k[(ti.cdiv(n, 64),)](out, 0)
    np.testing.assert_array_equal(out, np.full(n, 2.0, dtype=np.float32))


# (b) materialize：生成的 Triton 源码含顶层 return 语句；TIR dump 同。

def test_materialize_emits_return():
    @ti.jit
    def k():
        return
    src, dump = k.materialize()
    assert re.search(r"^\s*return$", src, re.M), src
    assert re.search(r"^\s*return$", dump, re.M), dump


def test_materialize_rejects_unvalidated_runtime_return():
    with pytest.raises(TilaError, match='TILA-TARGET-009.*|return under runtime if'):
        _early_exit_kernel().materialize()


# (c) 带值 return：TILA-SYN-021，提示写入输出 buffer。

def test_value_return_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            return x
    assert ei.value.code == "TILA-SYN-021"
    assert "output buffer" in ei.value.render()


# (d) return 之后的不可达代码不被检查（死代码中的类型错误不报告）。

def test_unreachable_after_return_not_checked():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          BLOCK: ti.Const[int] = 8):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
        return
        # 若被检查将触发 TILA-TYPE-017（Unit 不能赋值）——不可达，跳过：
        y = ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
    assert any("unreachable code after return skipped" in n
               for n in k.tk.notes), k.tk.notes


# (e) 双分支均 return：if 之后的代码不可达（跳过 + note），装饰成功且可运行。

def test_both_branches_return_skips_rest():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], c: ti.i32,
          BLOCK: ti.Const[int] = 8):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        if c > 0:
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 1.0, mask=m)
            return
        else:
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 2.0, mask=m)
            return
        # 不可达（两条路径都 return）——不检查：
        y = ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
    assert any("both branches" in n and "unreachable" in n
               for n in k.tk.notes), k.tk.notes
    out = np.full(8, -1.0, dtype=np.float32)
    k[(1,)](out, 1)
    np.testing.assert_array_equal(out, np.ones(8, dtype=np.float32))
    out2 = np.full(8, -1.0, dtype=np.float32)
    k[(1,)](out2, 0)
    np.testing.assert_array_equal(out2, np.full(8, 2.0, dtype=np.float32))


# static-if（Const 参数条件）双分支均 return：同样终止。

def test_static_if_both_branches_return():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], c: ti.i32,
          FLAG: ti.Const[int] = 1, BLOCK: ti.Const[int] = 8):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        if FLAG > 0:
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 1.0, mask=m)
            return
        else:
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 2.0, mask=m)
            return
        y = ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=m)
    assert any("unreachable" in n for n in k.tk.notes), k.tk.notes
    out = np.full(8, -1.0, dtype=np.float32)
    k[(1,)](out, 1, FLAG=1)
    np.testing.assert_array_equal(out, np.ones(8, dtype=np.float32))


# for 体内的 return：终止该次迭代的剩余语句并提前退出整个实例；
# 循环之后的语句仍可达（保守），照常检查与执行。

def test_return_inside_loop_ends_instance():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], c: ti.i32,
          BLOCK: ti.Const[int] = 8):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        for i in ti.range(0, c, 1):
            ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 1.0, mask=m)
            return                  # 第 0 次迭代即退出：2.0 不会写入
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32) + 9.0, mask=m)

    out = np.full(8, -1.0, dtype=np.float32)
    k[(1,)](out, 2)
    np.testing.assert_array_equal(out, np.ones(8, dtype=np.float32))

    out2 = np.full(8, -1.0, dtype=np.float32)
    k[(1,)](out2, 0)                # 空循环：尾随 store 写 9.0
    np.testing.assert_array_equal(out2, np.full(8, 9.0, dtype=np.float32))
