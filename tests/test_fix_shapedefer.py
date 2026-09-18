"""A7 修复验证：shape 相等判定两层化（docs/type-system.md §4.3）。

语法层（canon 相等）之外：两侧维只依赖 Const 参数时，等式作为约束
延迟到特化期（Stage 2）数值复核，而非 Stage-1 立即报错。此前仅 dot
内维走该路径；现在 broadcast 全路由（binop/cmp/where/mask/coords/
other）与 store 值形状检查同样延迟。

Stage-2 失败统一报 TILA-SHAPE-004（what 文本指明约束来源）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

import tila as ti  # noqa: F401  (smoke import)
from tila import frontend
from tila.checker import Checker
from tila.dims import Cst, Sym
from tila.errors import Loc, TilaError

N = ti.Dim("N")


# ---------------------------------------------------------------------------
# (a)/(b) repro：store 值形状 (128,) vs 访问形状 (BLOCK,) —— 修复前
# Stage-1 立即 TILA-SHAPE-012；修复后登记延迟约束，默认 BLOCK=128 时
# Stage-2 通过并正确执行。
# ---------------------------------------------------------------------------

@ti.jit
def k_store(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
            BLOCK: ti.Const[int] = 128):
    offs = ti.arange(0, BLOCK)
    m = offs < N
    v = ti.zeros((128,), ti.f32)
    ti.store(x, offs, v, mask=m)


def test_a_store_deferred_passes_stage1_and_runs():
    # Stage 1 通过且约束已登记（延迟而非立即报错）
    assert any(d["kind"] == "eq" and d["what"] == "store value shape"
               for d in k_store.tk.deferred)
    # 默认 BLOCK=128：Stage 2 代入数值 128 == 128 通过，interp 正确执行
    x = np.full(100, 7.0, dtype=np.float32)
    k_store[(1,)](x)
    assert np.allclose(x, 0.0)


def test_b_store_deferred_rejected_at_stage2():
    y = np.zeros(100, dtype=np.float32)
    with pytest.raises(TilaError) as ei:
        k_store[(1,)](y, BLOCK=64)      # 128 != 64：特化期拒绝
    assert ei.value.code == "TILA-SHAPE-004"
    assert "store value shape" in str(ei.value)


# ---------------------------------------------------------------------------
# (c) broadcast 路由：(BLOCK,) + (128,) 经二元运算的逐维 broadcast。
# ---------------------------------------------------------------------------

@ti.jit
def k_bcast(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
            BLOCK: ti.Const[int] = 128):
    offs = ti.arange(0, BLOCK)
    a = ti.zeros((BLOCK,), ti.f32)
    b = ti.zeros((128,), ti.f32)
    c = a + b                            # deferred: BLOCK == 128
    ti.store(x, offs, c, mask=offs < N)


def test_c_broadcast_route_default_passes():
    assert any(d["kind"] == "eq" and "operand shapes" in d["what"]
               for d in k_bcast.tk.deferred)
    x = np.full(100, 3.0, dtype=np.float32)
    k_bcast[(1,)](x)
    assert np.allclose(x, 0.0)


@pytest.mark.parametrize("mode", ["launch", "materialize"])
def test_c_broadcast_route_override_rejected_at_stage2(mode):
    if mode == "launch":
        with pytest.raises(TilaError) as ei:
            k_bcast[(1,)](np.zeros(100, dtype=np.float32), BLOCK=64)
    else:
        with pytest.raises(TilaError) as ei:
            k_bcast.materialize({"BLOCK": 64})
    assert ei.value.code == "TILA-SHAPE-004"
    assert "operand shapes" in str(ei.value)


# ---------------------------------------------------------------------------
# (d) 负例：纯常量维不等 / 运行期符号维不等 —— 不享受延迟，Stage 1
# 立即 TILA-SHAPE-003。
# ---------------------------------------------------------------------------

def test_d_const_const_pair_immediate_stage1():
    # BLOCK 参与 dim-0（可延迟），但 dim-1 是纯常量对 (32, 16)：语法层
    # 即定论 → 立即失败，不能把整对形状塞进约束池。
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(BLOCK: ti.Const[int] = 64):
            a = ti.zeros((BLOCK, 32), ti.f32)
            b = ti.zeros((16,), ti.f32)
            c = a + b
    assert ei.value.code == "TILA-SHAPE-003"


def _checker_with_const():
    def f(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
          BLOCK: ti.Const[int] = 128):
        pass
    return Checker(frontend.compile_stage1(f))


def test_d_runtime_dim_syms_reject_deferral():
    ck = _checker_with_const()
    # 运行期维符号（N/M）不在 Const 参数名集合内 → 立即失败
    assert ck._dim_eq_defer(Sym("N"), Sym("M"), Loc(0), "test") is False
    with pytest.raises(TilaError) as ei:
        ck._broadcast((Sym("N"),), (Sym("M"),), Loc(0))
    assert ei.value.code == "TILA-SHAPE-003"
    # 纯常量不等（无自由符号）：语法不等即定论，不延迟
    assert ck._dim_eq_defer(Cst(3), Cst(4), Loc(0), "test") is False
    # 对照：纯 Const 符号参与 → 延迟并登记约束
    assert ck._dim_eq_defer(Sym("BLOCK"), Cst(128), Loc(0),
                            "store value shape") is True
    d = ck.tk.deferred[-1]
    assert d["kind"] == "eq" and d["what"] == "store value shape"
