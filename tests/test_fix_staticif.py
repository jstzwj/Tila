"""A1 修复验证：constexpr-if 与 runtime-if 彻底分离（docs/type-system.md §5.2）。

条件纯由 Const 参数构成（模块常量已在 stage 1 折叠）⇒ StaticIf：
- 两分支独立检查并保留在 IR（TStaticIf），特化期由 Const 数值定值
  （Triton 路径条件引用 tl.constexpr 实参；interp 由 consts 求值）；
- 合并放宽：dtype/rank 相同且逐维 equal 或两侧均为纯 Const 符号式 →
  正常合并（代表类型取 then 侧）；不相容 → static-variant，使用点报
  TILA-TYPE-020；
- 条件混入任何运行期值（标量/维/pid）⇒ 绝不走该路径（soundness）。
"""

import numpy as np
import pytest

import tila as ti
from tila import tir as T
from tila.errors import TilaError

N = ti.Dim("N")
FLAG = True          # 模块级 bool 常量：stage 1 直接折叠


# (a) 文档模式：stage 1 通过，两分支均保留；两种特化的 launch 均正确 ------

def test_const_cond_if_is_static_if_stage1():
    """REPRO：此前 runtime-if 严格合并在此报 TILA-TYPE-020。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
        if BLOCK >= 128:
            offs = ti.arange(0, BLOCK)
        else:
            offs = ti.arange(0, 32)
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=offs < N)

    sifs = [s for s in k.tk.body if isinstance(s, T.TStaticIf)]
    assert len(sifs) == 1
    assert len(sifs[0].then_body) == 1          # 两分支都在（特化期定值）
    assert len(sifs[0].else_body) == 1
    assert not any(isinstance(s, T.TIf) for s in k.tk.body)
    assert any("static-if on Const parameters" in n for n in k.tk.notes)


def test_const_cond_if_launch_both_specializations():
    """分支 A 写 1.0 / 分支 B 写 2.0：BLOCK=256 走 then，BLOCK=64 走 else，
    N 的尾部行为由 mask 保证。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
        if BLOCK >= 128:
            offs = ti.arange(0, BLOCK)
            val = ti.zeros((BLOCK,), ti.f32) + 1.0
        else:
            offs = ti.arange(0, 32)
            val = ti.zeros((32,), ti.f32) + 2.0
        ti.store(x, offs, val, mask=offs < N)

    x = np.full(100, -1.0, dtype=np.float32)    # BLOCK=256：then，尾部 28 lane 屏蔽
    k[(1,)](x, BLOCK=256)
    assert list(x) == [1.0] * 100

    x = np.full(20, -1.0, dtype=np.float32)     # BLOCK=64：else（32 lane），尾部 12 lane 屏蔽
    k[(1,)](x, BLOCK=64)
    assert list(x) == [2.0] * 20

    x = np.full(200, -1.0, dtype=np.float32)    # BLOCK=512：then，尾部 lane 屏蔽
    k[(1,)](x, BLOCK=512)
    assert list(x) == [1.0] * 200


# (b) materialize：python if（条件引用 tl.constexpr 实参）保留在源码 ------

def test_materialize_keeps_python_if_over_constexpr():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
        if BLOCK >= 128:
            offs = ti.arange(0, BLOCK)
        else:
            offs = ti.arange(0, 32)
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=offs < N)

    src, dump = k.materialize()
    assert "if (BLOCK >= 128):" in src         # Triton trace 期解析该分支
    assert "BLOCK: tl.constexpr" in src
    assert "tl.arange(0, BLOCK)" in src        # 两分支均在（特化前）
    assert "tl.arange(0, 32)" in src
    assert "static_if (BLOCK >= 128):" in dump  # TIR dump 标注 static_if
    assert "both branches present pre-specialization" in dump


# (c) static-variant：两分支类型不相容 → 使用点 TILA-TYPE-020 -------------

def test_static_variant_use_raises():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              BLOCK: ti.Const[int] = 64):
            if BLOCK >= 128:
                offs = ti.zeros((BLOCK,), ti.f32)
            else:
                offs = ti.zeros((32,), ti.f16)     # dtype 不相容
            ti.store(x, ti.arange(0, 32), offs,
                     mask=ti.arange(0, 32) < N)     # 分支后使用 → 报错
    assert ei.value.code == "TILA-TYPE-020"
    assert "static" in str(ei.value)
    assert "Block[f32, (BLOCK)]" in str(ei.value)
    assert "Block[f16, (32)]" in str(ei.value)


def test_static_variant_rank_mismatch_also_variant():
    """rank 不相容（1D vs 2D）同样不享受放宽：无代表类型可用。"""

    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
              M: ti.Const[int] = 64):
            if M >= 128:
                offs = ti.zeros((M,), ti.f32)
            else:
                offs = ti.zeros((32, 4), ti.f32)     # rank-2 vs rank-1
            ti.store(x, ti.arange(0, 32), ti.zeros((32,), ti.f32),
                     mask=ti.arange(0, 32) < N)
            _ = offs
    assert ei.value.code == "TILA-TYPE-020"
    assert "static" in str(ei.value)


# (d) 回归：runtime-if 严格合并不变；模块常量 if 仍在 stage 1 折叠 ----------

def test_runtime_if_strict_merge_still_raises():
    """条件引用运行期标量 c：绝不走 static-deferred（soundness）。"""

    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], c: ti.i32,
              BLOCK: ti.Const[int] = 64):
            if c > 0:
                a = ti.zeros((BLOCK,), ti.f32)
            else:
                a = ti.zeros((32,), ti.f32)
            offs = ti.arange(0, BLOCK)
            ti.store(x, offs, a, mask=offs < N)
    assert ei.value.code == "TILA-TYPE-020"


def test_module_const_if_resolves_at_stage1():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
        if FLAG:
            offs = ti.arange(0, BLOCK)
        else:
            offs = ti.arange(0, 32)
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=offs < N)

    notes = "\n".join(k.tk.notes)
    assert "static branch resolved to then" in notes
    assert not any(isinstance(s, T.TStaticIf) for s in k.tk.body)


def test_dim_and_pid_conds_stay_runtime_if():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
        if N >= 64:
            offs = ti.arange(0, 32)
        else:
            offs = ti.arange(0, 32)
        ti.store(x, offs, ti.zeros((32,), ti.f32), mask=offs < N)
    assert any(isinstance(s, T.TIf) for s in k.tk.body)
    assert not any(isinstance(s, T.TStaticIf) for s in k.tk.body)


def test_mixed_const_runtime_cond_stays_runtime_if():
    """Const 参数与运行期值混合的条件：真运行期 if（严格合并）。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], c: ti.i32,
          BLOCK: ti.Const[int] = 64):
        if BLOCK >= 128 and c > 0:
            offs = ti.arange(0, 64)
        else:
            offs = ti.arange(0, 64)
        ti.store(x, offs, ti.zeros((64,), ti.f32), mask=offs < N)
    assert any(isinstance(s, T.TIf) for s in k.tk.body)
    assert not any(isinstance(s, T.TStaticIf) for s in k.tk.body)


def test_const_expr_and_boolop_cond_is_static_if():
    """Const 算术 + and/or 组合条件同样走 static-deferred。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
        if BLOCK % 4 == 0 and BLOCK >= 128:
            offs = ti.arange(0, BLOCK)
        else:
            offs = ti.arange(0, 32)
        ti.store(x, offs, ti.zeros((BLOCK,), ti.f32), mask=offs < N)
    assert len([s for s in k.tk.body if isinstance(s, T.TStaticIf)]) == 1
