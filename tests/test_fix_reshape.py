"""A8 修复验证：ti.reshape（docs/intrinsics.md §2.8，roadmap Phase 0-3）。

reshape(x, S2)：Block[T, S1] → Block[T, S2]，约束 numel(S1) ~ numel(S2)。

本实现的 numel 语义（分层判定，type-system.md §4.3）：
1. 两侧维度全常量（int_value 可算）→ Stage 1 立即数值比较；
   不等 → TILA-SHAPE-005 "reshape numel mismatch"。
2. 符号乘积 canon 等价（dims.equal：N*2 vs 2*N 形态）→ Stage 1
   直接通过，不进延迟池；
3. 其余且自由符号全为 Const 参数名 → Stage 1 延迟（kind="numel"），
   特化期（launch / materialize）代入 Const 值后数值比较；
4. 运行期维（ti.Dim）参与 → 无法验证：目标形状里的运行期名字更早被
   Const 语境拒绝（TILA-CONST-001 "reshape dimension must be
   compile-time known"），运行期 numel 约束在本版本根本写不出来；
   checker 另有防御分支（输入形状含运行期符号时立即 TILA-SHAPE-005）。

结果类型不携带事实（expr / contiguous_span 跨 reshape 失效）。
"""

import numpy as np
import pytest

import tila as ti
from tila.dims import Cst, Sym
from tila.errors import TilaError

N = ti.Dim("N")
N2 = ti.Dim("N2")


# ---------------------------------------------------------------------------
# (a) 端到端：1D → 2D 与 2D → 1D 的 arange 往返（interp 数值运行）
# ---------------------------------------------------------------------------

@ti.jit
def to_2d(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
          out: ti.Buffer[ti.f32, (N2, 4), ti.WriteOnly]):
    offs = ti.arange(0, 512)
    v = ti.load(x, offs, mask=offs < N)
    y = ti.reshape(v, (128, 4))
    r = ti.arange(0, 128)
    c = ti.arange(0, 4)
    ti.store(out, (r[:, None], c[None, :]), y,
             mask=(r[:, None] < N2) & (c[None, :] < 4))


@ti.jit
def to_1d(x: ti.Buffer[ti.f32, (N2, 4), ti.ReadOnly],
          out: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
    r = ti.arange(0, 128)
    c = ti.arange(0, 4)
    v = ti.load(x, (r[:, None], c[None, :]),
                mask=(r[:, None] < N2) & (c[None, :] < 4))
    y = ti.reshape(v, (512,))
    offs = ti.arange(0, 512)
    ti.store(out, offs, y, mask=offs < N)


def test_reshape_1d_to_2d_roundtrip():
    assert to_2d.tk.types["y"].dims == (Cst(128), Cst(4))
    assert to_2d.tk.types["y"].describe() == "Block[f32, (128, 4)]"
    x = np.arange(512, dtype=np.float32)
    out = np.zeros((128, 4), dtype=np.float32)
    to_2d[(1,)](x, out)
    assert np.array_equal(out, x.reshape(128, 4))


def test_reshape_2d_to_1d_roundtrip():
    assert to_1d.tk.types["y"].dims == (Cst(512),)
    x = np.arange(512, dtype=np.float32).reshape(128, 4)
    out = np.zeros(512, dtype=np.float32)
    to_1d[(1,)](x, out)
    assert np.array_equal(out, x.reshape(512))


# ---------------------------------------------------------------------------
# (b) Const 符号形状：Stage 1 延迟 numel，特化期代入比较
# ---------------------------------------------------------------------------

@ti.jit
def sym_reshape(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                out: ti.Buffer[ti.f32, (N2, 8), ti.WriteOnly],
                BLOCK: ti.Const[int] = 128):
    offs = ti.arange(0, 1024)
    v = ti.load(x, offs, mask=offs < N)
    y = ti.reshape(v, (BLOCK, 8))
    r = ti.arange(0, BLOCK)
    c = ti.arange(0, 8)
    ti.store(out, (r[:, None], c[None, :]), y,
             mask=(r[:, None] < N2) & (c[None, :] < 8))


def test_const_symbolic_deferred_then_pass():
    # Stage 1：(Cst 1024) vs (BLOCK, 8)——BLOCK 是 Const 符号 → 延迟
    assert any(d.get("kind") == "numel" for d in sym_reshape.tk.deferred)
    assert sym_reshape.tk.types["y"].dims == (Sym("BLOCK"), Cst(8))
    x = np.arange(1024, dtype=np.float32)
    out = np.zeros((128, 8), dtype=np.float32)
    sym_reshape[(1,)](x, out, BLOCK=128)          # 1024 == 128*8 → 通过
    assert np.array_equal(out, x.reshape(128, 8))


def test_const_symbolic_override_mismatch_stage2():
    x = np.arange(1024, dtype=np.float32)
    out = np.zeros((128, 8), dtype=np.float32)
    with pytest.raises(TilaError) as ei:
        sym_reshape[(1,)](x, out, BLOCK=64)       # 1024 != 64*8 → 特化期
    assert ei.value.code == "TILA-SHAPE-005"
    assert "reshape numel mismatch" in ei.value.title
    assert "1024" in ei.value.render() and "512" in ei.value.render()


# ---------------------------------------------------------------------------
# (c) 全常量不匹配：Stage 1 立即 TILA-SHAPE-005
# ---------------------------------------------------------------------------

def test_all_constant_mismatch_stage1():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly]):
            offs = ti.arange(0, 128)
            v = ti.load(x, offs, mask=offs < N)
            y = ti.reshape(v, (100, 4))           # 128 != 400
    assert ei.value.code == "TILA-SHAPE-005"
    assert ei.value.title == "reshape numel mismatch"


# ---------------------------------------------------------------------------
# (d) 运行期维与符号乘积 canon 等价
# ---------------------------------------------------------------------------

def test_canon_equal_symbolic_products_allowed():
    """2*BLOCK vs BLOCK*2：canon 等价 → Stage 1 直接通过，不进延迟池。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly], BLOCK: ti.Const[int]):
        v = ti.arange(0, BLOCK * 2)               # 形状 (2*BLOCK,)
        y = ti.reshape(v, (2, BLOCK))

    assert str(k.tk.types["v"].dims[0]) == "2*BLOCK"
    assert k.tk.types["y"].dims == (Cst(2), Sym("BLOCK"))
    assert not any(d.get("kind") == "numel" for d in k.tk.deferred)
    src, _ = k.materialize({"BLOCK": 64})
    assert "tl.reshape(v, (2, BLOCK))" in src


def test_runtime_dim_in_target_shape_rejected():
    """选定行为：运行期维 N 出现在目标形状 → Const 语境拒绝
    （TILA-CONST-001）。运行期 numel 约束（如 (N,)→(2,N)，符号上
    N vs 2*N 也确实不等价）在本版本无法表达，也无从验证。"""
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly]):
            offs = ti.arange(0, 128)
            v = ti.load(x, offs, mask=offs < N)
            y = ti.reshape(v, (2, N))
    assert ei.value.code == "TILA-CONST-001"
    assert "reshape dimension" in ei.value.title


# ---------------------------------------------------------------------------
# (e) lowering / TIR dump
# ---------------------------------------------------------------------------

def test_materialize_emits_tl_reshape():
    src, dump = to_2d.materialize()
    assert "tl.reshape(" in src
    assert "tl.reshape(v, (128, 4))" in src
    assert "reshape(v, (128, 4))" in dump          # TIR dump


def test_single_dim_shape_prints_as_tuple():
    """单项形状 (512,) 必须保持 tuple 文本（(512) 是 int，Triton 拒绝）。"""
    src, dump = to_1d.materialize()
    assert "tl.reshape(v, (512,))" in src
    assert "reshape(v, (512,))" in dump


# ---------------------------------------------------------------------------
# 负控制：签名与操作数类型
# ---------------------------------------------------------------------------

def test_bad_arity_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly]):
            offs = ti.arange(0, 128)
            v = ti.load(x, offs, mask=offs < N)
            y = ti.reshape(v)
    assert ei.value.code == "TILA-SYN-036"


def test_empty_shape_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly]):
            offs = ti.arange(0, 128)
            v = ti.load(x, offs, mask=offs < N)
            y = ti.reshape(v, ())
    assert ei.value.code == "TILA-SYN-036"


def test_non_block_operand_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(alpha: ti.f32):
            y = ti.reshape(alpha, (2, 4))
    assert ei.value.code == "TILA-SHAPE-005"


def test_zero_dim_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly]):
            offs = ti.arange(0, 128)
            v = ti.load(x, offs, mask=offs < N)
            y = ti.reshape(v, (0, 128))
    assert ei.value.code == "TILA-SHAPE-005"
