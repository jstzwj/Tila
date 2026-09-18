"""R4 修复验证：tl.multiple_of 自动发射（refinements.md §4 整除事实）。

对模式 offs = pid*STEP + arange(0, E)：lowering 拆分出 base = pid*STEP
并发射 tl.multiple_of(base, STEP)（base 恒被 STEP 整除，结构性成立）。
非匹配模式保持既有发射（仅 max_contiguous）。
"""

import tila as ti

N = ti.Dim("N")


@ti.jit
def add_kernel(
    x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    y: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    BLOCK: ti.Const[int, ti.PowerOfTwo] = 128,
):
    pid = ti.program_id(0)
    offs = pid * BLOCK + ti.arange(0, BLOCK)
    mask = offs < N
    a = ti.load(x, offs, mask=mask)
    b = ti.load(y, offs, mask=mask)
    ti.store(out, offs, a + b, mask=mask)


@ti.jit
def scalar_base_kernel(
    x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    K: ti.i32,
    BLOCK: ti.Const[int, ti.PowerOfTwo] = 128,
):
    offs = K + ti.arange(0, BLOCK)   # 运行期标量基：无可整除事实
    mask = offs < N
    ti.store(out, offs, ti.load(x, offs, mask=mask), mask=mask)


@ti.jit
def literal_step_kernel(
    x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
):
    pid = ti.program_id(0)
    offs = pid * 128 + ti.arange(0, 128)   # 字面量 STEP，无 Const 参数
    mask = offs < N
    ti.store(out, offs, ti.load(x, offs, mask=mask), mask=mask)


@ti.jit
def two_axis_kernel(
    x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    o1: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    o2: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    BM: ti.Const[int, ti.PowerOfTwo] = 64,
    BN: ti.Const[int, ti.PowerOfTwo] = 64,
):
    pid_m = ti.program_id(0)
    pid_n = ti.program_id(1)
    rm = pid_m * BM + ti.arange(0, BM)
    rn = pid_n * BN + ti.arange(0, BN)
    m1 = rm < N
    m2 = rn < N
    ti.store(o1, rm, ti.load(x, rm, mask=m1), mask=m1)
    ti.store(o2, rn, ti.load(x, rn, mask=m2), mask=m2)


def test_canonical_pattern_splits_with_multiple_of():
    """(a) 规范模式：拆分 base + multiple_of + max_contiguous 依序出现。"""
    src, _ = add_kernel.materialize({})
    lines = [ln.strip() for ln in src.splitlines()]
    base_i = lines.index("offs_base = (pid * BLOCK)")
    add_i = lines.index("offs = (offs_base + tl.arange(0, BLOCK))")
    mo_i = lines.index("tl.multiple_of(offs_base, BLOCK)")
    mc_i = lines.index("tl.max_contiguous(offs, BLOCK)")
    assert base_i < add_i < mo_i < mc_i


def test_non_matching_pattern_keeps_old_emission():
    """(b) 运行期标量基：不拆分、不发射 multiple_of。"""
    src, _ = scalar_base_kernel.materialize({})
    lines = [ln.strip() for ln in src.splitlines()]
    assert "offs = (K + tl.arange(0, BLOCK))" in lines
    assert "tl.max_contiguous(offs, BLOCK)" in lines
    assert "tl.multiple_of(" not in src
    assert "offs_base" not in src


def test_literal_step_splits_with_int():
    """(c) 字面量步长：拆分并以整数 128 发射 multiple_of。"""
    src, _ = literal_step_kernel.materialize({})
    lines = [ln.strip() for ln in src.splitlines()]
    assert "offs_base = (pid * 128)" in lines
    assert "offs = (offs_base + tl.arange(0, 128))" in lines
    assert "tl.multiple_of(offs_base, 128)" in lines


def test_materialize_is_deterministic():
    """(d) 两次 materialize 输出逐字节一致（base 名确定性派生）。"""
    src1, tir1 = add_kernel.materialize({})
    src2, tir2 = add_kernel.materialize({})
    assert src1 == src2
    assert tir1 == tir2


def test_two_hinted_vars_get_independent_bases():
    """(e) 两个 pid 轴：rm/rn 各自独立 base 名与 multiple_of。"""
    src, _ = two_axis_kernel.materialize({})
    lines = [ln.strip() for ln in src.splitlines()]
    assert "rm_base = (pid_m * BM)" in lines
    assert "rm = (rm_base + tl.arange(0, BM))" in lines
    assert "tl.multiple_of(rm_base, BM)" in lines
    assert "tl.max_contiguous(rm, BM)" in lines
    assert "rn_base = (pid_n * BN)" in lines
    assert "rn = (rn_base + tl.arange(0, BN))" in lines
    assert "tl.multiple_of(rn_base, BN)" in lines
    assert "tl.max_contiguous(rn, BN)" in lines
