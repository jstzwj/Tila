"""v0.4 K 循环与累加器（docs/v0.4-kloop.md）：R17 zeros / R18 range / R19 累加
与 φ、L7 join 单位元律、E20 矩阵（subcode）、作用域三围栏、任意 K 的
interpreter 差分（含 K=0 零迭代、非连续 b、多段累加）——最后一批是验收主场。"""

import numpy as np
import pytest

from tila.driver import compile_kernel
from tila.diagnostics import TilaError
from tila.interp import run_kernel
from tila.types import JoinL
from tila.types.layout import Identity, Mma, Zeros, equiv, normalize
from tila.types.shape import Const

MATMUL_LOOP = open("examples/matmul_loop.tila", encoding="utf-8").read()

# 公共前奏：1D 程序坐标
_PRELUDE_1D = """    pid = tila.program_id(0)
    offs = pid * 64 + tila.arange(0, 64)
"""

LOOPSIG = "b: tila.Tensor[tila.float32, N], B: tila.constexpr = 64"


def _c(body, params="", overrides=None):
    return compile_kernel(
        f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n", overrides)


def _expect(body, params="", subcode=None, code="E20", overrides=None):
    with pytest.raises(TilaError) as ei:
        _c(body, params, overrides)
    assert ei.value.code == code, ei.value.render()
    if subcode is not None:
        assert ei.value.subcode == subcode, ei.value.render()
    return ei.value


# ---------------------------------------------------------------------------
# L7：join 的单位元（layout 代数单元）
# ---------------------------------------------------------------------------

def test_l7_join_neutral_mma():
    z = Zeros((Const(64), Const(128)))
    m = Mma(64, 128, 64)
    assert normalize(JoinL(z, m)) == m
    assert normalize(JoinL(m, z)) == m


def test_l7_zeros_zeros():
    z = Zeros((Const(64), Const(128)))
    assert normalize(JoinL(z, Zeros((Const(64), Const(128))))) == z


def test_l7_not_in_equiv():
    z = Zeros((Const(64), Const(128)))
    assert not equiv(z, Mma(64, 128, 64))
    assert equiv(z, Zeros((Const(64), Const(128))))


def test_l7_join_identity():
    z = Zeros((Const(64),))
    assert normalize(JoinL(z, Identity((Const(64),)))) == Identity((Const(64),))


# ---------------------------------------------------------------------------
# R17 zeros：形态与纪律
# ---------------------------------------------------------------------------

def test_zeros_type_and_layout_term():
    body = (_PRELUDE_1D + """    z = tila.zeros((B,), tila.float32)
    tila.store(c, (offs,), z, mask=offs < N)
""")
    res = _c(body, "c: tila.Tensor[tila.float32, N], B: tila.constexpr = 64")
    assert "zeros (B) f32 : Tile<f32, (64,), L1>" in res.tir_dump
    assert "L1 = zeros(64)" in res.tir_dump


def test_zeros_literal_shape():
    body = """    z = tila.zeros((64,), tila.float32)
"""
    res = _c("    pid = tila.program_id(0)\n    offs = pid * 64 + tila.arange(0, 64)\n" + body +
             "    tila.store(c, (offs,), z, mask=offs < N)\n",
             "c: tila.Tensor[tila.float32, N]")
    assert "zeros (64) f32 : Tile<f32, (64,), L1>" in res.tir_dump
    assert "tl.zeros((64,), dtype=tl.float32)" in res.triton_source


def test_e20_zeros_non_tuple_shape():
    _expect("    z = tila.zeros(B, tila.float32)\n", LOOPSIG,
            subcode="ZerosForm")


def test_e20_zeros_runtime_sym_shape():
    # N 是运行期符号维：累加器 shape 必须静态
    _expect("    z = tila.zeros((N,), tila.float32)\n", LOOPSIG,
            subcode="ZerosForm")


def test_e20_zeros_rank3():
    _expect("    z = tila.zeros((B, B, B), tila.float32)\n", LOOPSIG,
            subcode="ZerosForm")


def test_e20_zeros_bool():
    _expect("    z = tila.zeros((B,), tila.bool)\n", LOOPSIG,
            subcode="ZerosForm")


def test_e20_zeros_arity():
    _expect("    z = tila.zeros((B,))\n", LOOPSIG, subcode="ZerosForm")


# ---------------------------------------------------------------------------
# R18 range / for：形态
# ---------------------------------------------------------------------------

def test_range_form_in_tir_and_triton():
    res = compile_kernel(MATMUL_LOOP)
    assert "%k0 = for 0 %K BK : Scalar(i32) [#k0]" in res.tir_dump
    assert "for k0 in range(0, K, BK):" in res.triton_source


def test_range_literal_end_and_step():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, 3, 2):
        acc += x
    tila.store(c, (offs,), acc, mask=offs < N)
""")
    res = _c(body, "b: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]")
    assert "= for 0 %t" in res.tir_dump          # 字面量 end 物化为 const 操作数
    assert "for k0 in range(0, 3, 2):" in res.triton_source


def test_e20_not_range_bare_python_range():
    _expect("    for k0 in range(0, N, 64):\n        x = 1\n",
            LOOPSIG, subcode="NotRange")


def test_e20_not_range_non_call_iterable():
    _expect("    for k0 in N:\n        x = 1\n", LOOPSIG, subcode="NotRange")


def test_e20_range_arity():
    _expect("    for k0 in tila.range(0, N):\n        x = 1\n",
            LOOPSIG, subcode="RangeForm")


def test_e20_range_start_not_zero():
    _expect("    for k0 in tila.range(1, N, 64):\n        x = 1\n",
            LOOPSIG, subcode="RangeForm")


def test_e20_range_step_zero():
    _expect("    for k0 in tila.range(0, N, 0):\n        x = 1\n",
            LOOPSIG, subcode="RangeForm")


def test_e20_range_step_runtime():
    _expect("    for k0 in tila.range(0, N, N):\n        x = 1\n",
            LOOPSIG, subcode="RangeForm")


def test_e20_range_end_tile():
    _expect("    for k0 in tila.range(0, tila.arange(0, 64), 64):\n        x = 1\n",
            LOOPSIG, subcode="RangeForm")


def test_e20_range_outside_for():
    _expect("    r = tila.range(0, N, 64)\n", LOOPSIG, subcode="RangePosition")


def test_e20_nested_loop():
    body = """    for k0 in tila.range(0, N, 64):
        for k1 in tila.range(0, N, 64):
            x = 1
"""
    _expect(body, LOOPSIG, subcode="NestedLoop")


# ---------------------------------------------------------------------------
# R19 累加纪律与围栏
# ---------------------------------------------------------------------------

def test_phi_forward_reference_and_exit_type():
    res = compile_kernel(MATMUL_LOOP)
    dump = res.tir_dump
    # φ：body 顶部，前向引用 back；类型 = 出口类型（Mma）
    assert "%acc.loop = phi %acc %acc.next : Tile<f32, (64,128), L4>" in dump
    assert "%acc.next = add %acc.loop %t6 : Tile<f32, (64,128), L4>" in dump
    # 循环后名字绑定出口值
    assert "%c16 = cast %acc.next f16" in dump
    assert "L4 = mma(64,128,64)" in dump


def test_double_accumulate_naming_and_equiv():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
        acc += x
    tila.store(c, (offs,), acc, mask=offs < N)
""")
    res = _c(body, "b: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]")
    dump = res.tir_dump
    assert "%acc.next = add %acc.loop %x" in dump
    assert "%acc.next2 = add %acc.next %x" in dump
    # φ 的 back 指向最后一个 += 结果
    assert "phi %acc %acc.next2" in dump


def test_two_sequential_loops_same_accumulator():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
    for k1 in tila.range(0, N, 64):
        acc += x
    tila.store(c, (offs,), acc, mask=offs < N)
""")
    res = _c(body, "b: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]")
    dump = res.tir_dump
    assert "%acc.loop = phi %acc %acc.next" in dump
    assert "%acc.loop2 = phi %acc.next %acc.next2" in dump


def test_e20_augassign_outside_loop():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    acc += x
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="AccumForm")


def test_e20_minus_equals():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc -= x
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="AccumForm")


def test_e20_accumulate_non_zeros_target():
    body = (_PRELUDE_1D + """    x = tila.load(b, (offs,), mask=offs < N)
    y = x + x
    for k0 in tila.range(0, N, 64):
        y += x
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="AccumForm")


def test_e20_accumulate_unbound():
    body = (_PRELUDE_1D + """    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="AccumForm")


def test_e20_accumulator_read_in_body():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
        y = acc + x
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="AccumRead")


def test_e20_accumulator_read_in_rhs():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += acc + x
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="AccumRead")


def test_e20_accumulator_stored_in_body():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
        tila.store(c, (offs,), acc, mask=offs < N)
""")
    _expect(body, "b: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]",
            subcode="AccumRead")


def test_accumulator_readable_after_loop():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
    y = acc + acc
    tila.store(c, (offs,), y, mask=offs < N)
""")
    res = _c(body, "b: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]")
    assert "%y = add %acc.next %acc.next" in res.tir_dump


def test_e20_loop_local_out_of_scope():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
    y = k0 + 1
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="LoopScope")


def test_e20_loop_body_local_out_of_scope():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    for k0 in tila.range(0, N, 64):
        x = tila.load(b, (offs,), mask=offs < N)
        acc += x
    tila.store(c, (offs,), x, mask=offs < N)
""")
    _expect(body, "b: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]",
            subcode="LoopScope")


def test_outer_names_readonly_in_body():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        x = x + x
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", code="E14")


def test_e02_accumulate_dtype_mismatch():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    xf = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += tila.cast(xf, tila.float16)
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", code="E02")


def test_e03_accumulate_shape_strict():
    body = (_PRELUDE_1D
            + "    offs32 = pid * 64 + tila.arange(0, 32)\n"
            + """    acc = tila.zeros((64,), tila.float32)
    for k0 in tila.range(0, N, 64):
        acc += tila.load(b, (offs32,), mask=offs32 < N)
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", code="E03")


def test_e05_accumulate_layout_after_first_join():
    # 第一次 += 后 acc 是 Identity 族；再 += Zeros tile → equiv 失败（L7 只在 join 侧）
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    z = tila.zeros((64,), tila.float32)
    for k0 in tila.range(0, N, 64):
        acc += x
        acc += z
""")
    _expect(body, "b: tila.Tensor[tila.float32, N]", code="E05")


# ---------------------------------------------------------------------------
# 差分（验收主场）：任意 K、零迭代、非连续 b、多段累加
# ---------------------------------------------------------------------------

def _run_matmul(a, b):
    res = compile_kernel(MATMUL_LOOP)
    c = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    return c.astype(np.float32)


@pytest.mark.parametrize("K", [0, 1, 40, 63, 64, 65, 130, 512, 4096])
def test_matmul_loop_arbitrary_k(K):
    # K=0 是零迭代恒等式：Loop(0, seed, F) = seed 且 C = A·B = 0
    # （size-0 数组的 RowMajor 契约空真，v0.4-kloop §12）
    rng = np.random.default_rng(42 + K)
    M, N = 100, 140
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    out = _run_matmul(a, b)
    assert np.allclose(out, ref, rtol=1e-2, atol=1e-2), \
        f"K={K}: max err {np.abs(out - ref).max()}"


def test_matmul_loop_transposed_b():
    rng = np.random.default_rng(7)
    M, K, N = 100, 130, 140
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16).T   # strides (1, K)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    out = _run_matmul(a, b)
    assert np.allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_matmul_loop_padded_b():
    rng = np.random.default_rng(8)
    M, K, N, PAD = 100, 130, 140, 160
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, PAD)).astype(np.float16)[:, :N]
    ref = a.astype(np.float32) @ b.astype(np.float32)
    out = _run_matmul(a, b)
    assert np.allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_e16_fp8_accumulator_rejected():
    """累加能力门（评审 §8）：fp8 种子的累加器 = storage-only 算术 → E16。"""
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.fp8e4m3fn)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
""")
    _expect(body, "b: tila.Tensor[tila.fp8e4m3fn, N]", code="E16")


def test_accumread_message_names_primitive_update():
    body = (_PRELUDE_1D + """    acc = tila.zeros((64,), tila.float32)
    x = tila.load(b, (offs,), mask=offs < N)
    for k0 in tila.range(0, N, 64):
        acc += x
        y = acc + x
""")
    e = _expect(body, "b: tila.Tensor[tila.float32, N]", subcode="AccumRead")
    assert "ordinary expression value" in e.message
    assert "left-hand target" in e.message


def test_two_loops_accumulate_differential():
    """两个顺序 for 累加同一 acc：equiv 分支的差分验证。"""
    src = """import tila


@tila.jit
def k2(a: tila.Tensor[tila.float16, M, K], b: tila.Tensor[tila.float16, K, N, (sb0, sb1)],
       c: tila.Tensor[tila.float16, M, N], BKH: tila.constexpr = 64):
    pid_m = tila.program_id(0)
    pid_n = tila.program_id(1)
    rm2 = tila.expand_dim(pid_m * 64 + tila.arange(0, 64), 1)
    rn2 = tila.expand_dim(pid_n * 128 + tila.arange(0, 128), 0)
    rk2 = tila.expand_dim(tila.arange(0, BKH), 0)
    rk3 = tila.expand_dim(tila.arange(0, BKH), 1)
    c_m = (rm2 < M) & (rn2 < N)
    acc = tila.zeros((64, 128), tila.float32)
    for k0 in tila.range(0, K, BKH):
        rka = rk2 + k0
        x = tila.load(a, (rm2, rka), mask=(rm2 < M) & (rka < K), other=0.0)
        y = tila.load(b, (rk3 + k0, rn2), mask=((rk3 + k0) < K) & (rn2 < N), other=0.0)
        acc += tila.dot(x, y)
    for k1 in tila.range(0, K, BKH):
        rkb = rk2 + k1
        x2 = tila.load(a, (rm2, rkb), mask=(rm2 < M) & (rkb < K), other=0.0)
        y2 = tila.load(b, (rk3 + k1, rn2), mask=((rk3 + k1) < K) & (rn2 < N), other=0.0)
        acc += tila.dot(x2, y2)
    tila.store(c, (rm2, rn2), tila.cast(acc, tila.float16), mask=c_m)
"""
    res = compile_kernel(src)
    rng = np.random.default_rng(3)
    M, K, N = 100, 130, 140
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)
    c = np.zeros((M, N), dtype=np.float16)
    run_kernel(res.kernel, {"a": a, "b": b, "c": c})
    ref = 2.0 * (a.astype(np.float32) @ b.astype(np.float32))
    assert np.allclose(c.astype(np.float32), ref, rtol=1e-2, atol=1e-2)


def test_store_inside_loop_legal():
    """store 在体内合法（非累加器值）：逐 K 块搬运。"""
    src = """import tila


@tila.jit
def copyk(a: tila.Tensor[tila.float32, K], c: tila.Tensor[tila.float32, K],
          B: tila.constexpr = 64):
    pid = tila.program_id(0)
    offs = pid * B + tila.arange(0, B)
    m = offs < K
    for k0 in tila.range(0, K, B):
        x = tila.load(a, (offs,), mask=m)
        tila.store(c, (offs,), x, mask=m)
"""
    res = compile_kernel(src)
    rng = np.random.default_rng(5)
    a = rng.standard_normal(130).astype(np.float32)
    c = np.zeros(130, dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "c": c})
    assert np.allclose(c, a)


def test_zero_trip_falls_back_to_seed():
    """K=0：零迭代 → acc = 种子（φ 语义）。"""
    src = """import tila


@tila.jit
def zloop(a: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N],
          B: tila.constexpr = 64):
    pid = tila.program_id(0)
    offs = pid * B + tila.arange(0, B)
    m = offs < N
    acc = tila.zeros((B,), tila.float32)
    x = tila.load(a, (offs,), mask=m)
    for k0 in tila.range(0, N - N, 64):
        acc += x
    tila.store(c, (offs,), acc, mask=m)
"""
    res = compile_kernel(src)
    rng = np.random.default_rng(6)
    a = rng.standard_normal(130).astype(np.float32)
    c = np.zeros(130, dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "c": c})
    assert np.allclose(c, np.zeros(130))


# ---------------------------------------------------------------------------
# 既有语义不受扰动（纯增量证明）
# ---------------------------------------------------------------------------

def test_matmul_loop_launcher_unchanged_shape():
    res = compile_kernel(MATMUL_LOOP)
    # launcher 无新参数类别：sym 仍只来自 shape/stride；grid 仍由 pid 模式推导
    assert "def matmul_loop_launch(a, b, c, BM: int = 64, BN: int = 128, BK: int = 64):" in res.triton_source
    assert "grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))" in res.triton_source
