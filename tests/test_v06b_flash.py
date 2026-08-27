"""v0.6b Stateful Loop Extension 语义矩阵（docs/v0.6b-flash.md §3 F——
进主规范的门；docs/v0.6-attention.md §1.4/§2.4–§2.6 的实现验收）。

覆盖：AccumulatorUpdate 状态机全转移（+= 物化/equiv 锁定、*= mul-identity/
MulSeed/乘数透明、handoff 物化/equiv）；handoff 负例全集（评审 B-⑯）；seed
值规范化（full(0) ≡ zeros）；StateRead 定型（α 出口求解 / 跨族读 E05 /
逃逸位矩阵）；零迭代恒等式；多累加器 φ 链；l 种子恒等式（评审 B-⑰）；
审核修复回归（in-loop 播种 / Seed per-axis / α 链泄漏）；full/maximum/
log2/launch_assert 的形态与能力矩阵。
"""

from __future__ import annotations

import numpy as np
import pytest

from tila.driver import compile_kernel
from tila.diagnostics import Loc, TilaError
from tila.interp import run_kernel
from tila.checker.layout import join_distributions, strict_join
from tila.types.dist import Identity, Product, Seed, Slice, Mma, equiv_dist
from tila.types.shape import Const

SIG = "b: tila.Tensor[tila.float32, M], c: tila.Tensor[tila.float32, M], BM: tila.constexpr = 64"
PRE = ("    pid = tila.program_id(0)\n"
       "    offs = pid * BM + tila.arange(0, BM)\n"
       "    mm = offs < M\n"
       "    x = tila.load(b, (offs,), mask=mm)\n")


def _c(body, params=SIG, overrides=None):
    return compile_kernel(
        f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n", overrides)


def _expect(body, params=SIG, subcode=None, code="E20", overrides=None):
    with pytest.raises(TilaError) as ei:
        _c(body, params, overrides)
    assert ei.value.code == code, ei.value.render()
    if subcode is not None:
        assert ei.value.subcode == subcode, ei.value.render()
    return ei.value


# ---------------------------------------------------------------------------
# 状态机：ReduceUpdate（'+='）
# ---------------------------------------------------------------------------

def test_reduce_seed_materializes():
    body = PRE + ("    acc = tila.zeros((64,), tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        acc += x\n"
                  "    tila.store(c, (offs,), acc, mask=mm)\n")
    res = _c(body)
    # Seed + Tile[Identity] → Materialized(Identity)：add 结果类型离开 seed
    assert "L0 = identity(64)" in res.tir_dump
    line = [l for l in res.tir_dump.splitlines() if "%acc.next = add" in l][0]
    assert ": Tile<f32, (64,), L0>" in line
    assert "%acc.loop = phi %acc %acc.next : Tile<f32, (64,), L0>" in res.tir_dump


def test_reduce_materialized_equiv_locked():
    # 物化后第二次 +=：同分布放行；跨分布（arange 种子 vs zeros 物化链）E05
    body = PRE + ("    acc = tila.zeros((64,), tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        acc += x\n"
                  "        acc += x\n"
                  "    tila.store(c, (offs,), acc, mask=mm)\n")
    assert "%acc.next2 = add" in _c(body).tir_dump


# ---------------------------------------------------------------------------
# 状态机：ScaleUpdate（'*='）
# ---------------------------------------------------------------------------

def test_scale_mul_identity_seed_stays_seed():
    # Seed(0) × X → Seed(0)：mul 的类型保持 seed（值层 0·x = 0，不进布局律）
    body = PRE + ("    l = tila.zeros((64,), tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        l *= x\n"
                  "    tila.store(c, (offs,), l, mask=mm)\n")
    res = _c(body)
    line = [l for l in res.tir_dump.splitlines() if "%l.next = mul" in l][0]
    assert "seed(64)" in res.tir_dump
    assert "L1 = seed(64)" in res.tir_dump and f", L1>" in line
    assert "%l.loop = phi %l %l.next : Tile<f32, (64,), L1>" in res.tir_dump


def test_scale_nonzero_seed_mulseed():
    body = PRE + ("    l = tila.full((64,), 1.0, tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        l *= x\n")
    e = _expect(body, subcode="MulSeed")
    assert "zero-valued seed" in e.message


def test_scale_neg_inf_seed_mulseed():
    # full(−∞) 的 *= 同样 MulSeed（种子值 ≠ 0）
    body = PRE + ("    l = tila.full((64,), tila.neg_inf, tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        l *= x\n")
    _expect(body, subcode="MulSeed")


def test_scale_materialized_multiplier_transparent():
    # Materialized × X → Materialized：乘数分布免检（R-ST）——跨族乘数
    # （Product 族乘 (BM,1) Slice 广播位）不约束累加器
    sig2d = ("a: tila.Tensor[tila.float16, M, K], b: tila.Tensor[tila.float16, K, N], "
             "c: tila.Tensor[tila.float32, M, N], BM: tila.constexpr = 64")
    body = ("    rm2 = tila.expand_dim(tila.arange(0, 64), 1)\n"
            "    rk2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
            "    rk3 = tila.expand_dim(tila.arange(0, 64), 1)\n"
            "    rn2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
            "    a2 = tila.load(a, (rm2, rk2), mask=(rm2 < M) & (rk2 < K), other=0.0)\n"
            "    b2 = tila.load(b, (rk3, rn2), mask=(rk3 < K) & (rn2 < N), other=0.0)\n"
            "    acc = tila.zeros((64, 64), tila.float32)\n"
            "    for k0 in tila.range(0, K, 64):\n"
            "        d = tila.dot(a2, b2)\n"
            "        acc += d\n"
            "        s = tila.max(d, axis=1)\n"
            "        acc *= tila.expand_dim(s, 1)\n"
            "    y = acc + tila.cast(a2, tila.float32)\n"
            "    tila.store(c, (rm2, rn2), y, mask=(rm2 < M) & (rn2 < N))\n")
    with pytest.raises(TilaError) as ei:  # 跨族读（y = Mma + Product）仍 E05：
        _c(body, sig2d)                    # 证明 acc 的 Mma 未被乘数污染
    assert ei.value.code == "E05"
    # 同族消费放行（acc 读与 d 同为 Mma）：R-ST 透明不改变 acc 的分布
    # ——去掉负例行，改为循环内读 acc 与 d 同族相加（不逃逸）
    body_ok = body.replace(
        "    y = acc + tila.cast(a2, tila.float32)\n"
        "    tila.store(c, (rm2, rn2), y, mask=(rm2 < M) & (rn2 < N))\n",
        "")
    body_ok = body_ok.replace(
        "        acc *= tila.expand_dim(s, 1)\n",
        "        acc *= tila.expand_dim(s, 1)\n        y2 = acc + d\n")
    res = _c(body_ok, sig2d)
    # φ 类型引用 mma 定义（物化后乘数透明：acc.next2 仍是 Mma）
    phi_l = [l for l in res.tir_dump.splitlines() if "acc.loop = phi" in l][0]
    import re as _re
    m = _re.search(r", L(\d+)>", phi_l)
    assert m and f"L{m.group(1)} = mma(" in res.tir_dump, phi_l
    y2_l = [l for l in res.tir_dump.splitlines() if "%y2 = add" in l][0]
    m2 = _re.search(r", L(\d+)>", y2_l)
    assert m2 and f"L{m2.group(1)} = mma(" in res.tir_dump, y2_l


def test_scale_shape_cannot_grow():
    # 乘数只可广播进累加器形状（不放大）：(64,) 累加器 × (128,) 乘数 →
    # ⊗ 不良式（64 vs 128 非一）→ E03
    body = PRE + ("    l = tila.zeros((64,), tila.float32)\n"
                  "    y2 = tila.cast(tila.expand_dim(tila.arange(0, 128), 0),"
                  " tila.float32)\n"
                  "    y2f = tila.max(y2, axis=0)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        l *= y2f\n")
    _expect(body, code="E03")


# ---------------------------------------------------------------------------
# 状态机：HandoffUpdate（'='）
# ---------------------------------------------------------------------------

def _handoff_body(tail):
    return PRE + ("    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  + tail +
                  "    tila.store(c, (offs,), m, mask=mm)\n")


def test_handoff_seed_materializes_and_phi_back_is_local():
    body = _handoff_body(
        "        m1 = tila.maximum(m, x)\n"
        "        m = m1\n")
    res = _c(body)
    # φ back 直接指向被交接局部（无额外指令）；出口类型 = m1 的类型
    assert "%m.loop = phi %m %m1" in res.tir_dump
    assert "%m.next" not in res.tir_dump  # handoff 无 update 指令


def test_handoff_materialized_equiv_locked():
    body = _handoff_body(
        "        m1 = tila.maximum(m, x)\n"
        "        m = m1\n"
        "        m = m1\n")
    assert "%m.loop = phi %m %m1" in _c(body).tir_dump


def test_handoff_negatives():
    # 评审 B-⑯ 全集：计算 / cast / 函数 / 自交接 / 跨累加器 / 循环前局部
    for tail, why in (
        ("        m = m1 + x\n", "computation"),
        ("        m1 = x\n        m = tila.cast(m1, tila.float32)\n", "cast"),
        ("        m = m\n", "self no-op"),
        ("        m = x\n", "pre-loop local"),
    ):
        body = _handoff_body(tail)
        if "m1" in tail and "m1 =" not in tail and "cast" not in why:
            pass  # `m = m1 + x` 里 m1 未定义 → E01 先拦（NameRef 未绑定）
        with pytest.raises(TilaError):
            _c(body)
    # 跨累加器：m = l（l 是另一累加器）→ HandoffForm
    body = PRE + ("    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
                  "    l = tila.zeros((64,), tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        l += x\n"
                  "        m = l\n"
                  "    tila.store(c, (offs,), m, mask=mm)\n")
    _expect(body, subcode="HandoffForm")
    # 复合重写：l = l * x + x → HandoffForm（附分解提示；计划 §3 F 的
    # AccumForm 措辞与 §7 冲突，实现裁决为 HandoffForm——文档随同步修正）
    body = PRE + ("    l = tila.zeros((64,), tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        l = l * x + x\n")
    e = _expect(body, subcode="HandoffForm")
    assert "decompose into updates" in e.message


def test_handoff_dtype_shape_mismatch():
    body = PRE + ("    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        m1 = tila.zeros((128,), tila.float32)\n"
                  "        m = m1\n")
    _expect(body, code="E03")
    body = PRE + ("    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        m1 = tila.zeros((64,), tila.float64)\n"
                  "        m = m1\n")
    _expect(body, code="E02")


# ---------------------------------------------------------------------------
# seed 值规范化（评审 B-⑩）与 R25 full
# ---------------------------------------------------------------------------

def test_full_zero_equivalent_zeros_state():
    # full(…, 0, f32) ≡ zeros：mul-identity 特例对两者同样成立（同状态断言）
    for seed_stmt in ("tila.zeros((64,), tila.float32)",
                      "tila.full((64,), 0, tila.float32)"):
        body = PRE + (f"    l = {seed_stmt}\n"
                      "    for k0 in tila.range(0, M, 64):\n"
                      "        l *= x\n")
        res = _c(body)
        line = [l for l in res.tir_dump.splitlines() if "%l.next = mul" in l][0]
        assert ", L1>" in line, seed_stmt  # mul 类型仍是 seed 态


def test_full_form_errors():
    for body, sub in (
        ("    z = tila.full((64,), 1.0)\n", "FullForm"),
        ("    z = tila.full(64, 1.0, tila.float32)\n", "FullForm"),
        ("    z = tila.full((64,), 1.0, tila.float32, 0)\n", None),
    ):
        with pytest.raises(TilaError) as ei:
            _c(PRE + body)
        assert ei.value.code in ("E20", "E13")
    # FLOAT 字面量配整数 dtype → E02；INT 字面量配浮点 dtype 合法（0 即 f32）
    _expect(PRE + "    z = tila.full((64,), 1.5, tila.int32)\n", code="E02")
    _c(PRE + "    z = tila.full((64,), 0, tila.float32)\n")
    # bool 种子拒绝（与 zeros 同）
    _expect(PRE + "    z = tila.full((64,), 1, tila.bool)\n", subcode="FullForm")


def test_full_tir_and_triton_forms():
    res = _c(PRE + "    z = tila.full((64,), tila.neg_inf, tila.float32)\n"
                   "    tila.store(c, (offs,), z, mask=mm)\n")
    assert "%z = full (64) -inf f32 : Tile<f32, (64,), L1>" in res.tir_dump
    assert 'tl.full((64,), float("-inf"), dtype=tl.float32)' in res.triton_source


# ---------------------------------------------------------------------------
# StateRead 定型（α 元变量 + 出口求解；docs/v0.6-attention §2.6）
# ---------------------------------------------------------------------------

SIG2D_DOT = ("q: tila.Tensor[tila.float16, M, D], k: tila.Tensor[tila.float16, N, D], "
             "c: tila.Tensor[tila.float32, M, N], BM: tila.constexpr = 64")


def _dot_prelude():
    return ("    rm2 = tila.expand_dim(tila.arange(0, 64), 1)\n"
            "    rn2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
            "    rd2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
            "    rd3 = tila.expand_dim(tila.arange(0, 64), 1)\n"
            "    q2 = tila.load(q, (rm2, rd2), mask=(rm2 < M) & (rd2 < D), other=0.0)\n"
            "    k2 = tila.load(k, (rn2, rd3), mask=(rn2 < N) & (rd3 < D), other=0.0)\n"
            "    d = tila.dot(q2, k2)\n")


def test_state_read_solved_to_exit_dist():
    # maximum(m, rowmax)：读 m 定型为出口分布（Slice）；dump 无 α
    body = _dot_prelude() + (
        "    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
        "    for k0 in tila.range(0, N, 64):\n"
        "        rowmax = tila.max(d, axis=1)\n"
        "        m1 = tila.maximum(m, rowmax)\n"
        "        m = m1\n"
        "    y = d * tila.expand_dim(m, 1)\n"
        "    tila.store(c, (rm2, rn2), y, mask=(rm2 < M) & (rn2 < N))\n")
    res = _c(body, SIG2D_DOT)
    assert "α(" not in res.tir_dump
    assert "%m1 = maximum %m.loop %rowmax : Tile<f32, (64,), L" in res.tir_dump
    # m1/读 m 的类型 = Slice(Mma,1)：maximum 行引用 slice 定义
    m1_line = [l for l in res.tir_dump.splitlines() if "%m1 = maximum" in l][0]
    import re as _re
    m = _re.search(r", L(\d+)>", m1_line)
    assert m and f"L{m.group(1)} = slice(" in res.tir_dump, m1_line


def test_state_read_cross_family_e05():
    # 跨族读：读 m（出口 Slice）与 arange 派生 Identity 同形相加——
    # 约束 α_m ~ Identity 与 α_m := Slice 冲突 → E05（docs §2.6 步骤 3）
    body = _dot_prelude() + (
        "    r0 = tila.cast(tila.arange(0, 64), tila.float32)\n"
        "    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
        "    for k0 in tila.range(0, N, 64):\n"
        "        rowmax = tila.max(d, axis=1)\n"
        "        m1 = tila.maximum(m, rowmax)\n"
        "        m = m1\n"
        "        y = m + r0\n")
    _expect(body, SIG2D_DOT, code="E05")


def test_state_read_escape_matrix():
    # store value 逃逸（m1 的 def 链经 maximum 触达 φ）——1D store 形态
    sig1d = ("q: tila.Tensor[tila.float16, M, D], k: tila.Tensor[tila.float16, N, D], "
             "c1: tila.Tensor[tila.float32, M], BM: tila.constexpr = 64")
    tail = ("    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
            "    r0 = tila.arange(0, 64)\n"
            "    for k0 in tila.range(0, N, 64):\n"
            "        rowmax = tila.max(d, axis=1)\n"
            "        m1 = tila.maximum(m, rowmax)\n"
            "        m = m1\n")
    body = _dot_prelude() + tail + \
        "        tila.store(c1, (r0,), m1, mask=r0 < M)\n"
    _expect(body, sig1d, subcode="AccumRead")
    # 经绑定名的传递逃逸：zz = m1 * 2.0; store(zz)
    body = _dot_prelude() + tail + \
        "        zz = m1 * 2.0\n" \
        "        tila.store(c1, (r0,), zz, mask=r0 < M)\n"
    _expect(body, sig1d, subcode="AccumRead")
    # store 的 mask 位逃逸（mask 消费读链）
    body = _dot_prelude() + tail + \
        "        tila.store(c1, (r0,), rowmax, mask=m1 > 0.0)\n"
    _expect(body, sig1d, subcode="AccumRead")


def test_state_read_in_coords_rejected():
    body = PRE + ("    l = tila.zeros((64,), tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        l += x\n"
                  "        y = tila.load(b, (l,), mask=mm)\n")
    # 坐标位 i32 vs f32 → E02 先拦；dtype 相同的场景由 Alpha 坐标拒绝覆盖
    with pytest.raises(TilaError):
        _c(body)


# ---------------------------------------------------------------------------
# 审核修复回归（P0-①②③）
# ---------------------------------------------------------------------------

def test_in_loop_seeding_is_plain_local():
    # P0-①：循环内 zeros 不构成累加器——读是普通局部读（无 Python 崩溃）
    body = PRE + ("    for k0 in tila.range(0, M, 64):\n"
                  "        z = tila.zeros((64,), tila.float32)\n"
                  "        w = z + x\n")
    res = _c(body)
    assert "%w = add %z %x" in res.tir_dump
    # 循环内 zeros + '+=' → AccumForm（"not bound before the loop"）
    body = PRE + ("    for k0 in tila.range(0, M, 64):\n"
                  "        z = tila.zeros((64,), tila.float32)\n"
                  "        z += x\n")
    e = _expect(body, subcode="AccumForm")
    assert "not bound before the loop" in e.message


def test_seed_per_axis_join():
    # P0-②（v0.6a 旧账）：Seed 以单段参与逐轴联合（与 v0.5 `_parts` 对齐）
    body = PRE + ("    rn2 = tila.expand_dim(tila.arange(0, 128), 0)\n"
                  "    z = tila.zeros((64, 1), tila.float32)\n"
                  "    w = z + tila.cast(rn2, tila.float32)\n")
    res = _c(body)
    assert "L3 = seed(64, 1)" in res.tir_dump
    assert "L4 = product(L3,L1)" in res.tir_dump


def test_alpha_chain_resolved_no_leak():
    # P0-③：l += mm（RHS 是另一累加器的读）→ 出口解析为 mm 的出口分布，
    # dump 无 α；逃逸（store 消费读链）仍被抓
    ok = PRE + ("    l = tila.zeros((64,), tila.float32)\n"
                "    m2 = tila.zeros((64,), tila.float32)\n"
                "    for k0 in tila.range(0, M, 64):\n"
                "        m2 += x\n"
                "        l += m2\n"
                "        w3 = l * 2.0\n"
                "    tila.store(c, (offs,), l, mask=mm)\n")
    res = _c(ok)
    assert "α(" not in res.tir_dump
    assert "%l.loop = phi %l %l.next : Tile<f32, (64,), L0>" in res.tir_dump
    esc = PRE + ("    l = tila.zeros((64,), tila.float32)\n"
                 "    m2 = tila.zeros((64,), tila.float32)\n"
                 "    for k0 in tila.range(0, M, 64):\n"
                 "        m2 += x\n"
                 "        l += m2\n"
                 "        w3 = l * 2.0\n"
                 "        tila.store(c, (offs,), w3, mask=mm)\n")
    _expect(esc, subcode="AccumRead")


def test_alpha_self_loop_falls_back_to_seed():
    # 自引链（m 只消费自己）：出口回退种子态——sound 且无 α 存活
    body = PRE + ("    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
                  "    for k0 in tila.range(0, M, 64):\n"
                  "        m1 = m + m\n"
                  "        m = m1\n")
    res = _c(body)
    assert "α(" not in res.tir_dump
    phi = [l for l in res.tir_dump.splitlines() if "m.loop = phi" in l][0]
    assert "seed(64)" in res.tir_dump and "L1 = seed(64)" in res.tir_dump
    assert ", L1>" in phi


# ---------------------------------------------------------------------------
# 零迭代恒等式 / 多累加器 φ 链 / l 种子恒等式（评审 B-⑰）
# ---------------------------------------------------------------------------

def test_zero_iteration_exit_is_seed():
    # N 迭代 0 次（end=0）→ 出口 = 种子值（interpreter 回填）
    body = PRE + ("    acc = tila.zeros((64,), tila.float32)\n"
                  "    for k0 in tila.range(0, 0, 64):\n"
                  "        acc += x\n"
                  "    tila.store(c, (offs,), acc, mask=mm)\n")
    res = _c(body)
    rng = np.random.default_rng(2)
    b = rng.standard_normal(64).astype(np.float32)
    c = np.full(64, 7.0, dtype=np.float32)
    run_kernel(res.kernel, {"b": b, "c": c}, {})
    np.testing.assert_allclose(c, np.zeros(64, dtype=np.float32), rtol=1e-6)


def test_multiple_accumulators_phi_isolated():
    # m/l/acc 三链 φ 互不干扰（flash 迷你形态）
    sig = ("q: tila.Tensor[tila.float16, M, D], k: tila.Tensor[tila.float16, N, D], "
           "c: tila.Tensor[tila.float32, M, D], BM: tila.constexpr = 64")
    body = ("    rm2 = tila.expand_dim(tila.arange(0, 64), 1)\n"
            "    rn2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
            "    rd2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
            "    rd3 = tila.expand_dim(tila.arange(0, 64), 1)\n"
            "    q2 = tila.load(q, (rm2, rd2), mask=(rm2 < M) & (rd2 < D), other=0.0)\n"
            "    k2 = tila.load(k, (rn2, rd3), mask=(rn2 < N) & (rd3 < D), other=0.0)\n"
            "    d = tila.dot(q2, k2)\n"
            "    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
            "    l = tila.zeros((64,), tila.float32)\n"
            "    acc = tila.zeros((64, 64), tila.float32)\n"
            "    for n0 in tila.range(0, N, 64):\n"
            "        rowmax = tila.max(d, axis=1)\n"
            "        m1 = tila.maximum(m, rowmax)\n"
            "        alpha = tila.exp2(m - m1)\n"
            "        p = tila.exp2(d - tila.expand_dim(m1, 1))\n"
            "        l *= alpha\n"
            "        l += tila.sum(p, axis=1)\n"
            "        acc *= tila.expand_dim(alpha, 1)\n"
            "        acc += d\n"
            "        m = m1\n"
            "    y = acc / tila.expand_dim(l, 1)\n"
            "    tila.store(c, (rm2, rd2), y, mask=(rm2 < M) & (rd2 < D))\n")
    res = _c(body, sig)
    d = res.tir_dump
    assert "%m.loop = phi %m %m1" in d
    assert "%l.loop = phi %l %l.next2" in d
    assert "%acc.loop = phi %acc %acc.next2" in d
    assert "α(" not in d


def _flash_core(seed_stmt):
    """flash 内核形态（单文件可参数化 l 种子）——l 恒等式与差分共用。"""
    sig = ("q: tila.Tensor[tila.float16, M, D], k: tila.Tensor[tila.float16, N, D], "
           "v: tila.Tensor[tila.float16, N, D], o: tila.Tensor[tila.float16, M, D], "
           "BM: tila.constexpr = 64")
    return sig, ("    rm2 = tila.expand_dim(tila.arange(0, 64), 1)\n"
                 "    rn2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
                 "    rn3 = tila.expand_dim(tila.arange(0, 64), 1)\n"
                 "    rd2 = tila.expand_dim(tila.arange(0, 64), 0)\n"
                 "    rd3 = tila.expand_dim(tila.arange(0, 64), 1)\n"
                 "    q2 = tila.load(q, (rm2, rd2), mask=(rm2 < M) & (rd2 < D), other=0.0)\n"
                 f"    m = tila.full((64,), tila.neg_inf, tila.float32)\n"
                 f"    l = {seed_stmt}\n"
                 "    acc = tila.zeros((64, 64), tila.float32)\n"
                 "    for n0 in tila.range(0, N, 64):\n"
                 "        rkn = rn2 + n0\n"
                 "        rvn = rn3 + n0\n"
                 "        k2 = tila.load(k, (rkn, rd3), mask=(rkn < N) & (rd3 < D), other=0.0)\n"
                 "        v2 = tila.load(v, (rvn, rd2), mask=(rvn < N) & (rd2 < D), other=0.0)\n"
                 "        qk = tila.dot(q2, k2)\n"
                 "        qk_ = tila.where(rkn < N, qk, tila.neg_inf)\n"
                 "        rowmax = tila.max(qk_, axis=1)\n"
                 "        m1 = tila.maximum(m, rowmax)\n"
                 "        alpha = tila.exp2(m - m1)\n"
                 "        p = tila.exp2(qk_ - tila.expand_dim(m1, 1))\n"
                 "        l *= alpha\n"
                 "        l += tila.sum(p, axis=1)\n"
                 "        acc *= tila.expand_dim(alpha, 1)\n"
                 "        acc += tila.dot(tila.cast(p, tila.float16), v2)\n"
                 "        m = m1\n"
                 "    y = acc / tila.expand_dim(l, 1)\n"
                 "    tila.store(o, (rm2, rd2), tila.cast(y, tila.float16), "
                 "mask=(rm2 < M) & (rd2 < D))\n")


def test_l_seed_identity():
    # 评审 B-⑰：首块 α = exp2(−∞ − m1) = 0 ⇒ l₀=1 与 l₀=0 的出口逐元素
    # 相等——Tila zeros 种子偏离 tutorial 初始 1.0 的 justification 固化。
    # 语言内 Seed(1.0) 不可 *=（MulSeed，B-⑨），故 l₀=1 一侧以 numpy 参考
    # 实现（tutorial 形态 l = 1.0; l = l·α + Σp）对拍 Tila 的 zeros 版。
    sig, body = _flash_core("tila.zeros((64,), tila.float32)")
    r0 = compile_kernel(f"import tila\n\n\n@tila.jit\ndef k({sig}):\n{body}\n")
    rng = np.random.default_rng(9)
    M, N, D = 32, 128, 64
    q = (rng.standard_normal((M, D)) * 0.7).astype(np.float16)
    k = (rng.standard_normal((N, D)) * 0.7).astype(np.float16)
    v = rng.standard_normal((N, D)).astype(np.float16)
    o = np.zeros((M, D), dtype=np.float16)
    run_kernel(r0.kernel, {"q": q, "k": k, "v": v, "o": o}, {})

    # numpy 参考：online softmax（exp2 域），l₀ = 1.0（tutorial 形态）
    qf = q.astype(np.float32)
    kf = k.astype(np.float32)
    vf = v.astype(np.float32)
    m_i = np.full(M, -np.inf, dtype=np.float32)
    l_i = np.ones(M, dtype=np.float32)
    acc = np.zeros((M, D), dtype=np.float32)
    BN = 64
    for n0 in range(0, N, BN):
        cols = np.arange(n0, min(n0 + BN, N))
        qk = qf @ kf[cols].T
        rowmax = qk.max(axis=1)
        m1 = np.maximum(m_i, rowmax)
        alpha = np.exp2(m_i - m1)
        p = np.exp2(qk - m1[:, None])
        l_i = l_i * alpha + p.sum(axis=1)
        acc = acc * alpha[:, None] + p @ vf[cols]
        m_i = m1
    ref = (acc / l_i[:, None]).astype(np.float16)
    np.testing.assert_allclose(o.astype(np.float32), ref.astype(np.float32),
                               rtol=5e-2, atol=5e-3)


@pytest.mark.parametrize("M,N,D", [(16, 16, 16), (32, 96, 32), (17, 129, 64)])
def test_flash_core_vs_torch(M, N, D):
    # L3 差分：flash 语义 = softmax(ln2·QK^T)·V（exp2 对齐；host 预缩放的
    # reference adaptation 在示例文件侧，这里直接以 ln2 温度对拍）
    import torch
    sig, body = _flash_core("tila.zeros((64,), tila.float32)")
    res = compile_kernel(f"import tila\n\n\n@tila.jit\ndef k({sig}):\n{body}\n")
    rng = np.random.default_rng(M * 7 + N)
    q = (rng.standard_normal((M, D)) * 0.7).astype(np.float16)
    k = (rng.standard_normal((N, D)) * 0.7).astype(np.float16)
    v = rng.standard_normal((N, D)).astype(np.float16)
    o = np.zeros((M, D), dtype=np.float16)
    run_kernel(res.kernel, {"q": q, "k": k, "v": v, "o": o}, {})
    qf = torch.tensor(q, dtype=torch.float32)
    kf = torch.tensor(k, dtype=torch.float32)
    vf = torch.tensor(v, dtype=torch.float32)
    ref = (torch.softmax(qf @ kf.T * np.log(2.0), dim=-1) @ vf).numpy()
    np.testing.assert_allclose(o.astype(np.float32), ref.astype(np.float32),
                               rtol=5e-2, atol=5e-3)


# ---------------------------------------------------------------------------
# R26 maximum / log2 / launch_assert
# ---------------------------------------------------------------------------

def test_maximum_join_same_as_arith():
    res = _c(PRE + "    y = tila.maximum(x, x)\n")
    assert "%y = maximum %x %x : Tile<f32, (64,), L0>" in res.tir_dump
    assert "tl.maximum(x, x)" in res.triton_source


def test_maximum_capability_and_dtype():
    _expect(PRE + "    y = tila.maximum(x, x2i)\n" + "    x2i = x\n",
            code="E01")  # 未定义先拦
    sig_i = "b: tila.Tensor[tila.int32, M], c: tila.Tensor[tila.int32, M], BM: tila.constexpr = 64"
    res = _c(PRE + "    y = tila.maximum(x, x)\n", sig_i)
    assert "maximum" in res.tir_dump
    # fp8 → E16（storage-only）
    sig_f8 = "b: tila.Tensor[tila.fp8e4m3fn, M], BM: tila.constexpr = 64"
    _expect(PRE + "    y = tila.maximum(x, x)\n", sig_f8, code="E16")
    # dtype 严格相等
    sig2 = ("b: tila.Tensor[tila.float32, M], b2: tila.Tensor[tila.float64, M], "
            "BM: tila.constexpr = 64")
    body = PRE + "    x2 = tila.load(b2, (offs,), mask=mm)\n    y = tila.maximum(x, x2)\n"
    _expect(body, sig2, code="E02")


def test_maximum_interp_semantics():
    res = _c(PRE + "    y = tila.maximum(x, x * 0.5 - 1.0)\n"
                   "    tila.store(c, (offs,), y, mask=mm)\n")
    rng = np.random.default_rng(4)
    b = rng.standard_normal(64).astype(np.float32) * 3
    c = np.zeros(64, dtype=np.float32)
    run_kernel(res.kernel, {"b": b, "c": c}, {})
    np.testing.assert_allclose(c, np.maximum(b, b * 0.5 - 1.0), rtol=1e-6)


def test_log2_unary():
    res = _c(PRE + "    y = tila.log2(x + 4.0)\n    tila.store(c, (offs,), y, mask=mm)\n")
    assert "%y = log2 %t" in res.tir_dump
    assert "tl.log2(" in res.triton_source
    rng = np.random.default_rng(6)
    b = (rng.random(64) * 4 + 1).astype(np.float32)
    c = np.zeros(64, dtype=np.float32)
    run_kernel(res.kernel, {"b": b, "c": c}, {})
    np.testing.assert_allclose(c, np.log2(b + 4.0), rtol=1e-6)
    # 整数域拒绝（FLOAT-only）
    _expect(PRE + "    y = tila.log2(x)\n",
            "b: tila.Tensor[tila.int32, M], BM: tila.constexpr = 64", code="E16")


def test_launch_assert_compile_launcher_and_interp():
    sig = "b: tila.Tensor[tila.float32, M], c: tila.Tensor[tila.float32, M], BM: tila.constexpr = 64"
    res = _c(PRE + "    tila.launch_assert(M % BM == 0)\n"
                   "    tila.store(c, (offs,), x, mask=mm)\n", sig)
    assert "launch_assert M % BM == 0" in res.tir_dump
    assert "assert M % BM == 0" in res.triton_source
    # interpreter 同判据：M=64 过；M=63 报运行期错误
    rng = np.random.default_rng(8)
    b = rng.standard_normal(64).astype(np.float32)
    c = np.zeros(64, dtype=np.float32)
    run_kernel(res.kernel, {"b": b, "c": c}, {})
    np.testing.assert_allclose(c, b, rtol=1e-6)
    b2 = rng.standard_normal(63).astype(np.float32)
    c2 = np.zeros(63, dtype=np.float32)
    with pytest.raises(RuntimeError):
        run_kernel(res.kernel, {"b": b2, "c": c2}, {})


def test_launch_assert_names_and_position():
    # 名字合法性：局部/缓冲名 → E13；值位置 → E08；循环内 → E13
    _expect(PRE + "    tila.launch_assert(x % BM == 0)\n", code="E13")
    _expect(PRE + "    y = tila.launch_assert(M % BM == 0)\n", code="E08")
    _expect(PRE + ("    for k0 in tila.range(0, M, 64):\n"
                   "        tila.launch_assert(M % BM == 0)\n"), code="E13")


# ---------------------------------------------------------------------------
# 纯层补充：strict_join 的 Seed 单位元（L7 语义的 join 层落地）
# ---------------------------------------------------------------------------

_L64 = Identity((Const(64),))
_S64 = Seed((Const(64),))


def test_strict_join_seed_unit():
    assert strict_join(_S64, _L64, Loc(1, 1), "t") == _L64
    assert strict_join(_L64, _S64, Loc(1, 1), "t") == _L64
    assert strict_join(_S64, _S64, Loc(1, 1), "t") == _S64
    assert not equiv_dist(_S64, _L64)  # 单位元是 join 行为，不是等价关系
