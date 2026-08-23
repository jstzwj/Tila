"""checker typing 规则测试：R2–R12 组合、TIR 不变量、签名/参数序、constexpr 特化。"""

import pytest

from tila.driver import compile_kernel
from tila.diagnostics import TilaError
from tila import tir

F32 = "a: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]"


def _c(body, params=F32, overrides=None):
    return compile_kernel(
        f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n", overrides)


PRELUDE = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
"""


# ---------------------------------------------------------------------------
# R12 字面量上下文规则
# ---------------------------------------------------------------------------

def test_r12_float_literal_matches_f16_tile():
    res = _c(PRELUDE.replace("x = tila.load(a, (offs,), mask=mask)",
                             "x = tila.load(a, (offs,), mask=mask)")
             .replace("a: tila.Tensor[tila.float32, N]", "a: tila.Tensor[tila.float16, N]")
             + "    z = x + 1.0\n    m2 = x > 0.0\n",
             params="a: tila.Tensor[tila.float16, N], c: tila.Tensor[tila.float16, N]")
    assert "Tile<f16, (128,), L0>" in res.tir_dump


def test_r12_int_tile_with_int_literal():
    res = _c("    offs = tila.arange(0, 128)\n"
             "    m = offs > 4\n",
             params="a: tila.Tensor[tila.int32, N], c: tila.Tensor[tila.int32, N]")
    assert "Tile<bool, (128,), L0>" in res.tir_dump


def test_r12_named_scalar_does_not_adapt():
    # 具名 i32 标量与 f32 tile 运算 → E02（只有字面量才按 R12 解释）
    with pytest.raises(TilaError) as ei:
        _c(PRELUDE + "    s = pid\n    z = x * s\n")
    assert ei.value.code == "E02"


# ---------------------------------------------------------------------------
# R4/R5/R6/R10
# ---------------------------------------------------------------------------

def test_r4_scalar_broadcast_both_sides():
    body = PRELUDE + """    s = 2.0
    t = x * s
    u = s * x
    tila.store(c, (offs,), u, mask=mask)
"""
    res = _c(body)
    assert "%t = mul %x %s" in res.tir_dump or "mul %x" in res.tir_dump


def test_r6_tile_tile_comparison():
    body = PRELUDE + """    y = x + x
    m = x < y
    tila.store(c, (offs,), y, mask=m)
"""
    res = _c(body)
    assert "%m" in res.tir_dump and "Tile<bool, (128,), L0>" in res.tir_dump


def test_r10_bool_and_or():
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    m1 = offs < N
    m2 = offs > 0
    m = m1 & m2
    x = tila.load(a, (offs,), mask=m)
    tila.store(c, (offs,), x, mask=m)
"""
    res = _c(body)
    assert "%m = and %m1 %m2 : Tile<bool, (128,), L0> [#m]" in res.tir_dump
    assert "m = m1 & m2" in res.triton_source


def test_r10_anonymous_cmps_use_m_counter():
    """匿名的比较结果按出现顺序 %m0、%m1…（黄金 batched dump 同款命名）。"""
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    m = (offs < N) & (offs > 0)
    x = tila.load(a, (offs,), mask=m)
    tila.store(c, (offs,), x, mask=m)
"""
    res = _c(body)
    assert "%m0 = lt %offs %N : Tile<bool, (128,), L0>" in res.tir_dump
    assert "%m1 = gt %offs" in res.tir_dump
    assert "%m = and %m0 %m1 : Tile<bool, (128,), L0> [#m]" in res.tir_dump


def test_r11_scalar_cast_chain():
    body = (PRELUDE + "    a2 = tila.cast(pid, tila.float32)\n"
            "    a3 = tila.cast(a2, tila.float16)\n"
            "    a4 = tila.cast(a3, tila.bool)\n"
            "    tila.store(c, (offs,), x, mask=mask)\n")
    res = _c(body)
    assert "cast %pid f32 : Scalar(f32) [#a2]" in res.tir_dump
    assert "cast %a2 f16 : Scalar(f16) [#a3]" in res.tir_dump


def test_address_is_internal_e07():
    """v0.3 定稿：Address 完全内部化——`p = a + offs` 的地址算术不可写（E07）；
    坐标只出现在 tila.load/tila.store 的实参位置（docs/v0.3-strides.md §1.2）。"""
    body = """    offs = tila.arange(0, 128)
    p = a + offs
"""
    with pytest.raises(TilaError) as ei:
        _c(body)
    assert ei.value.code == "E07"
    assert "compiler-owned" in ei.value.message


# ---------------------------------------------------------------------------
# constexpr（模型 B：检查逐值特化、发射保留名字）
# ---------------------------------------------------------------------------

ARANGE_BLOCK = """    r = tila.arange(0, BLOCK)
"""


def test_constexpr_binop_boundary():
    # BLOCK // 2 是合法 arange 边界（language-spec §7）：检查期求值
    res = _c("    r = tila.arange(0, BLOCK // 2)\n",
             params="a: tila.Tensor[tila.float32, N], BLOCK: tila.constexpr = 128")
    assert "Tile<i32, (64,), L0>" in res.tir_dump
    assert "%r = arange 0 BLOCK // 2 : Tile<i32, (64,), L0> [#r]" in res.tir_dump
    assert "tl.arange(0, BLOCK // 2)" in res.triton_source


def test_constexpr_binop_checked_per_value():
    # 逐值特化：BLOCK=66 时 66//2=33 非 2 的幂 → E06
    with pytest.raises(TilaError) as ei:
        _c(ARANGE_BLOCK,
           params="a: tila.Tensor[tila.float32, N], BLOCK: tila.constexpr = 128",
           overrides={"BLOCK": 66})
    assert ei.value.code == "E06"


def test_constexpr_promotes_to_scalar():
    body = """    t = BLOCK + 1
    r = tila.arange(0, BLOCK)
"""
    res = _c(body, params="a: tila.Tensor[tila.float32, N], BLOCK: tila.constexpr = 128")
    assert "%t0 = const 1 : Scalar(i32)" in res.tir_dump
    assert "%t = add %BLOCK %t0 : Scalar(i32) [#t]" in res.tir_dump


def test_arange_upper_bound_pow20():
    res = _c("    r = tila.arange(0, 1048576)\n")
    assert "Tile<i32, (1048576,)," in res.tir_dump
    with pytest.raises(TilaError) as ei:
        _c("    r = tila.arange(0, 2097152)\n")
    assert ei.value.code == "E06"


# ---------------------------------------------------------------------------
# 签名与参数顺序（triton-lowering §3：buffer → sym → constexpr）
# ---------------------------------------------------------------------------

def test_param_order_buffers_syms_scalars_constexpr():
    res = _c("    r = tila.arange(0, 128)\n",
             params="a: tila.Tensor[tila.float32, N], "
                    "b: tila.Tensor[tila.float32, M], "
                    "alpha, BLOCK: tila.constexpr = 8")
    sig = [l for l in res.triton_source.splitlines() if l.startswith("def k(")][0]
    assert sig == "def k(a, b, N, M, alpha, BLOCK: tl.constexpr):"
    header = res.tir_dump.splitlines()[0]
    assert header == ("func @k(a: Buffer<f32, (N,)>, b: Buffer<f32, (M,)>, "
                      "N: Scalar(i32), M: Scalar(i32), alpha: Scalar(i32), "
                      "BLOCK: Constexpr(i32)=8)")


def test_sym_dim_shared_across_buffers_is_runtime_contract():
    res = _c(PRELUDE + "    tila.store(c, (offs,), x, mask=mask)\n")
    assert "N = a.shape[0]" in res.triton_source
    assert "assert c.shape[0] == N" in res.triton_source


def test_static_dim_asserts_emitted():
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < 1024
    x = tila.load(a, (offs,), mask=mask)
    tila.store(c, (offs,), x, mask=mask)
"""
    res = _c(body, params="a: tila.Tensor[tila.float32, 1024], "
                          "c: tila.Tensor[tila.float32, 1024]")
    assert "assert a.shape[0] == 1024 and c.shape[0] == 1024" in res.triton_source
    assert "grid = (triton.cdiv(1024, 128),)" in res.triton_source


def test_no_program_id_means_single_program_grid():
    res = _c("    r = tila.arange(0, 128)\n",
             params="a: tila.Tensor[tila.float32, 1024], c: tila.Tensor[tila.float32, 1024]")
    assert "grid = (1,)" in res.triton_source


# ---------------------------------------------------------------------------
# TIR 不变量（ast.md §6）
# ---------------------------------------------------------------------------

def _ops(res):
    return [op for op in res.kernel.ops if op.id is not None]


def test_invariant_ids_unique_and_defined_before_use():
    from tila.driver import compile_kernel as ck

    res = ck(open("examples/add.tila", encoding="utf-8").read())
    seen = set()
    for op in _ops(res):
        assert op.id not in seen, f"duplicate id {op.id}"
        seen.add(op.id)
        for operand in _operand_ids(op):
            if operand in {p.name for p in res.kernel.params}:
                continue
            assert operand in seen, f"use of {operand} before definition"


def _operand_ids(op):
    ids = []
    for f in ("lhs", "rhs", "offs", "ptr", "value", "mask", "other", "operand"):
        v = getattr(op, f, None)
        if isinstance(v, str):
            ids.append(v)
    return ids


def test_invariant_store_ptr_is_addptr_and_base_is_buffer():
    res = compile_kernel(open("examples/add.tila", encoding="utf-8").read())
    by_id = {op.id: op for op in res.kernel.ops}
    buffers = {p.name for p in res.kernel.params if p.kind == "buffer"}
    for op in res.kernel.ops:
        if isinstance(op, tir.TStore):
            assert isinstance(by_id[op.ptr], tir.TAddPtr)
            assert op.ptr.startswith("%") is False
        if isinstance(op, tir.TAddPtr):
            assert op.base in buffers
        if isinstance(op, (tir.TLoad, tir.TStore)):
            src = by_id[op.ptr]
            assert isinstance(src, tir.TAddPtr)


def test_invariant_anonymous_used_once_and_layout_normal():
    from tila.types import normalize

    res = compile_kernel(open("examples/add.tila", encoding="utf-8").read())
    uses = {}
    for op in res.kernel.ops:
        for oid in _operand_ids(op):
            uses[oid] = uses.get(oid, 0) + 1
    for op in _ops(res):
        if op.src_name is None and not isinstance(op, (tir.TSymRef, tir.TConstParamRef)):
            assert uses.get(op.id, 0) == 1, f"匿名值 {op.id} 使用次数 != 1"
        if op.tila_type is not None and hasattr(op.tila_type, "layout"):
            assert normalize(op.tila_type.layout) == op.tila_type.layout, \
                "TIR 类型中的 layout 必须已是正规形式"


def test_alias_assignment_shares_id():
    body = PRELUDE + "    y = x\n    z = y + x\n    tila.store(c, (offs,), z, mask=mask)\n"
    res = _c(body)
    # y 与 x 共享同一 id（无新指令）；z 的操作数是 %x
    assert "%y = " not in res.tir_dump
    assert "%z = add %x %x" in res.tir_dump
