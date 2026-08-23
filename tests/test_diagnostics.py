"""诊断矩阵：E01–E17 每码一个最小复现，断言 code；并抽查 loc 与信息形态。

docs/type-checker.md §7 目录、§9 测试策略。错误信息必须包含两个操作数的
完整类型与来源行号（notes 固定形态）。
"""

import pytest

from tila.driver import compile_kernel
from tila.diagnostics import TilaError

HEADER = "import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n"


def _compile_body(body, params="", overrides=None):
    return compile_kernel(HEADER.format(params=params, body=body), overrides)


FILLER = "    r = tila.arange(0, 128)\n"


def _expect(body, code, params="", overrides=None):
    with pytest.raises(TilaError) as ei:
        _compile_body(body, params, overrides)
    assert ei.value.code == code, \
        f"expected {code}, got {ei.value.code}: {ei.value.message}"
    return ei.value


# ---------------------------------------------------------------------------
# checker 侧（E01–E10、E16、E17）
# ---------------------------------------------------------------------------

F32_ARGS = "a: tila.Tensor[tila.float32, N], b: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]"
PRELUDE = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    y = tila.load(b, (offs,), mask=mask)
"""


def test_E01_unbound_name():
    e = _expect("    z = x + y\n", "E01", params=F32_ARGS)
    assert "'x'" in e.message


def test_E02_dtype_mismatch_between_tiles():
    args = "a: tila.Tensor[tila.float32, N], b: tila.Tensor[tila.int32, N]"
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    yb = tila.load(b, (offs,), mask=mask)
    z = x + yb
"""
    e = _expect(body, "E02", params=args)
    rendered = e.render()
    assert "Tile<f32" in rendered and "Tile<i32" in rendered
    assert "tila.cast" in rendered  # 修复建议


def test_E02_store_narrowing_hint():
    args = "a: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float16, N]"
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    tila.store(c, (offs,), x, mask=mask)
"""
    e = _expect(body, "E02", params=args)
    assert "implicit narrowing" in " ".join(n.text for n in e.notes)


def test_E02_literal_category():
    # f16 tile + INT 字面量 → E02（提示写 1.0）
    args = "a: tila.Tensor[tila.float16, N], c: tila.Tensor[tila.float16, N]"
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    z = x + 1
    tila.store(c, (offs,), z, mask=mask)
"""
    e = _expect(body, "E02", params=args)
    assert "1.0" in " ".join(n.text for n in e.notes)


def test_E03_shape_mismatch():
    body = """    pid = tila.program_id(0)
    r1 = tila.arange(0, 128)
    r2 = tila.arange(0, 64)
    z = r1 + r2
"""
    e = _expect(body, "E03")
    rendered = e.render()
    assert "(128,)" in rendered and "(64,)" in rendered
    assert "dimension 0: 128 vs 64" in rendered


def test_E06_arange_not_pow2():
    _expect("    r = tila.arange(0, 100)\n", "E06")


def test_E06_arange_nonzero_start():
    _expect("    r = tila.arange(1, 128)\n", "E06")


def test_E13_load_arity():
    e = _expect("    x = tila.load(a)\n", "E13", params=F32_ARGS)
    assert "two positional" in e.message


def test_E07_load_first_arg_not_buffer():
    body = "    offs = tila.arange(0, 128)\n    x = tila.load(offs, (offs,))\n"
    e = _expect(body, "E07", params=F32_ARGS)
    assert "first argument" in e.message


def test_E07_scalar_comparison():
    body = """    pid = tila.program_id(0)
    m = pid < 4
"""
    _expect(body, "E07")


def test_E07_buffer_as_value():
    _expect("    z = a + 1\n", "E07", params=F32_ARGS)


def test_E08_store_in_value_position():
    body = PRELUDE + "    w = tila.store(c, (offs,), x, mask=mask)\n"
    _expect(body, "E08", params=F32_ARGS)


def test_E08_non_store_expr_stmt():
    body = PRELUDE + "    tila.load(c, (offs,))\n"
    _expect(body, "E08", params=F32_ARGS)


def test_E09_cast_buffer_operand():
    body = "    y = tila.cast(a, tila.float32)\n"
    _expect(body, "E09", params=F32_ARGS)


def test_E09_cast_literal_operand():
    _expect("    y = tila.cast(1, tila.float32)\n", "E09")


def test_E09_cast_bad_second_arg():
    _expect("    x0 = tila.load(a, (tila.arange(0, 128),))\n"
            "    y = tila.cast(x0, x0)\n", "E09", params=F32_ARGS)


def test_E10_arange_bound_runtime():
    body = """    pid = tila.program_id(0)
    r = tila.arange(0, N)
"""
    _expect(body, "E10", params=F32_ARGS)


def test_E16_fp8_arithmetic():
    args = ("a: tila.Tensor[tila.float8e4m3fn, N], b: tila.Tensor[tila.float8e4m3fn, N]")
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    y = tila.load(b, (offs,), mask=mask)
    z = x + y
"""
    e = _expect(body, "E16", params=args)
    assert "storage-only" in e.message
    assert "tila.cast" in e.message  # 错误信息给出正确写法


def test_E16_int_division():
    args = "a: tila.Tensor[tila.int32, N], b: tila.Tensor[tila.int32, N]"
    body = PRELUDE.replace("x = tila.load(a, (offs,), mask=mask)",
                           "x = tila.load(a, (offs,), mask=mask)").replace(
        "y = tila.load(b, (offs,), mask=mask)", "y = tila.load(b, (offs,), mask=mask)") + \
        "    q = x / y\n"
    _expect(body, "E16", params=args)


def test_E16_bool_arithmetic():
    body = PRELUDE + "    s = mask + mask\n"
    _expect(body, "E16", params=F32_ARGS)


def test_E16_int_bitand():
    args = "a: tila.Tensor[tila.int32, N], b: tila.Tensor[tila.int32, N]"
    body = PRELUDE + "    s = x & y\n"
    _expect(body, "E16", params=args)


def test_E14_reassignment():
    body = "    x = 1\n    x = 2\n"
    e = _expect(body, "E14", params=F32_ARGS)
    assert "single-assignment" in e.message


def test_E14_sym_dim_shadowing():
    body = PRELUDE + "    N = 5\n"
    _expect(body, "E14", params=F32_ARGS)


def test_E14_param_shadowing():
    body = PRELUDE + "    a = 1\n"
    _expect(body, "E14", params=F32_ARGS)


def test_E17_no_tiling_pattern():
    body = """    pid = tila.program_id(0)
    offs = pid + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    tila.store(a, (offs,), x, mask=mask)
"""
    _expect(body, "E17", params=F32_ARGS)


def test_E17_two_constexpr_same_axis():
    args = F32_ARGS + ", BLOCK: tila.constexpr = 128"
    body = """    pid = tila.program_id(0)
    t = pid * BLOCK + tila.arange(0, 128)
    u = pid * 64 + tila.arange(0, 128)
    offs = t + u
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
    tila.store(c, (offs,), x, mask=mask)
"""
    _expect(body, "E17", params=args)


def test_E17_no_bound_predicate():
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    x = tila.load(a, (offs,))
    tila.store(a, (offs,), x)
"""
    _expect(body, "E17", params=F32_ARGS)


# ---------------------------------------------------------------------------
# 转换/签名侧（E11–E15）
# ---------------------------------------------------------------------------

def test_E11_control_flow():
    with pytest.raises(TilaError) as ei:
        compile_kernel(HEADER.format(params=F32_ARGS,
                                     body="    if True:\n        x = 1\n"))
    assert ei.value.code == "E11"


def test_E11_from_import():
    src = "from tila import load\n"
    with pytest.raises(TilaError) as ei:
        compile_kernel(src)
    assert ei.value.code == "E11"


def test_E11_alias_import():
    src = "import tila as tl\n"
    with pytest.raises(TilaError) as ei:
        compile_kernel(src)
    assert ei.value.code == "E11"


def test_E11_syntax_error():
    with pytest.raises(TilaError) as ei:
        compile_kernel("def broken(:\n")
    assert ei.value.code == "E11"


def test_E11_subscript():
    with pytest.raises(TilaError) as ei:
        _compile_body("    z = a[0]\n", params=F32_ARGS)
    assert ei.value.code == "E11"


def test_E11_string_literal():
    with pytest.raises(TilaError) as ei:
        _compile_body('    s = "x"\n')
    assert ei.value.code == "E11"


def test_E11_user_function_call():
    with pytest.raises(TilaError) as ei:
        _compile_body("    z = foo(1)\n")
    assert ei.value.code == "E11"


def test_E12_rank3_annotation():
    # rank-2 已随 v0.2 二维预览解禁（docs/v0.2-preview-2d.md §2）
    with pytest.raises(TilaError) as ei:
        _compile_body(FILLER, params="a: tila.Tensor[tila.float32, M, N, K]")
    assert ei.value.code == "E12"


def test_E12_bool_buffer():
    with pytest.raises(TilaError) as ei:
        _compile_body(FILLER, params="a: tila.Tensor[tila.bool, 128]")
    assert ei.value.code == "E12"


def test_E12_unknown_dtype():
    with pytest.raises(TilaError) as ei:
        _compile_body(FILLER, params="a: tila.Tensor[tila.bfloat8, 128]")
    assert ei.value.code == "E12"


def test_E12_sym_conflicts_with_param():
    with pytest.raises(TilaError) as ei:
        _compile_body(FILLER,
                      params="a: tila.Tensor[tila.float32, N], N: tila.Tensor[tila.float32, 8]")
    assert ei.value.code == "E12"


def test_E12_reserved_param_name():
    with pytest.raises(TilaError) as ei:
        _compile_body(FILLER, params="tl: tila.Tensor[tila.float32, 128]")
    assert ei.value.code == "E12"


def test_E12_reserved_kernel_name():
    src = "import tila\n\n\n@tila.jit\ndef triton(a):\n    x = 1\n"
    with pytest.raises(TilaError) as ei:
        compile_kernel(src)
    assert ei.value.code == "E12"


def test_E13_unknown_intrinsic():
    with pytest.raises(TilaError) as ei:
        _compile_body("    z = tila.zzz(1)\n")
    assert ei.value.code == "E13"


def test_E13_dtype_ref_outside_cast():
    with pytest.raises(TilaError) as ei:
        _compile_body("    z = tila.float32\n")
    assert ei.value.code == "E13"


def test_E13_program_id_axis_2():
    # 轴集 {0,1}（v0.2 二维预览）；axis 2 仍拒绝
    with pytest.raises(TilaError) as ei:
        _compile_body("    pid = tila.program_id(2)\n")
    assert ei.value.code == "E13"


def test_E13_program_id_runtime_axis():
    with pytest.raises(TilaError) as ei:
        _compile_body("    pid = tila.program_id(N)\n", params=F32_ARGS)
    assert ei.value.code == "E13"


def test_E13_other_without_mask():
    body = """    offs = tila.arange(0, 128)
    x = tila.load(a, (offs,), other=0.0)
"""
    with pytest.raises(TilaError) as ei:
        _compile_body(body, params=F32_ARGS)
    assert ei.value.code == "E13"


def test_E13_unknown_kwarg():
    body = """    offs = tila.arange(0, 128)
    x = tila.load(a, (offs,), bogus=1)
"""
    with pytest.raises(TilaError) as ei:
        _compile_body(body, params=F32_ARGS)
    assert ei.value.code == "E13"


def test_E13_store_other_kwarg():
    body = PRELUDE + "    tila.store(c, (offs,), x, mask=mask, other=0.0)\n"
    with pytest.raises(TilaError) as ei:
        _compile_body(body, params=F32_ARGS)
    assert ei.value.code == "E13"


def test_E13_arange_wrong_arity():
    with pytest.raises(TilaError) as ei:
        _compile_body("    r = tila.arange(0)\n")
    assert ei.value.code == "E13"


def test_E13_override_unknown_constexpr():
    with pytest.raises(TilaError) as ei:
        _compile_body("    r = tila.arange(0, 128)\n", overrides={"NOPE": 4})
    assert ei.value.code == "E13"


def test_E14_tila_tensor_in_expression():
    with pytest.raises(TilaError) as ei:
        _compile_body("    z = tila.Tensor[tila.float32, 4]\n")
    assert ei.value.code == "E14"


def test_E15_non_int_default():
    with pytest.raises(TilaError) as ei:
        _compile_body("    r = tila.arange(0, BLOCK)\n",
                      params="BLOCK: tila.constexpr = 1.5")
    assert ei.value.code == "E15"


def test_E15_missing_default_without_override():
    with pytest.raises(TilaError) as ei:
        _compile_body("    r = tila.arange(0, BLOCK)\n",
                      params="BLOCK: tila.constexpr")
    assert ei.value.code == "E15"


def test_E15_no_default_but_override_ok():
    res = _compile_body("    r = tila.arange(0, BLOCK)\n",
                        params="BLOCK: tila.constexpr",
                        overrides={"BLOCK": 64})
    assert "Tile<i32, (64,)," in res.tir_dump


def test_error_rendering_has_loc_and_notes():
    e = _expect("    z = x + y\n", "E01", params=F32_ARGS)
    text = e.render()
    assert text.startswith("E01 UnboundName")
    assert "at " in text  # loc
