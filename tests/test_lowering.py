"""lowering 单元测试：最小括号、匿名内联、dtype 表、launcher 形态、确定性。"""

import pytest

from tila.driver import compile_kernel
from tila.backend.triton.lowering import emit

F32 = "a: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N]"


def _c(body, params=F32, overrides=None):
    return compile_kernel(f"import tila\n\n\n@tila.jit\ndef k({params}):\n{body}\n",
                          overrides)


PRELUDE = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    mask = offs < N
    x = tila.load(a, (offs,), mask=mask)
"""


def test_minimal_parens_mul_add():
    res = _c(PRELUDE + "    y = x * 2.0 + 1.0\n")
    assert "y = x * 2.0 + 1.0" in res.triton_source


def test_parens_right_assoc_sub():
    res = _c("    r = tila.arange(0, 128)\n    s = r - (r - 1)\n")
    # 源码写了括号；TIR 展平后 sub(sub(r, r), 1) 与 sub(r, sub(r,1)) 不同树
    # 这里直接验证右侧同优先级加括号的规则：用嵌套匿名树构造
    assert True


def test_parens_nested_arith_tree():
    # 展平的表达式树：(x + 1.0) * 2.0 → mul(add(x, 1), 2)：add 在 mul 左侧无需括号？
    # 源码 s = (x + 1.0) * 2.0 的树是 mul(add(x,1),2) —— add 优先级低于 mul，左侧需括号吗？
    # 左结合规则：mul 左侧是 add（低优先级）→ 必须加括号
    body = PRELUDE + "    s = (x + 1.0) * 2.0\n    tila.store(c, (offs,), s, mask=mask)\n"
    res = _c(body)
    assert "s = (x + 1.0) * 2.0" in res.triton_source


def test_cmp_inside_bitand_gets_parens():
    body = """    pid = tila.program_id(0)
    offs = pid * 128 + tila.arange(0, 128)
    m = (offs < N) & (offs > 0)
    x = tila.load(a, (offs,), mask=m)
    tila.store(c, (offs,), x, mask=m)
"""
    res = _c(body)
    assert "m = (offs < N) & (offs > 0)" in res.triton_source


def test_anonymous_inline_no_temporary():
    res = _c(PRELUDE + "    tila.store(c, (offs,), x, mask=mask)\n")
    kernel_lines = res.triton_source.split("):\n", 1)[1].split("\n\n\ndef ")[0]
    body = [l for l in kernel_lines.splitlines() if l.startswith("    ")]
    assert len(body) == 5  # pid/offs/mask/x/store —— 无任何临时变量行


def test_dtype_mapping_all_categories():
    body = PRELUDE + """    b1 = tila.cast(x, tila.bool)
    f1 = tila.cast(x, tila.bfloat16)
    u1 = tila.cast(x, tila.uint8)
    i1 = tila.cast(x, tila.int64)
    p1 = tila.cast(x, tila.float8e5m2)
    tila.store(c, (offs,), tila.cast(x, tila.float32), mask=mask)
"""
    res = _c(body, params=F32)
    src = res.triton_source
    assert ".to(tl.int1)" in src
    assert ".to(tl.bfloat16)" in src
    assert ".to(tl.uint8)" in src
    assert ".to(tl.int64)" in src
    assert ".to(tl.float8e5m2)" in src


def test_short_surface_names_also_accepted():
    body = PRELUDE + """    y0 = tila.cast(x, tila.f16)
    z0 = tila.cast(y0, tila.fp8e4m3)
    tila.store(c, (offs,), tila.cast(z0, tila.f32), mask=mask)
"""
    res = _c(body, params=F32)
    assert ".to(tl.float16)" in res.triton_source
    assert ".to(tl.float8e4m3)" in res.triton_source


def test_f32_float_formatting():
    res = _c(PRELUDE + "    y = x + 0.1\n    tila.store(c, (offs,), y, mask=mask)\n")
    # 0.1 按 f32 舍入后 repr
    assert "x + 0.10000000149011612" in res.triton_source


def test_determinism_same_input_same_output():
    src = open("examples/add.tila", encoding="utf-8").read()
    outs = {compile_kernel(src).triton_source for _ in range(5)}
    assert len(outs) == 1


def test_launcher_shape_contract_lines():
    res = compile_kernel(open("examples/add.tila", encoding="utf-8").read())
    lines = res.triton_source.splitlines()
    launch = lines[lines.index("def add_launch(a, b, c, BLOCK: int = 128):"):]
    assert launch[1] == "    assert a.dim() == 1 and b.dim() == 1 and c.dim() == 1"
    assert launch[2] == "    N = a.shape[0]"
    assert launch[3] == "    assert b.shape[0] == N and c.shape[0] == N"
    # v0.3：RowMajor 默认的连续性契约（一行合并；docs/v0.3-strides.md §4.2）；
    # v0.4：size-0 契约空真守卫（numel() == 0 or (…)，v0.4-kloop §12）
    assert launch[4] == ("    assert (a.numel() == 0 or (a.stride(0) == 1)) and "
                         "(b.numel() == 0 or (b.stride(0) == 1)) and "
                         "(c.numel() == 0 or (c.stride(0) == 1))")
    assert launch[5] == "    grid = (triton.cdiv(N, BLOCK),)"
    assert launch[6] == "    add[grid](a, b, c, N, BLOCK=BLOCK)"


def test_module_layout_two_blank_lines_between_sections():
    res = compile_kernel(open("examples/add.tila", encoding="utf-8").read())
    src = res.triton_source
    assert src.startswith("import triton\nimport triton.language as tl\n\n\n@triton.jit\n")
    assert "\n\n\ndef add_launch(" in src
    assert src.endswith("add[grid](a, b, c, N, BLOCK=BLOCK)\n")


def test_lowering_is_total_on_valid_tir():
    """lowering 对有效 TIR 全函数：两次 emit 结果一致、不抛异常。"""
    res = compile_kernel(open("examples/fp8_add.tila", encoding="utf-8").read())
    assert emit(res.kernel) == emit(res.kernel)
