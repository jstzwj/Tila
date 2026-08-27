"""TIR → Triton 源码 lowering（docs/triton-lowering.md）。

纯语法映射：不做任何 Tila 语义拒绝，对有效 TIR 全函数（total）且确定
（黄金测试逐字节比对）。模块固定三段：imports、@triton.jit kernel、launcher。
产物为 LF、无注释、无尾随空白、文件尾恰好一个换行。
"""

from __future__ import annotations

from ... import tir
from ...types.memory import Strided
from .printer import OpRenderer


def _kernel_signature(kernel: tir.TKernel) -> str:
    parts = []
    for p in kernel.params:
        if p.kind == "constexpr":
            parts.append(f"{p.name}: tl.constexpr")
        else:
            parts.append(p.name)
    return ", ".join(parts)


def _launcher(kernel: tir.TKernel) -> list:
    plan = kernel.launch_plan
    buffers = [p.name for p in kernel.params if p.kind == "buffer"]
    scalars = [p.name for p in kernel.params if p.kind == "scalar"]
    constexprs = [p for p in kernel.params if p.kind == "constexpr"]
    syms = [p.name for p in kernel.params if p.kind == "sym"]

    sig = list(buffers) + list(scalars)
    for p in constexprs:
        sig.append(f"{p.name}: int = {p.default}" if p.default is not None else f"{p.name}: int")

    lines = [f"def {kernel.name}_launch({', '.join(sig)}):"]

    # rank 断言（每张量必发）
    if plan.rank_asserts:
        conds = " and ".join(f"{b}.dim() == {r}" for b, r in plan.rank_asserts)
        lines.append(f"    assert {conds}")

    # 静态维断言（仅注解含静态维时发射；与 rank 断言同样合并为一行）
    if plan.static_dim_asserts:
        conds = [f"{b}.shape[{i}] == {v}" for b, i, v in plan.static_dim_asserts]
        lines.append(f"    assert {' and '.join(conds)}")

    # 符号维取值 + 同符号运行时契约（按符号名比较）
    first_binding = {}
    extra = []
    for b, i, s in plan.sym_dim_asserts:
        if s not in first_binding:
            first_binding[s] = (b, i)
        else:
            extra.append(f"{b}.shape[{i}] == {s}")
    for s in syms:
        if s in first_binding:
            b, i = first_binding[s]
            lines.append(f"    {s} = {b}.shape[{i}]")
    if extra:
        lines.append(f"    assert {' and '.join(extra)}")

    # 内存布局契约（v0.3-strides §4.2）：stride 绑定/断言 + RowMajor 连续性断言
    assigned = set(first_binding)
    mem_conds = []
    for b, i, sv in plan.stride_bindings:
        if isinstance(sv, int):
            mem_conds.append(f"{b}.stride({i}) == {sv}")
        elif sv not in assigned:
            lines.append(f"    {sv} = {b}.stride({i})")
            assigned.add(sv)
        else:
            mem_conds.append(f"{b}.stride({i}) == {sv}")
    for p in kernel.params:
        if p.kind != "buffer":
            continue
        if isinstance(p.tila_type.mem, Strided):
            continue
        r = len(p.tila_type.shape)
        inner = [f"{p.name}.stride({r - 1}) == 1"]
        for i in range(r - 1):
            inner.append(f"{p.name}.stride({i}) == {p.name}.shape[{i + 1}]")
        # size-0 数组的布局契约空真（v0.4-kloop §12）：torch 对空维报告的
        # stride 无意义（如 (100,0).stride() == (1,1)），无元素可错读。
        # 外层括号必须带：顶层以 and 连接，而 or 结合度更低
        mem_conds.append(f"({p.name}.numel() == 0 or ({' and '.join(inner)}))")
    if mem_conds:
        lines.append(f"    assert {' and '.join(mem_conds)}")

    # host 启动断言（v0.6b：launch_assert；条件由 checker 核验的延迟表达式，
    # 名字 = 符号维/constexpr——launcher 作用域内已绑定）
    for op in kernel.ops:
        if isinstance(op, tir.TLaunchAssert):
            lines.append(f"    assert {op.cond}")

    # grid（来自 launch_plan；lowering 只发射不推导）
    if plan.axes:
        axis_exprs = [f"triton.cdiv({a.dim}, {a.tiling})" for a in plan.axes]
        grid = "(" + ", ".join(axis_exprs) + ("," if len(axis_exprs) == 1 else "") + ")"
    else:
        grid = "(1,)"
    lines.append(f"    grid = {grid}")

    call_args = [p.name for p in kernel.params if p.kind != "constexpr"]
    call_args += [f"{p.name}={p.name}" for p in constexprs]
    lines.append(f"    {kernel.name}[grid]({', '.join(call_args)})")
    return lines


def emit(kernel: tir.TKernel) -> str:
    """TKernel → Triton 模块源码字符串（total、确定）。"""
    renderer = OpRenderer(kernel)
    body = renderer.body_lines()

    parts = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        f"@triton.jit",
        f"def {kernel.name}({_kernel_signature(kernel)}):",
    ]
    parts.extend(f"    {line}" for line in body)
    parts.extend(["", ""])
    parts.extend(_launcher(kernel))
    return "\n".join(parts) + "\n"
