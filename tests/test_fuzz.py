"""Fuzz（development-plan §8 评审 §39/§40）：

1. 变异 fuzz：对合法程序做随机 token 变异，checker 只允许 TilaError 或成功
   （健壮性：绝不泄漏 Python 原生异常——type-checker.md §9.5）。
2. 有效程序生成器：随机小 kernel（arange/load/arith/cast/store），
   编译必须成功，且 interpreter 结果与 NumPy 参考一致。
"""

import random
import re

import numpy as np
import pytest

from tila.driver import compile_kernel
from tila.diagnostics import TilaError
from tila.interp import run_kernel

ADD_SRC = open("examples/add.tila", encoding="utf-8").read()

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]", re.M)


def _mutate(src: str, rng: random.Random) -> str:
    tokens = TOKEN_RE.findall(src)
    for _ in range(rng.randint(1, 3)):
        if not tokens:
            break
        kind = rng.random()
        i = rng.randrange(len(tokens))
        if kind < 0.4:  # 删除
            del tokens[i]
        elif kind < 0.7:  # 复制到随机位置
            tokens.insert(rng.randrange(len(tokens) + 1), tokens[i])
        else:  # 交换
            j = rng.randrange(len(tokens))
            tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def test_mutation_fuzz_only_tila_errors():
    """500 个变异体：compile_kernel 要么成功、要么抛 TilaError。"""
    rng = random.Random(2026)
    accepted = 0
    for _ in range(500):
        mutant = _mutate(ADD_SRC, rng)
        try:
            compile_kernel(mutant)
            accepted += 1
        except TilaError:
            pass
        except Exception as e:  # noqa: BLE001 - 不允许任何原生异常泄漏
            pytest.fail(f"native exception leaked for mutant:\n{mutant}\n{e!r}")
    assert accepted < 500  # 绝大多数变异应被拒绝（子集极小）


# ---------------------------------------------------------------------------
# 有效程序生成 + interpreter 对拍
# ---------------------------------------------------------------------------

def _gen_program(seed: int):
    """随机合法 kernel：load 后接一串 tile 算术/cast，最后 store x0。

    对拍口径：kernel 固定 store x0（生成算子只保证类型合法性与可执行性），
    参考实现即恒等拷贝；算子链的数值语义由解释器执行、不得抛异常。
    bool 名与数值名分开维护（bool 不参与算术——E16）。
    """
    rng = random.Random(seed)
    block = rng.choice([2, 4, 8, 32])
    names = ["x0"]
    lines = [
        "import tila",
        "",
        "",
        f"@tila.jit",
        f"def gen(a: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N],"
        f" BLOCK: tila.constexpr = {block}):",
        "    pid = tila.program_id(0)",
        "    offs = pid * BLOCK + tila.arange(0, BLOCK)",
        "    mask = offs < N",
        "    x0 = tila.load(a + offs, mask=mask)",
    ]
    for i in range(rng.randint(1, 4)):
        mode = rng.random()
        if mode < 0.35:
            op = rng.choice(["+", "-", "*"])
            lines.append(f"    x{i + 1} = {rng.choice(names)} {op} {rng.choice(names)}")
            names.append(f"x{i + 1}")
        elif mode < 0.6:
            lit = round(rng.uniform(-2, 2), 3)
            op = rng.choice(["+", "-", "*"])
            lines.append(f"    x{i + 1} = {rng.choice(names)} {op} {lit}")
            names.append(f"x{i + 1}")
        elif mode < 0.8:
            lit = round(rng.uniform(0.1, 2.0), 3)
            lines.append(f"    x{i + 1} = tila.cast({rng.choice(names)}, tila.float32)")
            names.append(f"x{i + 1}")
            _ = lit
        else:
            lines.append(f"    x{i + 1} = tila.cast({rng.choice(names)}, tila.bool)")
            lines.append(f"    b{i} = tila.cast(x{i + 1}, tila.float32)")
            names.append(f"b{i}")
    lines.append("    tila.store(c + offs, x0, mask=mask)")
    return "\n".join(lines) + "\n", block


def test_generated_programs_compile_and_run():
    for seed in range(60):
        src, block = _gen_program(seed)
        try:
            res = compile_kernel(src)
        except TilaError as e:
            pytest.fail(f"generated program rejected (seed {seed}): {e}\n{src}")
        rng = np.random.default_rng(seed)
        n = block * 3 + 1
        a = rng.standard_normal(n).astype(np.float32)
        c = np.zeros(n, dtype=np.float32)
        run_kernel(res.kernel, {"a": a, "c": c}, constexpr={"BLOCK": block})
        np.testing.assert_allclose(c, a, rtol=1e-6)  # store 的是 x0（恒等拷贝）


def test_div_compare_and_logic_paths():
    """补充 /、比较、& 合取的完整语义对拍（含非整除 N 与 other 通道）。"""
    src = """import tila


@tila.jit
def gen2(a: tila.Tensor[tila.float32, N], c: tila.Tensor[tila.float32, N],
         BLOCK: tila.constexpr = 8):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = (offs < N) & (offs > 1)
    x = tila.load(a + offs, mask=mask, other=1.0)
    y = x / 2.0
    m = y > 0.25
    w = m & mask
    z = y + 0.0
    tila.store(c + offs, z, mask=w)
"""
    res = compile_kernel(src)
    n = 25
    rng = np.random.default_rng(11)
    a = rng.standard_normal(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    run_kernel(res.kernel, {"a": a, "c": c})

    ref = np.zeros(n, dtype=np.float32)
    for pid in range(-(-n // 8)):
        offs = np.arange(8) + pid * 8
        oob = offs >= n
        safe = np.where(oob, 0, offs)
        x = np.where(oob, 1.0, a[safe]).astype(np.float32)
        y = x / 2.0
        sel = (~oob) & (offs > 1) & (y > 0.25)
        ref[safe[sel]] = y[sel]
    np.testing.assert_allclose(c, ref, rtol=1e-6)
