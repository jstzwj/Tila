"""test_errors.py — 类型系统检查有效性演示。

Triton 把大量 dtype/shape 错误留到运行期（甚至静默给出错结果）；Tila 在
@tila.jit 编译期（首次调用触发编译）就把它们全部拒绝：精确到文件行列的定位、
两个操作数的完整类型与定义位置、可直接照抄的修复建议——且绝不产出任何
Triton 代码。运行：python test_errors.py
"""
from __future__ import annotations

import numpy as np
import tila
from tila import TilaError


# ---------------------------------------------------------------------------
# 六个典型的"错误操作"kernel（对照 Triton 中的行为见各处注释）
# ---------------------------------------------------------------------------

@tila.jit
def mixed_dtypes(                                   # Triton：f32 + i32 静默提升，结果错得悄无声息
    a: tila.Tensor[tila.float32, N],
    b: tila.Tensor[tila.int32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(a + offs, mask=mask)
    y = tila.load(b + offs, mask=mask)
    z = x + y                                      # ← E02: Tile<f32> ⊕ Tile<i32>
    tila.store(c + offs, z, mask=mask)


@tila.jit
def shape_mismatch(                                 # Triton：运行期才报 broadcasting 错
    a: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    x = tila.load(a + offs)
    y = x + x
    half = tila.cast(tila.arange(0, 64), tila.float32)   # 同为 f32，排除 dtype 因素
    z = y + half                                   # ← E03: (128,) ⊕ (64,)
    tila.store(c + offs, z)


@tila.jit
def fp8_arithmetic(                                 # Triton：fp8 算术行为依后端而定
    a: tila.Tensor[tila.float8e4m3fn, N],
    b: tila.Tensor[tila.float8e4m3fn, N],
    c: tila.Tensor[tila.float8e4m3fn, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(a + offs, mask=mask)
    y = tila.load(b + offs, mask=mask)
    z = x + y                                      # ← E16: fp8 是存储 dtype，禁止算术
    tila.store(c + offs, z, mask=mask)


@tila.jit
def reassignment(                                  # Triton/Python：合法；Tila：单赋值 SSA
    a: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    x = tila.load(a + offs)
    x = x + 1.0                                    # ← E14: 名字只能绑定一次
    tila.store(c + offs, x)


@tila.jit
def bad_arange(                                    # Triton：arange(0,100) 运行期才失败
    a: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
):
    offs = tila.arange(0, 100)                     # ← E06: 长度必须是 2^k
    x = tila.load(a + offs)
    tila.store(c + offs, x)


@tila.jit
def typo_name(                                     # Triton：NameError 运行期才爆
    a: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < n                                # ← E01: 'n' 未绑定（想要 N？）
    x = tila.load(a + offs, mask=mask)
    tila.store(c + offs, x, mask=mask)


# ---------------------------------------------------------------------------
# 两个"按错误信息里的 hint 修正后"的版本——hint 照抄即可通过
# ---------------------------------------------------------------------------

@tila.jit
def mixed_dtypes_fixed(
    a: tila.Tensor[tila.float32, N],
    b: tila.Tensor[tila.int32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(a + offs, mask=mask)
    y = tila.load(b + offs, mask=mask)
    z = x + tila.cast(y, tila.float32)             # ← E02 hint 给出的写法
    tila.store(c + offs, z, mask=mask)


@tila.jit
def fp8_add_fixed(
    a: tila.Tensor[tila.float8e4m3fn, N],
    b: tila.Tensor[tila.float8e4m3fn, N],
    c: tila.Tensor[tila.float8e4m3fn, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x0 = tila.load(a + offs, mask=mask)
    y0 = tila.load(b + offs, mask=mask)
    x = tila.cast(x0, tila.float16)                # ← E16 hint 给出的写法
    y = tila.cast(y0, tila.float16)
    z = x + y
    z0 = tila.cast(z, tila.float8e4m3fn)
    tila.store(c + offs, z0, mask=mask)


# ---------------------------------------------------------------------------
# 演示驱动
# ---------------------------------------------------------------------------

def report(title, call):
    print(f"── {title} " + "─" * max(0, 64 - len(title)))
    try:
        call()
        print("!! 竟然编译通过（不应发生）")
    except TilaError as e:
        print(e.render())
    print()


def main() -> None:
    f32 = lambda n: np.random.randn(n).astype(np.float32)
    i32 = lambda n: np.random.randint(-100, 100, n).astype(np.int32)

    print("=" * 66)
    print("Tila 编译期类型检查：错误在生成任何 Triton 代码之前被拒绝")
    print("=" * 66)
    print()

    N = 512
    report("① 混 dtype 运算（E02，hint 直接给出 cast 写法）",
           lambda: mixed_dtypes(f32(N), i32(N), f32(N)))
    report("② shape 不匹配（E03，逐维给出冲突）",
           lambda: shape_mismatch(f32(N), f32(N)))
    report("③ FP8 算术（E16，存储 dtype 纪律）",
           lambda: fp8_arithmetic(f32(N).astype(np.float16), f32(N).astype(np.float16),
                                  f32(N).astype(np.float16)))
    report("④ 重赋值（E14，单赋值 → 直线 SSA）",
           lambda: reassignment(f32(N), f32(N)))
    report("⑤ arange 非 2 的幂（E06，Triton 约束前移）",
           lambda: bad_arange(f32(128), f32(128)))
    report("⑥ 名字拼错（E01，先使用后定义）",
           lambda: typo_name(f32(N), f32(N)))

    print("─" * 66)
    print("按 hint 修正后：同样的操作，显式 cast 即可通过并运行正确")
    print("─" * 66)

    a, b, c = f32(N), i32(N), f32(N)
    mixed_dtypes_fixed(a, b, c)                    # 编译 + CPU 解释器执行
    expected = a + b.astype(np.float32)
    print(f"①'  cast 修正版  == numpy 参考 : {np.allclose(c, expected)}")

    try:
        import ml_dtypes

        a8 = (f32(N) * 0.5).astype(ml_dtypes.float8_e4m3fn)
        b8 = (f32(N) * 0.5).astype(ml_dtypes.float8_e4m3fn)
        c8 = np.zeros(N, dtype=ml_dtypes.float8_e4m3fn)
        fp8_add_fixed(a8, b8, c8)
        want = (a8.astype(np.float16) + b8.astype(np.float16)).astype(ml_dtypes.float8_e4m3fn)
        print(f"③'  fp8 修正版  == numpy 参考 : {np.array_equal(c8, want)}")
    except ImportError:
        print("③'  （fp8 解释器需要 ml_dtypes，跳过）")


if __name__ == "__main__":
    main()
