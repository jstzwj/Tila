"""规范示例：向量加法（docs/surface-language.md §7 的端到端走查载体）。

无 GPU 时走 CPU reference interpreter（numpy）；有 triton + CUDA 张量时
编译到 Triton 并在 GPU 上执行。两条路径共享同一套静态检查。
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

    offs = pid * BLOCK + ti.arange(0, BLOCK)   # Block[i32, (BLOCK,)]
    mask = offs < N                             # Mask[(BLOCK,)] { offs < N }

    a = ti.load(x, offs, mask=mask)             # bounds: ProvenSafe
    b = ti.load(y, offs, mask=mask)

    ti.store(out, offs, a + b, mask=mask)       # f32 精确匹配，WriteOnly ✓


def main():
    import numpy as np

    n = 1000
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.zeros_like(x)

    BLOCK = 128
    add_kernel[(ti.cdiv(n, BLOCK),)](x, y, out, BLOCK=BLOCK)

    ref = x + y
    assert np.allclose(out, ref), "vector add mismatch"
    print(f"add_kernel: {n} elements OK (interpreter/CUDA auto-selected)")


if __name__ == "__main__":
    main()
