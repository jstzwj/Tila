"""规范示例：逐行 softmax——标量行坐标、单块整行 masked load、
带轴 max/sum 归约到标量后的标量广播。

masked lane 用 other=-1.0e30 填充（neg_inf 模式）：既不会污染行最大值，
经 exp 后也恰好贡献 0，分母因此无需再掩蔽。
"""

import tila as ti

M = ti.Dim("M")
N = ti.Dim("N")


@ti.jit
def softmax_kernel(
    x: ti.Buffer[ti.f32, (M, N), ti.ReadOnly],
    o: ti.Buffer[ti.f32, (M, N), ti.WriteOnly],
    BN: ti.Const[int, ti.PowerOfTwo] = 256,
):
    row = ti.program_id(0)                  # 标量行坐标：grid = (M,)
    offs = ti.arange(0, BN)                 # Block[i32, (BN,)] 整行一个块
    m = offs < N                            # 行尾掩蔽（N < BN 时）

    v = ti.load(x, (row, offs), mask=m, other=-1.0e30)
    mx = ti.max(v, 0)                       # 行最大值（标量）
    e = ti.exp(v - mx)                      # masked lane → exp(-1e30) = 0
    s = ti.sum(e, 0)                        # 行归一化因子（标量）

    ti.store(o, (row, offs), e / s, mask=m)


def main():
    import numpy as np

    m, n = 128, 1823                        # n 非二次幂：行尾掩蔽生效
    rng = np.random.default_rng(0)
    x = rng.standard_normal((m, n)).astype(np.float32)
    o = np.zeros_like(x)

    BN = 2048                               # 覆盖整行的最小二次幂块
    softmax_kernel[(m,)](x, o, BN=BN)

    e = np.exp(x - x.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(o, ref, atol=1e-6), "softmax mismatch"
    print(f"softmax: {m}x{n} OK")


if __name__ == "__main__":
    main()
