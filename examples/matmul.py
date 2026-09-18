"""规范示例：分块 matmul——2D 坐标、外积 mask、dot 内维约束、K 循环。"""

import tila as ti

M = ti.Dim("M")
N = ti.Dim("N")
K = ti.Dim("K")


@ti.jit
def matmul_kernel(
    a: ti.Buffer[ti.f16, (M, K), ti.ReadOnly],
    b: ti.Buffer[ti.f16, (K, N), ti.ReadOnly],
    c: ti.Buffer[ti.f32, (M, N), ti.WriteOnly],
    BM: ti.Const[int, ti.PowerOfTwo] = 64,
    BN: ti.Const[int, ti.PowerOfTwo] = 64,
    BK: ti.Const[int] = 32,
):
    pid_m = ti.program_id(0)
    pid_n = ti.program_id(1)

    rm = pid_m * BM + ti.arange(0, BM)
    rn = pid_n * BN + ti.arange(0, BN)
    rk = ti.arange(0, BK)

    m_mask = rm < M
    n_mask = rn < N

    acc = ti.zeros((BM, BN), ti.f32)

    for k0 in ti.range(0, K, BK):
        k_mask = (k0 + rk) < K        # K 尾块谓词（bounds 证明必需）
        a_tile = ti.load(a, (rm[:, None], (k0 + rk)[None, :]),
                         mask=m_mask[:, None] & k_mask[None, :])
        b_tile = ti.load(b, ((k0 + rk)[:, None], rn[None, :]),
                         mask=k_mask[:, None] & n_mask[None, :])
        acc = ti.dot(a_tile, b_tile, acc=acc)

    ti.store(c, (rm[:, None], rn[None, :]), acc,
             mask=m_mask[:, None] & n_mask[None, :])


def main():
    import numpy as np

    m, n, k = 128, 64, 96
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    c = np.zeros((m, n), dtype=np.float32)

    BM, BN = 64, 64
    grid = (ti.cdiv(m, BM), ti.cdiv(n, BN))
    matmul_kernel[grid](a, b, c, BM=BM, BN=BN, BK=32)

    ref = a.astype(np.float32) @ b.astype(np.float32)
    assert np.allclose(c, ref, atol=2e-2), "matmul mismatch"
    print(f"matmul: {m}x{k}x{n} OK")


if __name__ == "__main__":
    main()
