import triton
import triton.language as tl


@triton.jit
def matmul_loop(a, b, c, M, K, N, sb0, sb1, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm2 = tl.expand_dims(pid_m * BM + tl.arange(0, BM), 1)
    rn2 = tl.expand_dims(pid_n * BN + tl.arange(0, BN), 0)
    rk2 = tl.expand_dims(tl.arange(0, BK), 0)
    rk3 = tl.expand_dims(tl.arange(0, BK), 1)
    c_m = (rm2 < M) & (rn2 < N)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rka = rk2 + k0
        rkb = rk3 + k0
        x = tl.load(a + (rm2 * K + rka), mask=(rm2 < M) & (rka < K), other=0.0)
        y = tl.load(b + (rkb * sb0 + rn2 * sb1), mask=(rkb < K) & (rn2 < N), other=0.0)
        acc = acc + tl.dot(x, y)
    c16 = acc.to(tl.float16)
    tl.store(c + (rm2 * N + rn2), c16, mask=c_m)


def matmul_loop_launch(a, b, c, BM: int = 64, BN: int = 128, BK: int = 64):
    assert a.dim() == 2 and b.dim() == 2 and c.dim() == 2
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    assert b.shape[0] == K and c.shape[0] == M and c.shape[1] == N
    sb0 = b.stride(0)
    sb1 = b.stride(1)
    assert (a.numel() == 0 or (a.stride(1) == 1 and a.stride(0) == a.shape[1])) and (c.numel() == 0 or (c.stride(1) == 1 and c.stride(0) == c.shape[1]))
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_loop[grid](a, b, c, M, K, N, sb0, sb1, BM=BM, BN=BN, BK=BK)
