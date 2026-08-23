import triton
import triton.language as tl


@triton.jit
def matmul(a, b, c, M, K, N, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rm2 = tl.expand_dims(rm, 1)
    rk2 = tl.expand_dims(rk, 0)
    rk3 = tl.expand_dims(rk, 1)
    rn2 = tl.expand_dims(rn, 0)
    a_idx = rm2 * K + rk2
    b_idx = rk3 * N + rn2
    c_idx = rm2 * N + rn2
    a_m = (rm2 < M) & (rk2 < K)
    b_m = (rk3 < K) & (rn2 < N)
    c_m = (rm2 < M) & (rn2 < N)
    x = tl.load(a + a_idx, mask=a_m, other=0.0)
    y = tl.load(b + b_idx, mask=b_m, other=0.0)
    acc = tl.dot(x, y)
    tl.store(c + c_idx, acc, mask=c_m)


def matmul_launch(a, b, c, BM: int = 64, BN: int = 128, BK: int = 64):
    assert a.dim() == 2 and b.dim() == 2 and c.dim() == 2
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    assert b.shape[0] == K and c.shape[0] == M and c.shape[1] == N
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul[grid](a, b, c, M, K, N, BM=BM, BN=BN, BK=BK)
