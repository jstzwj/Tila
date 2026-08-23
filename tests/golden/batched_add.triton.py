import triton
import triton.language as tl


@triton.jit
def batched_add(a, b, c, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    rows2 = tl.expand_dims(rows, 1)
    cols2 = tl.expand_dims(cols, 0)
    mask = (rows2 < M) & (cols2 < N)
    x = tl.load(a + (rows2 * N + cols2), mask=mask)
    y = tl.load(b + (rows2 * N + cols2), mask=mask)
    z = x + y
    tl.store(c + (rows2 * N + cols2), z, mask=mask)


def batched_add_launch(a, b, c, BM: int = 64, BN: int = 128):
    assert a.dim() == 2 and b.dim() == 2 and c.dim() == 2
    M = a.shape[0]
    N = a.shape[1]
    assert b.shape[0] == M and b.shape[1] == N and c.shape[0] == M and c.shape[1] == N
    assert (a.numel() == 0 or (a.stride(1) == 1 and a.stride(0) == a.shape[1])) and (b.numel() == 0 or (b.stride(1) == 1 and b.stride(0) == b.shape[1])) and (c.numel() == 0 or (c.stride(1) == 1 and c.stride(0) == c.shape[1]))
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    batched_add[grid](a, b, c, M, N, BM=BM, BN=BN)
