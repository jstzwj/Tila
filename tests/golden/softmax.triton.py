import triton
import triton.language as tl


@triton.jit
def softmax(a, o, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rm2 = tl.expand_dims(pid * BM + tl.arange(0, BM), 1)
    rn2 = tl.expand_dims(tl.arange(0, BN), 0)
    m2 = (rm2 < M) & (rn2 < N)
    x = tl.load(a + (rm2 * N + rn2), mask=m2).to(tl.float32)
    neg = tl.where(m2, x, float("-inf"))
    mx = tl.max(neg, 1)
    mx2 = tl.expand_dims(mx, 1)
    e = tl.where(m2, tl.exp((x - mx2)), 0.0)
    s = tl.sum(e, 1, dtype=tl.float32)
    s2 = tl.expand_dims(s, 1)
    y = tl.where(m2, (e / s2), 0.0)
    tl.store(o + (rm2 * N + rn2), y.to(tl.float16), mask=m2)


def softmax_launch(a, o, BM: int = 64, BN: int = 128):
    assert a.dim() == 2 and o.dim() == 2
    M = a.shape[0]
    N = a.shape[1]
    assert o.shape[0] == M and o.shape[1] == N
    assert (a.numel() == 0 or (a.stride(1) == 1 and a.stride(0) == a.shape[1])) and (o.numel() == 0 or (o.stride(1) == 1 and o.stride(0) == o.shape[1]))
    grid = (triton.cdiv(M, BM),)
    softmax[grid](a, o, M, N, BM=BM, BN=BN)
