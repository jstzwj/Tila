import triton
import triton.language as tl


@triton.jit
def attention(q, k, v, o, M, D, N, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rm2 = tl.expand_dims(pid * BM + tl.arange(0, BM), 1)
    rn2 = tl.expand_dims(tl.arange(0, BN), 0)
    rn3 = tl.expand_dims(tl.arange(0, BN), 1)
    rd2 = tl.expand_dims(tl.arange(0, BD), 0)
    rd3 = tl.expand_dims(tl.arange(0, BD), 1)
    m_qk = rn2 < N
    q2 = tl.load(q + (rm2 * D + rd2), mask=(rm2 < M) & (rd2 < D), other=0.0)
    k2 = tl.load(k + (rn2 * D + rd3), mask=(rn2 < N) & (rd3 < D), other=0.0)
    v2 = tl.load(v + (rn3 * D + rd2), mask=(rn3 < N) & (rd2 < D), other=0.0)
    qk = tl.dot(q2, k2)
    qk_ = tl.where(m_qk, qk, float("-inf"))
    mx = tl.max(qk_, 1)
    mx2 = tl.expand_dims(mx, 1)
    p = tl.exp((qk_ - mx2))
    s = tl.sum(p, 1, dtype=tl.float32)
    s2 = tl.expand_dims(s, 1)
    w = p / s2
    y = tl.dot(w.to(tl.float16), v2)
    tl.store(o + (rm2 * D + rd2), y.to(tl.float16), mask=(rm2 < M) & (rd2 < D))


def attention_launch(q, k, v, o, BM: int = 64, BN: int = 64, BD: int = 64):
    assert q.dim() == 2 and k.dim() == 2 and v.dim() == 2 and o.dim() == 2
    M = q.shape[0]
    D = q.shape[1]
    N = k.shape[0]
    assert k.shape[1] == D and v.shape[0] == N and v.shape[1] == D and o.shape[0] == M and o.shape[1] == D
    assert (q.numel() == 0 or (q.stride(1) == 1 and q.stride(0) == q.shape[1])) and (k.numel() == 0 or (k.stride(1) == 1 and k.stride(0) == k.shape[1])) and (v.numel() == 0 or (v.stride(1) == 1 and v.stride(0) == v.shape[1])) and (o.numel() == 0 or (o.stride(1) == 1 and o.stride(0) == o.shape[1]))
    grid = (triton.cdiv(M, BM),)
    attention[grid](q, k, v, o, M, D, N, BM=BM, BN=BN, BD=BD)
