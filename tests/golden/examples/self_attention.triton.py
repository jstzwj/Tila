import triton
import triton.language as tl


@triton.jit
def _tila_maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def self_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    scale: tl.float32,
    M: tl.int32,
    q_stride0: tl.int32,
    q_stride1: tl.int32,
    N: tl.int32,
    k_stride0: tl.int32,
    k_stride1: tl.int32,
    v_stride0: tl.int32,
    v_stride1: tl.int32,
    o_stride0: tl.int32,
    o_stride1: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr
):
    scale = tl.cast(scale, tl.float32)
    M = tl.cast(M, tl.int32)
    q_stride0 = tl.cast(q_stride0, tl.int32)
    q_stride1 = tl.cast(q_stride1, tl.int32)
    N = tl.cast(N, tl.int32)
    k_stride0 = tl.cast(k_stride0, tl.int32)
    k_stride1 = tl.cast(k_stride1, tl.int32)
    v_stride0 = tl.cast(v_stride0, tl.int32)
    v_stride1 = tl.cast(v_stride1, tl.int32)
    o_stride0 = tl.cast(o_stride0, tl.int32)
    o_stride1 = tl.cast(o_stride1, tl.int32)
    pid = tl.program_id(0)
    rm_base = (pid * BM)
    rm = (rm_base + tl.arange(0, BM))
    tl.multiple_of(rm_base, BM)
    tl.max_contiguous(rm, BM)
    rn = tl.arange(0, BN)
    tl.max_contiguous(rn, BN)
    rd = tl.arange(0, BD)
    tl.max_contiguous(rd, BD)
    m_ok = (tl.cast(rm, tl.int32) < tl.cast(M, tl.int32))
    q_tile = tl.load(q_ptr + tl.cast(q_stride0, tl.int64) * tl.cast(rm[:, None], tl.int64) + tl.cast(q_stride1, tl.int64) * tl.cast(rd[None, :], tl.int64), mask=m_ok[:, None], other=0.0)
    row_max = (tl.zeros((BM, 1), tl.float32) - 1e+30)
    row_sum = tl.zeros((BM, 1), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for _tila_loop_n0 in range(tl.cast(0, tl.int64), tl.cast(N, tl.int64), tl.cast(BN, tl.int64)):
        n0 = tl.cast(_tila_loop_n0, tl.int32)
        rv = (n0 + rn)
        tl.max_contiguous(rv, BN)
        n_ok = (tl.cast(rv, tl.int32) < tl.cast(N, tl.int32))
        causal = (tl.cast(rm[:, None], tl.int32) >= tl.cast(rv[None, :], tl.int32))
        keep = (causal & n_ok[None, :])
        k_t = tl.load(k_ptr + tl.cast(k_stride0, tl.int64) * tl.cast(rv[None, :], tl.int64) + tl.cast(k_stride1, tl.int64) * tl.cast(rd[:, None], tl.int64), mask=(n_ok[None, :] & (tl.cast(rd[:, None], tl.int32) >= tl.cast(0, tl.int32))), other=0.0)
        s = (tl.dot(q_tile, k_t) * scale)
        s = tl.where(keep, s, -1e+30)
        row = tl.reduce(s, 1, _tila_maximum)
        m_tile = row[:, None]
        m_new = tl.where((m_tile > row_max), m_tile, row_max)
        alpha = tl.exp((row_max - m_new))
        p = tl.where(keep, tl.exp((s - m_new)), 0.0)
        v_tile = tl.load(v_ptr + tl.cast(v_stride0, tl.int64) * tl.cast(rv[:, None], tl.int64) + tl.cast(v_stride1, tl.int64) * tl.cast(rd[None, :], tl.int64), mask=n_ok[:, None], other=0.0)
        acc = (acc * alpha)
        row_sum = ((row_sum * alpha) + tl.sum(p, axis=1, dtype=tl.float32)[:, None])
        acc = tl.dot(tl.cast(p, tl.float16), v_tile, acc)
        row_max = m_new
    tl.store(o_ptr + tl.cast(o_stride0, tl.int64) * tl.cast(rm[:, None], tl.int64) + tl.cast(o_stride1, tl.int64) * tl.cast(rd[None, :], tl.int64), (acc / row_sum), mask=m_ok[:, None])
