import triton
import triton.language as tl


@triton.jit
def _tila_maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def fused_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale: tl.float32,
    H: tl.int32,
    L: tl.int32,
    q_stride0: tl.int32,
    q_stride1: tl.int32,
    q_stride2: tl.int32,
    k_stride0: tl.int32,
    k_stride1: tl.int32,
    k_stride2: tl.int32,
    v_stride0: tl.int32,
    v_stride1: tl.int32,
    v_stride2: tl.int32,
    o_stride0: tl.int32,
    o_stride1: tl.int32,
    o_stride2: tl.int32,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr
):
    sm_scale = tl.cast(sm_scale, tl.float32)
    H = tl.cast(H, tl.int32)
    L = tl.cast(L, tl.int32)
    q_stride0 = tl.cast(q_stride0, tl.int32)
    q_stride1 = tl.cast(q_stride1, tl.int32)
    q_stride2 = tl.cast(q_stride2, tl.int32)
    k_stride0 = tl.cast(k_stride0, tl.int32)
    k_stride1 = tl.cast(k_stride1, tl.int32)
    k_stride2 = tl.cast(k_stride2, tl.int32)
    v_stride0 = tl.cast(v_stride0, tl.int32)
    v_stride1 = tl.cast(v_stride1, tl.int32)
    v_stride2 = tl.cast(v_stride2, tl.int32)
    o_stride0 = tl.cast(o_stride0, tl.int32)
    o_stride1 = tl.cast(o_stride1, tl.int32)
    o_stride2 = tl.cast(o_stride2, tl.int32)
    pid = tl.program_id(0)
    ph = tl.program_id(1)
    rm_base = (pid * BM)
    rm = (rm_base + tl.arange(0, BM))
    tl.multiple_of(rm_base, BM)
    tl.max_contiguous(rm, BM)
    rn = tl.arange(0, BN)
    tl.max_contiguous(rn, BN)
    rd = tl.arange(0, BD)
    tl.max_contiguous(rd, BD)
    m_ok = (tl.cast(rm, tl.int32) < tl.cast(L, tl.int32))
    q_tile = tl.load(q_ptr + tl.cast(q_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(q_stride1, tl.int64) * tl.cast(rm[:, None], tl.int64) + tl.cast(q_stride2, tl.int64) * tl.cast(rd[None, :], tl.int64), mask=m_ok[:, None], other=0.0)
    qs = (sm_scale * 1.4426950408889634)
    row_max = (tl.zeros((BM, 1), tl.float32) - 1e+30)
    row_sum = tl.zeros((BM, 1), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    if (CAUSAL == 1):
        for _tila_loop_n0 in range(tl.cast(0, tl.int64), tl.cast((pid * BM), tl.int64), tl.cast(BN, tl.int64)):
            n0 = tl.cast(_tila_loop_n0, tl.int32)
            rv = (n0 + rn)
            tl.max_contiguous(rv, BN)
            n_ok = (tl.cast(rv, tl.int32) < tl.cast(L, tl.int32))
            k_t = tl.load(k_ptr + tl.cast(k_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(k_stride1, tl.int64) * tl.cast(rv[None, :], tl.int64) + tl.cast(k_stride2, tl.int64) * tl.cast(rd[:, None], tl.int64), mask=n_ok[None, :], other=0.0)
            s = (tl.dot(q_tile, k_t) * qs)
            row = tl.reduce(s, 1, _tila_maximum)
            m_tile = row[:, None]
            m_new = tl.where((m_tile > row_max), m_tile, row_max)
            alpha = tl.math.exp2((row_max - m_new))
            p = tl.math.exp2((s - m_new))
            v_tile = tl.load(v_ptr + tl.cast(v_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(v_stride1, tl.int64) * tl.cast(rv[:, None], tl.int64) + tl.cast(v_stride2, tl.int64) * tl.cast(rd[None, :], tl.int64), mask=n_ok[:, None], other=0.0)
            l_ij = tl.sum(p, axis=1, dtype=tl.float32)
            acc = (acc * alpha)
            acc = tl.dot(tl.cast(p, tl.float16), v_tile, acc)
            row_sum = ((row_sum * alpha) + l_ij[:, None])
            row_max = m_new
        for _tila_loop_n0 in range(tl.cast((pid * BM), tl.int64), tl.cast(((pid * BM) + BM), tl.int64), tl.cast(BN, tl.int64)):
            n0 = tl.cast(_tila_loop_n0, tl.int32)
            rv = (n0 + rn)
            tl.max_contiguous(rv, BN)
            n_ok = (tl.cast(rv, tl.int32) < tl.cast(L, tl.int32))
            causal = (tl.cast(rm[:, None], tl.int32) >= tl.cast(rv[None, :], tl.int32))
            keep = (causal & n_ok[None, :])
            k_t = tl.load(k_ptr + tl.cast(k_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(k_stride1, tl.int64) * tl.cast(rv[None, :], tl.int64) + tl.cast(k_stride2, tl.int64) * tl.cast(rd[:, None], tl.int64), mask=n_ok[None, :], other=0.0)
            s = ((tl.dot(q_tile, k_t) * qs) + tl.where(keep, 0.0, -1000000.0))
            row = tl.reduce(s, 1, _tila_maximum)
            m_tile = row[:, None]
            m_new = tl.where((m_tile > row_max), m_tile, row_max)
            alpha = tl.math.exp2((row_max - m_new))
            p = tl.math.exp2((s - m_new))
            v_tile = tl.load(v_ptr + tl.cast(v_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(v_stride1, tl.int64) * tl.cast(rv[:, None], tl.int64) + tl.cast(v_stride2, tl.int64) * tl.cast(rd[None, :], tl.int64), mask=n_ok[:, None], other=0.0)
            l_ij = tl.sum(p, axis=1, dtype=tl.float32)
            acc = (acc * alpha)
            acc = tl.dot(tl.cast(p, tl.float16), v_tile, acc)
            row_sum = ((row_sum * alpha) + l_ij[:, None])
            row_max = m_new
    else:
        for _tila_loop_n0 in range(tl.cast(0, tl.int64), tl.cast(L, tl.int64), tl.cast(BN, tl.int64)):
            n0 = tl.cast(_tila_loop_n0, tl.int32)
            rv = (n0 + rn)
            tl.max_contiguous(rv, BN)
            n_ok = (tl.cast(rv, tl.int32) < tl.cast(L, tl.int32))
            k_t = tl.load(k_ptr + tl.cast(k_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(k_stride1, tl.int64) * tl.cast(rv[None, :], tl.int64) + tl.cast(k_stride2, tl.int64) * tl.cast(rd[:, None], tl.int64), mask=n_ok[None, :], other=0.0)
            s = ((tl.dot(q_tile, k_t) * qs) + tl.where(n_ok[None, :], 0.0, -1000000.0))
            row = tl.reduce(s, 1, _tila_maximum)
            m_tile = row[:, None]
            m_new = tl.where((m_tile > row_max), m_tile, row_max)
            alpha = tl.math.exp2((row_max - m_new))
            p = tl.math.exp2((s - m_new))
            v_tile = tl.load(v_ptr + tl.cast(v_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(v_stride1, tl.int64) * tl.cast(rv[:, None], tl.int64) + tl.cast(v_stride2, tl.int64) * tl.cast(rd[None, :], tl.int64), mask=n_ok[:, None], other=0.0)
            l_ij = tl.sum(p, axis=1, dtype=tl.float32)
            acc = (acc * alpha)
            acc = tl.dot(tl.cast(p, tl.float16), v_tile, acc)
            row_sum = ((row_sum * alpha) + l_ij[:, None])
            row_max = m_new
    tl.store(o_ptr + tl.cast(o_stride0, tl.int64) * tl.cast(ph, tl.int64) + tl.cast(o_stride1, tl.int64) * tl.cast(rm[:, None], tl.int64) + tl.cast(o_stride2, tl.int64) * tl.cast(rd[None, :], tl.int64), (acc / row_sum), mask=m_ok[:, None])
