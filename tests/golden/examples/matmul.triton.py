import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.int32,
    K: tl.int32,
    a_stride0: tl.int32,
    a_stride1: tl.int32,
    N: tl.int32,
    b_stride0: tl.int32,
    b_stride1: tl.int32,
    c_stride0: tl.int32,
    c_stride1: tl.int32,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr
):
    M = tl.cast(M, tl.int32)
    K = tl.cast(K, tl.int32)
    a_stride0 = tl.cast(a_stride0, tl.int32)
    a_stride1 = tl.cast(a_stride1, tl.int32)
    N = tl.cast(N, tl.int32)
    b_stride0 = tl.cast(b_stride0, tl.int32)
    b_stride1 = tl.cast(b_stride1, tl.int32)
    c_stride0 = tl.cast(c_stride0, tl.int32)
    c_stride1 = tl.cast(c_stride1, tl.int32)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm_base = (pid_m * BM)
    rm = (rm_base + tl.arange(0, BM))
    tl.multiple_of(rm_base, BM)
    tl.max_contiguous(rm, BM)
    rn_base = (pid_n * BN)
    rn = (rn_base + tl.arange(0, BN))
    tl.multiple_of(rn_base, BN)
    tl.max_contiguous(rn, BN)
    rk = tl.arange(0, BK)
    tl.max_contiguous(rk, BK)
    m_mask = (tl.cast(rm, tl.int32) < tl.cast(M, tl.int32))
    n_mask = (tl.cast(rn, tl.int32) < tl.cast(N, tl.int32))
    acc = tl.zeros((BM, BN), tl.float32)
    for _tila_loop_k0 in range(tl.cast(0, tl.int64), tl.cast(K, tl.int64), tl.cast(BK, tl.int64)):
        k0 = tl.cast(_tila_loop_k0, tl.int32)
        k_mask = (tl.cast((k0 + rk), tl.int32) < tl.cast(K, tl.int32))
        a_tile = tl.load(a_ptr + tl.cast(a_stride0, tl.int64) * tl.cast(rm[:, None], tl.int64) + tl.cast(a_stride1, tl.int64) * tl.cast((k0 + rk)[None, :], tl.int64), mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)
        b_tile = tl.load(b_ptr + tl.cast(b_stride0, tl.int64) * tl.cast((k0 + rk)[:, None], tl.int64) + tl.cast(b_stride1, tl.int64) * tl.cast(rn[None, :], tl.int64), mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)
        acc = tl.dot(a_tile, b_tile, acc)
    tl.store(c_ptr + tl.cast(c_stride0, tl.int64) * tl.cast(rm[:, None], tl.int64) + tl.cast(c_stride1, tl.int64) * tl.cast(rn[None, :], tl.int64), acc, mask=(m_mask[:, None] & n_mask[None, :]))
