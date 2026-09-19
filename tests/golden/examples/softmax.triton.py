import triton
import triton.language as tl


@triton.jit
def _tila_maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def softmax_kernel(
    x_ptr,
    o_ptr,
    M: tl.int32,
    N: tl.int32,
    x_stride0: tl.int32,
    x_stride1: tl.int32,
    o_stride0: tl.int32,
    o_stride1: tl.int32,
    BN: tl.constexpr
):
    M = tl.cast(M, tl.int32)
    N = tl.cast(N, tl.int32)
    x_stride0 = tl.cast(x_stride0, tl.int32)
    x_stride1 = tl.cast(x_stride1, tl.int32)
    o_stride0 = tl.cast(o_stride0, tl.int32)
    o_stride1 = tl.cast(o_stride1, tl.int32)
    row = tl.program_id(0)
    offs = tl.arange(0, BN)
    tl.max_contiguous(offs, BN)
    m = (tl.cast(offs, tl.int32) < tl.cast(N, tl.int32))
    v = tl.load(x_ptr + tl.cast(x_stride0, tl.int64) * tl.cast(row, tl.int64) + tl.cast(x_stride1, tl.int64) * tl.cast(offs, tl.int64), mask=m, other=-1e+30)
    mx = tl.reduce(v, 0, _tila_maximum)
    e = tl.exp((v - mx))
    s = tl.sum(e, axis=0, dtype=tl.float32)
    tl.store(o_ptr + tl.cast(o_stride0, tl.int64) * tl.cast(row, tl.int64) + tl.cast(o_stride1, tl.int64) * tl.cast(offs, tl.int64), (e / s), mask=m)
