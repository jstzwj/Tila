import triton
import triton.language as tl


@triton.jit
def snapshot_constants(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    a_stride0: tl.int32,
    b_stride0: tl.int32,
    c_stride0: tl.int32,
    d_stride0: tl.int32
):
    a_stride0 = tl.cast(a_stride0, tl.int32)
    b_stride0 = tl.cast(b_stride0, tl.int32)
    c_stride0 = tl.cast(c_stride0, tl.int32)
    d_stride0 = tl.cast(d_stride0, tl.int32)
    tl.store(a_ptr + tl.cast(a_stride0, tl.int64) * tl.cast(0, tl.int64), tl.cast(tl.full((), 11878, tl.uint16), tl.float16, bitcast=True))
    tl.store(b_ptr + tl.cast(b_stride0, tl.int64) * tl.cast(0, tl.int64), tl.cast(tl.full((), 15821, tl.uint16), tl.bfloat16, bitcast=True))
    tl.store(c_ptr + tl.cast(c_stride0, tl.int64) * tl.cast(0, tl.int64), tl.cast(tl.full((), 2147483648, tl.uint32), tl.float32, bitcast=True))
    tl.store(d_ptr + tl.cast(d_stride0, tl.int64) * tl.cast(0, tl.int64), tl.cast(tl.full((), 4591870180066957722, tl.uint64), tl.float64, bitcast=True))
