import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N: tl.int32,
    x_stride0: tl.int32,
    y_stride0: tl.int32,
    out_stride0: tl.int32,
    BLOCK: tl.constexpr
):
    N = tl.cast(N, tl.int32)
    x_stride0 = tl.cast(x_stride0, tl.int32)
    y_stride0 = tl.cast(y_stride0, tl.int32)
    out_stride0 = tl.cast(out_stride0, tl.int32)
    pid = tl.program_id(0)
    offs_base = (pid * BLOCK)
    offs = (offs_base + tl.arange(0, BLOCK))
    tl.multiple_of(offs_base, BLOCK)
    tl.max_contiguous(offs, BLOCK)
    mask = (tl.cast(offs, tl.int32) < tl.cast(N, tl.int32))
    a = tl.load(x_ptr + tl.cast(x_stride0, tl.int64) * tl.cast(offs, tl.int64), mask=mask, other=0.0)
    b = tl.load(y_ptr + tl.cast(y_stride0, tl.int64) * tl.cast(offs, tl.int64), mask=mask, other=0.0)
    tl.store(out_ptr + tl.cast(out_stride0, tl.int64) * tl.cast(offs, tl.int64), (a + b), mask=mask)
