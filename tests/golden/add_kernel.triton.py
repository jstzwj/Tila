import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    x_stride0,
    y_stride0,
    out_stride0,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs_base = (pid * BLOCK)
    offs = (offs_base + tl.arange(0, BLOCK))
    tl.multiple_of(offs_base, BLOCK)
    tl.max_contiguous(offs, BLOCK)
    mask = (offs < N)
    a = tl.load(x_ptr + x_stride0 * offs, mask=mask, other=0.0)
    b = tl.load(y_ptr + y_stride0 * offs, mask=mask, other=0.0)
    tl.store(out_ptr + out_stride0 * offs, (a + b), mask=mask)
