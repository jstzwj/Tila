import triton
import triton.language as tl


@triton.jit
def saxpy(x, y, out, N, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(x + offs, mask=mask)
    yv = tl.load(y + offs, mask=mask)
    a = alpha.to(tl.float32)
    t = xv * a
    z = t + yv
    tl.store(out + offs, z, mask=mask)


def saxpy_launch(x, y, out, alpha, BLOCK: int = 128):
    assert x.dim() == 1 and y.dim() == 1 and out.dim() == 1
    N = x.shape[0]
    assert y.shape[0] == N and out.shape[0] == N
    grid = (triton.cdiv(N, BLOCK),)
    saxpy[grid](x, y, out, N, alpha, BLOCK=BLOCK)
