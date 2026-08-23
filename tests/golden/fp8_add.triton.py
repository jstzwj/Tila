import triton
import triton.language as tl


@triton.jit
def fp8_add(a, b, c, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x0 = tl.load(a + offs, mask=mask)
    y0 = tl.load(b + offs, mask=mask)
    x = x0.to(tl.float16)
    y = y0.to(tl.float16)
    z = x + y
    z0 = z.to(tl.float8e4m3fn)
    tl.store(c + offs, z0, mask=mask)


def fp8_add_launch(a, b, c, BLOCK: int = 128):
    assert a.dim() == 1 and b.dim() == 1 and c.dim() == 1
    N = a.shape[0]
    assert b.shape[0] == N and c.shape[0] == N
    grid = (triton.cdiv(N, BLOCK),)
    fp8_add[grid](a, b, c, N, BLOCK=BLOCK)
