import triton
import triton.language as tl


@triton.jit
def masked_add(a, b, c, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(a + offs, mask=mask, other=0.0)
    y = tl.load(b + offs, mask=mask, other=0.0)
    z = x + y
    tl.store(c + offs, z, mask=mask)


def masked_add_launch(a, b, c, BLOCK: int = 128):
    assert a.dim() == 1 and b.dim() == 1 and c.dim() == 1
    N = a.shape[0]
    assert b.shape[0] == N and c.shape[0] == N
    grid = (triton.cdiv(N, BLOCK),)
    masked_add[grid](a, b, c, N, BLOCK=BLOCK)
