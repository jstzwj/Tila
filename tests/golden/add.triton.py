import triton
import triton.language as tl


@triton.jit
def add(a, b, c, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(a + offs, mask=mask)
    y = tl.load(b + offs, mask=mask)
    z = x + y
    tl.store(c + offs, z, mask=mask)


def add_launch(a, b, c, BLOCK: int = 128):
    assert a.dim() == 1 and b.dim() == 1 and c.dim() == 1
    N = a.shape[0]
    assert b.shape[0] == N and c.shape[0] == N
    assert (a.numel() == 0 or (a.stride(0) == 1)) and (b.numel() == 0 or (b.stride(0) == 1)) and (c.numel() == 0 or (c.stride(0) == 1))
    grid = (triton.cdiv(N, BLOCK),)
    add[grid](a, b, c, N, BLOCK=BLOCK)
