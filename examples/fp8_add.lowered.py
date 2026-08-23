# examples/fp8_add.lowered.py — fp8_add.tila 期望生成的 Triton 代码
#
# tila.cast（R11）lowering 为 .to(tl.<dtype>)；其余与 add.lowered.py 同形。
# dtype 映射：fp8e4m3fn -> tl.float8e4m3fn（triton-lowering.md §6）。

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


# 由 Tila launcher 生成的主机侧调用（示意；断言与 grid 来自 checker 的 launch_plan）：
#
#     def fp8_add_launch(a, b, c, BLOCK=128):
#         assert a.dim() == 1 and b.dim() == 1 and c.dim() == 1
#         N = a.shape[0]
#         assert b.shape[0] == N and c.shape[0] == N
#         grid = (triton.cdiv(N, BLOCK),)
#         fp8_add[grid](a, b, c, N, BLOCK=BLOCK)