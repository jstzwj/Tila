# examples/add.lowered.py — Tila v0.1 期望生成的 Triton 代码
#
# 与 add.tila 逐行对应。lowering 是纯语法映射：
# dtype / shape / layout 检查已全部在 Tila 编译期完成，本阶段不会失败。
# 表面语言中的 tila.load(a + offs) 在这里展开为 tl.load(a + offs)——
# 指针算术（a + offs）只存在于生成的代码，不出现在 Tila 源码里。

import triton
import triton.language as tl


@triton.jit
def add(a, b, c, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # Scalar(i32)

    offs = pid * BLOCK + tl.arange(0, BLOCK)     # Tile[i32, (BLOCK,), L0]
    mask = offs < N                              # Tile[bool, (BLOCK,), L0]

    x = tl.load(a + offs, mask=mask)             # Tile[f32, (BLOCK,), L0]
    y = tl.load(b + offs, mask=mask)             # Tile[f32, (BLOCK,), L0]

    z = x + y                                    # Tile[f32, (BLOCK,), L0]

    tl.store(c + offs, z, mask=mask)             # ()


# 由 Tila launcher 生成的主机侧调用（示意；断言与 grid 来自 checker 的 launch_plan）：
#
#     def add_launch(a, b, c, BLOCK=128):
#         assert a.dim() == 1 and b.dim() == 1 and c.dim() == 1
#         N = a.shape[0]
#         assert b.shape[0] == N and c.shape[0] == N
#         grid = (triton.cdiv(N, BLOCK),)
#         add[grid](a, b, c, N, BLOCK=BLOCK)