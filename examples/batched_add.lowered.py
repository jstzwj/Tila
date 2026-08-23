# examples/batched_add.lowered.py — batched_add.tila 期望生成的 Triton 代码
#
# 与 Tila 源码逐行 1:1（expand_dim → tl.expand_dims，& → &）：
# 二维索引与 mask 由 expand_dim + 原生 size-1 广播 + 普通比较拼出，
# 无 meshgrid/bounds，也无"使用处分解"（docs/v0.2-preview-2d.md §5）。

import triton
import triton.language as tl


@triton.jit
def batched_add(a, b, c, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)

    rows2 = tl.expand_dims(rows, 1)
    cols2 = tl.expand_dims(cols, 0)

    idx = rows2 * N + cols2
    mask = (rows2 < M) & (cols2 < N)

    x = tl.load(a + idx, mask=mask)
    y = tl.load(b + idx, mask=mask)

    z = x + y

    tl.store(c + idx, z, mask=mask)


# 由 Tila launcher 生成的主机侧调用（示意）：
#
#     def batched_add_launch(a, b, c, BM=64, BN=128):
#         M, N = a.shape
#         assert b.shape == (M, N) and c.shape == (M, N)   # 同符号契约
#         grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
#         batched_add[grid](a, b, c, M, N, BM=BM, BN=BN)