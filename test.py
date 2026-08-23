"""完整可运行示例：@tila.jit 定义 kernel，直接调用执行。

- np.ndarray   → CPU reference interpreter（无需 GPU）
- torch.Tensor(CUDA) → 编译出的 Triton kernel 在 GPU 上运行
"""
from __future__ import annotations   # 必须是第一条语句：符号维 N 在注解里，需 PEP 563 延迟求值

import numpy as np
import torch
import tila


@tila.jit
def add(
    a: tila.Tensor[tila.float32, N],
    b: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(a + offs, mask=mask)
    y = tila.load(b + offs, mask=mask)
    z = x + y
    tila.store(c + offs, z, mask=mask)


# ---- ① CPU：numpy 数组（reference interpreter）--------------------------------
a = np.random.randn(1000).astype(np.float32)
b = np.random.randn(1000).astype(np.float32)
c = np.zeros(1000, dtype=np.float32)

add(a, b, c)                    # ← 先创建张量，再调用
print("CPU interpreter == numpy.add :", np.allclose(c, a + b))

# ---- ② GPU：torch CUDA 张量（Triton kernel）-----------------------------------
ta = torch.randn(1000, device="cuda")
tb = torch.randn(1000, device="cuda")
tc = torch.zeros_like(ta)

add(ta, tb, tc, BLOCK=64)       # 新的 constexpr 组合 = 一次新的完整编译（自动缓存）
torch.cuda.synchronize()
print("GPU triton    == torch.add  :", torch.allclose(tc, ta + tb))
print("compilation cache:", len(add._cache), "specialization(s)")
