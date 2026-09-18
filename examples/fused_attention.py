"""Fused attention（Triton 官方 06-fused-attention tutorial 的 Tila 移植）。

对应官方 `_attn_fwd` + `_attn_fwd_inner`（FlashAttention v2 结构）：

- **两段 KV 循环**：stage 1 覆盖对角带以下 [0, pid·BM)，完全不做 causal
  掩蔽；stage 2 只处理对角带 [pid·BM, (pid+1)·BM)，用加性 bias
  `+ where(keep, 0, -1e6)` 掩蔽（官方同款技巧，掩蔽列 exp2 下溢为精确 0，
  无需二次掩 p）；
- **exp2 域**：host/kernel 侧把 sm_scale 乘 log2(e) 折入，归约内全部
  用 ti.exp2（f32 硬件 exp2 更快且无 exp 的额外缩放）；
- **FA2 更新顺序**：m_ij 合并 running max → alpha 校正 → p → l_ij →
  acc 先乘 alpha 再进 dot 累加 → 循环尾更新 l、m；
- **2D grid**：(cdiv(L, BM), H)——axis 0 是 m 块（cdiv 契约），axis 1
  精确等于 head 数（精确维契约，义务 `ph < H` 由 grid 事实证明）。

与官方的差异（设计边界内）：TensorDescriptor/TMA → 带 n_ok 掩码的
stride 寻址 load；`tl.maximum` → `ti.where`；`-inf` → `-1e30`（f32 下
数值等价且避免 -inf - -inf = NaN）；FP8/Hopper/LSE/autotune 未排期。

m/l 初值语义与官方一致：row_max = -1e30 时首块 alpha = exp2(下溢) = 0，
乘法链自然消去初值，无需特判。
"""

import tila as ti

L = ti.Dim("L")    # 序列长（自注意：query/key/value 同长）
H = ti.Dim("H")    # head 数
BD = ti.Dim("BD")  # head 维（与同名 Const 参数绑定）

LOG2E = 1.4426950408889634    # 1 / ln(2)：exp2 域换算（模块级常量折入）


@ti.jit
def fused_attention_kernel(
    q: ti.Buffer[ti.f16, (H, L, BD), ti.ReadOnly],
    k: ti.Buffer[ti.f16, (H, L, BD), ti.ReadOnly],
    v: ti.Buffer[ti.f16, (H, L, BD), ti.ReadOnly],
    o: ti.Buffer[ti.f32, (H, L, BD), ti.WriteOnly],
    sm_scale: ti.f32 | ti.Positive,
    CAUSAL: ti.Const[int] = 1,               # 1 = causal（默认），0 = 全注意力
    BM: ti.Const[int, ti.PowerOfTwo] = 64,
    BN: ti.Const[int, ti.PowerOfTwo] = 64,
    BD: ti.Const[int, ti.PowerOfTwo] = 64,
):
    pid = ti.program_id(0)                   # m 块索引
    ph = ti.program_id(1)                    # head 索引（grid 契约：ph < H）

    rm = pid * BM + ti.arange(0, BM)         # (BM,)  query 行
    rn = ti.arange(0, BN)                    # (BN,)  key/value 行
    rd = ti.arange(0, BD)                    # (BD,)  head 维

    m_ok = rm < L
    q_tile = ti.load(q, (ph, rm[:, None], rd[None, :]),
                     mask=m_ok[:, None])     # (BM, BD) f16，常驻寄存器

    qs = sm_scale * LOG2E                    # exp2 域 scale

    row_max = ti.zeros((BM, 1), ti.f32) - 1.0e30
    row_sum = ti.zeros((BM, 1), ti.f32)
    acc = ti.zeros((BM, BD), ti.f32)

    if CAUSAL == 1:
        # ---- stage 1：对角带以下，无 causal 掩蔽（官方 STAGE 1） ----
        for n0 in ti.range(0, pid * BM, BN):
            rv = n0 + rn
            n_ok = rv < L
            k_t = ti.load(k, (ph, rv[None, :], rd[:, None]),
                          mask=n_ok[None, :])            # (BD, BN) f16 = K^T
            s = ti.dot(q_tile, k_t) * qs                 # (BM, BN) f32
            row = ti.max(s, 1)
            m_tile = row[:, None]
            m_new = ti.where(m_tile > row_max, m_tile, row_max)
            alpha = ti.exp2(row_max - m_new)             # ≤ 1，首块下溢为 0
            p = ti.exp2(s - m_new)
            v_tile = ti.load(v, (ph, rv[:, None], rd[None, :]),
                             mask=n_ok[:, None])         # (BN, BD) f16
            l_ij = ti.sum(p, 1)
            acc = acc * alpha
            acc = ti.dot(ti.cast[ti.f16](p), v_tile, acc=acc)
            row_sum = row_sum * alpha + l_ij[:, None]
            row_max = m_new

        # ---- stage 2：对角带，causal 掩蔽经加性 bias（官方 STAGE 2） ----
        for n0 in ti.range(pid * BM, pid * BM + BM, BN):
            rv = n0 + rn
            n_ok = rv < L
            causal = rm[:, None] >= rv[None, :]
            keep = causal & n_ok[None, :]
            k_t = ti.load(k, (ph, rv[None, :], rd[:, None]),
                          mask=n_ok[None, :])
            s = ti.dot(q_tile, k_t) * qs + ti.where(keep, 0.0, -1.0e6)
            row = ti.max(s, 1)
            m_tile = row[:, None]
            m_new = ti.where(m_tile > row_max, m_tile, row_max)
            alpha = ti.exp2(row_max - m_new)
            p = ti.exp2(s - m_new)                       # 掩蔽列下溢为 0
            v_tile = ti.load(v, (ph, rv[:, None], rd[None, :]),
                             mask=n_ok[:, None])
            l_ij = ti.sum(p, 1)
            acc = acc * alpha
            acc = ti.dot(ti.cast[ti.f16](p), v_tile, acc=acc)
            row_sum = row_sum * alpha + l_ij[:, None]
            row_max = m_new
    else:
        # ---- 非 causal：单循环全序列（官方 STAGE 3 路径；官方假设
        # L % BN == 0 故无掩蔽，这里通用 L 用同一加性 bias 掩尾块） ----
        for n0 in ti.range(0, L, BN):
            rv = n0 + rn
            n_ok = rv < L
            k_t = ti.load(k, (ph, rv[None, :], rd[:, None]),
                          mask=n_ok[None, :])
            s = ti.dot(q_tile, k_t) * qs + ti.where(n_ok[None, :], 0.0,
                                                    -1.0e6)
            row = ti.max(s, 1)
            m_tile = row[:, None]
            m_new = ti.where(m_tile > row_max, m_tile, row_max)
            alpha = ti.exp2(row_max - m_new)
            p = ti.exp2(s - m_new)
            v_tile = ti.load(v, (ph, rv[:, None], rd[None, :]),
                             mask=n_ok[:, None])
            l_ij = ti.sum(p, 1)
            acc = acc * alpha
            acc = ti.dot(ti.cast[ti.f16](p), v_tile, acc=acc)
            row_sum = row_sum * alpha + l_ij[:, None]
            row_max = m_new

    ti.store(o, (ph, rm[:, None], rd[None, :]), acc / row_sum,
             mask=m_ok[:, None])


def _reference(q, k, v, scale, causal):
    """fp32 参考实现（与 examples/self_attention.py 同源）。"""
    import numpy as np

    h, l, d = q.shape
    qf, kf, vf = (x.astype(np.float32) for x in (q, k, v))
    s = np.einsum("hmd,hnd->hmn", qf, kf) * scale
    rows = np.arange(l)[None, :, None]
    cols = np.arange(l)[None, None, :]
    if causal:
        s = np.where(cols <= rows, s, -np.inf)
    s -= s.max(axis=2, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=2, keepdims=True)
    return np.einsum("hmn,hnd->hmd", p, vf)


def main():
    import numpy as np

    l, h, d = 96, 4, 64                    # 96 非块整除：尾块 + causal 掩蔽
    rng = np.random.default_rng(0)
    q = rng.standard_normal((h, l, d)).astype(np.float16)
    k = rng.standard_normal((h, l, d)).astype(np.float16)
    v = rng.standard_normal((h, l, d)).astype(np.float16)
    scale = d ** -0.5

    BM, BN, BD = 64, 64, 64
    grid = (ti.cdiv(l, BM), h)

    for causal in (1, 0):
        o = np.zeros((h, l, d), dtype=np.float32)
        fused_attention_kernel[grid](q, k, v, o, scale,
                                     CAUSAL=causal, BM=BM, BN=BN, BD=BD)
        ref = _reference(q, k, v, scale, causal=causal)
        assert np.allclose(o, ref, atol=2e-3), \
            f"fused attention mismatch (causal={causal})"
        mode = "causal" if causal else "full"
        print(f"fused attention ({mode}, 2-stage, exp2): {h}x{l}x{d} OK")


if __name__ == "__main__":
    main()
