"""Self-attention（causal，online-softmax 单遍 Flash 风格）。

演示组合：2D 分块 load（含换坐标序取 K^T，无需 trans）、dot、
带轴归约 sum/max、exp、where 掩蔽、(M,1)/(M,N) 广播、K 循环携带
多个归纳变量（running max / normalizer / acc）。
"""

import tila as ti

M = ti.Dim("M")   # query 序列长
N = ti.Dim("N")   # key/value 序列长（causal 自注意下 host 保证 M == N）
BD = ti.Dim("BD")  # head 维：与同名 Const 参数 BD 绑定（launch 时由张量
#               shape 与 Const 实参共同取值 64）；rd = arange(0, BD) 的
#               lane 事实即可闭合 bounds 义务，无需运行期检查。


@ti.jit
def self_attention_kernel(
    q: ti.Buffer[ti.f16, (M, BD), ti.ReadOnly],
    k: ti.Buffer[ti.f16, (N, BD), ti.ReadOnly],
    v: ti.Buffer[ti.f16, (N, BD), ti.ReadOnly],
    o: ti.Buffer[ti.f32, (M, BD), ti.WriteOnly],
    scale: ti.f32 | ti.Positive,
    BM: ti.Const[int, ti.PowerOfTwo] = 64,
    BN: ti.Const[int, ti.PowerOfTwo] = 64,
    BD: ti.Const[int, ti.PowerOfTwo] = 64,
):
    pid = ti.program_id(0)

    rm = pid * BM + ti.arange(0, BM)      # (BM,)  query 行
    rn = ti.arange(0, BN)                 # (BN,)  key/value 行
    rd = ti.arange(0, BD)                 # (BD,)  head 维（编译期常界）

    m_ok = rm < M
    q_tile = ti.load(q, (rm[:, None], rd[None, :]),
                     mask=m_ok[:, None])                # (BM, BD) f16

    row_max = ti.zeros((BM, 1), ti.f32) - 1.0e30        # 行 running max
    row_sum = ti.zeros((BM, 1), ti.f32)                 # 行 running 归一化
    acc = ti.zeros((BM, BD), ti.f32)                    # 输出累加器

    for n0 in ti.range(0, N, BN):
        rv = n0 + rn                                   # (BN,)
        n_ok = rv < N
        causal = rm[:, None] >= rv[None, :]            # (BM, BN)
        keep = causal & n_ok[None, :]

        # K^T tile：行坐标 rv 沿块列广播、列坐标 rd 沿块行广播，
        # 直接取出 (BD, BN) 转置块，无需 trans 内建
        k_t = ti.load(k, (rv[None, :], rd[:, None]),
                      mask=n_ok[None, :] & (rd[:, None] >= 0))  # (BD, BN)

        s = ti.dot(q_tile, k_t) * scale                # (BM, BN) f32
        s = ti.where(keep, s, -1.0e30)

        row = ti.max(s, 1)                             # (BM,)
        m_tile = row[:, None]                          # (BM, 1)
        # 新 running max 必须与旧值合并：若某 KV tile 整块被 causal
        # 掩蔽（m_tile = -1e30），直接采用会使 alpha = exp(+1e30) 溢出。
        m_new = ti.where(m_tile > row_max, m_tile, row_max)
        alpha = ti.exp(row_max - m_new)                # 旧 max 校正因子 ≤ 1
        p = ti.where(keep, ti.exp(s - m_new), 0.0)     # (BM, BN) f32

        v_tile = ti.load(v, (rv[:, None], rd[None, :]),
                         mask=n_ok[:, None])           # (BN, BD) f16

        acc = acc * alpha
        row_sum = row_sum * alpha + ti.sum(p, 1)[:, None]
        acc = ti.dot(ti.cast[ti.f16](p), v_tile, acc=acc)
        row_max = m_new

    ti.store(o, (rm[:, None], rd[None, :]), acc / row_sum,
             mask=m_ok[:, None])


def main():
    import numpy as np

    m = n = 96                          # 96 非块整除：尾块 + causal 掩蔽
    d = 64
    rng = np.random.default_rng(0)
    q = rng.standard_normal((m, d)).astype(np.float16)
    k = rng.standard_normal((n, d)).astype(np.float16)
    v = rng.standard_normal((n, d)).astype(np.float16)
    o = np.zeros((m, d), dtype=np.float32)
    scale = d ** -0.5

    BM, BN, BD = 64, 64, 64
    grid = (ti.cdiv(m, BM),)
    self_attention_kernel[grid](q, k, v, o, scale, BM=BM, BN=BN, BD=BD)

    # 参考实现：fp32 精度的 causal softmax(QK^T) V
    qf, kf, vf = (x.astype(np.float32) for x in (q, k, v))
    s = (qf @ kf.T) * scale
    rows = np.arange(m)[:, None]
    cols = np.arange(n)[None, :]
    s = np.where(cols <= rows, s, -np.inf)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    ref = p @ vf

    assert np.allclose(o, ref, atol=2e-3), "self-attention mismatch"
    print(f"self-attention (causal, online softmax): {m}x{n}x{d} OK")


if __name__ == "__main__":
    main()
