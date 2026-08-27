# attention.py — 可运行的完整 attention fragment 验证（kernel 与
# examples/attention.tila 相同）
#
# v0.6a layout-calculus 里程碑的验收载体：softmax(QK^T)·V，单 tile。
# 四条新增语义的完整走查（docs/v0.6-attention.md §0.2）：
#   marginal(Mma) → Slice（分支六：dot 结果的 max/sum = 亲代切片）、
#   R-PT（where 的谓词不进布局代数：Product 族掩码作用于 Mma 值）、
#   R-BT（one-sided 广播透明：(BM,1) Slice 经 expand 与 (BM,BN) Mma 合并）、
#   Slice provenance（Slice ≢ Identity，softmax 轨迹闭环于 Mma 族）。
# k 以坐标转置入场（load(k, (rn2, rd3)) 直接得到 (BD, BN)——无 trans 内建）。
#
# 包络：M 任意（grid 轴 0）；N ≤ BN（softmax 行耦合，N > BN 属 v0.6b online
# softmax）；D ≤ BD（收缩维补零安全）。block 组合受目标 SM 共享内存约束
# （本机 TITAN Xp 48KB：BN=BD=128 的 dot 需 64KB——典型配置 BD=64 不受限）。
# 数值卫生：qk_ − mx ≤ 0 恒成立（exp 无上溢）；每行 ≥ 1 有效列（N ≥ 1）→
   # mx > −∞ → 掩码列 e^-inf = 0、s ≥ 1（无除零）；i ≥ M 的行 semantically
# arbitrary（store 被掩码收割）。
#
# 验证矩阵：{CPU 解释器, GPU(Triton)} × M/N/D 非整除掩码（含 1×1×64 /
# 33×64×16 / 100×127×64）× 极值数据（−65504、全负行、±3e4 大 logits），
# 参照 torch 的 f32 参考。

from __future__ import annotations

import numpy as np

import tila


@tila.jit
def attention(
    q: tila.Tensor[tila.float16, M, D],
    k: tila.Tensor[tila.float16, N, D],
    v: tila.Tensor[tila.float16, N, D],
    o: tila.Tensor[tila.float16, M, D],
    BM: tila.constexpr = 64,
    BN: tila.constexpr = 64,
    BD: tila.constexpr = 64,
):
    pid = tila.program_id(0)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)   # : Tile<i32, (BM,1), L0>
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)              # : Tile<i32, (1,BN), L0>
    rn3 = tila.expand_dim(tila.arange(0, BN), 1)              # : Tile<i32, (BN,1), L0>
    rd2 = tila.expand_dim(tila.arange(0, BD), 0)              # : Tile<i32, (1,BD), L0>
    rd3 = tila.expand_dim(tila.arange(0, BD), 1)              # : Tile<i32, (BD,1), L0>
    m_qk = rn2 < N                                            # : Tile<bool, (1,BN), L0>

    q2 = tila.load(q, (rm2, rd2), mask=(rm2 < M) & (rd2 < D), other=0.0)  # : Tile<f16, (BM,BD), Product>
    k2 = tila.load(k, (rn2, rd3), mask=(rn2 < N) & (rd3 < D), other=0.0)  # : Tile<f16, (BD,BN), Product> ← 坐标转置
    v2 = tila.load(v, (rn3, rd2), mask=(rn3 < N) & (rd2 < D), other=0.0)  # : Tile<f16, (BN,BD), Product>

    qk = tila.dot(q2, k2)                          # : Tile<f32, (BM,BN), Mma₁>  dot 边界
    qk_ = tila.where(m_qk, qk, tila.neg_inf)       # : Tile<f32, (BM,BN), Mma₁>  R-PT
    mx = tila.max(qk_, axis=1)                     # : Tile<f32, (BM,),   Slice(Mma₁,1)>  分支六
    mx2 = tila.expand_dim(mx, 1)                   # : Tile<f32, (BM,1),  Slice(Mma₁,1)>
    p = tila.exp(qk_ - mx2)                        # : Tile<f32, (BM,BN), Mma₁>  R-BT
    s = tila.sum(p, axis=1)                        # : Tile<f32, (BM,),   Slice(Mma₁,1)>
    s2 = tila.expand_dim(s, 1)                     # : Tile<f32, (BM,1),  Slice(Mma₁,1)>
    w = p / s2                                     # : Tile<f32, (BM,BN), Mma₁>  R-BT
    y = tila.dot(tila.cast(w, tila.float16), v2)   # : Tile<f32, (BM,BD), Mma₂>  第二次 dot 边界
    tila.store(o, (rm2, rd2), tila.cast(y, tila.float16),
               mask=(rm2 < M) & (rd2 < D))         # : ()  内存边界（R9'）


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

# (M, N, D, BN, BD)：N ≤ BN、D ≤ BD（单 tile 包络）；M/N/D 非整除验证边界
CASES = ((1, 1, 64, 64, 64), (33, 64, 16, 64, 64), (64, 64, 64, 64, 64),
         (64, 33, 33, 64, 64), (100, 127, 64, 128, 64), (128, 128, 64, 128, 64))
RTOL, ATOL = 2e-2, 2e-3


def _data(rng, M, N, D):
    q = ((rng.standard_normal((M, D)) + 0.3) * 200.0).astype(np.float16)
    k = ((rng.standard_normal((N, D)) - 0.2) * 200.0).astype(np.float16)
    q[0, 0] = -65504.0                  # f16 最低值
    k[N // 2] = -60000.0                # 全负键行
    v = rng.standard_normal((N, D)).astype(np.float16)
    return q, k, v


def _ref(q, k, v):
    import torch

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).float()

    a = t(q) @ t(k).transpose(0, 1)     # logits（±3e4 量级：exp 安全由构造保证）
    return (torch.softmax(a, dim=-1) @ t(v)).numpy().astype(np.float16)


def _check(tag, o, ref):
    err = float(np.abs(o.astype(np.float32) - ref.astype(np.float32)).max())
    ok = np.allclose(o.astype(np.float32), ref.astype(np.float32),
                     rtol=RTOL, atol=ATOL)
    print(f"{'PASS' if ok else 'FAIL'}  {tag:<30} max|err| = {err:.3e}")
    assert ok, f"{tag}: 结果与 torch 参考不一致 (max err {err})"


def main():
    rng = np.random.default_rng(0)
    for M, N, D, BN, BD in CASES:
        q_np, k_np, v_np = _data(rng, M, N, D)
        ref = _ref(q_np, k_np, v_np)
        print(f"--- M = {M}, N = {N}, D = {D} (BN = {BN}, BD = {BD}) ---")

        # ---- 路径 1：CPU reference interpreter ----
        o = np.zeros((M, D), dtype=np.float16)
        attention(q_np, k_np, v_np, o, BN=BN, BD=BD)
        _check("CPU  interpreter", o, ref)

        # ---- 路径 2：GPU（torch CUDA → 生成的 Triton launcher）----
        try:
            import torch

            if not torch.cuda.is_available():
                print("SKIP  GPU（CUDA 不可用）")
                continue
            q = torch.from_numpy(q_np).cuda()
            k = torch.from_numpy(k_np).cuda()
            v = torch.from_numpy(v_np).cuda()
            out = torch.zeros(M, D, device="cuda", dtype=torch.float16)
            attention(q, k, v, out, BN=BN, BD=BD)
            torch.cuda.synchronize()
            _check("GPU  triton", out.cpu().numpy(), ref)
        except ImportError:
            print("SKIP  GPU（torch 未安装）")


if __name__ == "__main__":
    main()
