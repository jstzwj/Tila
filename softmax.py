# softmax.py — 可运行的完整 softmax 验证（kernel 与 examples/softmax.tila 相同）
#
# v0.5 归约与逐元素内建的验收载体：f16 进出、f32 内部（入口 cast——强类型
# 不隐式提升）。五个语义原语的完整走查：
#   exp（逐元素，律 L8 记法）、where（三方广播 + 值选择，非严格）、
#   sum/max（分布的边缘化，marginal）、num_programs 未用但同批放行、
#   语境字面量（0.0 / tila.neg_inf 按 f32 语境定型，R24）。
# 掩码 max 的哨兵是 tila.neg_inf（max 的归约零元）——对 f16 最低值（−65504）
# 与全负行不污染；N > BN 的 online softmax 需非 zeros 种子（v0.6）。
#
# 验证矩阵：{CPU 解释器, GPU(Triton)} × M/N 非整除掩码（含 1×1 / 65×129 /
# 130×300）× 极值数据（−65504、全负行、+30000），参照 torch.softmax。

from __future__ import annotations

import numpy as np

import tila


@tila.jit
def softmax(
    a: tila.Tensor[tila.float16, M, N],
    o: tila.Tensor[tila.float16, M, N],
    BM: tila.constexpr = 64,
    BN: tila.constexpr = 128,
):
    pid = tila.program_id(0)                    # : Scalar(i32)
    rm2 = tila.expand_dim(pid * BM + tila.arange(0, BM), 1)   # : Tile<i32, (64,1),  L0>
    rn2 = tila.expand_dim(tila.arange(0, BN), 0)              # : Tile<i32, (1,BN),  L1>
    m2 = (rm2 < M) & (rn2 < N)                  # : Tile<bool, (64,BN), Product(L0,L1)>

    x = tila.cast(tila.load(a, (rm2, rn2), mask=m2), tila.float32)  # : Tile<f32, (64,BN), L2>
    neg = tila.where(m2, x, tila.neg_inf)       # : Tile<f32, (64,BN), L2>   掩码列不进 max
    mx = tila.max(neg, axis=1)                  # : Tile<f32, (64,),   L0>   边缘化
    mx2 = tila.expand_dim(mx, 1)                # : Tile<f32, (64,1),  L0>
    e = tila.where(m2, tila.exp(x - mx2), 0.0)  # : Tile<f32, (64,BN), L2>
    s = tila.sum(e, axis=1)                     # : Tile<f32, (64,),   L0>   边缘化
    s2 = tila.expand_dim(s, 1)                  # : Tile<f32, (64,1),  L0>
    y = tila.where(m2, e / s2, 0.0)             # : Tile<f32, (64,BN), L2>   掩码行无 NaN
    tila.store(o, (rm2, rn2), tila.cast(y, tila.float16), mask=m2)  # : ()


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

# (M, N, BN)：BN >= N（单 tile 覆盖整行）；M/N 非 BM/BN 整除验证边界 mask
CASES = ((1, 1, 128), (63, 127, 128), (64, 128, 128), (65, 129, 256),
         (100, 140, 256), (130, 300, 512))
RTOL, ATOL = 2e-2, 2e-3


def _data(rng, M, N):
    a = rng.standard_normal((M, N)).astype(np.float16)
    a[0, 0] = -65504.0                 # f16 最低值：−∞ 哨兵不污染 max
    a[M // 2] = -60000.0               # 全负行
    a[0, min(1, N - 1)] = 30000.0      # 大正值：max-减法防 exp 溢出
    return a


def _ref(a):
    import torch

    return torch.softmax(torch.from_numpy(a.astype(np.float32)), dim=-1) \
        .numpy().astype(np.float16)


def _check(tag, o, ref):
    err = float(np.abs(o.astype(np.float32) - ref.astype(np.float32)).max())
    ok = np.allclose(o.astype(np.float32), ref.astype(np.float32),
                     rtol=RTOL, atol=ATOL)
    print(f"{'PASS' if ok else 'FAIL'}  {tag:<34} max|err| = {err:.3e}")
    assert ok, f"{tag}: 结果与 torch.softmax 不一致 (max err {err})"


def main():
    rng = np.random.default_rng(0)
    for M, N, BN in CASES:
        a_np = _data(rng, M, N)
        ref = _ref(a_np)
        print(f"--- M = {M}, N = {N} (BN = {BN}) ---")

        # ---- 路径 1：CPU reference interpreter ----
        o = np.zeros((M, N), dtype=np.float16)
        softmax(a_np, o, BN=BN)
        _check(f"CPU  interpreter", o, ref)

        # ---- 路径 2：GPU（torch CUDA → 生成的 Triton launcher）----
        try:
            import torch

            if not torch.cuda.is_available():
                print("SKIP  GPU（CUDA 不可用）")
                continue
            a = torch.from_numpy(a_np).cuda()
            out = torch.zeros(M, N, device="cuda", dtype=torch.float16)
            softmax(a, out, BN=BN)
            torch.cuda.synchronize()
            _check("GPU  triton", out.cpu().numpy(), ref)
        except ImportError:
            print("SKIP  GPU（torch 未安装）")


if __name__ == "__main__":
    main()
