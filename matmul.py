# matmul.py — 可运行的完整 matmul 验证（kernel 与 examples/matmul_loop.tila 相同）
#
# v0.4 K-loop：K 维沿 tila.range 循环分块累加——任意 K（K 不是 BK 的倍数由
# mask + other=0.0 补零；正确性包络不再受 K <= BK 限制）。
# acc 由 tila.zeros 播种（Zeros 种子 + L7 join 单位元 → φ 类型 = 首次累加的
# Mma 分布），体内唯一跨迭代通道是 acc += tila.dot(x, y)（受限累加器纪律，
# docs/v0.4-kloop.md）。
# v0.3：b 声明全符号 strides (sb0, sb1)——转置、padding 等非连续布局一律合法；
# 线性化由编译器发射，表面上没有地址算术。a/c 省略 strides = 隐式 RowMajor。
#
# 验证矩阵：{CPU 解释器, GPU(Triton)} × {连续 b, 转置 b, padded b} ×
# {K=50（< BK 单迭代重掩码）, K=512, K=4096（大循环）}，
# 参照 torch.matmul / numpy f32 matmul；M/N 非 BM/BN 整除验证边界 mask。

from __future__ import annotations

import numpy as np

import tila


@tila.jit
def matmul(
    a: tila.Tensor[tila.float16, M, K],
    b: tila.Tensor[tila.float16, K, N, (sb0, sb1)],
    c: tila.Tensor[tila.float32, M, N],
    BM: tila.constexpr = 64,
    BN: tila.constexpr = 128,
    BK: tila.constexpr = 64,
):
    # 注：局部变量的类型注解（AnnAssign）不在 Tila 子集内（E11）——类型推导
    # 由 checker 完成，类型信息以行尾注释标注（# : 类型）。
    pid_m = tila.program_id(0)                    # : Scalar(i32)
    pid_n = tila.program_id(1)                    # : Scalar(i32)

    rm = pid_m * BM + tila.arange(0, BM)          # : Tile[i32, (64,),   L0]
    rn = pid_n * BN + tila.arange(0, BN)          # : Tile[i32, (128,),  L1]
    rk = tila.arange(0, BK)                       # : Tile[i32, (64,),   L2]

    rm2 = tila.expand_dim(rm, 1)                  # : Tile<i32, (64,1),  L0>
    rk2 = tila.expand_dim(rk, 0)                  # : Tile<i32, (1,64),  L0>
    rk3 = tila.expand_dim(rk, 1)                  # : Tile<i32, (64,1),  L0>
    rn2 = tila.expand_dim(rn, 0)                  # : Tile<i32, (1,128), L1]

    c_m = (rm2 < M) & (rn2 < N)                   # : Tile[bool, (64,128)]

    acc = tila.zeros((BM, BN), tila.float32)      # : Tile<f32, (64,128), Zeros 种子>
    for k0 in tila.range(0, K, BK):               # : Scalar(i32)   归纳变量
        rka = rk2 + k0                            # : Tile<i32, (1,64),  L0>
        rkb = rk3 + k0                            # : Tile<i32, (64,1),  L0>
        x = tila.load(a, (rm2, rka),
                      mask=(rm2 < M) & (rka < K), other=0.0)   # : Tile<f16, (64,64)>
        y = tila.load(b, (rkb, rn2),
                      mask=(rkb < K) & (rn2 < N), other=0.0)   # : Tile<f16, (64,128)>
        acc += tila.dot(x, y)                     # : Tile<f32, (64,128), Mma>  φ 链

    tila.store(c, (rm2, rn2), acc, mask=c_m)      # : ()   R9'：内存边界布局无关


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

M, N, PAD = 100, 140, 160          # M/N 非 BM/BN 整除；PAD > N
KS = (50, 512, 4096)               # 50 < BK（单迭代重掩码）；512/4096 多迭代
RTOL, ATOL = 1e-2, 1e-2


def _check(tag, c, want):
    err = float(np.abs(np.asarray(c) - want).max())
    ok = np.allclose(np.asarray(c), want, rtol=RTOL, atol=ATOL)
    print(f"{'PASS' if ok else 'FAIL'}  {tag:<44} max|err| = {err:.3e}")
    assert ok, f"{tag}: 结果与参照不一致 (max err {err})"


def main():
    rng = np.random.default_rng(0)
    for K in KS:
        a_np = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
        bt_np = (rng.standard_normal((N, K)) * 0.5).astype(np.float16)   # b 的转置源
        pad_np = (rng.standard_normal((K, PAD)) * 0.5).astype(np.float16)
        print(f"--- K = {K} ---")

        # ---- 路径 1：CPU reference interpreter（numpy → 解释器）----
        for tag, b in (
            ("CPU  b 连续 (strides=(N,1))", bt_np.T.copy()),
            ("CPU  b 转置视图 (strides=(1,N))", bt_np.T),
            ("CPU  b padded 切片 (strides=(PAD,1))", pad_np[:, :N]),
        ):
            c = np.zeros((M, N), dtype=np.float32)
            matmul(a_np, b, c)
            _check(tag, c, a_np.astype(np.float32) @ b.astype(np.float32))

        # ---- 路径 2：GPU（torch CUDA → 生成的 Triton launcher）----
        try:
            import torch

            if not torch.cuda.is_available():
                print("SKIP  GPU（CUDA 不可用）")
                continue
            a_t = torch.from_numpy(a_np).cuda()
            bt_t = torch.from_numpy(bt_np).cuda()
            pad_t = torch.from_numpy(pad_np).cuda()
            for tag, b in (
                ("GPU  b 连续 (strides=(N,1))", bt_t.t().contiguous()),
                ("GPU  b 转置视图 (strides=(1,N))", bt_t.t()),
                ("GPU  b padded 切片 (strides=(PAD,1))", pad_t[:, :N]),
            ):
                c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
                matmul(a_t, b, c)
                torch.cuda.synchronize()
                _check(tag, c.cpu().numpy(),
                       a_t.float().cpu().numpy() @ b.float().cpu().numpy())
        except ImportError:
            print("SKIP  GPU（torch 未安装）")


if __name__ == "__main__":
    main()
