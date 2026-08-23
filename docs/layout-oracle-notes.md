# Stage 0 Triton Oracle 笔记（layout 抽象验证）

状态：实测记录（2026-08-24，triton 3.7.1 / TITAN Xp / CUDA 12.9）。
数据：`build/oracle/add_b{32,64,128,256}_w{1,2,4,8}_{fp16,bf16,fp32}.mlir`（48 组合，
由 `tools/dump_triton_layout.py --all` 生成；每份含 TTIR 与 TritonGPU IR）。

回答的问题（`development-plan.md` §2）：

> **"Tila 的 layout 抽象（v0.1 只有 identity）是否够用？"**

## 结论

**够用，且比预期更强。** 三个实测事实：

1. **单 encoding 不变量**。全部 48 个组合中，kernel 内每一个张量值——
   `tt.make_range`（arange 种子）、`tt.splat`（标量/指针广播）、`arith.addi/cmpi`
   （索引与 mask）、`tt.addptr`、`tt.load`、`arith.addf`、`tt.store`——
   共享**同一个** concrete encoding（`#ttg.blocked<...>`），无一处 layout 转换
   （无 `ttg.convert_layout`）。

   ⇒ Tila 的 L1–L4 擦除律（load/标量广播/cast/join 不改变分布）在 1D 情形
   与 Triton 实际行为一致；"等价 = 种子相等"的判定口径成立。
   两次独立 `tl.load` 经同一 `offs` 索引得到完全相同的 encoding——
   这正是 type-system §8 里 `z = x + y` 无需任何转换就能通过的现实依据。

2. **encoding 参数是 (BLOCK, num_warps, dtype) 的函数**，观察到的规律：

   - 元素守恒：`sizePerThread × threadsPerWarp × warpsPerCTA = BLOCK`
     （threadsPerWarp 恒 32，warpsPerCTA = num_warps）。
   - `order = [0]` 恒成立（1D 唯一维度序）。
   - dtype 只影响向量化宽度上限（16B/线程）：BLOCK=256、num_warps=1 时
     fp16/bf16 给 `sizePerThread=8`（8×2B=16B），fp32 给 `4`（4×4B=16B）；
     其余组合三 dtype 参数一致。

3. **跨配置 encoding 漂移，但等价关系内部稳定**。同一 kernel 换 num_warps
   即改变 concrete encoding（如 BLOCK=128：w1→spt4、w2→spt2、w4→spt1）。
   这印证了 type-system §3.2 的定位：DistLayout 是**等价/来源描述符**，
   不是 concrete 分布的转述；num_warps 是发射/启动参数，不进 Tila 类型。
   "Tila 等价 ⇒ Triton 分配相同 concrete encoding"在**单次编译内**成立
   （事实 1），跨编译配置不成立也不需要成立（事实 3）。

## 对桥接问题（type-system §5）的含义

- 1D：恒为 `#ttg.blocked<sizePerThread=[s], threadsPerWarp=[32],
  warpsPerCTA=[w], order=[0]>`，可实现性无约束——任何 2^k BLOCK 都能配平
  （s = BLOCK / (32w) 不为整数时 Triton 自行调整 s 与每 CTA 重复次数）。
- **2D（v0.2 预览实测，batched_add BM=64×BN=128、num_warps=4）**：

  ```
  #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32],
                           warpsPerCTA = [4, 1], order = [1, 0]}>
  ```

  三个观察：① `order` 开始起作用（`[1,0]`：列维最快，对应 row-major 连续方向），
  印证 1D 笔记中"order 在 2D 才有意义"的预判；② **整个 2D kernel 仍然只有
  一个 distinct encoding**——`Product(L0, L1)` 语义分布积对应的 concrete
  分布是单一的 2D blocked，expand_dim / size-1 广播 / `&` 全程无 layout
  转换；③ 元素配平沿两维独立成立（64 = spt1×tpw1×wpc0，128 = 4×32×1），
  4 元素/线程 × f32 = 16B 向量化上限仍然生效。
  ⇒ "Tila 等价 ⇒ 单次编译内相同 concrete encoding"在 2D 预览范围内同样成立。
- 差分测试（development-plan §8）已覆盖本阶段：同一 TIR 走 interpreter 与
  Triton/GPU 双路径逐元素一致（`tests/test_gpu_integration.py`，含 2D）。

## matmul fragment 的 MMA 观测（H1 假说检验，v0.2 片段二）

对 `examples/matmul.tila`（BM=64 BN=128 BK=64，num_warps=4）dump TTGIR：

```
#blocked  = blocked<{spt=[1,8], tpw=[4,8],  wpc=[4,1], order=[1,0]}>   # a 链（load x）
#blocked1 = blocked<{spt=[1,8], tpw=[2,16], wpc=[4,1], order=[1,0]}>   # b 链（load y）
#blocked2 = blocked<{spt=[4,4], tpw=[1,32], wpc=[4,1], order=[1,0]}>   # dot 结果（本机）
#blocked3 = blocked<{spt=[1,4], tpw=[1,32], wpc=[4,1], order=[1,0]}>   # c 链（索引/mask/store）
#shared   = swizzled_shared<{vec=1, perPhase=1, maxPhase=1, order=[1,0]}>  # dot 操作数 staging
convert_layout × 1；local_alloc/local_load × 2（dot 操作数经 shared memory 入场）
```

三个结论：

1. **家族级分离成立**。dot 结果自成一类 encoding（本机 TITAN Xp 无 Tensor Core，
   Triton 以 FMA 下探、结果驻留独立的 `#blocked2`；在有 Tensor Core 的机器上
   预期为 `#ttg.mma_layout`，待复测）。它与全部种子推导值的 encoding 不同——
   Tila 的 `Mma ≢ Product` 在 concrete 层面同样成立；store 侧 R9' 的坐标语义
   被 Triton 接受（blocked2 值写回 blocked 索引，无报错）。
2. **H1 的强形式被证伪（重要）**。"同 Tila 正规形式 ⇒ 同 concrete encoding"
   不成立：b 链与 c 链在 Tila 中同为 `product(L0,L1)`，但 Triton 分别分配
   `#blocked1`（tpw=[2,16]）与 `#blocked3`（tpw=[1,32]）——Triton 按**数据流
   与形状**各自择参，不按等价类。这证实 `type-system.md` §3.2 的立场：
   "Tila 等价 ⇒ 相同 concrete encoding" 不可作为规范承诺；它的正确角色是
   **"无隐藏转换"的启发式检查**（若 Tila 允许的混算引发大量 convert_layout，
   说明抽象漏掉了真实代价）。
3. **本 kernel 的唯一 convert_layout 来自 dot 操作数入场**（MMA 的固有代价，
   经 swizzled shared memory staging）——非 Tila 判等失误所致。dot 边界
   "构造性转换"的建模（blocked 进 MMA 出）与实测一致。

对差分的影响：无。翻译语义下 GPU 结果与 torch.matmul/interpreter 一致
（`tests/test_gpu_integration.py`）；encoding 漂移只关乎性能与桥接研究，
不影响正确性验收。

## 复现

```
scripts\run_gpu_tests.bat          # 前置：编译一次 cuda_utils（Windows 需 MSVC + CC=cl）
python tools/dump_triton_layout.py --all --out build/oracle/
```

典型输出（`add_b128_w4_fp32.mlir` 的 TTGIR，encoding 定义与逐值引用）：

```
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
...
%offs_0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
%mask_3 = arith.cmpi slt, %offs_2, %mask : tensor<128xi32, #blocked>
%x_5 = tt.load %x_4, %mask_3 : tensor<128x!tt.ptr<f32>, #blocked>
%z = arith.addf %x_5, %y_7 : tensor<128xf32, #blocked>
```
