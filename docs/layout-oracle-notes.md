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

---

## 第二轮：归约的边缘化观测（H2 假说检验，v0.5-reduce §8.1）

日期：2026-08-24。工具：`tools/oracle_round2_reduce.py`
（`scripts\run_oracle_round2.bat`）。实验域（评审强化矩阵）：
op ∈ {sum, max} × axis ∈ {0, 1} × num_warps ∈ {1, 2, 4, 8} × dtype ∈
{fp16, fp32}，2D tile (64, 128)，triton-windows 3.7.1 / TITAN Xp（SM 6.1），
共 32 组，产物落 `build/oracle_round2/`。

**问题（H2，v0.5-reduce §2.6）**：Tila 声称 Product(L0,L1) --reduce_k-->
L_(1-k)（幸存轴保留自身分布）。Triton 的 `tl.sum/tl.max` 结果 encoding
是否与之相容？强形式 = 结果 encoding 与独立构造的幸存轴 1D tile
（`tl.arange(0, L)`）同族；弱形式 = 不同族但单次 `convert_layout` 到达。

### 观测（32/32 组一致）

1. **reduce 结果 encoding 恒为输入分布的切片**：

   ```
   %s = "tt.reduce"(%x) <{axis = 1}> (…) : (tensor<64x128xf32, #blocked>)
                                    -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
   ```

   axis=0 时 `dim = 0`。不是独立 1D 构造的 blocked 族——**强形式证伪**。

2. **`expand_dims(slice, k)` 直接返回 parent**：

   ```
   %s2 = tt.expand_dims %s {axis = 1} : tensor<64xf32, #ttg.slice<…>>
                                    -> tensor<64x1xf32, #blocked>
   ```

   L9（expand_k(marginal_k(L)) ≡ L，v0.5-reduce §2.6 的 intended law）在
   TTGIR 层被逐字实现——"边缘化取因子、升维还原因子"是恒等往返。

3. **softmax 形状轨迹 0 次 convert_layout**（全部 32 组）：load → reduce →
   expand → 逐元素乘 → store 全链无分布转换。归约结果的 slice 布局在
   轨迹内天然可用——**弱形式成立且升级为零转换**。

4. **结果与"新鲜"幸存轴 tile 相遇时按需单次转换**：reduce kernel 中唯一
   的 `convert_layout` 位于 1D store（`slice → #blocked1` 与 1D 指针
   arange 对齐）。Tila 语义层判等（结果 = L0）与 concrete 按需转换相容。

### 判定与含义

- **H2 = 强形式证伪、弱形式成立（0-convert 轨迹）**——与 H1 的结局模式
  同构：家族级（语义级）成立、具体 encoding 强形式不成立。
- 证伪的方式比预期更有利：slice-of-parent 就是"幸存轴继承其在复合分布中
  的位置"的字面具体化——Tila 的 Product-因子语义声明是它的语义影子，
  marginal 规则按原样实现（不降级为"语义声明 + 显式 convert 层"）。
- dtype 观测（附带，v0.5-reduce §4 的依据）：`tl.sum` 有 `dtype` 形参且
  默认把 int<32 提升为 i32/u32 返回；`tl.max` 无 `dtype` 形参，对一切
  <32 位 dtype（f16/bf16/i8/i16）内部提升 f32/i32 **并以提升后 dtype
  返回**。Tila 的"结果 dtype = 输入 dtype"由 lowering 显式适配（sum 恒
  传 `dtype=`；max 对 <32 位 cast 恢复）。

### 复现

```
scripts\run_oracle_round2.bat           # 或 python tools\oracle_round2_reduce.py
type build\oracle_round2\A_sum_axis1_w4_fp32.mlir    # reduce 结果 slice 布局
```

---

## 第三轮：Mma 边缘化与 attention 轨迹（H3-S/H3-C 假说检验，v0.6-attention §2.7/§8.1）

日期：2026-08-24。工具：`tools/oracle_round3_attention.py`
（`scripts\run_oracle_round3.bat`）。实验域：kernel {A,B,C,D} × 形状
(BM,BN,D) ∈ {(64,64,64),(64,128,64),(128,64,64)} × num_warps {1,4}
（f16 入 / f32 累加），triton-windows 3.7.1 / TITAN Xp（SM 6.1），共 6 组，
产物落 `build/oracle_round3/`。

**问题（H3，判据分离——评审 §9/§10 采纳）**：Tila 声称 marginal(Mma, k) =
Slice(Mma, k)（分支六）+ 读透明四规则（R-BT/R-PT）。拆两个独立判据：
H3-S（语义）= reduce-over-dot 结果 encoding 与亲代切片同构、expand 还原
亲代、跨矩阵稳定；H3-C（代价）= attention 正则轨迹在 dot 操作数入场之外
的 convert 计数 ≤ 4/kernel（convert 计数不判语义生死，只判桥接质量）。

四个 kernel（docs/v0.6-attention.md §1.1/§1.2 的逐行同构）：A reduce-over-dot
（dot → tl.max(·,1) → store 1D）；B expand-join（A + expand → `qk * s2` →
store 2D）；C fragment 全轨迹（where(-inf) → max → exp → sum → div →
cast → 第二 dot → store）；D flash body（for n0 循环内 maximum/α 重定标
`acc *= α₂`、`acc += dot(p,v)`，循环外终归一化除法）。

### 观测（6/6 组一致）

1. **H3-S 逐字成立**：reduce 结果 encoding 恒为亲代切片——
   `tt.dot … -> tensor<64x64xf32, #blocked2>` 的 max 结果 =
   `#ttg.slice<{dim = 1, parent = #blocked2}>`（axis=0 时 dim=0）；
   `tt.expand_dims %s {axis=1}` 直接返回亲代 `#blocked3`。与 H2（Product
   输入）的结构完全同构——`Slice(Mma,k)` 的 provenance 语义声明是
   TTGIR 层行为的语义影子。
2. **H3-C strong**：converts(A/B/C/D) = 1/1/1/1（bn128 组 D=3），而各
   kernel 均含 2 个 dot（本机基线 = 每 dot 1 次操作数入场，H1 实测）——
   **入场之外 0–1 次额外**：C（attention fragment 全轨迹：where-cond 跨
   族、exp/sum、(BM,1) 广播除法、第二次 dot）与 D（跨 Mma rescale、
   终归一化除法、maximum 链、循环携带）的全部"读透明位点"合计 ≤ 1 次
   convert，远低于 C_max=4。
3. **附带核实**：`tl.full((BM,), float("-inf"), dtype=tl.float32)` 编译
   通过（flash 的 max 种子发射事实）；`tl.maximum` 以 `arith.maxnumf`
   落地、同 dtype 进出无提升。

### 判定与含义

- **H3-S pass + H3-C strong = 完美档**（§2.7 决策表）：分支六与读透明
  规则按设计实现，不降级；attention 的值级正确性由 GPU 差分独立验收
  （`tests/test_gpu_integration.py`，11 项含极值数据）。
- 与 H1/H2 的结局**不同且更好**：H1/H2 均为"强形式证伪、家族级成立"，
  H3 的结构声明（切片/还原）与代价声明（有界）**双双原样成立**——因为
  分支六从一开始就按 provenance（而非独立等价物）措辞，语义声明与
  concrete 行为之间的缝隙在 H3 为零。
- **SM 6.1 限定沿用**：本机 dot 以 FMA 下探（blocked 族 encoding）；结构
  结论（切片/还原/计数）按 encoding 无关设计，mma_layout 的 encoding 级
  复核待 TC 机器（与 H1 同款挂起项）。实验期记录：初始矩阵第三组
  (128,128,64) 的 D kernel（双 128 维 dot 循环）在本机编译超 1 小时未
  完成，换 (128,64,64) 覆盖 BM=128 方向；BM=BN=128 的值级正确性由 GPU
  差分覆盖。
- 实现期解析修正两处（工具 bug，非实验结论变化）：layout 定义行键名
  双井号（resolve 空转，靠名字相等侥幸判对 A）；expand 选择须匹配
  reduce 结果类型（排除先于 reduce 的坐标 expand_dims）。

### 复现

```
scripts\run_oracle_round3.bat           # 或 python tools\oracle_round3_attention.py
type build\oracle_round3\A_bm64_bn64_d64_w4.mlir    # reduce 结果 slice 布局
type build\oracle_round3\summary.json
```
