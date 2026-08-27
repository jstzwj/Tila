# Tila 开发路线（Stage 0–6）

状态：定稿（2026-08 评审修订）。本文定义从零实现 Tila 的完整路线：Stage 0–6 的先后顺序、每个阶段的实验/验收、第一竖切清单与测试体系。评审的核心建议之一是**不要先写 parser**——先做 Triton oracle + TIR + reference interpreter，这三样一旦确定，frontend → checker → lowering 基本自然收敛。

---

## 1. 总目标

**v0.1 目标**（评审 §42 原句）：

> *Demonstrate that a strongly typed, layout-aware subset of Triton can be compiled to Triton with a total and semantics-preserving lowering.*

实验程序集：**add、saxpy、masked_add、fp8_add**。验证四件事：syntax correctness（子集校验）、type soundness（E 码矩阵）、lowering totality（黄金测试）、GPU correctness（differential）。

**设计立场**（评审 §43）：Tila 的真正研究点是

> *Can we define a small abstract layout calculus that is expressive enough to statically guarantee correctness of a useful class of Triton kernels, while remaining independent of Triton's concrete layout encodings?*

即 `Tila = Strong Types + Shape Algebra + Abstract Layout Algebra + Total Triton Lowering`，而不是 `Tila = prettier Triton`。

## 2. Stage 0：Triton Oracle（不写编译器）

**目标**：只用现成 Triton 回答一个问题——"Tila 的 layout 抽象（v0.1 只有 `identity`）是否够用？" 不要靠脑补，直接从 Triton concrete behavior 反推。

**实验设计**：手写 add 的 Triton kernel（与 `examples/add.lowered.py` 同形），固定实验域：

```
BLOCK ∈ {32, 64, 128, 256}
num_warps ∈ {1, 2, 4, 8}
dtype ∈ {fp16, bf16, fp32}
```

对每个组合 dump TTIR 与 TritonGPU IR，观察：

| 观察点 | 问题 |
|---|---|
| `tl.arange(0, BLOCK)` | 什么 encoding？（BlockedEncoding / LinearEncoding / …） |
| `tl.load(...)` | 结果 encoding？与 arange 是否一致？ |
| `+`（elementwise） | operand/result encoding 是否一致？ |
| size-1 broadcast（`tl.expand_dims` 后） | encoding 如何变化？ |
| mask | 是否同 layout？ |

**工具**：`tools/dump_triton_layout.py`——输入 `(shape, dtype, num_warps, blk_config)`，输出每个中间值 shape + encoding 的清单；批量收集上表组合，产出笔记 `docs/layout-oracle-notes.md`。

**验收**：能明确回答"v0.1 的 identity 抽象（等价 = 种子相等）在 1D 情形下是否被 Triton 的实际行为支持"。此阶段不写任何 Tila 编译器代码。

## 3. Stage 1：Minimal Tila（第一竖切）

**目标**：只支持 1D、常量块、f32/fp16、`program_id`/`arange`/`load`/`store`/`+`/`<`/`mask`，从 Python source 一直走到 GPU，把 `add.tila` 完整跑通。

**先写这 7 个东西**（评审 §46，映射到新目录）：

1. `tools/oracle_add.py`——手写 Triton add，GPU 实测基准（Stage 0 的延续物）；
2. `docs/semantic-model.md`——语义模型（值类别、执行模型、内建合同）**先行冻结**，它是 checker 与 lowering 的共同契约；
3. `src/tila/types/`——`dtype.py` / `shape.py` / `type.py`（v0.6a 重构后：
   `layout.py` 拆为 `dist.py` + `memory.py` + `join.py`——DistExpr 五 term +
   normalize_dist/equiv_dist、MemoryLayout、纯结构谓词，见 `type-system.md` §3）；
4. `src/tila/tir/`——TIR 节点与 canonical dump；
5. `src/tila/checker/`——规则驱动的 check_kernel；
6. `src/tila/backend/triton/lowering.py`——total lowering；
7. `tests/test_add.py`——黄金 TIR + 黄金 Triton + GPU differential。

**不要**在竖切跑通前实现 E01–E17 的全部错误——第一遍之前根本不知道哪些错误边界是真正合理的语言边界。诊断系统从第一天就存在（`TilaError` 结构固定），但错误分支随竖切生长。

**验收**：`add.tila` → 黄金 `add.tir.txt` + `add.triton.py` 逐字节一致；GPU 上与 `torch.add` 逐元素一致。

## 4. Stage 2：Strong Typing

加入：完整 dtype 域与能力表、无隐式提升（E02）、shape 系统与广播（E03）、`tila.cast`、符号维 `N`、TIR 全值定型（不变量 6）。

**验收**：E 码矩阵成型（每码一个最小复现 + 断言 code/loc/信息形态）；fp8_add 走通（storage-only 纪律）。

## 5. Stage 3：Layout Algebra

加入：`identity`/`broadcast`/`product` 三 term 的表达力、normalize（律 L1–L4）、equiv；把 Stage 0 的 oracle 结论固化成律。

**验收**：正式证明 `checker accepted ⇒ lowering valid`（在 TIR 不变量与 lowering 表之间建立对应）；`tests/types/layout/` 的律测试全绿。

## 6. Stage 4：2D

按 `docs/v0.2-preview-2d.md`：`expand_dim`、size-1 广播、`&`、`Product` 与律 L5；解禁 rank-2 注解与 `program_id(1)`。

**验收**：`batched_add` 走通；matmul fragment 预览（`dot` 的 MMA 约束是桥接问题的第一个实例，若进入则同时启动桥接研究）。

## 7. Stage 5：Backend Validation

处理 `num_warps`、block shape、layout realizability 等 Triton 限制：`backend/triton/validator.py` + B01–Bxx（`triton-lowering.md` §8）。

**验收**：E 码（语言错误）与 B 码（后端限制）严格分开；任何 B 码失败都发生在 lowering 之前（"check backend → fail"，不破坏 lowering total）。

## 8. Stage 6：Differential Compiler Testing

- **Reference interpreter**（评审 §40，最值得新增的一项）：TIR → NumPy/PyTorch 解释器（CPU），作为独立 oracle：

  ```
                    TIR
                   /    \
                  ▼      ▼
        Reference Interpreter  Triton Lowering
                  │              │
                  ▼              ▼
                 CPU            Triton → GPU
                               CPU result == GPU result
  ```

  没有它，Tila 只有 `Tila → Triton → GPU` 一条路径，无从证明"Tila 语义与 Triton implementation 一致"。
- **GPU differential**：随机 input / 随机 N（含非 2 次幂）/ 随机 mask / 随机 dtype，与 PyTorch 对拍（`torch.testing.assert_close`）。
- **Fuzzing**（评审 §39）：语法极小，适合随机 AST 生成——随机生成小 kernel（arange/load/load/add/mul/compare/and/store），验证 "Tila compile vs reference interpreter" 与 "Triton result vs reference result" 双对拍（10 万级；无 GPU 环境时只跑到 interpreter 对拍）。
- 黄金 TIR / 黄金 Triton 纳入 CI 逐字节比对。

## 9. 测试体系（分层）

```
tests/
  frontend/
    valid/       合法程序集（add、saxpy、masked_add、fp8_add、cast 链…）
    invalid/     E11–E15 每码一个最小复现
  types/
    dtype/      能力表、cast 矩阵（含 FP8 两端点）
    shape/      相等、⊗ 广播、符号维
    broadcast/  E03 边界
    layout/     normalize/equiv 律测试（L1–L4；v0.2 加 L5/L6）
  checker/
    arithmetic/ E02/E16 矩阵
    load/       R8 各前提
    store/      R9 各前提（含单位语境 E08）
  tir/
    canonical/  dump 格式
  lowering/
    golden/     add.tir.txt、add.triton.py（逐字节）
  backend/
    triton/     B 码验证
  integration/
    gpu/        标记 gpu；torch 对拍
  fuzz/         随机 kernel 生成 + 双对拍脚本
```

测试不是只有 expected output：source → AST → TIR → Triton → GPU result 全链路可测（`development-plan.md` §9 与 `type-checker.md` §9 的测试策略互补）。

## 10. 里程碑与验收

| 里程碑 | 验收 |
|---|---|
| Stage 1 末 | `add.tila` 黄金一致 + GPU 与 `torch.add` 一致（N ∈ {1, 127, 128, 129, 1000}；BLOCK ∈ {32, 128}） |
| Stage 3 末 | checker accepted ⇒ lowering valid 的对应建立；E 码矩阵全绿 |
| Stage 6 末 | interpreter == GPU（同一 TIR 双路径）；fuzz 10 万 kernel 无原生异常、无双路径分歧 |
| v0.1 整体（评审 §42） | add / saxpy / masked_add / fp8_add 四程序全部：语法正确、类型健全、lowering total、GPU 正确 |

## 11. 开放问题（研究方向）

1. **桥接问题**：Tila 等价/来源代数 ↔ Triton concrete encoding（BlockedEncoding、LinearEncoding、CTALayout、MMA layout）——是否、以及如何精化为可证的 realization 映射；`dot` 的 MMA 约束是第一个实例（`type-system.md` §5）。
2. **解释器语义范围**：reference interpreter 覆盖到哪一层（masked load 的 other 语义、fp8 舍入模式对齐）才能与 Triton 严格对拍。
3. **layout 成本的显式化**：`convert(L → L')` 与代价模型（shuffle / shared memory round-trip）何时进入（v0.2+）。
4. **维度绑定索引**：索引 tile 记录其来源维，把"`idx` 里的 `N` 是否是 a 的 dim1"从用户责任变成编译期检查（v0.3 候选，`v0.2-preview-2d.md` §6）。
5. **全运算 shape 代数（DimExpr）**：shape 表达式目标全集已定为 DimExpr——`+ − × floordiv ceildiv mod max min`，`/` 禁用（`type-system.md` §2.1，2026-08-24 两轮评审定案）。两层判定（确定性重写 + 约束感知证明）、must_equal 只认 ProvenEqual、SMT 仅可插拔 fallback 且不进主路径。分阶段启用：线性片段先行，非线性项（`N*M`、`//`、`%`、max/min）随后；首批消费者 `cat`（R-cat）、reshape（numel 守恒证明）与维度绑定索引（上条）。
6. **shape 约束系统**：`tila.assume` 提案（编译期 shape 约束，非运行期断言）、维度正性（N ≥ 1）自动约束、`ProvenNotEqual` 的诊断利用（`type-system.md` §2.1，阶段二）。