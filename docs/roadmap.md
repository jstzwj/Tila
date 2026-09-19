# Tila 路线图

状态：执行路线摘要（2026-09-19；M2 按 ADR-011 更新）

当前能力事实来源：[status.md](status.md)

详细任务、ADR 和退出标准：[../plan.md](../plan.md)

本文只说明里程碑顺序、能力归属和阶段边界。具体能力是否已经可用，以
`status.md` 为准；详细实施内容以 `plan.md` 为准。旧文档使用的
“Phase 0–3”编号不再作为当前计划术语。

---

## 1. 路线原则

1. **先一致，再正确，再验证，最后扩展**：先让规范/API/状态一致，再冻结
   类型和内存模型，随后完成 CPU 与 GPU 正确性闭环；
2. **Soundness 优先**：Unknown 不得升级为 ProvenSafe；未经证明的优化事实
   不得发射给后端；
3. **小支持域先闭环**：先固定一个 Triton/CUDA target 和有限 dtype/shape
   集合，再扩展 target、泛型和 layout；
4. **状态与设计分离**：规范中出现不等于已经实现，README 只承诺
   `status.md` 中的 `Implemented` 能力；
5. **每个阶段有退出标准**：实现、测试、诊断、文档和后端验证必须同时完成。

---

## 2. 里程碑总览

| 里程碑 | 状态 | 主题 | 主要退出结果 |
|---|---|---|---|
| M0 | 已完成 | 基线冻结与文档/API 对账 | 当前语法可构造，所有能力有唯一状态，文档不超前承诺 |
| M1 | 已完成 | 核心语言与内存模型定型 | Ptr/Buffer/RegionId/Extent、refinement 和 intrinsic registry 定型；[退出审计通过](m1-exit-audit.md) |
| M2 | 已完成 | CPU 正确性与静态证明闭环 | M2-01 至 M2-08 完成；证明、数据流、性质审计、explain/常量接口与固定 GPU 对照已落地，未验证范围见 status.md |
| M3 | 进行中 | Triton/CUDA 后端闭环 | M3-06 审计后已取得 CPU 托管首次成功并补测 dtype 矩阵；自动 GPU CI 缺失，结论仍 NOT READY；未测形状/架构不作承诺 |
| M4 | 基础 IR 实施中 | Effect、Atomic、Race、Uniformity | ADR-016 Accepted，M4-01b 局部 effect/定义引用/verifier 已实现；汇总待迁移，atomic/race/uniformity 暂缓 |
| M5 | 计划中 | 泛型、特化与 Target Capability | TypeVar、capability、target database、完整 FP8 支持 |
| M6 | 计划中 | 优化事实、性能诊断与 Layout | hint provenance、结构性性能 warning、layout 评审 |

M0/M1/M2 已完成。M3 的通过范围与剩余阻塞见 [退出审计](m3-exit-audit.md)；新增
GitHub 托管 CPU CI 不替代 GPU 持续验收。M4 基础 IR 的实施不改变 M3 未完成状态；
M4–M6 的设计/计划不表示相应能力已经可用。

---

## 3. 容易混淆的能力归属

### 3.1 FP8

| 层次 | 当前状态 | 归属 |
|---|---|---|
| `f8e4m3fn/f8e5m2` 名字、storage dtype 建模、禁止直接算术 | `Partial`，当前已有 | M0 基线事实，不代表完整 FP8 |
| 通用 Triton lowering/GPU 环境验证基础 | 未闭环 | M3 前置验证 |
| load→cast→compute→cast→store、舍入/溢出语义、FP8 dot 与 target gating | `Designed` | **M5 完整归属** |

因此，“当前有 FP8 dtype”不能表述为“已经支持 FP8 kernel”；完整 FP8 只在
M5 的 target capability 和真实 GPU 验证完成后开放。

### 3.2 Effect 与并发安全

| 层次 | 当前状态 | 归属 |
|---|---|---|
| kernel 级 `Read/Write[region]` 汇总 | `Implemented` | M0 基线事实 |
| `where` eager memory warning（`TILA-EFFECT-007`） | `Implemented` | M0 基线事实 |
| per-instruction effect IR、独立 effects 策略 | `Designed` | **M4** |
| atomic、inter-program race、uniformity/barrier | `Designed` | **M4** |

当前实现不是“完全不检查 effect”，也不是“完整 effect system”：它具有汇总和
一项 eager-where 检查；完整 effect/并发语义统一归 M4。

### 3.3 Target Capability

| 层次 | 当前状态 | 归属 |
|---|---|---|
| CUDA tensor 缺少 Triton 等基础后端错误 | `Partial`，当前已有 | M0 基线事实 |
| 固定 Python/PyTorch/Triton/CUDA/compute-capability 支持矩阵、device contract、GPU CI | `Partial`：本地矩阵/门禁已实现，自动 GPU CI 已撤下 | **M3 基础闭环** |
| 可查询 target database、操作/dtype capability、`@tila.requires` | `Designed` | **M5 完整归属** |

M3 负责证明“一个固定 target 上能够正确工作”；M5 才提供语言级、可扩展的
target capability 系统。完整 `TILA-TARGET` 能力归 M5。

### 3.4 TypeVar

| 层次 | 当前状态 | 归属 |
|---|---|---|
| 设计文档中的 `TypeVar` 示例 | `Designed`，不可执行 | **M5** |
| dtype TypeVar、bound/capability、一致绑定、诊断与特化缓存 | `Designed` | **M5** |
| shape-kinded TypeVar | 不在首版范围 | M5 以后重新评估 |

当前公共 API 没有 `tila.TypeVar` 或 `tila.Float`。在 M5 退出前，TypeVar 不能
出现在“当前语法”示例中。

---

## 4. M0：基线冻结与文档/API 对账

目标：准确回答“现在能写什么”，并让当前示例、公共 API 和文档陈述一致。

主要工作：

- 修复跨平台 CLI/example 可运行性；
- 建立零 skipped 的开发测试基线；
- 建立 `status.md` 唯一状态表；
- README 只承诺闭环能力；
- 统一本路线图中的 FP8/Effect/Target/TypeVar 阶段；
- 为 Markdown 当前语法示例建立 smoke test；
- 对齐 `tila.__all__`、frontend intrinsic allowlist 和 checker handlers。

M0 不负责扩充语言表面积。发现文档超前时，默认先降为 `Partial/Designed`，
而不是临时增加未经设计的实现。

退出标准：

- 当前 API 和 intrinsic 全部出现在状态表；
- README 无错误数量、源码行数或超前 GPU 声明；
- 当前语法示例可构造，未来语法有显式标记；
- Windows/Linux CLI 与官方 examples smoke 全绿。

---

## 5. M1：核心语言与内存模型定型

目标：冻结 bounds、effect 和 race 都依赖的类型模型。

主要工作：

- 按 [ADR-002](adr/002-region-id-and-extent.md) 将内存身份 `RegionId`、bounds
  数值范围 `Extent` 和 alias relation 拆分；
- 按 [ADR-001](adr/001-ptr-public-syntax.md) 实现 Ptr 公共参数顺序，并按
  [ADR-003](adr/003-buffer-stride-address-space.md) 实现 Buffer 的隐式 Global、
  launch-bound strides 与受限 `buf.ptr`；
- 明确 `buf.ptr`、元素 offset、未来 byte offset 的能力继承；
- 按 [ADR-004](adr/004-refinement-construction-syntax.md) 统一
  `Range/MultipleOf/Aligned` 的下标语法和合法性校验；
- 按 [ADR-005](adr/005-const-type-domain.md) 将 0.2.x Const 参数域固定为
  `Const[int]`，并分离 staged bool 与 runtime bool；
- 将 Access/AddressSpace 从字符串协议迁移为受控类型；
- 按 [ADR-006](adr/006-intrinsic-registry.md) 建立 intrinsic registry，集中管理
  名字、表面形式、阶段、effect、TIR 覆盖、target 和状态关联；
- 稳定核心错误码和诊断契约（M1-06 已完成机器 registry、统一渲染与
  非 debug traceback 门禁）。

退出标准：RegionId 不参与 bounds 数值比较，Extent 不作为 alias 身份；公共
类型构造、类型打印、规范和测试完全一致。2026-09-19 的
[M1 Exit Audit](m1-exit-audit.md) 已逐项验证这些条件。

---

## 6. M2：CPU 正确性与静态证明闭环

目标：在不依赖 GPU 的环境中建立高可信语义基线。

主要工作：

- 先固定 ADR-007 的整数溢出/除法/移位/cast 语义，数学整数与有限位宽分开；
- 按 [ADR-011](adr/011-smt-proof-and-trust.md) 建立 predicate DAG 与统一结果，
  拆分 verdict 和静态/契约/用户假设来源，unsafe 表示为 Exempted；
- Z3 作为默认通用证明引擎，保留小型快速路径，替代强制 DNF 展开；
- 查询/累计资源预算、语义与信任来源缓存、sat 反例可达性和 Unknown 原因；
- 完善分支/循环的数据流与 facts sound intersection；
- 明确整数、归约、NaN、bf16 等 interpreter 语义；
- 为所有 Implemented intrinsic 提供 interpreter 路径；
- 增加 property/metamorphic tests 和 explain golden。

进度：M2-01 已完成，新增保守整数门禁及独立 GPU 整数 differential；详见
[ADR-007](adr/007-integer-semantics.md)。现有本地 GPU 证据不代替 M3 的 CI 退出条件。

测试期间审计新旧证明路径差异，再切换默认并退役复杂手写推理。独立推进
布尔 tile mask 设计；Const bool、宿主整数白名单和显式舍入常量先评审再开放，
不随 SMT 自动改变 ADR-005 的 0.2.x 边界。提前衔接固定 GPU 环境的小型
整数/cast/mask/归约对照，无环境时保持 GPU 未验证，完整支持承诺仍归 M3。

退出标准：CPU 环境零 skipped；默认 SMT 与小型快速路径结果可解释，信任来源、
豁免、反例与预算覆盖；完成新旧差异审计和整数边界测试；interpreter 成为稳定的
规范 oracle。SMT 是已接受的待实现设计，不是当前能力。

---

## 7. M3：Triton/CUDA 后端闭环

目标：从“能够生成 Triton 字符串”升级为“在一个明确 target 上持续正确”。

主要工作：

- 固定 Python/PyTorch/Triton/CUDA/compute-capability 支持矩阵；
- 校验所有 tensor device、dtype、shape、stride、alignment 和 launch option；
- 建立 TIR verifier 与穷尽 lowering，禁止未知节点静默生成注释；
- 扩展 add/matmul/attention/fused-attention 的 TIR/Triton/explain golden；
- 建立真实 GPU CI 与 CPU/GPU differential；
- 让 alignment、multiple-of、contiguous hint 具有 proof provenance；
- 将 `num_warps` 从硬编码变为受限 launch option。

退出标准：固定支持矩阵中，官方 examples 的 GPU differential 持续通过；所有
Implemented intrinsic 的 Triton lowering 可真实编译；任何 hint 都能追溯事实。

---

## 8. M4：Effect、Atomic、Race 与 Uniformity

目标：在 M1 内存身份模型和 M3 GPU 闭环上建立最小并发安全系统。

主要工作：

- effect 成为每条 TIR 内存指令的原生字段；
- kernel effect summary 从指令聚合，不维护平行事实列表；
- `where` eager-effect 使用独立策略开关；
- 从 `atomic_add` 开始加入类型化 order/scope 和 target 检查；
- 对 affine program-id 地址实现安全/冲突/Unknown race 分级；
- 定义 Program/CTA/Warp/Varying uniformity lattice，并服务 barrier。

退出标准：effect 使用 RegionId；atomic、race、uniformity 有独立错误码、策略和
CPU/GPU 测试；复杂无法判定情形保持保守 warning。

---

## 9. M5：泛型、特化与 Target Capability

目标：提供真正受控的语言级泛型和硬件能力系统。

主要工作：

- dtype TypeVar、能力 bound、跨参数一致绑定和冲突诊断；
- 公共 `Float/Int/UInt/Numeric/DotInput/AtomicTarget/StorageFloat`；
- target database 根据后端、架构和 Triton 版本选择 capability；
- `@tila.requires` 声明附加要求，但不能绕过实际 target 检查；
- 完整特化缓存键与失效规则；
- FP8 load/cast/compute/store、dot、舍入/溢出和 GPU 验证闭环。

退出标准：TypeVar 文档示例真实运行；不支持的特化在 launch 前拒绝；只有通过
固定 GPU 验证的 bf16/FP8 组合才标为 `Implemented`。

---

## 10. M6：优化事实、性能诊断与 Layout

目标：保持正确性语义不变，系统消费已经证明的事实。

主要工作：

- 每条优化事实记录来源、推导规则、依赖 contract 和 lowering 消费点；
- 增加不连续访问、stride、tile、归约形态等结构性 warning；
- 性能 warning 永远不改变类型正确性；
- 完成 Triton layout 稳定性、dot operand 联合约束和收益评估；
- 只有评审通过后，才决定 Layout 是否进入公共 `Block` 类型。

退出标准：hint 有完整 provenance 和负测试；Layout 的公共化有独立设计评审与
真实 benchmark 证据。

---

## 11. 长期研究项

以下不属于当前 M0–M6 的完成承诺：

- 直接 PTX 或多后端；
- shared-memory typestate；
- warp specialization；
- 用户控制具体 TritonGPU encoding；
- 任意 dependent types；
- 复杂跨过程 alias/race analysis；
- shape-kinded 泛型。

这些能力必须在核心语言、proof 和 GPU 支持矩阵稳定后单独立项。

---

## 12. 与旧代码库的关系

2026-08 基线的 `.tila` 语法、DistExpr layout、旧坐标寻址原语和
E01–E21/B 诊断体系已经归档到 `trash/`，不再代表当前 Tila。

2026-09 的新语义实现已经直接位于 `src/tila`；不存在并行的新实现目录，也
不再计划通过第二套包名迁移。后续重构在现有模块中逐项进行，并由 golden、
interpreter 和诊断测试保持行为可审查。

保留并继续演进的工程资产包括：

- Python AST frontend；
- typed HIR/TIR；
- Triton source lowering；
- NumPy reference interpreter；
- launcher、CLI、诊断和测试组织。

---

## 13. 测试策略

各里程碑共同使用以下验证层：

1. **Unit**：dtype、DimExpr、facts、solver、类型构造；
2. **Rule matrix**：每条语言规则的正例、反例和稳定诊断；
3. **Golden**：canonical TIR、Triton source、explain/proof；
4. **Interpreter**：官方 examples 和极端 shape/dtype/stride；
5. **Differential**：固定 target 上 CPU interpreter vs GPU；
6. **Property/fuzz**：shape、mask、表达式和控制流组合；
7. **Documentation gates**：README、状态表、公共 API 和示例不漂移。

某项能力只有满足 `plan.md` 的完成定义后，才能在 `status.md` 中升级为
`Implemented`。
