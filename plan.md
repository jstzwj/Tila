# Tila 语言与实现完善计划

状态：执行计划 v2；2026-09-19 M3-06 审计与限定矩阵补测完成，CPU 托管首个 run 已通过；隔离 GPU CI 缺失，M3 仍 NOT READY

后续：2026-09-20 f4add24 的托管 CPU CI 已通过 953 项；M4-01c/d 已从 TIR 派生
控制流 effect 汇总、固定可选详细输出并验收绑定隔离；M4-02 已实现 where 独立效应检查。
本地 CPU 1264 项通过；GPU 417 节点（561 案例）通过。最小 atomic_add CPU/GPU 已实现，
uniformity 已有内部分析与可选输出，M3 仍未完成。M4-03a 已冻结最小 atomic_add
设计（ADR-017）；M4-03b/c 已完成前端/IR/CPU 与固定 RTX 3090 GPU 验收。
M4-03 已提交为 `eb0867f` 并推送；M4-04b 已冻结 ADR-018 为 Accepted，
M4-04c 已接入独立 Race 策略、启动门禁、同次 tile store lane 检查、稳定输出与绑定隔离。
M4-04d 已完成 819 次小域枚举/变形分析及限定覆盖退出审计。
M4-05b/c 已冻结 ADR-019，实现内部值/控制分析、可选输出及绑定隔离。
2026-09-20 暂缓 M4-05d 及下一轮功能扩展，优先执行 [C0 正确性收口](docs/correctness-closure.md)：

1. C0-01：修复负号规范化导致的错误 Safe、按实际 ABI 值检查入口精化；
2. C0-02：冻结 ADR-020，收紧静态分支类型合并，统一 Ptr checked i64 地址语义；
3. C0-03：逐条审计 Safe 规则，补独立枚举/SMT、CPU/GPU 对照及门禁测试；
4. C0-04：全量回归、golden、固定 GPU 严格复验，登记证据与兼容性变化后再评审恢复扩展。

C0-01..04 本地限定验收已完成；证据与剩余边界见上述审计。功能扩展保持暂停，
恢复前先评审语义收紧和新增反例。C0 提交 4aa13e9 已取得托管 CPU 1217 项成功记录；
[C1 独立边界复核](docs/c1-correctness-review.md)限定验收完成，修复权限/cast 遗漏并收紧 GPU return 支持。
C1 提交 785df52 的托管 CPU CI 已通过 1264 项并核对附件。
后续先评审兼容性变化与下一轮审计范围；不自动恢复 M4-05d。

基线日期：2026-09-19（M0/M1/M2 已完成，M3 进行中）

适用范围：语言规范、前端、类型系统、静态证明、TIR、Triton 后端、运行时、解释器、测试与文档。

---

## 1. 目标

Tila 的目标是一门以 Python 语法承载、面向 GPU kernel、编译到 Triton 的静态强类型 DSL。项目的核心竞争力应集中在四件事：

1. **类型正确性**：dtype、字面量、转换、shape、broadcast、控制流在编译期判定；
2. **内存正确性**：Buffer/Ptr capability 与逐轴 bounds proof 可审计、默认严格；
3. **可预测的 GPU lowering**：通过检查的 TIR 必须确定性地生成合法 Triton；
4. **事实复用**：类型和证明得到的 alignment、multiple-of、contiguous 等事实可安全地反馈给优化。

本计划同时完善“语言是什么”和“实现是否兑现语言承诺”。短期不以扩充内建数量为目标，先建立一个小而一致、可验证、可以真实运行在 GPU 上的语言核心。

### 1.1 完成定义

一个能力只有同时满足以下条件，才能在 README 中标记为“已实现”：

- 规范中有明确、无歧义的语法和静态语义；
- 公共 Python API 可以真实构造该语法；
- frontend/checker/TIR/lowering/interpreter 或 GPU 路径形成闭环；
- 正例、反例、诊断和至少一个端到端测试齐全；
- 文档示例可复制运行；
- 若依赖 target，则至少在一个明确版本的 Triton + CUDA 环境验证。

仅有设计文档的能力标记为 `Designed`；有部分代码但未闭环的标记为 `Partial`；不再使用“规范已经写了”代替“实现已经完成”。

---

## 2. 当前基线

### 2.1 已经具备

- `inspect.getsource → ast.parse → HIR → typed TIR` 的静态前端；
- Scalar、Block、Mask、Const、Buffer、Ptr 的基本内部表示；
- widening-only 数值转换与字面量语境化；
- 基础 broadcast、reshape、dot、归约和控制流检查；
- Buffer/Ptr load/store、capability 检查和 bounds obligation；
- 区间/谓词/cdiv 战术组成的无第三方依赖证明 fast path；
- `assume`、`unsafe_load/store`、launch contract 与 `launch_auto`；
- Triton 源码生成、NumPy reference interpreter、CLI；
- add、matmul、self-attention、fused-attention 示例；
- 当前本地测试基线：1264 passed、零 skipped；含 88 项 Race、64 项 uniformity 及 C0/C1 正确性专项。GPU 严格验收 417 节点/561 案例通过，含 7 项真实 CUDA Race 门禁和 22 项 C0/C1 对照及负测试；不作为 uniformity 同步证据；race=warn 允许明确 Unknown，不表示全部案例无竞争。托管 CI 记录见 docs/cpu-ci.md，各提交单独验收。

### 2.2 当前主要缺口

- M0/M1 已完成状态表、公共语法对账、RegionId/Extent 拆分和 intrinsic registry；
- ADR-007 的整数语义/保守门禁、有限位宽 SMT 与受限数据流证明已实现；
- 证明已使用共享 DAG、ProofResult、默认 Z3、Int/BitVec、预算及有界进程内缓存；
- 一般加载内容/循环/数据流可达性仍保守近似，SAT 不自动升级为 ProvenUnsafe；
- alignment 已反馈一维 stride-1 Buffer/Ptr 基地址 hint，更一般布局仍保守；
- GPU device/target 门禁和固定组合 differential 已验证；自动 GPU CI 已撤下，dtype/shape 全组合证据不足；
- 完整 FP8、target capability 与泛型仍是未来能力；
- effect 已有局部元数据、控制流派生汇总及 M4-01d 可选详细审计与隔离验收；race/uniformity/atomic 尚未形成系统；
- M2-05 性质/差异审计、M2-06 explain golden、M2-07a/b/c/d、M2-08 固定 GPU 对照已完成；更广 fuzz 和持续 GPU 验收继续归后续阶段。

---

## 3. 执行原则

### 3.1 先收敛，再扩展

按以下顺序推进：

```text
规范与 API 对齐
  → 类型与内存模型定型
  → CPU/checker 正确性闭环
  → Triton/CUDA 闭环
  → effect/race/uniformity
  → 泛型与 target capability
  → 性能事实与 layout
```

在前一阶段退出标准没有满足前，不把后一阶段能力宣传为已实现。

### 3.2 Soundness 优先

- bounds、shape、dtype 和 capability 只有被证明时才通过；
- Unknown 不得静默升级为 ProvenSafe；
- 任何发射给 Triton 的 hint 都必须能追溯到一条可靠事实；
- GPU 与 interpreter 结果不一致时，以缩小语言承诺为默认处理，不用特殊分支掩盖差异。

### 3.3 规范是可测试接口

- 文档中的每段规范示例都应进入 doctest、compile test 或 example smoke test；
- 错误码、诊断字段和阶段归属是兼容性的一部分；
- 每个语义规则至少有一个正例和一个最小反例。

### 3.4 避免永久双实现

不新建长期并存的 `tila2`。需要重构时在现有模块中建立小接口、逐项迁移，并用测试保持行为稳定。

---

## 4. 全局里程碑

| 里程碑 | 主题 | 退出结果 |
|---|---|---|
| M0 | 基线冻结与文档/API 对账 | 所有公开能力有状态，规范示例可构造 |
| M1 | 核心语言与内存模型定型 | Ptr/Buffer/Region/Extent、类型与诊断一致 |
| M2 | CPU 正确性闭环 | checker、proof、interpreter、CLI 全绿且可审计 |
| M3 | Triton/CUDA 闭环 | 固定环境真实编译执行，CPU/GPU differential 通过 |
| M4 | Effect 与并发安全 | effect IR、where、atomic、race、uniformity 分阶段可用 |
| M5 | 泛型与 target capability | TypeVar、requires、特化缓存、硬件能力表闭环 |
| M6 | 优化事实与 layout | hint soundness、性能诊断、layout 研究性落地 |

M0–M3 是近期主线；M4–M6 必须在前面基线稳定后开始。

---

## 5. M0：基线冻结与文档/API 对账

目标：让用户能够准确回答“现在能写什么”，并保证文档中的当前语法真实可用。

### M0.1 建立能力状态清单

新增 `docs/status.md`，按下列状态列出每个类型、语法和内建：

- `Implemented`：端到端完成；
- `Partial`：列出具体缺失路径；
- `Designed`：规范存在、尚未实现；
- `Deferred`：有意不在近期范围；
- `Removed`：不再支持并说明替代方案。

清单至少覆盖：dtype、Const、Block、Mask、Ptr、Buffer、控制流、每个 intrinsic、bounds、effects、target、interpreter、GPU。

验收：README 的“已实现”列表只能引用 `Implemented` 项；CI 校验 README 中测试数量不再手工硬编码，或直接去掉具体数字。

### M0.2 修正文档中的不可用语法

需要作出并落实以下决定：

1. Ptr 公共语法采用五参数规范，还是 v0 四参数简化形式；
2. Buffer 是否允许用户显式声明 strides/address space；
3. `Range`、`MultipleOf` 统一使用 `[]` 还是构造调用 `()`；
4. `cast[U](x)` 是否是唯一形式，是否真的支持 `cast(x, U)`；
5. `Const` 当前只支持 `int`，还是补齐 `bool`/数值 dtype；
6. `static_assert` 是否支持消息字符串；
7. TypeVar 示例全部标成未来语法，直到 M5 完成。

首选原则：公共语法尽量与类型总览一致；若短期实现成本过高，则缩小规范，不保留“看起来已经可用”的未来语法。

验收：抽取所有 Markdown Python 代码块，至少完成语法构造 smoke test；当前语法块不得因 `AttributeError`、`TypeError` 或缺失导出失败。

### M0.3 公共命名空间对齐

- 建立唯一的 intrinsic registry 或公开名字表；
- `tila.exp2` 等 frontend 可识别的名字必须真实存在于模块；
- `__all__`、frontend allowlist、checker handler、文档签名表由同一来源校验；
- 对 `any/all` 的方法形式和是否存在函数形式作出明确决定；
- IDE/introspection 下的 API 与 kernel AST 语义一致。

验收：自动测试对比公共导出、frontend allowlist 和 registry，不允许单边新增。

### M0.4 修复 CLI 与示例可移植性

- Windows 下安全配置 UTF-8 stdout/stderr，或只使用 ASCII 状态符号；
- `python -m tila --help`、`check`、`dump`、`explain` 在 Windows/Linux 均成功；
- 四个 example 脚本退出码必须为 0；
- CLI 捕获并格式化源文件不存在、模块 import 失败、输出目录不可写等错误。

验收：Windows 和 Linux smoke job；不允许 UnicodeEncodeError。

### M0.5 清理过期路线图

- 删除 `src/tila2` 迁移计划；
- 区分 FP8“存储类型闸门”和“完整 GPU 支持”；
- 把 hardware target、TypeVar、effect 的阶段归属统一；
- 将测试策略分成“当前覆盖”和“目标覆盖”。

### M0 退出标准

- 所有文档内部无已知阶段矛盾；
- 所有“当前可用”的语法示例可编译；
- README 不再夸大测试、GPU、FP8、effect 或 alignment 状态；
- Windows/Linux CLI smoke 全绿。

---

## 6. M1：核心语言与内存模型定型

目标：冻结后续 effect、race 和 bounds 都依赖的类型模型。

### M1.1 拆分 Region 身份与 Extent

ADR-001/002 已接受，公共类型与内部身份模型分别为：

```text
Ptr[
  Elem,
  AddressSpace,
  Access,
  Extent,
  Alignment
]

PtrType.region_id: ParamRegion | BufferRegion | InternalRegion | UnknownRegion
```

- `RegionId`：编译器生成，供 effect/race 使用，不是公共类型参数；
- `Extent`：bounds proof 使用的元素数量或受限 shape；
- 同长度的两个指针不应自动成为同一区域；
- 同一区域的派生指针继承 RegionId，offset 只改变可访问范围表达式；
- Buffer 参数拥有稳定 RegionId，并从 shape/strides 推导地址域；不同外部参数仍默认 `MayAlias`。

短期若不支持完整多维 Ptr extent，必须明确限制为 contiguous 1D，而不是用一个字段同时表达两个概念。

验收：两个同长度不同 Buffer 的 effect region 不同；两个同 Region 派生指针相同；bounds 只消费 Extent，不消费 RegionId。

### M1.2 冻结 Buffer/Ptr 公共参数顺序

ADR-001/003 已决定：Ptr 完整形式显式携带 AddressSpace，Buffer v0 公共形式为
`Buffer[T, Shape, Access, Alignment]`，隐式且仅支持 Global；Buffer stride 不进入
公共参数，由 launch 以元素为单位绑定。

- `buf.ptr` 仅允许 rank-1、stride-1，继承 BufferRegion、Extent 与能力；
- 多维或非连续 Buffer 使用坐标访问，不做隐式 flatten；
- Shared/Local 构造在真正实现分配/生命周期前明确拒绝；
- `p - offset` 已从 v0 规范删除；未来开放前另行定义负 offset 与 bounds 规范化；
- `byte_offset` 在实现前保持明确的 Deferred 诊断。

### M1.3 类型表示去除字符串协议

Access/AddressSpace 已从字符串协议迁移为不可伪造的 enum/数据类型：

- `Access.ReadOnly/WriteOnly/ReadWrite`；
- `AddressSpace.Global/Shared/Local`；
- `RegionId`、`Extent`、`Alignment` 使用独立类型；
- Buffer stride 绑定显式区分 `BoundStrides` 与 `UnboundStrides`；
- 类型相等、打印和序列化集中实现。

验收：不存在任意字符串进入 PtrT/BufferT 后在深层 `_access_name` 才失败的路径。

### M1.4 Const 与 refinement 语义闭环

ADR-004 已决定带参数 refinement 只使用下标语法；ADR-005 已决定 0.2.x
唯一可声明的编译期参数域是 `Const[int]`，且：

- `Range[lo, hi]` 为闭区间，端点为整数字面量且 `lo <= hi`；
- `MultipleOf[k]` 的 `k` 为正整数；
- `Aligned[k]` 为正的 2 次幂，单位为字节且只用于 memory alignment；
- 明确哪些 refinement 仅在 launch 检查，哪些进入 proof facts；
- `Positive/NonNegative/Range/MultipleOf` 的事实注入不能只做值检查；
- 模块级 bool、由 Const int 比较得到的 staged bool 和 runtime bool 的边界写成测试矩阵；
- `Const[bool]`/`Const[dtype]` 不属于 0.2.x 公共参数语法。

### M1.5 intrinsic registry

按 ADR-006 建立单一注册结构，每项至少包含：

```text
name
surface forms
phase/status
arity/keywords
checker handler
effect constructor
bounds rule
reachable TIR ops
backend expectations
target requirements
```

复杂 intrinsic 仍可使用专用 handler，但名字、参数形态、阶段和实现覆盖由 registry 管理。

验收：新增 intrinsic 时缺少 checker/lowering/interpreter 状态会在测试中失败；文档签名表可从 registry 检查或生成。

### M1.6 诊断契约

- 为现有错误码建立稳定目录；
- 同一错误原因不得在不同路径随机使用不同 code；
- 每个错误至少包含 location、found/expected、关键事实、一个可操作修复建议；
- 区分语法错误、类型错误、launch contract、target/backend 错误；
- 禁止捕获用户错误后暴露内部 Python traceback，除非 `--debug`。

### M1 退出标准

- Region/Extent 不再混用；
- Ptr/Buffer/refinement 语法和内部类型一致；
- intrinsic registry 成为单一事实来源；
- 所有公开类型构造有正反测试与稳定诊断。

---

## 7. M2：CPU 正确性与静态证明闭环

目标：在不依赖 GPU 的环境中，把语言语义、proof 和 interpreter 做成高可信基线。

### M2.1 整数语义与统一证明接口（先行）

先完成 ADR-007：固定溢出、负数除法/取模、除零、移位、cast 和 shape/grid
到索引类型转换的语义，再统一 checker、interpreter 与 lowering。数学整数
与有限位宽整数分别建模；只有证明中间运算不溢出后才能使用数学整数推理。
显式 i64 不能替代范围证明，未定义或未覆盖语义返回 Unknown。

按 [ADR-011](docs/adr/011-smt-proof-and-trust.md) 建立统一 predicate DAG、
obligation、ProofResult 和 provenance 接口。结论为 ProvenSafe / ProvenUnsafe /
Unknown / Exempted，独立记录静态事实、已检查 launch 契约和用户 assume 依赖。
SafeUnderContract 保留显示兼容；unsafe 不再伪装成 ProvenSafe。

要求：

- 每个安全结论输出可复核依据与信任来源，不承诺最小证明；
- Unknown 区分缺事实、不支持、超时及资源耗尽；
- ProvenUnsafe 需要精确模型或已确认可达的反例；保守近似下未确认的 sat 为 Unknown；
- assume 的派生事实/hint 继承依赖，unsafe 只局部豁免；
- 路径不可达、矛盾用户假设和无条件安全在报告中明确区分。

### M2.2 Z3 作为默认通用引擎

- 定义独立 solver protocol，Z3 为默认实现；切换时纳入标准依赖及锁文件，
  缺失时报告配置错误，不静默回落到较弱的默认验证；
- 保留 And/Or/Not 及共享子表达式，不强制 DNF 展开；正确保留 lane/broadcast
  关系、路径条件和符号身份；仅保留常量折叠/直接事实匹配等小型快速路径；
- 查询 `facts ∧ path ∧ mask ∧ ¬in_bounds`；Const/shape 使用 Int，kernel
  有限位宽运算按 ADR-007 编码；需要绑定的 Const/launch 事实延迟到 Stage 2；
- 设置单查询 timeout/rlimit、kernel 累计预算和公式构建大小限制；默认数值以
  基准确定，超预算返回 Unknown，不裁剪析取分支后宣称安全；
- 缓存包含义务、事实及信任来源、path/mask、Const/launch 绑定、整数编码版本、
  求解器版本/配置；命中仍重验 launch 契约，Unknown 不跨更高预算永久复用；
- 测试期双跑新旧路径并审计差异，尤其“旧拒绝、新判安全”；验证后切换默认
  并退役复杂手写 DNF 逻辑，避免长期双实现。

验收：覆盖 unsat、可达 sat、近似伪反例、unknown、timeout、总预算耗尽、
布尔组合压力、整数边界/中间溢出及缓存隔离；所有新安全结论可追溯到执行语义。

### M2.3 控制流数据流分析

- 完成 runtime-if/static-if 的定义赋值和分支合并矩阵；
- 明确 loop-carried 变量、不变量和循环局部；
- 支持或明确拒绝 `+=`，不保留“文档说 v1、checker表现不确定”的状态；
- return 后不可达代码处理一致；
- facts 在分支合并时取 sound intersection，循环使用保守不动点。

### M2.4 Interpreter 作为规范 oracle

- 为所有 Implemented intrinsic 建立 interpreter 路径；
- 明确整数溢出、除法、NaN、bf16/FP8、归约精度语义；
- 对 non-contiguous NumPy/torch CPU tensor、stride、view 增加覆盖；
- debug 模式执行 `assume` 断言和 bounds 检查；
- interpreter 不得默默使用与 Triton 不同的广播或 dtype promotion。

### M2.5 Property-based 与变形测试

增加以下自动生成测试：

- dtype conversion matrix；
- shape/broadcast 对称性和结合场景；
- DimExpr canon/equality；
- 布尔 DAG/SMT implication、否定、共享表达式与预算压力；
- grid/cdiv/尾块；
- reshape numel；
- interpreter 与 NumPy reference；
- `unsafe` 只豁免局部 obligation，不泄漏事实。

可选采用 Hypothesis；随机测试必须固定 seed 并在失败时输出最小复现源码。

### M2.6 Explain 与审计稳定化

`tila explain` 输出稳定章节：

- 参数和推导类型；
- facts 及来源；
- effects；
- obligations；
- 每个 obligation 的 verdict、信任依赖、求解路径和候选反例/Unknown 原因；
- unsafe/assume/launch-contract 汇总；
- 将要发射的 optimization hints 及依据。

为 explain 建立 golden，避免“证明结果正确但解释漂移或缺事实”。

### M2.7 独立的公共接口设计

2026-09-19 以下独立 ADR 均 Accepted 且实现完成。

| 子项 | ADR 与推荐方向 | 后续门禁 |
|---|---|---|
| M2-07a | [ADR-012](docs/adr/012-boolean-tile-mask.md)：布尔 tile 消费者适配，保持 Mask 独立 | DONE；39 项专项、68 组 CPU/GPU 对照；身份/广播、审计 golden 对齐 |
| M2-07b | [ADR-013](docs/adr/013-const-bool-domain.md)：exact bool、独立参数域及类型标签键 | DONE；自 0.3.0.dev0 生效，保留 0.2.x 历史边界；47 项专项、11 组 CPU/GPU 对照 |
| M2-07c | [ADR-014](docs/adr/014-host-integer-normalization.md)：显式 host_int 白名单转换 | DONE；39 项专项与可执行文档示例；类型身份白名单、无损值、使用点门禁、缓存等价 |
| M2-07d | [ADR-015](docs/adr/015-rounded-typed-constants.md)：显式 RNE 浮点常量 | DONE；39 项专项、两份 golden、46 组 CPU/GPU 按位对照；f16/bf16/f32/f64 |

实施顺序建议 a → b；c/d 独立排期，不阻塞 a，也不因 ADR 成文而标已实现。

- 在 SMT 和信任模型稳定后，设计并实现布尔 tile 的 `mask=`/组合支持；
  比较携带边界谓词，加载的布尔值通常不给边界事实；保持 shape 检查与
  scalar-if 限制，不直接合并整个 Mask/Block bool 类型体系。
- 优先评审 Const bool 的独立参数域及带类型标签缓存键；开放前修订 ADR-005
  的版本边界，当前 0.2.x ExactInt 承诺保持有效。
- 较低优先级评估 NumPy integer 白名单、范围检查与规范化，拒绝任意 `__int__`。
- 保持严格浮点字面量规则，先设计明确 dtype/舍入的常量构造，再按实际 kernel
  体验评估默认规则；这些便利性设计不阻塞核心 SMT 迁移，也不自动变为已实现。

### M2.8 提前衔接 GPU 语义验证

推进 ADR-009 固定最小 GPU 环境，针对整数/cast/mask/归约增加小型 CPU/GPU
对照；不必等 M3 全部完成才验证。无 runner 时记录未验证与阻塞，不用 CPU
通过替代 GPU 证据；完整 GPU CI、示例 differential 与支持承诺仍归 M3。

### M2 退出标准

- CPU 环境安装 dev extra 后零 skipped；
- 默认 SMT 与小型快速路径遵守同一语义和信任规则；各结论、豁免及预算有系统测试；
- 新旧差异已审计，复杂 DNF 迁移完成；整数边界、假设污染、缓存重验均有反例测试；
- interpreter 覆盖所有 Implemented intrinsic；
- examples、CLI、explain、golden 全绿；
- fuzz/property 测试没有已知 soundness 缺陷。

---

## 8. M3：Triton/CUDA 后端闭环

目标：证明 Tila 不仅“能生成字符串”，而且能在明确支持的 GPU 环境正确运行。

### M3.1 固定支持矩阵

在 CI 和文档中明确：

- Python 版本；
- Triton 版本范围；
- PyTorch 版本范围；
- CUDA toolkit/driver 下限；
- 首个支持的 NVIDIA compute capability；
- 操作和 dtype capability 表。

`pyproject.toml` 的 optional dependency 必须与该矩阵一致，不能使用无限制的 `triton`。

### M3.2 Launch 参数与 device contract

- 校验所有 tensor 在同一 device；
- 校验 CPU/CUDA 混用并给出 TILA-TARGET 诊断；
- 校验 dtype、rank、shape、stride、alignment、storage offset；
- grid 每轴范围、空 grid、超过 Triton 限制的情况显式诊断；
- `num_warps` 作为类型化/受限 launch option，而不是硬编码为 4；
- 特化缓存键包含源码 fingerprint、dtype、Const、target、debug 和影响生成代码的选项。

### M3.3 Lowering 合法性

- 所有 TIR 节点使用穷尽分派，未知节点不得生成 `# ?...` 后继续；
- lowering 前运行 TIR verifier；
- 对 Triton API 版本差异建立兼容层；
- 支持的 bf16 操作、归约和 `exp2` 生成代码在真实 Triton 编译验证；FP8 参数/中间值明确拒绝，完整 FP8 执行归 M5；
- source map 保留 Tila 源位置，后端错误能够映射回用户代码。

### M3.4 CPU/GPU differential

每个核心示例在多组参数下比较 interpreter 与 GPU：

- add：整除/非整除 N，多 dtype；
- matmul：M/N/K 尾块、非连续 stride；输入限 f16，f16/f32 acc 与输出有独立 dot 对照；bf16/f32/FP8 输入明确拒绝，不宣称未实现的精度组合；
- self-attention：causal、不同序列长度；
- fused-attention：多 head、两段 KV、exp2；
- 极值、NaN/Inf、零长度允许范围按规范覆盖。

误差容限按 dtype/算法单独定义，不能全局使用宽松 `allclose`。

### M3.5 Golden 扩展

为每个官方 example 建立：

- canonical TIR golden；
- Triton source golden；
- explain/proof golden；
- CPU output test；
- 可选 GPU differential test。

### M3.6 Alignment 与优化 hint 闭环

- alignment launch 校验成功后生成对应 Triton assumption/hint；
- `multiple_of`、`max_contiguous` 每个发射点记录事实来源；
- hint 不改变解释器语义；
- 加入负测试：无法证明 alignment/multiple-of 时绝不发射；
- 对生成 hint 进行 GPU 编译和结果验证。

### M3 退出标准

2026-09-19 [M3-06 审计](docs/m3-exit-audit.md)结论为 NOT READY：以下条件在
已列举子集有本地证据，后续 dtype 矩阵已补测；按不增加功能的范围明确 FP8 等
拒绝契约。拒绝路径不能充当执行支持证据，M3-01 持续 GPU 验收仍未关闭。

- 支持矩阵中的至少一个真实 CUDA 环境全绿；
- 四个官方示例 CPU/GPU differential 通过；
- 所有 Implemented intrinsic 的 Triton lowering 可编译；
- 混合 device、unsupported target、非法 launch option 有清晰诊断；
- 任何 optimization hint 都有 proof provenance。

---

## 9. M4：Effect、Atomic、Race 与 Uniformity

目标：在 Region/Extent 模型稳定后，逐层启用并发与副作用分析。

### M4.1 Per-instruction effect IR

- 冻结设计：[ADR-016](docs/adr/016-instruction-effect-ir.md)，Accepted；M4-01b/c/d 已实现，含详细输出及缓存隔离验收；
- M4-01 先让每个现有 load/store（含 unsafe）节点携带 effect；atomic 留待 M4.3；
- kernel effect summary 从指令聚合生成，不再由 checker 单独维护平行列表；
- effect 使用 RegionId，不使用 extent 名称；
- mask 与路径分开，访问身份与定义引用稳定；分支/早退、零次循环和循环携带值有
  明确组合规则，未知信息保留 may-effect；未来函数调用组合不在本批范围；
- 实施依次为 ADR 评审、节点/引用/verifier、控制流 summary 迁移、输出与缓存验收。

### M4.2 `where` eager-effect 检查

M4-02 已实现：[ADR-010](docs/adr/010-effect-diagnostic-policy.md) Accepted。
检查只遍历值操作数中的实际读取节点，不追溯已计算值的定义；嵌套 where 按最近
值分支归属，附 site/源位置与互补 mask 改写建议。静态提示不作可达性消除。

- `where` 两侧含内存操作时默认 warning；
- 独立的 `--effects {off,warn,error}`，不复用 bounds safety 开关；
- 诊断提供 masked load/store 改写建议；
- 文档明确 eager evaluation，不把它描述成短路分支。

### M4.3 Atomic

[ADR-017](docs/adr/017-minimal-atomic-add.md) 已 Accepted（M4-03a 设计，M4-03b/c 前端/IR/CPU 与限定 GPU 已实现）。
首版限定 Global ReadWrite、i32/u32/f32、标量或一维 tile、Relaxed/GPU；支持
Buffer/Ptr 两种寻址，返回旧值，masked-off 返回同 dtype 零。重复地址逐元素
更新，CPU 固定参考顺序不约束 GPU 调度。浮点采用目标 f32 原子数值规则，
并发验证合法结果/误差包络，不要求逐位等于 CPU。

- 最小集合从 `atomic_add` 开始；
- value dtype 与 pointer element dtype 精确匹配；
- order/scope 使用类型化枚举；
- target capability 在特化期校验；
- interpreter 提供确定性参考模型，GPU 提供真实测试。

### M4.4 Race 分析

[ADR-018](docs/adr/018-minimal-race-analysis.md) 已 Accepted，M4-04b/c 精确查询与启动门禁已实现。
以 Effect IR 派生动态事件对，按实际字节区间及独立 alias 事实判断；SAT 只有在
有效访问共同可达时才确认冲突。首版跨 program 仿射子集，同 program lane 覆盖
单独列明，不能由跨 program 通过声称全局无竞争。同次 store 重复 lane 可确认；
未知同 program 顺序保留 Unknown。独立 off/warn/error 默认 warn，确认冲突始终拒绝（off 不检查）。

分三个等级：

1. 可证明不相交：无诊断；
2. 可证明冲突：error；
3. 无法判定：warning，可显式抑制并进入审计报告。

第一版只分析 affine program-id 地址；不要为了覆盖复杂 pattern 而牺牲 soundness。
兼容 atomic 之间允许同址更新，但 Atomic 与普通访存的冲突不能自动豁免；
atomic 自身合法性始终检查；实施范围与 Unknown 边界见 [Race 分析](docs/race-analysis.md)。

### M4.5 Uniformity

- [ADR-019](docs/adr/019-minimal-uniformity.md) 已 Accepted；M4-05b 内部分析与 verifier 已实现，整体 Partial；
- 首版区分 LaunchUniform/ProgramUniform/Varying/Unknown，逻辑元素不等同物理线程；
- 分离值一致性与控制参与、共同迭代和提前退出；复用 ValueRef、Predicate DAG 和 Effect IR；
- 先提供内部分析与可选审计，未知情况保守，不通过 assume/Race Safe 升级；
- 未来同步消费者同时校验值/控制要求与 target 映射，不能用审计策略关闭硬性门禁；
- barrier/shared memory、CTA/Warp 物理 scope 和优化消费另立契约，不在 M4-05a 内实现。

### M4 退出标准

- effect 是 TIR 原生字段；
- where、atomic、race、uniformity 各自有独立开关和错误码；
- affine race 正/反例及 GPU atomic 测试齐全；
- RegionId 不再与 shape/extent 混淆。

---

## 10. M5：泛型、特化与 Target Capability

目标：在核心语言稳定后加入真正可控的泛型，而不是 Python 级伪泛型。

### M5.1 TypeVar

- 支持 dtype TypeVar，首版不支持 shape-kinded TypeVar；
- bound 可为能力类或显式 dtype 集；
- TypeVar 只在特化期绑定，Stage 1 生成约束；
- 相同 TypeVar 在多个参数中必须一致；
- 诊断展示绑定结果和冲突来源。

### M5.2 Capability 集合

建立真实公共对象：

- Float、Int、UInt、Numeric；
- DotInput、DotAccumulator；
- AtomicTarget；
- StorageFloat。

能力集合不能只存在于文档或私有 tuple 中。

### M5.3 Target database

- 后端、compute capability、Triton 版本共同选择 capability；
- dot/atomic/FP8/shared-memory 等从表中查询；
- `@tila.requires(...)` 只声明额外前提，不绕过实际 target 检查；
- 不支持时使用稳定的 TILA-TARGET 诊断。

### M5.4 特化缓存

缓存键至少包含：

```text
source fingerprint
bound TypeVars/dtypes
Const values
target capability
alignment class
debug/safety-affecting lowering options
```

定义缓存失效规则和可观察统计，增加同一 kernel 多 dtype、多 Const 的隔离测试。

### M5.5 FP8 完整闭环

- 明确支持的 FP8 格式，不写“四种”等模糊表述；
- storage-only 算术限制保持；
- load → cast → compute → cast → store CPU/GPU 对齐；
- FP8 dot 只在 capability 表允许时开放；
- 明确舍入、饱和、NaN 和溢出规则。

### M5 退出标准

- 文档中的 TypeVar 示例真实运行；
- capability 是公共且可测试的语言概念；
- target 不支持的泛型特化在 launch 前拒绝；
- FP8 只在经过真实 GPU 验证的组合上标记 Implemented。

---

## 11. M6：优化事实、性能诊断与 Layout

目标：在正确性不变的前提下，系统利用类型和证明事实改善生成代码。

### M6.1 事实 provenance

每条优化事实记录：

- 事实内容；
- 来源源码位置；
- 推导规则；
- 依赖的 launch contract/refinement；
- 消费该事实的 lowering 点。

`explain` 能展示“为什么发射这个 hint”。

### M6.2 结构性性能诊断

先实现只 warning、不影响类型正确性的检查：

- 明显不连续访问；
- 无法向量化的 stride；
- 可疑 tile 大小；
- 归约形态与 warp 数不匹配；
- 重复加载等局部模式。

每条 warning 必须给出可操作建议，并允许按 code 抑制。

### M6.3 Layout 研究门槛

Layout 进入用户类型语法前必须完成：

- Triton layout/encoding 的版本稳定性评估；
- dot operand layout 的联合约束模型；
- layout 对类型相等、分支合并、reshape/trans 的影响；
- 至少两个真实 kernel 的性能收益数据；
- 不暴露具体 Triton 内部 encoding 的抽象边界。

在这些条件满足前，Layout 只作为 TIR/lowering 内部属性，不进入 `Block[T, S, Layout]` 公共语法。

### M6 退出标准

- optimization hint 有完整 provenance 和负测试；
- 性能诊断永远不伪装成类型错误；
- Layout 是否进入语言经过独立设计评审和 benchmark 证明。

---

## 12. 横向工程工作

### 12.1 CI 矩阵

GitHub 托管 Linux/Python 3.11 CPU 回归与 golden 首个 run 已通过，JUnit 已下载
核实。自动 GPU CI 已撤下，用户目前没有独立 GPU runner，保留本地严格验收。
配置与复现见 [CPU CI](docs/cpu-ci.md)。以下是目标矩阵，不是当前已运行的 job 清单：

| Job | 内容 |
|---|---|
| lint/type | 格式、静态检查、导出/registry 一致性 |
| cpu-py310 | 最低 Python 版本、全量 interpreter/checker |
| cpu-latest | 最新支持 Python、文档示例、CLI |
| windows | UTF-8 CLI、路径、examples smoke |
| gpu | 固定 Triton/PyTorch/CUDA，golden + differential |
| solver | 默认 Z3、Int/BitVec 语义、反例可达性、timeout/资源预算与缓存测试 |

### 12.2 测试分层

- unit：dtype、DimExpr、facts、solver；
- rule matrix：每条语言规则正例/反例；
- diagnostics：code、位置、found/expected、fix；
- golden：TIR、Triton、explain；
- interpreter：数值语义；
- differential：CPU vs GPU；
- regression：每个修复必须留下最小复现；
- fuzz/property：表达式、shape、mask 和控制流组合。

### 12.3 版本与兼容策略

在 0.x 阶段允许语法调整，但必须：

- 每次破坏性修改写入 changelog；
- 文档和代码在同一提交更新；
- 能给迁移提示的旧语法至少保留一个版本的定向诊断；
- TIR dump 格式若用于 golden，不承诺跨 minor 稳定，但变更必须评审。

### 12.4 代码质量

- 为 checker.py 拆分清晰子模块：typing、control-flow、intrinsics、memory；
- 将 proof 与 runtime contract 求值解耦；
- 禁止 unknown TIR 节点静默 lowering；
- 公共 dataclass 使用不可变或受控构造，避免非法类型状态；
- 关键 soundness 函数写明前置条件、不变量和失败语义。

### 12.5 文档质量门禁

- 当前规范、未来设计、实现说明分别标注；
- 示例从真实文件或测试抽取，减少复制漂移；
- README 只做定位、安装、最小示例和状态摘要；
- 完整语义放在规范文档，阶段计划以本文件为准；
- 所有文档链接和章节引用由 CI 检查。

---

## 13. 近期执行批次

Batch A/B 和 C 的核心内存模型已在 M0/M1 完成，以下保留历史范围；C 中的
device 一致性/launch 扩展归 M3。当前从 Batch D 开始，并提前衔接 E 的最小 GPU 验证。

### Batch A：状态与可用性修复

1. 修复 Windows CLI/示例 UTF-8；
2. 让 dev 环境安装后测试零 skipped；
3. 新增 `docs/status.md`；
4. 修正 README 的测试数量、行数和成熟度描述；
5. 删除路线图中过期的 `tila2` 迁移段；
6. 给所有 Markdown 当前示例加 compile smoke。

退出：用户按 README 可以完整安装、运行、检查和理解当前能力。

### Batch B：API/规范对齐

1. 决定并实现 Ptr/Buffer 最终 v0 参数形式；
2. 统一 Range/MultipleOf/Aligned 语法和校验；
3. 补齐 `exp2` 等公共导出；
4. 决定 Const/cast/static_assert 的实际支持范围；
5. TypeVar 未来示例移出当前规范；
6. 建立 intrinsic registry 与一致性测试。

退出：文档中的每个当前 API 都可由 Python 构造，frontend/checker 不存在隐形名字。

### Batch C：内存模型修正

1. 设计评审 RegionId/Extent；
2. 迁移 PtrT/BufferT/TIR/effects/obligations；
3. 修正 `buf.ptr` 和 pointer arithmetic；
4. 增加 alias identity 与 bounds 独立测试；
5. 补 device 一致性和 launch 参数验证。

退出：bounds 和未来 race 分析拥有正确、稳定的数据模型。

### Batch D：证明与解释

1. ADR-007 整数语义和有限位宽边界测试；
2. predicate DAG、ProofResult 与信任来源接口；
3. 默认 Z3、资源预算、缓存和新旧差异审计；
4. explain golden、property-based bounds/shape 测试与小型 GPU 语义对照；
5. 独立推进布尔 tile mask 和 Const/字面量易用性评审。

退出：每个内存访问结论可复核，Unknown 的原因清晰。

### Batch E：GPU 闭环

1. 固定 Triton/PyTorch/CUDA 支持矩阵；
2. 扩充四个 example golden；
3. 建立 GPU CI；
4. CPU/GPU differential；
5. alignment/hint 发射闭环；
6. `num_warps` 与 target capability 基础接口。

退出：Tila 的核心卖点在真实 GPU 上有持续验证，而不仅是源码生成测试。

---

## 14. 暂不做的事项

在 M3 完成前，暂不投入：

- 完整 shared-memory typestate；
- warp specialization；
- 多后端或直接 PTX；
- 任意 dependent types；
- 用户可控 concrete Triton encoding；
- 大量 convenience intrinsics；
- 复杂跨过程 alias analysis；
- 为 benchmark 特例硬编码 checker/lowering。

这些工作会扩大表面积，但不能解决当前最关键的规范一致性、soundness 和 GPU 验证问题。

---

## 15. 风险与应对

| 风险 | 后果 | 应对 |
|---|---|---|
| 文档愿景继续先于实现扩张 | 用户无法判断能力真假 | status 表 + 文档 smoke + README 门禁 |
| Region/Extent 不拆分 | race/alias 模型先天错误 | 在 M4 前完成 M1.1 |
| 只测生成文本、不测 GPU | Triton API 或数值语义错误长期隐藏 | M3 固定 GPU CI + differential |
| checker 单文件继续膨胀 | 规则互相污染、难以验证 | registry + 分模块重构，行为由 golden 锁定 |
| SMT 被当成万能证明器 | 编码错误、编译变慢、伪反例 | 整数语义先行，默认 SMT 配合小型快速路径、总预算、信任来源与可达性审查 |
| 过早开放 Layout/FP8 | target 组合爆炸 | capability 表与真实 GPU 测试作为开放门槛 |
| interpreter 与 GPU 语义漂移 | differential 不可信 | 明确 dtype/归约/溢出语义，逐 intrinsic 对齐 |

---

## 16. 发布门槛建议

### 0.2.x：语言核心校准版

- 完成 M0；
- 修复公开 API/文档硬错误；
- Windows/Linux CLI 可用；
- 不承诺 TypeVar、race、atomic、完整 FP8。

### 0.3.0：内存与证明模型版

- 完成 M1–M2；
- RegionId/Extent 定型；
- 默认 SMT、整数语义、信任来源、预算与 proof trace；
- CPU 测试零 skipped。

### 0.4.0：GPU 验证版

- 完成 M3；
- 固定首个支持矩阵；
- 官方 examples 的 GPU differential 持续通过；
- alignment/hints 可审计。

### 0.5.0：并发语义版

- 完成 M4 的最小闭环；
- effect IR、where 检查、首批 atomic、affine race 分析。

### 0.6.0：泛型与能力版

- 完成 M5；
- TypeVar、target capability、完整特化缓存；
- 经过验证的 bf16/FP8 组合。

---

## 17. 下一步

M0/M1、**M2-01 至 M2-08 已完成**：
ADR-012 至 ADR-015 分别覆盖布尔 tile mask、Const bool、宿主整数转换与显式舍入常量。
M2-08 已整合现有专项并扩展整数/cast/mask/归约的固定环境 CPU/GPU 对照，
统一入口及局限见 [GPU 审计](docs/m2-gpu-audit.md)。下一步推进 M3 的持续 GPU 验收。
Const bool 已自 0.3.0.dev0 开放，不回移至 0.2.x，不放宽 ExactInt。
M2-06 的版本化审计格式与 golden 边界见 [explain 审计](docs/explain-audit.md)。
M2-05 的有限穷举范围、差异台账与重放方式见 [证明审计](docs/m2-proof-audit.md)。
Mask/Const 易用性独立处理。
M2-04 的不变量与可达性采用明确的保守子集，不包含一般递推求解，
具体边界见 [数据流与解释器](docs/dataflow-interpreter.md)。

ADR-009 已固定本地 GPU 环境，M2-08 的有限语义对照已完成。
M3-01 已有固定组合支持矩阵和本地证据；隔离 GPU 持续验收仍待完成，不因本地通过升级为 DONE。

---

## 18. 必须先完成的设计决策

以下决策会影响公共语法或核心 IR，必须在对应实现开始前形成简短 ADR，并同时给出语法示例、类型表示和迁移方案。

| ADR | 状态 | 决策 | 最晚决策点 | 决定/推荐方向 |
|---|---|---|---|---|
| [ADR-001](docs/adr/001-ptr-public-syntax.md) | Accepted | Ptr 公共类型参数 | Batch B | 完整形式为 `Ptr[T, AddressSpace, Access, Extent, Alignment]`；`RegionId` 不对用户开放，常用形式提供无歧义简写 |
| [ADR-002](docs/adr/002-region-id-and-extent.md) | Accepted | RegionId 与 Extent | Batch B | 强制拆分；RegionId 只表示身份，Extent 只服务 bounds，外部参数默认 `MayAlias` |
| [ADR-003](docs/adr/003-buffer-stride-address-space.md) | Accepted | Buffer 的 stride/address-space 语法 | Batch B | 公共形式固定为 `Buffer[T, Shape, Access, Alignment]`；Global 隐式、stride 由 launch 绑定；`buf.ptr` 仅限 rank-1 stride-1 |
| [ADR-004](docs/adr/004-refinement-construction-syntax.md) | Accepted | refinement 构造语法 | Batch B | 带参数 refinement 统一使用下标语法；Range 为闭区间，MultipleOf 为正整数，Aligned 为正的 2 次幂字节对齐 |
| [ADR-005](docs/adr/005-const-type-domain.md) | Accepted | Const 类型域 | Batch B | 0.2.x 只支持 `Const[int]`；比较结果是 staged bool 而非可声明的 `Const[bool]`，bool/dtype Const 延后 |
| [ADR-006](docs/adr/006-intrinsic-registry.md) | Accepted | intrinsic registry 形态 | Batch B | registry 管机器元数据和完整性门禁，复杂规则委托专用 checker，backend 覆盖以 typed TIR 为边界 |
| [ADR-007](docs/adr/007-integer-semantics.md) | Accepted | 整数溢出与除法语义 | M2-01 已完成 | runtime 回绕、floor 商余数、显式转换、定义域及索引门禁；基础 CPU/GPU 对照 |
| ADR-008 | Accepted | reduction 精度 | M2-08 已实现 | 输入、累加、返回 dtype 与 NaN 策略明确建模 |
| ADR-009 | Accepted | 本地 target 验收基线 | M2-08 已验证 | 固定 RTX 3090 + Triton/PyTorch/CUDA；正式支持和 CI 仍归 M3 |
| [ADR-010](docs/adr/010-effect-diagnostic-policy.md) | Accepted | where effect 严格度 | M4-02 已实现 | effects 独立 off/warn/error；race 仍待独立设计 |
| [ADR-011](docs/adr/011-smt-proof-and-trust.md) | Accepted | 默认 SMT 与信任来源 | M2 证明迁移前 | Z3 + 布尔 DAG + 小型快速路径；Int/BitVec 分离、Exempted、预算及反例可达性 |
| [ADR-012](docs/adr/012-boolean-tile-mask.md) | Accepted | 布尔 tile 与 Mask | M2-07a 已完成 | 受限消费者适配，保留类型与未知谓词身份 |
| [ADR-013](docs/adr/013-const-bool-domain.md) | Accepted | Const bool | M2-07b 已完成 | 自 0.3.0.dev0 开放 exact bool、独立参数域、带类型标签的缓存键 |
| [ADR-014](docs/adr/014-host-integer-normalization.md) | Accepted | 宿主整数转换 | M2-07c 已完成 | 显式 host_int 白名单；ExactInt 入口不变 |
| [ADR-015](docs/adr/015-rounded-typed-constants.md) | Accepted | 显式舍入常量 | M2-07d 已完成 | 目标 dtype、直接 RNE、规范位模式；拒绝非有限值及溢出 |
| [ADR-016](docs/adr/016-instruction-effect-ir.md) | Accepted | 逐指令 Effect IR 与派生汇总 | M4-01b/c/d 已实现 | 局部元数据、派生 path/mask/loop、可选详细输出与隔离验收已落地；不涉及 atomic/race/uniformity 实现 |
| [ADR-017](docs/adr/017-minimal-atomic-add.md) | Accepted | 最小 atomic_add | M4-03b/c 完成 | CPU 参考与固定 RTX 3090 GPU 并发验收分别取证，见 docs/atomic-gpu.md |
| [ADR-018](docs/adr/018-minimal-race-analysis.md) | Accepted | 最小 Race 分析 | M4-04b/c 完成 | 精确子集、同次 store lane、启动策略、诊断与缓存隔离；未覆盖顺序/数据流保持 Unknown |
| [ADR-019](docs/adr/019-minimal-uniformity.md) | Accepted | 最小 Uniformity | M4-05b/c 完成 | 值/控制分析与 verifier，可选详细输出/golden/隔离；同步消费未接入，不引入同步 API |
| [ADR-020](docs/adr/020-correctness-closure.md) | Accepted | 正确性收口 | C0-01..04 完成 | 保真符号、实际 ABI 精化、保守 shape 合并、checked i64 指针域及 Safe 规则审计 |

每份 ADR 至少回答：

1. 当前问题和不变量；
2. 候选方案及被拒绝原因；
3. 最终语法与内部表示；
4. soundness/兼容性影响；
5. 正例、反例和诊断；
6. 对现有示例、golden 和缓存的迁移影响。

---

## 19. 任务编号与执行台账

任务编号采用 `<里程碑>-<两位序号>`，例如 `M0-03`。任务状态只允许：

- `TODO`：尚未开始；
- `DOING`：已有负责人和活动分支；
- `BLOCKED`：注明 ADR、环境或上游依赖；
- `REVIEW`：实现完成，等待设计/代码评审；
- `DONE`：验收命令和文档同步均完成。

### 19.1 首轮台账

| ID | 状态 | 工作项 | 依赖 | 验收摘要 |
|---|---|---|---|---|
| M0-01 | DONE | Windows CLI/stdout UTF-8 | 无 | help/check/explain/examples 无编码异常；cp1252 子进程回归已覆盖 |
| M0-02 | DONE | dev 依赖与零 skipped 基线 | 无 | M1 退出历史基线 427 passed；当前基线见 §2.1/status.md，零 skipped |
| M0-03 | DONE | 建立 `docs/status.md` | M0-02 | 类型、语法、intrinsic、proof、后端与未来能力均有唯一状态；公开 API/allowlist 有漂移测试 |
| M0-04 | DONE | README 状态对账 | M0-03 | README 只承诺状态表中的闭环能力；测试数量/行数不硬编码，Partial/Designed 集中列为限制 |
| M0-05 | DONE | roadmap 阶段对账 | M0-03 | roadmap 使用 M0–M6；FP8/effect/target/TypeVar 归属唯一，过期并行迁移方案已移除 |
| M0-06 | DONE | Markdown 当前示例 smoke | M0-03 | Python fence 分为 current/diagnostic/future/generated；current exec 真实通过 Stage 1，所有块有语法 smoke |
| M0-07 | DONE | 公共导出与 frontend 名字检查 | 无 | 公共占位符/frontend/checker 共用名字 registry；`exp2` 已导出，隐形手工 hint 入口已移除 |
| M1-01 | DONE | 实现 Ptr 规范参数语法 | ADR-001/002 已接受 | 完整五参数与紧凑/短别名均规范化；类型打印固定五项；Global v0、Extent、Alignment 与 launch 契约有回归；RegionId 不可由用户伪造 |
| M1-02 | DONE | RegionId/Extent IR 迁移 | ADR-002 已接受 | 四类结构化 RegionId；bounds obligation 仅含 Extent；effect 携带 RegionId；alias relation 独立建模并由 launch 存储区间细化；TIR/explain 分栏输出 |
| M1-03 | DONE | 实现 Buffer 规范语法与 `buf.ptr` 边界 | ADR-003 已接受 | 规范四项打印；公共 spec 不被参数绑定污染；stride/address space 不可伪造；rank-1 在 check、stride-1 在 launch 强制；转置 Buffer 坐标访问回归通过 |
| M1-04 | DONE | refinement 语法统一 | ADR-004 已接受 | 唯一下标构造；参数域/适用位置/明显矛盾检查；规范打印；Range/MultipleOf/PowerOfTwo 与地址 Alignment 事实传播均有回归 |
| M1-05 | DONE | intrinsic registry 完整元数据 | ADR-006 已接受 | 不可变 catalog 派生导出/表面形式；统一 arity/keyword 门禁；显式 checker map；effect/bounds、TIR backend、target、status/docs 与缓存 revision 自动一致性检查 |
| M1-06 | DONE | 诊断契约与 Const 边界收口 | ADR-005/M1-05 | 机器诊断 registry；统一 location/phase/fix 渲染；ExactInt 全入口门禁；deferred static_assert；CLI 非 debug 无 traceback |
| M1-07 | DONE | M1 Exit Audit | M1-01..06 | 四项退出标准逐项通过；审计中移除字符串/`None` 哨兵类型协议并收回 `p - offset` 超前声明；`docs/m1-exit-audit.md` 记录证据与 M2 交接边界 |
| M2-01 | DONE | ADR-007 与整数语义对齐 | M1 完成 | 回绕/floor 商余数/移位/cast/索引门禁与反例；452 项 CPU 回归、23 组本地 GPU 对照；完整 SMT 仍归 M2-03 |
| M2-02 | DONE | DAG 与 ProofResult/信任来源 | ADR-011、M2-01 | 471 项 CPU 回归；共享 And/Or/Not、path/lane/broadcast、不可变 ProofResult、Exempted、来源/作用域隔离及契约重验；19 项专门回归 |
| M2-03 | DONE | 默认 Z3、预算与缓存 | M2-02 | 505 项 CPU 回归；34 项 SMT 专项；固定 Z3 4.16.0.0，Int/BitVec、否定/析取、候选反例、预算/缓存隔离与重放查询；边界见 docs/smt-prover.md |
| M2-04 | DONE | 数据流与 interpreter 收口 | M2-01/02/03 | 537 项 CPU 回归；32 项新专项；活跃分支/整数 phi、简单不变量/零次出口、标量反例重放、非连续/广播/bf16/f64/debug；23 组 GPU 整数对照；保守边界见 docs/dataflow-interpreter.md |
| M2-05 | DONE | 差异审计与性质测试 | M2-03/04 | 604 项 CPU 回归，新增 67 项专项；小位宽合法输入对穷举、SMT 对照/差异台账、固定种子布尔/控制流/广播、预算/缓存/来源隔离、失败见证重放；确认无复杂 DNF 执行入口，详见 docs/m2-proof-audit.md |
| M2-06 | DONE | explain 与审计 golden | M2-03/05 | 628 项 CPU 回归，新增 24 项专项/15 份 golden；audit explain v1、信任来源/反例分类、预算修复建议、缓存与模型附件边界、CLI/错误 SMT 重放及两类 hint 依据；见 docs/explain-audit.md |
| M2-07 | DONE | Mask/Const/常量接口独立设计 | M2 核心模型、ADR-005 版本评审 | a/b/c/d 均完成；a 为 39 项专项/68 组 GPU 对照，b 为 47 项专项/11 组 GPU 对照，c 为 39 项宿主转换专项，d 为 39 项专项/两份 golden/46 组 GPU 按位对照；ADR-012 至 015 与状态表同步 |
| M2-08 | DONE | 小型 CPU/GPU 语义对照 | M2-01、ADR-009 本地基线 | 统一 runner、214 案例、失败重放；ADR-008 归约契约；不替代持续 CI |
| M3-01 | IN_PROGRESS | GPU 支持矩阵 | ADR-009 | 固定依赖锁、本地 GPU 验收及重放保留；自动 GPU workflow/机器注册已撤下；M3-06 新增托管 CPU CI 配置不替代隔离 GPU 持续验收 |
| M3-02 | DONE | launch/target 检查与缓存收口 | M3-01 | grid/零启动、集中 capability、结构与静态 target verifier、源码/ABI/布局缓存隔离、hint/alignment 负测试；本地 CPU 877/GPU 112 节点通过，边界见 docs/launch-target.md |
| M3-03 | DONE | 编译后资源诊断与 source map | M3-02 | 语句映射、稳定编译/加载诊断、附件编译重放、shared-memory/线程硬门禁；寄存器/spill 仅作性能信息；3 份 golden 与真实 GPU 负测试；范围见 docs/backend-diagnostics.md |
| M3-04 | DONE | 操作/dtype 支持矩阵与编译覆盖审计 | M3-03 | 22 项 intrinsic 证据索引、36 项 GPU 新案例、窄 exp/exp2 修复、FP8 target 门禁；五个示例 15 份 golden；CPU 910/GPU 151 节点通过；见 docs/gpu-operation-audit.md |
| M3-05 | DONE | alignment 契约到 hint 的发射闭环 | M3-04 | 一维 stride-1 Buffer/Ptr 的基地址字节提示，独立 checked 来源、explain golden、缓存隔离与逐次契约校验；CPU 926/GPU 160 节点通过；范围见 docs/alignment-hints.md |
| M3-06 | DONE | M3 退出审计及矩阵补测 | M3-01..05 现有成果 | docs/m3-exit-audit.md / m3-matrix-followup.md：NOT READY；托管 CPU 首个 run 通过，补测后本地 CPU 937/GPU 300 节点通过；dot f16 修复，zeros/reshape/add dtype 证据补齐；隔离 GPU 持续验收仍缺失 |
| M4-01a | DONE | 逐指令 Effect IR 设计评审 | ADR-002/006/007/011 | ADR-016 已评审冻结为 Accepted |
| M4-01b | DONE | 节点 effect、定义引用与 verifier | ADR-016 已接受 | 局部不可变元数据、词法定义点引用、穷尽遍历与独立重算校验；16 项专项，CPU 953/GPU 300 节点通过；边界见 docs/effect-ir.md |
| M4-01c | DONE | 控制流上下文与 TIR 派生汇总 | M4-01b | mask/path/loop 分栏，早退/Const/零次循环保守；移除 checker 平行列表；19 项专项，CPU 972/GPU 300 节点通过；fused-attention explain 删除默认特化中不可达的两条 Read |
| M4-01d | DONE | effect 输出与缓存迁移验收 | M4-01c | `--show-effects` / `tila.effect-details.v1`，12 项专项；修复失败/空启动残留 alias/alignment；CPU 984、GPU 300 节点/444 案例通过；不含 atomic/race/uniformity |
| M4-02 | DONE | where 急切求值 effect 检查 | M4-01、ADR-010 | 独立 off/warn/error、已验证 TIR 的嵌套读取与定义复用区分、最近值分支归属、互补 mask 修复建议；18 项专项与诊断 golden；CPU 1002、GPU 300 节点/444 案例通过 |
| M4-03a | DONE | 最小 Atomic 设计 ADR | M4-01/02 | ADR-017 Accepted；固定签名、旧值/mask/重复地址、dtype/order/scope、AtomicInfo、explain v2 迁移及 CPU/GPU 验收契约；无运行时实现 |
| M4-03b | DONE | Atomic 前端、Effect IR 与 CPU 参考 | ADR-017 | enum/签名、TAtomicAdd/AtomicInfo、verifier/bounds/summary/where、explain v2、36 项 CPU 专项及 golden；GPU 路径以 TARGET-012 明确拒绝；见 docs/atomic-cpu.md |
| M4-03c | DONE | Atomic GPU lowering 与严格验收 | M4-03b | 固定 RTX 3090、88 项 atomic 专项；重复地址/旧值、f32 合法历史/FTZ/误差界、短路/嵌套求值、门禁与缓存、source golden 和重放；CPU 1040、GPU 388 节点/532 案例通过；不等于完成 M3 持续 GPU CI |
| M4-04a | DONE | 最小 Race 分析设计 ADR | M4-01..03 | ADR-018 Proposed；明确跨 program/同 program 覆盖、字节重叠、alias/路径共同可达性、策略/预算及反例；不含运行时实现 |
| M4-04b | DONE | Race 精确子集与成对查询 | ADR-018 Accepted | 内部 analyze_races；字节地址/动态身份、跨 program/固定循环、共同可达 witness、预算与 Unknown；39 项专项、CPU 1079；无缓存，不接 launch 策略；见 docs/race-analysis.md |
| M4-04c | DONE | Race 诊断与绑定隔离 | M4-04b | 同次 store lane、off/warn/error、RACE-001..003、race-details.v1、绑定/缓存隔离及3份 golden；新增31 CPU/7 GPU专项，CPU1110、GPU395/539通过；详见 docs/race-launch-audit.md |
| M4-04d | DONE | Race 差异审计 | M4-04b/c | 新增18项专项、747个不同输入/819次分析；修复布尔 mask 取反编码，故障注入/单输入重放；CPU1128、GPU395/539通过；docs/race-exit-audit.md 限定覆盖收口，不代替 uniformity 或持续 GPU CI |
| M4-05a | DONE | 最小 Uniformity 设计 ADR | M4-01/04、ADR-019 Proposed | 四层级、值/控制分离、定义边复用、传播规则与正反例、消费/诊断边界；仅文档，无运行时功能 |
| M4-05b | DONE | Uniformity 评审冻结与内部分析 | ADR-019 Accepted | 内部值/控制摘要、定义入边、固定点/预算、重算 verifier；43 项专项、CPU1170通过；见 docs/uniformity-analysis.md；无同步 API/公共策略 |
| M4-05c | DONE | Uniformity 详细输出与隔离 | M4-05b | show-uniformity、details.v1、3份golden、21项专项、CPU1191通过；Const/预算/历史launch/缓存隔离；逻辑消费fixture不等于同步API；见 docs/uniformity-audit.md |
| M4-05d | TODO | Uniformity 小域与退出审计（暂缓） | C0 评审、M4-05b/c | 枚举/变形、故障注入；区分内部分析与真实 target 消费证据，不代替整个 M4 或 M3 验收 |
| C0-01 | DONE | 错误 Safe 与入口精化 | 已确认反例 | 保真符号规范键；实际 ABI 舍入后检查精化；保留 numpy.float64 宿主兼容 |
| C0-02 | DONE | 静态类型与地址语义 | ADR-020 Accepted | 保守 shape 合并、嵌套 variant 传播、checked i64 指针位移及每步启动门禁 |
| C0-03 | DONE | Safe 规则与跨层审计 | C0-01/02 | 8 类出口清单；715 表达式/57,915 次求值、450 组直接/区间枚举、独立 SMT 蕴含、缓存隔离 |
| C0-04 | DONE | 限定收口验收 | C0-03 | CPU1217、GPU402节点/546案例全部通过且零跳过；新增 explain golden；详见 docs/correctness-closure.md，不代表完整编译器形式化证明 |
| C1 | DONE | 独立边界复核与修复 | C0-01..04 | WriteOnly/Mask cast 双层拒绝、合法 cast 保形、Const return lowering；runtime/loop return GPU 门禁；CPU1264、GPU417节点/561案例通过，见 docs/c1-correctness-review.md |

后续每完成一个 Batch，就在此台账追加下一批工作，不提前维护数百个可能变化的微任务。

### 19.2 单任务完成定义

每个任务标记 `DONE` 前必须满足：

- 实现和设计决定一致；
- 新增或修改的语义有正例、反例、诊断测试；
- 受影响的 interpreter/lowering 路径已覆盖，或明确标记不适用；
- README、规范、status、roadmap 中的相关陈述同步；
- 不引入新的 skipped/xfail；
- `pytest -q` 通过；
- 若影响生成代码，相关 TIR/Triton/explain golden 已评审；
- 若影响 GPU，固定支持矩阵上的 differential 已通过；
- 变更没有顺手扩大到无关语言功能。

---

## 20. 首个迭代的具体拆分

建议首个迭代由三个独立、可审查的变更集组成。

### 变更集 1：可运行性基线

范围：`M0-01`、`M0-02`。

- 修复 Windows 文本输出；
- 检查 console 编码不可修改时的 fallback；
- 把 `ml_dtypes` 纳入实际使用的 dev 安装流程；
- 固化基线测试命令；
- 对四个 examples 增加 subprocess exit-code smoke。

不包含：任何类型规则、Ptr/Buffer API 或文档语义修改。

验收命令：

```bash
pytest -q
python -m tila --help
python -m tila check examples/add_kernel.py
python -m tila explain examples/add_kernel.py
python examples/add_kernel.py
python examples/matmul.py
python examples/self_attention.py
python examples/fused_attention.py
```

### 变更集 2：状态事实来源

范围：`M0-03`、`M0-04`、`M0-05`。

- 建立能力状态表；
- README 只引用状态表中的 Implemented 项；
- roadmap 只表达顺序与验收，不重复虚构当前状态；
- 明确 FP8 storage gate、effect partial check、target capability 的真实阶段；
- 记录当前 GPU 测试缺口。

不包含：为了让文档描述成立而临时补功能。发现缺失时将能力降级为 Partial/Designed。

### 变更集 3：公共 API 审计门禁

范围：`M0-06`、`M0-07`。

- 枚举全部 `tila.*` 当前公共名字；
- 比对文档、`__all__`、frontend allowlist、checker handler；
- 为当前规范代码块建立 smoke test；
- 未来语法代码块显式标为不可执行设计示例；
- 先补无争议缺口（如 `exp2` placeholder），有兼容影响的语法留给 ADR。

### 首个迭代退出评审问题

1. 新用户是否能只看 README 成功跑通 CPU 路径？
2. status 表是否能回答任一内建是否可用以及在哪条后端可用？
3. 文档中是否还存在未标记的未来语法？
4. 当前 API 名字是否存在 frontend-only 或 Python-only 的分裂？
5. 测试是否在标准 dev 安装下零 skipped？

全部回答“是/没有”后，才进入 ADR-001 至 ADR-006 和 Batch B。
