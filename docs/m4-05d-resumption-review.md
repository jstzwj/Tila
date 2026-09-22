# M4-05d 恢复评审

日期：2026-09-22。对象：C0/C1/C2 正确性收口后，是否恢复 Uniformity 小域与退出审计。
本次交付是恢复评审与实施边界，不是 M4-05d 的实现或退出验收。

## 结论与前提

评审结论：**通过，可以恢复 M4-05d 内部分析审计**。C2 提交 `7edd591` 的
[托管 CPU CI](https://github.com/jstzwj/Tila/actions/runs/35686086849)已通过 1412 项，
零失败/错误/跳过；JUnit、golden、完整 C2 与 Race 摘要已下载核对。
运行于 GitHub Actions 托管 `ubuntu-24.04`，不连接开发者机器。
计划标为可启动，本轮不恢复其他功能扩展。
Uniformity 仍为 Partial，M4-05d 尚未实施，M3 的隔离 GPU CI 缺口仍独立存在。

依据是 [C2 限定退出](c2-correctness-exit-audit.md)、
[ADR-020](adr/020-correctness-closure.md) 的兼容性收紧，以及已经 Accepted 的
[ADR-019](adr/019-minimal-uniformity.md)。本次重新核对了 `uniformity.py` 的
transfer、定义边、固定点、控制摘要与 verifier，以及 `uniformity_audit.py` 的
逻辑 requirement evaluator；已有 64 项专项继续纳入全量 CPU 回归。
这不是对分析器正确性的独立证明；新阶段正是补足这一证据缺口。

| 评审项 | 现状与恢复约束 |
| --- | --- |
| 错误 Safe、ABI 精化、分支类型、指针地址域 | C0/C1 已修复并收紧契约，C2 在有限组合内复验；新 oracle 必须沿用机器整数与 checked i64 地址语义 |
| Uniformity 的前提 | 只对合法且已定义的执行作保证；不得把 L/P、Full 或逻辑 Satisfied 当成 bounds、数值或 Race Safe |
| 信任与隔离 | 仅已验证 TIR 与本次 Const；不采样内存，不消费 assume/Race Safe，不复用历史绑定，无分析缓存 |
| 输出与消费 | 值与控制分开；Varying 不等于已确认分歧；Unknown 不允许满足硬要求；真实消费者仍为 not-installed |
| GPU 支持收紧 | runtime if/loop 内 return 仍被 GPU verifier 拒绝；可以审计 CPU 语义，但不能记录成 GPU 支持正例 |
| 剩余证据缺口 | 缺独立小域观察、变形与分析器规则故障注入；这是 M4-05d 的任务，不在本评审中标记完成 |

## 为什么仍需独立观察

`verify_uniformity` 重算分析结果后做结构比较，可以拒绝缺失、伪造、过期摘要，
但它与生成摘要的分析器共享规则。如果 transfer 本身错误，重算也可能得到同一个
错误保证。新 oracle 必须基于具体执行的值和到达事件检查保证，不能再次调用
transfer、join 或 `_control` 来生成“预期正确答案”。复用 TIR 身份和执行入口可以，
独立参考运算与观察断言必须能发现其中的偏差。

## 后续实施顺序与退出要求

1. **先建立小域执行观察。** 固定种子，枚举少量 program、逻辑 tile 元素、
   Const/宿主标量和零/非零循环边界。每个观察以定义 site、program、逻辑元素和
   嵌套循环迭代序号标识动态实例；按实例比较，不能跨迭代混合值。
   L 检查所有实际求值者的相同表示，P 检查同 program 内的相同表示。
   对控制单独记录到达、退出及参与者集合；Full 采用 ADR-019 的 scope 与条件语义，
   不要求所有程序点必然可达，也不能仅从已经到达的样本推断完整参与。
   不可达样本不能冒充执行成功；V/U 不要求构造实际不同值的反例。
2. **再扩展变形与控制流组合。** 覆盖布尔等价、机器整数回绕、广播/reshape、
   完整与部分归约、where 急切求值、重绑定、selector 合并、Const 特化、零次循环、
   携带值和提前 return。先限定无内存依赖的可执行小域；load/atomic、浮点运算、
   未支持控制等边界核验应保留 Unknown。变形必须保持类型和已定义行为，不能
   用实数恒等式要求机器运算等价，也不要求等价程序得到完全相同的保守层级。
3. **注入错误保证并检验隔离。** 故意将 P/V/U 提升成 L/P，遗漏 merge selector，
   错误恢复 Full 或忽略零次循环，要求独立观察能识别反例。另测缺失/伪造摘要、
   预算未收敛、Const Bool/Int 域、跨绑定与修改 TIR 后的旧摘要。保留合法 L/P 与
   完整参与正例，防止全改 Unknown 的实现通过验收。预算/缺失/未建模物理 scope
   不得获得 Satisfied；不可达消费为 NotApplicable。
4. **最后形成可复现退出记录。** 保存种子、源码/TIR、绑定、预算、动态实例、
   观察值/参与者与被违反的事实，提供单案例重放；明确是否做了最小化，不能虚称。
   输出枚举域、实际观察数、L/P 与控制保证覆盖、Unknown 和不适用数，运行全部
   回归与 golden，并取得对应提交的托管 CPU CI 与附件后再判定限定退出。

发现新的错误强保证时，先修复并补反例再继续退出审计，不通过提高预算、更新
golden 或将运行失败一概归为 Unknown 来隐藏问题。有限枚举未发现反例不等于
全程序证明；oracle 与 CPU 路径的共享部分也须在退出记录中列明。

## 明确保留的边界

本阶段不增加 barrier/shared memory、同步策略 env、活动 UNIFORM 诊断、优化消费
或 Warp/CTA 物理映射；内部逻辑消费者 fixture 不构成 GPU 同步验收。
不放宽 C0/C1 的类型/权限/return 限制，不要求消除 bounds 或 uniformity 的 Unknown。
已有 RTX 3090 数值、atomic、Race 证据不替代 uniformity 同步证据；即使 M4-05d
限定退出，整个 M4 及缺少隔离 GPU 持续验收的 M3 仍不能宣称完成。
