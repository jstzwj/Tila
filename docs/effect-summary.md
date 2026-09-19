# M4-01c：控制流上下文与派生 Effect 汇总

日期：2026-09-20；前置为 [ADR-016](adr/016-instruction-effect-ir.md) 和
[M4-01b 节点模型](effect-ir.md)。本阶段不新增 atomic、race、uniformity 或 solver 查询。

## 单一来源

`TKernel.effect_summary(consts=None)` 从经过元数据验证的 TIR 重新计算；checker 的
四处 effect append 和可变 kernel effects 字段已移除。`TKernel.effects` 是只读
tuple 兼容视图，保留可能发生的 Read/Write 及原 RegionId，保留不同访问点的重复项。
它不是调用次数统计，也不证明必然访问或无竞争。

每份 `EffectSummary.accesses` 保存独立的 path、mask、LoopContext 序列及访问
节点/不可变 MemoryEffect；死路径或 false mask 的记录仍保留供审计，只有粗粒度
effects 投影排除这些访问。源指令始终保有 effect 元数据，不能因不可达而删除元数据。
详细记录采用结构遍历次序，不是 GPU 的动态事件轨迹，也不能用于优化重排。

路径是 program 到达条件；mask 是访问自身的 lane 选择，保留 tile shape。二者
不混成一个谓词，不在此阶段推断逐 lane 的路径/地址不相交关系。循环上下文单独
保存结构身份、induction 定义引用、起终点/步长操作数及可能进入的条件。

## 控制流与保守边界

- 分支先记录条件表达式的读取，再派生正/反路径；后续路径只合并能够继续的前驱。
  return 只结束当前 program；两侧都 return 时，后续访问为死路径。
- literal/Const 能确定条件时，只计入存活分支。默认不代入 Const 默认值：无实参
  为 symbolic，部分绑定为 partial，完整绑定为 specialized。调用者只传 Const，
  不接收运行时 scalar/launch alias/用户假设作为特化事实。
- 支持字面量/Const 值、简单已知值传播、布尔组合和受限 staged 运算；不把运行期
  固定位宽算术当 Python 无界整数折叠。单表达式求值预算耗尽或操作不支持即未知，
  保留可能访问。无新增 SMT 查询。
- 未编码的条件使用共享 Predicate DAG 中的 opaque unknown，关联使用处定义
  身份与循环作用域。不同读取、重绑定或循环出口不会因变量同名而合并。
- 循环界限的读取在进入循环前记录。已知零次循环不计入体内效果；未知次数保留
  零次路径和体内可能效果，不展开循环或推测固定执行次数。
- 循环携带值在体入口保守失去具体值；循环之后不泄漏某次迭代的条件。非零且
  每次必定 return 的循环可以排除后续访问；条件 return 的多次迭代可达性不能
  精确表示时，出口保留独立的 unknown continuation。
- where 不短路，两个操作数的读取都记录。访问的 false mask 不抹掉求值其地址、
  mask、other 或 store value 所需的嵌套读取。
- assume 不是分支，矛盾的 assume 也不删除默认汇总中的访问；unsafe 只豁免 bounds。
  当前 frontend 拒绝字面量 assume(False)，测试同时覆盖合法的矛盾符号假设，以及
  转换后含 false TAssume 的内部 TIR。

汇总没有缓存，也不修改原 TIR、旧访问身份或 launch 快照。每次计算重新验证元数据；
IR 改写后需显式重新绑定，否则拒绝陈旧元数据。失败/空启动和交替 Const 绑定不影响
之后的静态汇总。[M4-01d](effect-audit.md)现已实现可选详细格式与 cache/overlay 验收。

## 输出与验证

dump/旧 check 报告使用符号投影；explain 使用已解析、已验证的 Const 绑定投影。
现有 effects 章节与 Read/Write 文本格式不变，仅内容按可达性修正。fused-attention
默认 CAUSAL=1，因此其 explain golden 删除非 causal 分支的两条 K/V Read；这是
明确审核的唯一旧 golden 变化，TIR 和 Triton source golden 未改写。

新增 19 项控制流专项，覆盖早退、Const 切换、零次/未知/嵌套循环、循环界限读取、
携带值/重绑定、where/other 急切读取、assume/unsafe、不变的静态汇总与元数据拒绝。
完整验收结果见 [状态表](status.md)。测试中的可执行简单控制流与 CPU 实际输出对照；
GPU 使用既有严格 300 节点/444 案例，未扩张 target 支持范围。

本次完整本地验收：CPU **972 项全部通过**，严格 GPU **300 节点全部通过**、
覆盖 444 案例，均零跳过。CPU JUnit 为 `artifacts/cpu/m4-01c-results.xml`；GPU
accepted 报告为 `artifacts/ci-gpu/20260919T162626Z-42y4bghv/report.json`，原始附件
仅本地保存，不作为公开机器环境附件上传。

前置提交 f4add24 的 [托管 CPU CI](https://github.com/jstzwj/Tila/actions/runs/35454545217)
已通过 953 项、零跳过，JUnit 附件已下载核实；该记录不替代本次工作区的远端验收。
M3 仍缺少独立 GPU 持续验收，本轮不会重新接入开发者机器。
