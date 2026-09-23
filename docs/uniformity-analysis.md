# M4-05b：内部值与控制一致性分析

日期：2026-09-20。契约 [ADR-019](adr/019-minimal-uniformity.md) 已评审冻结为
Accepted。本页保留 M4-05b 内部派生分析与 verifier 的记录。后续
[M4-05c](uniformity-audit.md) 已接入可选 explain/CLI 详细输出；当前能力仍为
Partial，未接入 launch 或同步消费者，不增加 barrier/shared memory API。

## 入口与事实

`src/tila/uniformity.py` 提供内部 `analyze_uniformity(kernel, consts=None, *, config=...)`
与 `verify_uniformity(kernel, summary, consts=None, *, config=...)`。
调用方传入已检查的 TIR；入口再运行结构与 Effect IR verifier，拒绝未知节点、
损坏 effect 和错误定义引用。Const 按精确 bool/int 域及 refinement 验证；不接收
tensor、宿主运行时值、Race/Bounds 证明结果或历史 launch overlay。

结果含不可变 facts 元组、原 EffectSummary 的 AccessContext、出口控制、特化阶段、
Const 和预算。所有事实为静态语义来源；`requires_defined_execution=True` 明确表示
只描述合法且已定义执行的一致性，不能取代数值/边界检查。
AccessContext/LoopContext 复用原 TIR 节点；不得修改摘要或在修改 TIR 后直接消费旧摘要。
`verify_uniformity` 重新派生并逐字段检查，拒绝缺失/伪造层级、入边、scope 和绑定。

层级为 LaunchUniform、ProgramUniform、Varying、Unknown。Varying 表示没有
program 内一致保证，不等于已经发现不同值。load/atomic 旧值和 dot 初始 Unknown；
未建立目标位模式契约的浮点逐元素运算和 cast 也 Unknown。浮点参数/typed constant/
复制仍可有一致性事实。整数纯运算按依赖传播，不做不可靠的代数消元。
部分轴浮点归约为 Unknown；完整归约只按逻辑单一结果契约派生 ProgramUniform，
不推导目标线程的共同参与。

## 定义图与控制

复用 ValueRef、`effect_ir.OPERANDS`、ConditionSource、Predicate DAG、LoopContext
和 AccessContext；没有第二套表达式 AST。每个参数、赋值、merge、循环携带与归纳
定义有独立身份。TName 通过定义点引用连边，重复使用已读值不增加访问事件。
派生的定义边图描述现有结构化 TIR 的入边，不假设当前 TIR 是完整 SSA。
每种已有节点必须在 transfer 表登记；表与 Effect IR 漂移会明确失败。

值层级和控制上下文分别输出。控制含 selection_level、launch/program participation、
reachability、path、selectors、loops、returned 和 iteration_sources；访问 mask
仍在原 AccessContext 中独立保留。Full 表示对应 scope 在执行此点时参与一致，
不表示该点必然执行；Unreachable 的参与标为 None。

分支 merge 包含 selector 依赖；两分支完整落入后继才允许结构化重合。
单侧 return 保留剩余路径：常量值可以是 LaunchUniform，而其控制仅为
program 内 Full、launch Conditional。内存相关条件的控制保持 Unknown。
Const 可选择值入边和标记死分支，但不跨绑定复用结果。
`where` 两侧仍全部遍历；常量条件只选择结果值，不能删除未选侧 effect。

循环建立显式入口/回边，通过有限层级的单调固定点传播携带值。已知零次循环
选择入口值；已知非空循环可选择体出口；未知次数保留零次与非零路径。
归纳值按同次迭代的范围依赖传播，不表示跨迭代不变。
循环内提前 return 首版保守污染该循环参与和后继控制；已知零次不受死循环体影响。
未支持的范围参与关系即使经过后续 if 合并也保持 Unknown。
M4-05d 进一步发现：缺失入口定义的循环出口合并引用不能提供 Full 控制保证，
即使循环后继的独立语句正常重合。修复仅将该合并引用的控制降为 Unknown，
见[退出审计](uniformity-exit-audit.md)。

## 预算与隔离

`UniformityConfig` 默认 max_nodes=10000、max_depth=128、max_iterations=128。
这些是分析节点/递归/全图固定点轮次的确定性上限，不是 wall-clock/SMT 预算；
TIR 验证与原 effect 派生先行，不通过零预算跳过合法性检查。
预算耗尽时报告 construction-budget、depth-budget 或 fixed-point-budget，
已收集的所有值事实降为 Unknown，控制保证也失效；结构预算可能留下不完整事实表，
消费者必须检查 incomplete_reason，缺失事实不得默认安全。

无缓存、无全局绑定状态、不写 TIR、不调用 SMT 或 GPU。循环固定点在完整收敛前
不会发布中间强保证。对同一 kernel 的不同 Const/预算分别重新分析。
公共输出和 launch 绑定的隔离验收已由后续 M4-05c 完成。

## 本地验收与边界

`tests/test_uniformity.py` 共 **43 项专项**：四层级、读取复用、地址/expand/reshape、
完整/部分/布尔归约、急切 where、定义重绑定、selector 合并、提前 return、
Const bool/int 域、零次/嵌套/相互携带循环、未知迭代参与、assume 不加强、浮点/dot
保守性、预算、伪造摘要及损坏 TIR/transfer 漂移拒绝。
不能在当前源码表达的嵌套 assume 谓词通过显式构造并验证 TIR 检查，未放宽前端语法。

本地 CPU 全量 **1170 passed，零跳过**，JUnit 为
`artifacts/cpu/m4-05b-results.xml`。相对 M4-04d 的 1128 项，M4-05a 移除了旧文档中
一个尚无契约的 barrier 代码示例测试，本轮新增 43 项，故当前总数为 1128 - 1 + 43。
尚未取得本轮远端 CPU CI 记录；不沿用旧提交的成功状态。

```bash
PYTHONPATH=src python -m pytest -q tests/test_uniformity.py
```

本阶段未改变 GPU lowering，也不以已有 GPU 数值/atomic/Race 验收作为 uniformity
同步证据。当前没有真实同步消费者或 CTA/Warp 参与者映射，整个 M4 未完成；
M3 仍因隔离 GPU 持续验收缺失而 NOT READY。
后续 M4-05c 已固定可选详细输出、golden 与绑定隔离，下一步为 M4-05d 退出审计。
