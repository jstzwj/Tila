# ADR-018：最小 Race 分析与诊断契约

- 状态：**Accepted**（2026-09-20，M4-04b 评审冻结）
- 实施状态：**Partial**；M4-04b/c 已实现精确子集、同次 store lane 检查、公共策略、启动门禁、诊断与缓存；一般数据流和同 program 顺序仍保守。
- 前置：ADR-002、007、010、011、016、017。
- 范围：单次 kernel launch 内 Global 内存的仿射访问冲突；复用 Effect IR、定义引用、
  Predicate DAG 和有预算的整数证明。先实现跨 program 分析。
- 非目标：跨 kernel/stream/host 的同步分析、barrier/shared memory、完整 GPU 内存模型、
  uniformity、非仿射全覆盖、新 atomic 操作，以及利用 race 结论重排或优化程序。

冻结评审：首批只提供读取 TIR 与绑定元数据的内部 `analyze_races`，不在 launch
自动运行。依赖 load/atomic 旧值、merge/loop-carried 定义和循环提前退出的可达性不
确认为真实冲突。有限位宽加减/常数乘/cast 可用与 BitVec 等价的显式取模编码，
逐步保留回绕；不得省略中间规范化。实现范围及验收见 [Race 精确子集](../race-analysis.md)。
绑定仅描述内部分析实例；JIT 包装层的 assume_launch、target 和完整启动契约不在本接口内，
不得直接将内部结果当作已完成 launch 检查。M4-04c 接入时必须先验证全部实际契约。

M4-04c 已完成上述启动接入，详见 [Race 启动验收](../race-launch-audit.md)。
不改变未覆盖情况保持 Unknown 的约束；默认 warn 会允许带明确告警的 Unknown。
M4-04d 已完成[限定覆盖退出审计](../race-exit-audit.md)，不扩大上述保证范围。

## 1. 现状与关键决定

M4-03 已有 Read/Write/Atomic、逐访问 site、path/mask/loop 和绑定 alias 信息。
它们描述可能访问，不直接证明两个动态事件可以共同发生，也没有 happens-before 图。
现有 `_runtime_alias_facts` 根据实际对象和保守存储区间细化关系：区间不相交可给
NoAlias；区间重叠仍是 MayAlias，不能直接认定实际访问冲突。

本设计采用成对的动态访问查询，先分析跨 program 的普通访存冲突。
同一 program 内 lane 重复地址单独报告覆盖范围，不能因跨 program 查询通过就宣称
整个 kernel 无竞争。CPU 顺序解释器只用于求值/事件核对，不能作为并发安全 oracle。

## 2. 动态访问身份、配对与顺序

动态事件身份为 `(site_id, pid[3], logical_lane, iteration_tuple)`。
logical_lane 是访问 tile 的逻辑元素索引，不等于 CUDA thread/warp ID；标量只有一个元素。
循环 site 对应多个动态事件；循环迭代和两个查询副本的 program/lane 变量必须独立。
公共输入、Const、实际 grid/layout 绑定则共享，不能给同一次 launch 两套不同输入。

所有可能访问 site 两两配对，包括一个 site 与自身。先按下表筛选：

| 访问对 | 处理 |
|---|---|
| Read / Read | 不构成数据竞争；可记录只读豁免理由 |
| Write / Read、Write / Write | 查询潜在重叠；即使写入相同值也不豁免 |
| Atomic / Read、Atomic / Write | 查询潜在重叠；原子标签不保护普通访问 |
| Atomic / Atomic | 仅对已验证、相同完整元素范围且兼容的 ADR-017 atomic_add 排除该对的 race；仍保留并发数值非确定性 |

Atomic/Atomic 部分字节重叠、不同元素宽度或未知配置不能据此豁免。
类型/target/自然对齐校验始终先执行，race 策略不能放松这些条件。

跨 program 查询要求 `pid_A != pid_B`（三轴向量至少一轴不同）。不同 program 的
源码先后没有跨 program 排序保证，循环迭代也不能据此建立全局顺序。
首版不推断通信协议或用户自建同步关系。

同一 program 的同一逻辑执行链遵循语言求值顺序，不能把普通顺序读写报成竞争。
同一次普通 tile store 的不同 lane 重复地址是单独的冲突检查义务，不用物理线程布局
解释成安全；M4-04b 先标 Unknown/未覆盖，M4-04c 再实现精确重复地址检查。
同 program 跨 site/迭代的 lane 关系尚无足够顺序模型时保持 Unknown；不假定
“所有 lane 相互独立”或“同一 program 所有访问都串行”。

## 3. 地址、alias 与有限整数语义

以半开字节区间 `[address, address + sizeof(dtype))` 判定重叠，不能只比较元素索引。
Buffer 地址为实际 view 基址加坐标与字节 stride 的线性组合；Ptr 继承 region 和
元素 offset，并显式乘 dtype 大小。view 基址已含 storage offset，不能重复累加。
异 dtype view 的部分字节重叠也必须保留，不能用 dtype 不同推导 NoAlias。

不同 RegionId 默认 MayAlias；Extent 只提供有效访问范围，不是内存身份。
静态相同来源可用共同基址推理；不同参数只有实际绑定或已验证的独立存储证据才能
收紧。MustAlias 也不代表两个派生 offset 相同。未知基址差值允许保守过近似：
unsat 可证明不相交，sat 只有候选意义，不能虚构两个用户参数确实重叠。

首个精确子集：标量或一维 tile，三轴 pid、逻辑 lane、已绑定 Const/标量整数系数、
固定正 stride 和已知 view 相对偏移的仿射地址；静态可判定分支、精确整数比较 mask。
循环仅在归纳变量/范围/step 精确且无未知 loop-carried 地址和退出条件时进入精确域。
复杂 phi、间接寻址、依赖内存或 atomic 旧值的地址/控制流不承诺精确判定。

所有索引遵守 ADR-007。`pid * B + lane` 必须计入中间回绕和窄化，不能直接当作
数学整数仿射式。只有已证明无溢出才用 Int 化简；否则使用已有精确 BitVec 编码，
或使用逐步显式取模的等价有限位宽编码；未覆盖运算返回 Unsupported/Unknown。
地址、shape/grid 转换的定义域门禁先执行；不能用
未定义操作或越界访问构造所谓已确认 race。unsafe bounds 豁免不消除 race 义务，
但候选无法独立确认访问有效时不升级为确定冲突。

## 4. 查询、路径与可达性

成对查询的概念形式：

```text
shared_launch_facts
  ∧ domain(A) ∧ domain(B) ∧ distinct_programs(A, B)
  ∧ path(A) ∧ mask(A) ∧ path(B) ∧ mask(B)
  ∧ overlap(byte_range(A), byte_range(B))
```

path、mask、loop 保持独立展示，编码时按各自 lane/广播映射组合。
`pid == 0` 与 `pid != 0` 两个分支在同一 program 内互斥，但两个不同 program
可以分别走两侧；不得把两份谓词错误地使用同一个 pid，进而推导全局互斥。
不展开循环/所有 program 对；用两个符号执行实例，并对迭代条件作有预算编码。

复用真实控制流的早退、Const 分支和零次循环规则；where 仍急切求值；外层 false
mask 不擦除嵌套访存。`assume` 不作为默认 race 查询的消除依据；初版只接受
StaticFact 和每次验证的 CheckedLaunchContract，仍在审计中列出被排除的用户假设。
不能把依赖 UserAssumption 的 bounds 结论升级成无条件的 race 前提。

不支持的地址/guard 必须作保守过近似，或者直接 Unknown；不能用不可靠关联收窄。
特别是 load、atomic 返回值和 loop-carried 条件，两个副本不能随意共享未知值身份。
若 guard 过近似后仍 unsat，可以证明这一对无冲突；sat 必须进一步确认共同可达。

## 5. 结果与覆盖范围

沿用 ProofResult 的 verdict 与信任来源分离原则，增加 race 专用 pair/witness 信息，
不借用 bounds 义务或 Exempted 表达 atomic 兼容性。内部结构草案：

```text
RacePairResult {
  pair: (site_A, site_B), domain: InterProgram | IntraProgram,
  verdict: ProvenSafe | ProvenUnsafe | Unknown,
  reason, dependencies, coverage,
  candidate_witness?, confirmed_witness?, replay_query?
}
RaceReport { stage, binding_scope, policy, pairs, unchecked_domains, suppressions }
```

| 结果 | 含义 |
|---|---|
| unsat | 该覆盖域、绑定和前提下无冲突；区分 NoAlias、地址不交、不可达与兼容 atomic |
| sat + 共同可达的有效访问对 | ProvenUnsafe；存在合法输入下可发生的无排序冲突，不声称每次调度都观测到错误结果 |
| sat，但 alias/地址/路径/循环仍为过近似 | Unknown，显示候选，不显示“已确认” |
| 超时、预算耗尽、未覆盖、不支持编码 | Unknown，稳定原因；不得丢掉未检查 pair |

确认 witness 至少提供共享输入与绑定、两个 site、pid/lane/iteration、相对字节区间、
各自 path/mask 成立及实际访问有效的证据。精确无内存依赖的子集可以用求值器核对；
CPU 对两个 program 的单独执行不能证明依赖共享可变内存的路径可以同时发生。
后一类保留 Unknown，不要求实现并发状态空间搜索。

符号阶段无实际 grid/绑定时保留 pending 条件报告；依赖假想 grid 或 alias 的候选
不在装饰时直接拒绝所有 launch。绑定后再确认；空 grid 没有动态访问，只说明本次
launch 为零事件，不抹去静态报告，也不跳过已有参数/target 合法性检查。
汇总必须分别列出跨 program、同 program 的覆盖情况。只完成前者时，禁止输出
无条件的“kernel race-free”；全局成功需要所有相关域和 pair 都已覆盖。

## 6. 独立策略与诊断（M4-04c 已实现）

采用 `TILA_RACE=off|warn|error` 与 CLI `--race`（显式 CLI 优先，否则环境，
默认 warn）。不复用 bounds safety 或 where 的 effects 策略；非法配置明确报错。

| 策略 | 已确认冲突 | Unknown |
|---|---|---|
| off | 不运行 race 查询，不宣称已检查 | 不运行 race 查询，报告 disabled/未检查 |
| warn（默认） | error，launch 前拒绝 | warning，允许执行并保留风险记录 |
| error | error，launch 前拒绝 | error，launch 前拒绝 |

off 是显式关闭 race 分析，不是安全证明；不关闭 verifier、bounds、atomic 或 target
门禁。初版不增加逐行抑制语法；关闭配置写入 RaceReport 的 suppressions，不能
只消音后把结果显示成 Safe。编译缓存命中仍按当前策略应用报告，off 的结果不能给
后续 warn/error 使用。将来若增加单点抑制，需要独立设计 site 身份与失效规则。

`TILA-RACE-001` 用于确认冲突，`TILA-RACE-002` 用于 Unknown；003 用于非法策略/访问对预算。
按策略决定后者 warning/error，稳定展示两个源位置、访问种类、region、path、mask、
loop、信任来源、原因及修复建议。预算遥测与原始 SMT 放在可选附件，不进入 golden。
建议只引用已有能力：按 pid/lane 分区、消除重复地址，或在算法确实是累加时使用
受支持的 atomic_add；不能建议不存在的 atomic_store，也不能把加 atomic 当作通用修复。

新增可选 `--show-races`/`tila.race-details.v1`，保留现有 effect-details.v2 和默认
explain 章节；设计实施时配套 golden。默认输出不暴露运行时绝对指针、对象 id、
机器路径；witness 使用 region/view 相对偏移。完整本地重放附件遵守既有隐私边界。

## 7. 预算、缓存与绑定隔离

复用 ADR-011 的 solver/整数编码配置，但 race 使用独立 session，不能耗尽 bounds
预算。首版沿用其单查询、总时间、节点、深度与查询字节限制；额外限制访问对枚举，
拟定每 session 最多 4096 个候选 pair（含可快速排除项），达到上限产生汇总 Unknown，
记录未检查范围。稳定 site 顺序枚举，不预先构建完整二次方 pair 列表。
预算默认值是待实现时测量的初值，不承诺硬实时；Unknown 不缓存成永久结论。

先实现无 race 缓存的正确性路径，再接独立有界结果缓存。键至少含分析/整数语义
版本、TIR/定义引用、pair 与分析域、两份 path/mask/loop、dtype/字节宽度、布局、Const、
grid、已验证标量/alias/相对 view 偏移、target、solver/预算和信任来源。
缓存结果与显示策略分离；相对地址归一化只能在 alias/重叠关系保持等价时复用。
每次 launch 重验绑定后再查缓存；失败、空启动、切换 alias/view 不得残留旧 overlay。
没有已验证的绑定等价性时宁可不缓存，不用参数名字或 RegionId 单独作为地址事实键。

## 8. 实施拆分与验收

- **M4-04a（DONE）**：形成 Proposed ADR、同步状态与计划，不宣称 race 已实现。
- **M4-04b（DONE）**：评审冻结 ADR；实现字节地址/动态身份适配、跨 program pair 查询与精确
  witness 确认、预算边界；不支持域显式 Unknown。先无缓存，复用现有 TIR 数据源。
- **M4-04c（DONE）**：同次 tile store 重复 lane 地址检查、独立策略/诊断/explain golden、
  launch 前拒绝与绑定/缓存隔离；未知 intra-program 顺序仍单独列明。
- **M4-04d（DONE）**：小域枚举与变形审计、支持范围和[退出审计](../race-exit-audit.md)；不以 M4-04 完成替代 uniformity
  或 M3 的隔离 GPU 持续验收。

至少包含下列性质与反例：

1. `pid * B + lane` 分区及尾 mask 无冲突；有限位宽中间回绕不能被 Int 化简漏掉。
2. 两个 program 写常量地址、跨 program Read/Write、自配对均捕获；同值写不豁免。
3. `pid == 0` 单写者可证明安全；不同 program 分别走互补分支仍可能冲突。
4. 相同 Extent/不同 Region、同对象绑定、重叠切片、相交包围区间但实际 stride 不交、
   异 dtype 部分字节重叠；冷/热缓存及绑定切换结论一致。
5. 兼容 atomic/atomic 允许同址；atomic/普通访问不豁免；off/unsafe 不放宽 atomic 合法性。
6. where 两侧、false mask 的嵌套访问、提前 return、Const 分支、零次/未知循环、
   loop-carried 地址；assume(false) 不制造无竞争结论。
7. 同 program 普通顺序访问不能误报；同一 store 多 lane 重复地址不被遗漏；
   只通过跨 program 分析的 grid=1 不能显示全局安全。
8. SAT 候选与确认 witness 分开；依赖内存内容或可能不可达的双路径保留 Unknown。
9. 零预算、pair 截断、超时、未支持节点、off→error、失败/空 launch 无事实泄漏。
10. 固定种子的小 grid/lane/整数域枚举实际事件，验证 Safe 无漏报、Unsafe 有有效事件对；
    保存最小反例、归一化输入和查询。CPU 用于事件求值，安全 GPU 样例用于保持数值语义；
    已确认冲突用 launch spy 验证未启动，不依赖执行危险 kernel 后偶然出现错误数值。

## 9. 排除方案

拒绝不同 Region 自动 NoAlias、SMT sat 自动报错、依靠用户 assume 消除访问、
CPU 串行结果充当并发证明、普通写相同值自动豁免，以及将所有 atomic 与普通访存
混用当安全。暂不实现通用并发模型检查器；精确子集之外保留可解释的 Unknown。
