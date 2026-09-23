# ADR-019：最小 Uniformity 分析与消费契约

- 状态：**Accepted**（2026-09-20，M4-05b 评审冻结）。
- 实施状态：**Partial**；M4-05b/c 内部分析、可选详细输出与隔离已实现，同步消费未接入。
- 前置：ADR-007、008、011、012、016、017、018。
- 范围：单次 launch 内 TIR 值依赖和结构化控制流的一致性；首先提供可审计分析。
- 非目标：barrier/shared memory API、Warp/CTA 布局模型、跨 launch/stream/host 同步、
  任意内存不变性证明、warp specialization、依靠一致性重排或优化代码。

冻结评审补充：所有事实以合法且已定义的执行为前提，摘要必须显式携带此限制，
不能代替 bounds/数值门禁。Full 指在该 scope 执行该点时参与一致，是否到达该点
另由 reachability 表达。首版对浮点运算/cast 的位模式契约采取保守 Unknown，
不从数学确定性推导目标一致性；参数、常量和复制仍可拥有层级。
部分轴浮点归约不推导跨输出元素的位模式相同，首版为 Unknown；完整归约的
ProgramUniform 只来自逻辑单一结果契约，仍不表示物理线程参与一致。
未支持的循环参与关系污染后继控制，不因走到结构化合并点就恢复 Full。
内部分析只接收 Const，不接收内存值、Race 结论或历史绑定；无缓存、无 SMT。

## 1. 一致性的对象与层级

一致性回答“在指定比较域内，值是否具有相同表示”，不回答值是什么，也不回答
地址是否安全、访问是否无竞争。浮点按 dtype/位模式比较（包括 signed zero/NaN），
不是用浮点 `==` 定义等价。只对合法且已定义的运算应用传播规则。

首版比较逻辑 tile 元素，不把元素编号等同于物理 CUDA thread/lane。
每个事实必须关联定义点、求值位置及动态循环实例；只比较同一逻辑求值实例，
不意味着某值在不同循环迭代中不变。

| 层级 | 保证 | 典型来源 |
| --- | --- | --- |
| LaunchUniform（L） | 同一 launch 中，所有实际求值该定义的 program 和逻辑元素具有同一值 | literal、typed constant、Const、按值传入的宿主标量、num_programs |
| ProgramUniform（P） | 同一 program、同一求值实例内，全部逻辑元素具有同一值；不同 program 可不同 | program_id、广播后的 program_id |
| Varying（V） | 已理解的逻辑元素依赖；不保证 program 内一致 | arange、多元素位置相关表达式 |
| Unknown（U） | 缺少适用规则、定义来源或分析资源，无法建立保证 | 首版 load/atomic 返回值、未知合并/布局/操作 |

保证从强到弱为 `L → P → V → U`；join 取最弱者。V 是“允许变化”，不是存在两个
不同值的数学证明：`arange * 0` 可以保守为 V，不能据此声称已经确认 divergent。
U 必须有原因，不能以标量 shape、缺失字段或空列表默认成 L/P。
内部可用尚无贡献的 bottom 求固定点，但它不是可输出的安全结论。

L 不等于 Const：宿主标量在 launch 内一致，却仍是运行时值；不能用 L 提升 staging。
`program_id(axis)` 为 P；不同 axis 不影响其 program 内一致性。
shape、stride、grid 等经当前 launch 校验的按值元数据为 L；grid=1 不隐式改变
静态层级。空 launch 标注 empty-launch，不用“没有执行”给所有定义制造 L。

load 初始一律 U，即使索引为 L/P、参数为 ReadOnly、同地址、结果是标量、Race Safe，
或 mask=false；这些条件单独都不构成统一的读取/材料化语义证明。
atomic_add 的旧值同样为 U，不能从 order/scope 或原子性推导一致。
未来如放宽，必须有单独的读取材料化、内存可见性与并发修改契约。

## 2. 值与控制流分别记录

值事实 `value_level` 与求值点上下文 `control` 独立。
例如 P 条件的分支中定义常量 1：常量值本身仍为 L，但其定义点只在部分 program
执行；不能将 L 当作所有 program 都到达此处的证明。

控制上下文至少包含：

- `selection_level`：控制此处执行的条件/循环范围依赖的 join；入口为 L。
- `participation`：在指定 scope 和同一动态实例上，Full、Conditional 或 Unknown。
  Full 必须附带 scope；P 条件可让 program 内参与一致，但不是 launch 全部参与。
- `reachability`：Reachable / Unreachable / Unknown，独立于层级。
- `fallthrough` 与 `returned` 的分离：到达后继的参与者集合，以及退出原因。
- 复用的 path、loop 定义引用；mask 单独记录，不把 masked memory 操作变成控制分支。

首版支持的 scope 是逻辑 Launch/Program；不输出 Warp/CTA convergence 保证。
证明程序点可达不等于所有参与者可达；相同分支条件的层级也不等于相同迭代次数。
不宣称 general happens-before 或 scheduler 顺序。

## 3. 传播规则表

下表的控制前提独立保留；值传播不能擦掉前提。未知 dtype/shape/操作规则返回 U。

| 节点或操作 | 首版值规则 | 条件与保守边界 |
| --- | --- | --- |
| literal、TConstant、Const | L | 保留 dtype/typed constant 位模式和 Const 类型标签 |
| 宿主 scalar、TPid、TNumPrograms | 分别 L、P、L | 只用参数契约，不检查 tensor 内容 |
| TArange | V | 首版不为长度 0/1 特判强化 |
| TZeros | L | 已通过 shape/dtype 校验 |
| 宿主 Buffer/Ptr 基址、TBufPtr | L | 仅地址值契约，不推导地址中的数据；当前绑定合法性独立检查 |
| TPAdd | join(base, offset) | 只传播地址值依赖，不推出 bounds/alias/读取一致性 |
| 纯确定性逐元素算术、比较、bool、cast | 所有输入的 join | 遵守 ADR-007/015；不以实数代数化简消掉有限位宽/NaN 依赖 |
| broadcast/expand/reshape | 保留输入层级 | 已验证形状的复制/重排；不删除 V/U，按轴精化不在首版 |
| TReduce，结果仍有多个元素 | 保留输入层级 | 不因消去一个轴就声称所有剩余元素相同；U 仍 U |
| TReduce，所有逻辑元素归约为一个值 | 已知 L/P/V 输入可得 P；U 保留 U | 仅已注册且具有 program 内单一结果契约的 sum/max/布尔归约；规则须核对 dtype、空域和执行前提；不直接得 L |
| TDot | U | 首版不从标量 acc 或常量矩阵推导；后续单独建立 transfer |
| TWhere | join(condition, a, b) | 两侧急切求值；两侧 effect/control 均保留，不能当 if |
| TWhere 特化常量条件 | 可选择结果侧层级 | 只改变值选择；不消除另一侧的 load/atomic 或合法性检查 |
| 同一 ValueRef 的复用 | 查该定义事实 | 不重走 load、不使用当前同名变量的事实 |
| runtime 分支后的 merge | join(selector, 各可落入后继的定义值) | 不能只 join 两边常量而忘记 selector；仅同一已验证定义引用可省去 selector |
| Const 分支 | 特化后只传播实际分支 | 未绑定 Const 不选分支；相同绑定域之外不复用摘要 |
| load/atomic 结果 | U | 地址、mask、other 分别分析，不能反推结果的一致性 |

TAssign 产生新定义事实；TStore/TAtomicStmt 不产生可复用值，记录求值点控制及
操作数依赖，仍访问其全部嵌套操作数。TAssume 不加强事实；其谓词的求值与嵌套
effect 不能遗漏。if/static-if/for/return 按下述结构化控制规则处理。

`where(c, x, x)` 可在两侧是同一 ValueRef 时保留 x 的值事实，但 c 和两侧操作数
求值合法性仍独立；不能把两个语法相同的 load 当作相同定义。
浮点纯操作只有在对应后端契约保证同输入具有一致表示时才适用通用规则；否则 U。
归约得到 P 表示逻辑结果性质，不证明归约执行点满足任何未来 collective 的前提。

### 分支、循环和提前 return

1. 进入分支：control 加入 selector 依赖；分支内已定义值按上表传播。
   未选择的分支只有在已验证 Const/static 路径下才标 Unreachable；用户 assume 不剪枝。
2. 合并：分别处理值定义与 fallthrough。没有 return/其他退出且两个分支均完整落入
   后继时，结构化合并可恢复进入 if 前的参与上下文；值仍按 merge 规则。
   仅一侧落入后继时必须保留相应路径，禁止无条件恢复 Full。
3. 循环：分别记录 start/end/step 的层级、循环内 control、归纳变量及 carried 定义。
   全 L 范围的第 k 次归纳值可为 L，P 范围最多 P；它们不代表跨迭代不变。
   V/U 范围不建立共同迭代实例，归纳/迭代参与事实为 U，不创造 lane 循环 API。
4. 循环携带值：首版使用有限层级单调固定点；join 入口值、更新值及选择更新的
   control。零次循环出口必须包含入口定义；只有验证至少一次执行才能省去零次路径。
   结构预算/迭代预算耗尽时，对受影响事实置 U，不能输出上轮尚未收敛的强保证。
5. 提前 return：出口记录实际剩余参与域。循环中 return 会影响后续迭代和循环后继，
   首版未能证明共同迭代/退出时将相关控制事实置 U；不会把一次迭代的 path 泄漏出去。
   program_id 条件 return 可保留剩余 program 内参与一致，绝不推导 launch Full。

Tila 当前的 runtime if 不接受 tile mask；上述 V 控制规则用于描述分析/未来消费
边界，不放宽前端语法。不为了测试分歧控制而把非法源码悄悄接受为新语言功能。

## 4. 数据模型与信任来源

增加由已验证 TIR 派生的不可变分析结果，不向 checker 添加平行手工列表。
直接引用 ADR-016 `ValueRef(kind, site, name, vtype)`；按 `definition/merge/loop/
induction/parameter` 分析，不能用变量名作唯一键。现有 TIR 不是完整 SSA/phi 图：
merge 和 loop 引用的入边需沿结构化块派生一个定义边表，不能假设 ValueRef 已含入边。

穷尽遍历复用 `effect_ir.OPERANDS` 的节点清单，明确每种节点的 transfer；已知但
首版不支持的节点输出 U+reason，未知 TIR 节点由 verifier 明确拒绝。
复用 `ConditionSource`、Predicate DAG、`AccessContext.path/mask/loops` 和 site 身份；
但 EffectSummary 只覆盖访问点，不能替代纯表达式、merge 和所有控制后继的分析。
允许新增结构化控制摘要，不能新增第二套表达式 AST 或重复解析源码。

拟议事实字段：definition/site、value_level、control、reason、dependencies、source。
依赖来源只接受静态语义和已验证的当前 launch 元数据。
`UserAssumption`、Race Safe、Bounds Safe、unsafe 豁免都不能升级一致性。
ProofResult 的 Safe/Unsafe/Unknown 与此处的层级是不同概念，不相互转换。
首版无需 SMT、内存值采样或 GPU 执行；分析自身预算独立，不挤占 bounds/race session。

初版不引入持久分析缓存。若后续缓存，键必须覆盖 TIR/定义边、Const 的值和类型、
规则版本、预算，以及消费需要的 target/launch 参数；符号结果和已绑定事实分离。
任何依赖当前绑定的事实必须逐次校验，失败/空 launch 不残留上次结果。

## 5. 消费门禁、诊断与 explain

现有 load/store/atomic_add 不因本提案新增一致性要求；scalar if 的类型限制也不
等同于已经实施 uniformity 检查。M4-05b/c 先提供分析和审计，不增加 barrier。

未来消费者必须显式声明：所需 scope、值要求、控制参与/共同动态实例要求，以及
target 对逻辑 program 与物理参与者的映射契约。候选包括 CTA barrier、warp
collective、shared-memory 协作协议；均需后续 ADR 才能加入。仅有 ProgramUniform
条件不自动满足 CTA barrier；Uniformity 本身也不证明 shared-memory race freedom。

消费检查必须同时满足值与控制前提。V 表示未获保证，不报“已证明分歧”；U 表示
分析不足。两者都不得通过硬性同步消费门禁；门禁在 lowering/编译/启动之前生效。
未来 off/warn/error 策略仅控制可选审计提示，不能关闭真实消费者的合法性检查。
不存在消费者时，不因正常 arange 的 V 或正常 load 的 U 对每个 kernel 发告警。

拟议诊断边界（**仅设计，不加入活动 registry/CLI/env**）：

| 码 | 含义 | 消费时行为 |
| --- | --- | --- |
| TILA-UNIFORM-001 | 所需一致性/参与保证不成立，例如要求 P 而只得到 V | 拒绝；说明 required/actual，不能将 V 称为已确认 divergent |
| TILA-UNIFORM-002 | Unknown、未收敛、缺失 target 映射等导致无法建立前提 | 拒绝并列 reason；不能因 warn/off 放行 |
| TILA-UNIFORM-003 | 未来审计策略或预算配置非法 | 配置诊断；实现策略入口时再注册 |

verifier 校验引用归属、入边覆盖、transfer 穷尽性、事实完整性及 scope 合法性；
不能信任调用者手写的 L/P 标注，不能用默认 L 修补缺失事实。未知新节点明确失败。
已知不支持的分析是 U；损坏 IR 是 verifier 错误，不能混为一个 Unknown 原因。

M4-05c 已接入可选 `--show-uniformity` 与 `tila.uniformity-details.v1`：按稳定 TIR site 顺序
输出参数/定义层级、值依赖、独立 control、loop/退出状态、消费者要求与结果、来源、
Unknown 原因及源位置。默认 explain 不变；无内存地址、tensor 内容、对象 id、
时间戳或 SMT 文本。内部标 symbolic/partial/specialized；公共 explain 沿用完整 Const
解析，不接受历史 launch 绑定。Unreachable 与未完成分析分别显示，不能伪装为证明通过。
具体字段和三份 golden 见 [Uniformity 审计](../uniformity-audit.md)。
本轮采用显式展示开关，无消费者时不新增 off/warn/error 策略、环境变量或活动错误码；
以后消费者硬性门禁不受展示策略影响。内部 requirement evaluator 仅为逻辑契约测试入口。

## 6. 正反例与验收表

以下是规则断言/伪码，不是新增 API；`need_program_full()` 仅代表未来消费者要求。

| 场景 | 预期值/控制事实或拒绝边界 |
| --- | --- |
| `Const B`、宿主 `n`、`num_programs(0)` | L；n 不因此变为编译期常量 |
| `program_id(0) + n`，再 broadcast | P，不能报 LaunchUniform |
| `arange(0, B) + program_id(0)` | V；不需要证明具体元素不同 |
| `load(x, 0)` / 同地址 atomic 旧值 | U，即使 x 为 ReadOnly 或 Race Safe |
| `sum(arange(0, B), axis=0)` 的合法完整归约 | 值 P；控制仍继承求值点，不能自动获得 Full |
| 矩阵部分轴归约、`reshape(arange(...))` | 不提升 V；标量形状之外也不按剩余轴猜测 |
| `where(pid == 0, 1, 2)` | P，不因两边常量为 L 就得到 L |
| `where(mask, load(a,...), load(b,...))` | U，两个读取都保留；mask 不构成控制分支 |
| pid 条件的两分支赋不同常量，之后 merge | P；两侧完整 fallthrough 可恢复入口控制 |
| pid 条件 return；随后常量 1 | 值 L，控制仅剩部分 program；不能满足 launch Full |
| `for j in range(0, N)`，N 为 L；无提前退出 | 同第 k 次归纳变量 L；不能称 j 跨迭代不变 |
| N 可为零的 carried 变量 | 出口 join 初值和更新；不能使用只在循环体定义的值 |
| 内存相关 return 或循环携带条件 | 未建模的参与/迭代事实 U，阻止未来同步消费者 |
| `assume(...)`、Race Safe、换同名变量绑定 | 不升级事实；重绑定按新 ValueRef 分析 |
| V 控制下的 `need_program_full()` | 即使该点值为 L 也拒绝；当前非法 tile-if 仍由前端拒绝 |

实现验收必须覆盖：规则表全节点、定义复用/重绑定、分支选择依赖、零次/固定点循环、
提前退出、预算耗尽、Const bool/int 域隔离、未知节点与损坏引用、未支持 load、
值/控制分离、symbolic/bound 输出隔离及 golden。小域枚举只验证 L/P 保证没有
反例；V 不要求每次输入实际变化。以伪造消费者元数据测试门禁，不能增设假 barrier API。
没有真实同步消费者前，不声称完成 GPU 同步安全验收。

## 7. 实施拆分与退出条件

- **M4-05a（DONE）**：Proposed ADR、规则/反例、状态与计划；无运行时修改。
- **M4-05b（DONE）**：已评审冻结为 Accepted；实现内部值层级与控制/定义边摘要、固定点和预算、
  verifier 与反例。无公共策略/同步 API；所有不支持情况有 U 原因。
- **M4-05c（DONE）**：可选详细输出、golden、符号/绑定与缓存隔离；显式展示开关为独立审计入口。
  硬性消费要求用测试 fixture 验证，实际消费者仍需单独批准的语言/target 契约。
- **M4-05d（限定 DONE）**：小域枚举/变形、错误保证的故障注入、退出审计；明确哪些仅为分析保证，
  哪些有真实后端消费证据。不能用内部分析通过代替未实现的同步功能。

M4-05d 的[限定退出审计](../uniformity-exit-audit.md)补足独立 Python 参考与具体执行观察。
评审时发现缺失入边的合并引用虽然值为 Unknown，控制仍可能被错误标为 Full；
现已只将该引用的控制降为 Unknown，不污染合法后继点。真实同步消费仍未安装。

M4-05a 完成不等于 uniformity 已实现，也不关闭整个 M4；M3 的隔离 GPU 持续验收
仍独立阻塞。未来物理 scope、同步消费者、优化消费分别立项，不隐含授权于本 ADR。
M4-05b 内部实现与验收见 [Uniformity 分析](../uniformity-analysis.md)。
