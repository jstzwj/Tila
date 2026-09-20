# M4-04b：跨 program 精确子集与成对查询

本页保留 M4-04b 阶段的接口与证据。M4-04c 已增加同次 tile store lane 检查，
并接入启动策略、缓存与稳定输出，当前行为见 [启动验收记录](race-launch-audit.md)。

[ADR-018](adr/018-minimal-race-analysis.md) 已 Accepted。内部入口为
`tila.race.analyze_races(tk, grid=..., bindings=..., consts=..., scalars=...)`，
不导出到 `tila` 公共 API，不改变现有 kernel 的编译、执行或默认诊断。
读取真实 NumPy/Torch（含 CUDA）tensor 元数据，不读取 tensor 内容，不运行 kernel。

## 当前覆盖

- 复用 verifier、Effect IR 的 AccessContext、定义点引用和 Predicate DAG。site 自配对
  与不同 site 的配对均检查；两个实例分别使用三轴 pid、逻辑 lane 和循环迭代变量。
- 精确输入是显式绑定的 Const/整数或 bool 标量、标量/一维 Buffer 或连续 Ptr、正
  整元素 stride、实际 view 基址/字节宽度；基址包含 view offset。缺失绑定保持 pending/Unknown。
- 支持 pid、num_programs、arange、加减、常数乘、整数 cast、比较与布尔组合；每一步
  有限整数运算保留取模回绕。已绑定常数起止/正 step、无携带地址与未知退出的循环使用
  符号归纳变量，不枚举循环次数。零次循环、Const 分支和 return 使用既有派生上下文。
- 按实际字节区间判断 overlap，覆盖不同参数 alias、重叠 view、stride 空洞、异 dtype
  部分字节重叠。不同 RegionId 不自动 NoAlias；不消费长期 `runtime_aliases` overlay。
- Read/Read 无竞争；验证配置及自然对齐后，仅相同 dtype、完整元素范围的兼容
  atomic_add/atomic_add 可以排除。Atomic/普通访存照常查询。

返回不可变 RaceReport/RacePairResult，包含 verdict、reason、来源、site 对、候选或
确认 witness、SMT-LIB 重放查询。witness 记录共享输入/grid、两个 pid/lane/循环归纳值、参数名和
view 相对字节偏移；查询只使用相对基址差，不输出绝对指针。

## 确认边界

unsat 证明这一对在指定跨 program 域内无冲突。SAT 要升级为 ProvenUnsafe，必须
满足精确地址和路径、通过既有 numeric gate，证明 kernel 中所有活跃访问有效，
并以独立 Python 整数求值核对两个具体事件。失败/未覆盖仍是 Unknown，不从已有
bounds Safe 或 unsafe 豁免借用有效性结论，不用 CPU 串行运行结果证明并发安全。

间接寻址、merge/loop-carried 值、依赖内存或 atomic 返回值的路径、循环提前退出、
动态循环范围、rank>1、多余未支持表达式保持 Unknown。guard 可保守过近似；若仍
unsat 可证明该对无冲突，SAT 则只能给候选。含运行期短路或未覆盖值运算定义域的
kernel 当前不确认 SAT，防止 may-effect 过近似制造真实冲突。assume 不参与消除访问。

`report.verdict` **只汇总 InterProgram**；`unchecked_domains` 始终列出 IntraProgram。
grid=1/空 grid 无不同 program，并不证明 tile 内重复 lane 地址安全。此阶段尚未接入
JIT 包装层的 assume_launch、target 或完整启动契约，不等同于可执行 launch 的安全证明。
调用者必须显式提供分析实例的绑定；不是从历史 launch 猜测或继承事实。

## 预算与隔离

独立 session 使用 ProofConfig 默认预算和 Z3 锁，不占用 bounds session 的查询计数，
没有 race 缓存。`RaceConfig.max_pairs` 默认 4096，流式生成 pair，截断保留
`incomplete_reason` 与 `unchecked_pairs`。默认每对最多 8000 个构建节点、128 层深度，
整个分析最多 128 次查询/2000ms，单次最多 100ms/200000 rlimit；查询序列化大小也受限。
零预算、超时、solver unknown 均不能升级为 Safe。verifier/summary 仍是既有前置遍历，
预算不是整个 frontend 的硬实时上限。

每次重新读取绑定，不修改 TIR/summary/alias overlay，也不保存 solver 可变断言。
元数据错型、越界标量/refinement、非法 grid、effect 缺失直接拒绝；尚不支持的布局
返回 Unknown。后续策略、启动拒绝、缓存及稳定 explain/schema 属 M4-04c。

## 验收

`tests/test_race.py` 覆盖分区、自配对、三轴 grid、单写者、互补分支、Const/return、
定义重绑定、固定/携带/提前退出循环、alias/stride/view、混合 dtype 字节重叠、
Atomic 与普通访问、where/嵌套 false mask、assume、间接寻址、无效先前访问、
整数窄化及 i32 中间溢出、预算耗尽和查询重放。

固定种子 204018 排列 27 组小 grid/step 域；逐 program/lane 枚举实际事件作为独立
oracle，要求 Safe 无漏报、Unsafe 的 witness 在事件集中。该子集审计不代替 M4-04d
后续更广的变形测试与失败最小化。现有 GPU lowering/执行未改变，不增加 GPU 支持声明。
M3 的隔离 GPU CI 缺口不变。

本地专项 **39 passed**，完整 CPU 回归 **1079 passed、零 skipped**。
完整记录：`artifacts/cpu/m4-04b-results.xml`。当前修改尚无新的 GitHub 托管 CPU CI
记录；既有 M4-03 GPU 388 节点/532 案例是此前执行路径证据，不作为新 Race 验收。
