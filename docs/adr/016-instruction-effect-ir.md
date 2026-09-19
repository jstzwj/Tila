# ADR-016：逐指令 Effect IR 与派生汇总

- 状态：**Accepted**（2026-09-19 冻结；M4-01b/c/d 已实现）
- 日期：2026-09-19
- 前置：[ADR-002](002-region-id-and-extent.md)、[ADR-006](006-intrinsic-registry.md)、
  [ADR-007](007-integer-semantics.md)、[ADR-011](011-smt-proof-and-trust.md)
- 范围：现有 load/store/unsafe_load/unsafe_store 的内部表示与汇总；不增加公共语法。
- 非目标：本轮不实现 atomic、race、uniformity、barrier、优化重排或新的诊断策略。
  ADR-010 的 effects/race 严格度议题保持独立。

## 问题与现状

冻结评审：使用结构路径作为出现点身份；源位置首版只承诺已知行号；定义引用的
merge/loop 标签是保守身份，不是 SSA phi 或可达性证明。M4-01b 在 checker 完成
TIR 后统一绑定局部元数据，verifier 独立重算检查；保留原聚合列表和输出。
2026-09-20 M4-01c 已实现控制流 path/mask/loop 汇总并移除平行列表；
实现边界见 [派生汇总](../effect-summary.md)；M4-01d 已实现
[详细输出与隔离验收](../effect-audit.md)。

冻结前 `tir.TEffect(op, region_id)` 仅表示 Read/Write；checker 的 Buffer/Ptr 访问
处理器向 `TKernel.effects` 手工追加记录。`TLoad`/`TStore` 本身没有 effect 字段，
dump 和 explain 消费此独立列表。它适合粗粒度展示，却不能回答某次访问受哪个
mask、路径或循环约束，也不能保证 IR 改写后汇总仍与实际指令一致。

TIR 目前不是完整 SSA：`TName` 只保存变量名，`tk.types` 是最终定义快照；load
可能嵌在表达式中，不能仅遍历顶层语句，也不能把同名变量的不同定义混在一起。
本提案建立 effect 需要的定义引用，不要求本阶段把整个编译器改成 SSA。

## 不变量

1. 每个实际内存访问出现点恰有一个原生 effect。嵌套 load 不遗漏，重复使用已
   求得的值不重复记访问；两个独立 load 即使地址相同也不是同一个事件。
2. RegionId 表示来源身份，Extent 仍只服务 bounds；不同 RegionId 不推出 NoAlias。
3. Effect 表示**可能发生的读写**，不是执行次数、必然发生、无越界或无竞争证明。
   `unsafe_*` 仍产生正常 Read/Write；其 bounds 豁免不能擦除 effect。
4. 指令是唯一语义来源；路径、循环上下文和 kernel 汇总只能派生，不再让 checker
   同时维护一份独立、可修改的 kernel effects 列表。
5. 未知条件、未知地址、未知来源或预算不足均保留可能访问，不能解释成 Pure。
   内存 effect 不是所有可观察行为：debug assert、trap、return 仍由原 IR 表达；
   空内存 effect 集不能授权删除指令或重排。

## 建议数据模型

以下为内部字段草案，不是已存在的 Python API 或序列化承诺：

```text
MemoryEffect {                 # TLoad.effect / TStore.effect，不可变
  site_id: AccessSiteId,
  kind: Read | Write,
  region_id: RegionId,
  address_space: AddressSpace,
  element_dtype: DType,
  location: SourceLocation
}

AccessContext {                # 由 typed TIR 遍历产生，不写回第二份语义
  site: 指向访问节点及其 effect,
  address: BufferAccess | PointerAccess | UnknownAddress,
  path: Guard,
  mask: Guard,
  loops: tuple[LoopContext],
  value_refs: 当前定义点引用,
  bounds_refs: 当前访问的 obligation IDs（可为空）
}

Guard = KnownPredicate(shared Predicate DAG, origins)
      | OpaqueCondition(ValueRef, polarity, shape)
ValueRef = 参数引用 | 定义点引用 | 显式未知的 merge/loop-carried 引用
LoopContext = (loop_id, induction_ref, start_ref, end_ref, step_ref, parent)
KernelEffectSummary = 按遍历顺序保留的 AccessContext + 粗粒度 Read/Write 投影
```

`buffer/coords/ptr/mask/unsafe` 继续由现有访问节点持有；AccessContext 引用它们，
不得另存一份可漂移的 mask 或地址字段。MemoryEffect 的 region/dtype/space 属于
局部已检查元数据，verifier 必须与访问操作数的类型及来源交叉校验。

### 身份、位置与值引用

- AccessSiteId 按函数内的确定性结构路径分配，例如 `body/2/then/0/value/load`；
  同行多次访问仍区分，不用 Python 对象 id、临时目录或 GPU 地址作为稳定身份。
- 位置保留原文件/函数、源行及可获得的列；未知位置明确标记，不猜测父语句精度。
  最初迁移只承诺已有可确认的语句行；source map 可以引用同一访问身份。
- 特化保留存活节点的来源 site；新生成/复制的访问必须有新的出现点 ID 和原始
  来源关系。循环不展开，单个静态 site 代表零次或多次动态访问。
- 引用绑定到使用处的定义点，而非当前名字。重新赋值、分支合并和循环携带值
  不能回查 `tk.types` 的最终条目猜测旧值。不能精确恢复时给出 Opaque/Unknown，
  不把它当作循环不变量；定义引用与已有 Predicate/索引表达式通过适配层关联。
- 内存节点必须有唯一的求值出现点所有者。共享纯表达式或谓词 DAG 可以去重；
  不允许靠对象 identity 将两次内存求值合并。一个 TName 的多次使用不重新遍历
  其生产者。对结构异常或非法共享内存节点由 verifier 拒绝。

### Region 与地址

Buffer 访问使用参数的 BufferRegion；`buf.ptr` 和派生指针继承该 RegionId。
裸 Ptr 使用 ParamRegion；无法恢复来源则是 UnknownRegion。运行时 alias 事实
不改写静态 RegionId，不把两个参数合并成一个 effect site。

BufferAccess 引用坐标、布局/stride 符号与实际 view 基地址；PointerAccess 引用
指针定义及已有元素 offset。归一化字节地址时显式带 `sizeof(dtype)`，按 ADR-007
保留有限整数语义，不能用无界整数代数绕开溢出门禁。view 基址已含 storage offset，
不能再重复相加。地址不能归一化时仍保留 region-level Read/Write。

Extent 不进入 effect key；bounds obligation 仅按 ID 关联以供诊断。相同 Extent
不合并 RegionId；同 Region 的不同 offset 保留独立记录；地址不相交判断属于后续
消费者。默认外部参数 MayAlias，只有独立、有效的 alias 证据才能收紧。

## Mask、路径与信任来源

路径回答“这个 program 是否走到这里”，mask 回答“到达后哪些 lane 访问内存”。
二者保持分栏；需要合并时，先按访问 tile 的 lane 映射广播，再计算 path ∧ mask。
无 mask 对应 true，不等于来自用户的安全证明。bool tile 与 Mask 使用 ADR-012
的既有映射，不建立第二套布尔系统。

分支条件中的实际比较可进入共享 Predicate DAG；依赖 load、复杂 phi 或尚无编码
的条件保留 OpaqueCondition 和定义引用。Opaque 是保留条件的未知表达，不能把
所有未知条件合并成一个符号，尤其不能跨循环迭代认定为相同值。

`assume(p)` 不是真实的 if，不将 p 作为访问发生的无条件路径门禁。默认 summary
不依赖 UserAssumption 消除访问。StaticFact、CheckedLaunchContract、UserAssumption
的来源遵守 ADR-011，并与 ProofResult 分离；bounds 的 ProvenSafe/Exempted 也不
意味着访问没有 effect。

M4-01 不新增 SMT 查询，不利用一般不可达性推理删访问。仅允许依据已知常量、
Const 特化结果及确定的控制转移识别不可达；其余情况保留 may-effect。后续若增加
带假设的条件化报告，必须单独标注条件/来源，不能覆盖默认汇总。

即使外层 load 的 mask 为 false，计算其地址、mask、other 的子表达式仍可能包含
需要记录的读取。应先遍历已定义的操作数求值，再决定外层访问是否有活跃 lane，
不能跳过整棵子树。该规则同样适用于 store 的 value 和条件表达式。

## 控制流组合

遍历返回 `(accesses, fallthrough)`：前者是有序访问记录，后者描述是否/在什么
条件下继续执行当前块。它不是简单对所有子树求集合并集。

| 结构 | 规则 |
|---|---|
| 顺序 s1; s2 | 先收集 s1；s2 使用 s1 的 fallthrough。变量环境按定义更新，不按名字覆盖旧访问引用 |
| TIf | 先收集 cond 的访问；then 使用 path ∧ cond，else 使用 path ∧ ¬cond；汇合只合并可继续的分支，读取条件只发生一次 |
| TStaticIf | 未特化保留两侧带 guard 的记录；Const 特化后只汇总保留分支。符号 summary 和 specialized summary 必须注明阶段 |
| TReturn | 此 program 的 fallthrough=false；不影响此前访问，不代表所有 program 返回。后续语句只有仍活跃的分支能到达 |
| TFor | 收集循环控制表达式的求值；循环体附带 start/end/step 与 induction/loop-carried 定义引用；静态 site 不按次数展开 |
| 已知零次循环 | 体内无动态访问；控制表达式求值仍保留。原始静态清单可显示 dead body，但不计入该特化的 may-summary |
| 未知次数/循环内 return | 保留循环体可能访问与零次路径；循环后的路径保守包含正常结束情形，不能因体内可能 return 就全删，也不能断言必定到达 |
| where | 依现有 TIR 求值两侧；选择条件不成为任一侧的访问 guard；只有各 load/store 自带的 mask 与外围真实控制流限制访问 |

循环中的 return 退出当前 program，不是 break。循环内存写可能影响后续迭代的
读取条件；不得把基于内存或 loop-carried 值的条件当成固定谓词跨迭代合并。
无法表示“所有此前迭代都未 return”时使用未知 continuation，保留之后可能发生
的访问；不构造未经证明的精确循环可达性。

表达式遍历必须遵循现有 typed TIR 的求值结构，覆盖全部有操作数的节点，包括
load 地址/other、store value、if 条件和循环界限。不以源代码文本搜索 load/store，
也不以简单字段名白名单代替穷尽访问器。若某节点的求值规则尚不明确，应拒绝分析
并补齐契约，不能自行规定新求值顺序来绕过 CPU/backend 差异。

## 汇总、特化、缓存与输出

1. checker 构建局部 effect 元数据；TIR verifier 验证 kind、RegionId、dtype、
   mask shape、site 唯一性及节点覆盖完整性。普通/unsafe load 都是 Read，store
   都是 Write；未知 TIR 节点不得默认为 Pure。
2. 共用 `summarize_effects(typed_kernel)` 遍历器派生访问记录；特化/控制流改写后
   必须重新计算。初版优先重算，不维护易漂移的跨改写缓存。
3. kernel 级 Read/Write 投影只是报告视图；需要集合时可按 kind/RegionId 去重，
   但详细记录不丢失地址、条件、顺序或重复访问。集合不能用于访问计数和重排。
4. `TKernel.effects` 可暂时保留只读兼容属性，按事件顺序投影为旧 TEffect 列表，
   包括重复项；移除 checker 的四处 append，禁止外部修改该派生视图。
5. 原始/特化 summary 与单次 launch overlay 分开。CheckedLaunchContract/alias
   overlay 每次绑定重验，不写回长期静态节点；失败和空启动不能留下旧 overlay。
   空 grid 的实际访问为零，但不能抹掉 kernel 本身的静态可能效果。
6. 如后续缓存 summary，键必须覆盖 TIR revision、定义/控制流、Const、相关类型/
   布局、分析版本和信任来源；runtime overlay 不跨绑定复用。引入字段时更新 TIR/
   intrinsic 语义指纹，不能复用旧解释。

保持现有 explain 的 `effects:` 章节位置；先保留兼容 Read/Write 行。新增详细
访问格式应独立标注版本，稳定输出 site、源位置、kind、region、path、mask、loop
及未知原因，不暴露对象地址或 SMT 原文。当前 audit explain v1 和 golden 不在
本轮设计中静默修改；实施时单独评审新增输出及合法的可达性差异。

## 实施拆分与验收

M4-01a：设计评审（本文，Accepted）；M4-01b：节点元数据、定义引用、穷尽 visitor
及 verifier；M4-01c：控制流上下文与 summary、移除平行列表；M4-01d：输出/缓存
迁移及 golden。后续 `where` 策略、atomic、race、uniformity 单独立项。

至少覆盖以下反例再宣布实现完成：

- 相同长度的不同参数 RegionId 不合并；同 Buffer 与其 ptr/offset 共享来源但
  保留访问点；UnknownRegion 与 MayAlias 不产生独立性结论。
- 同行两次 load 保留两个 site；一个 load 值重复使用只记一次；嵌套 load、条件
  中 load、store value 和 `other` 中 load 不遗漏。
- `where(p, load(a), load(b))` 两侧仍可能读取；外层 false mask 不抹掉操作数
  自身的访问；用户 assume(false) 不把默认 summary 变成空。
- 分支早退后只有存活路径继续；名字重绑定不污染先前地址/guard；两侧都 return
  后无后续访问。静态 if 特化不会继承被丢弃分支的 effects。
- 零次循环、未知次数、循环携带指针/谓词和循环内 return 均保持保守；一个循环
  site 不伪装成只执行一次，循环体信息不泄漏到不成立的出口路径。
- `unsafe_*` 的 effect 不消失；mask 不相交或 bounds 安全不冒充 race 安全。
- 特化前后、绑定切换、失败/空 launch、不同 hash seed 的输出稳定且不泄漏事实。
- 遍历覆盖与 TIR registry 对账；新/未知节点和缺失 effect 被明确拒绝。纯 effect
  迁移不改变 CPU 数值结果或生成的 Triton 程序；按风险运行既有 CPU/golden 与 GPU gate。

M4-01b 不修改现有数值语义或 golden 格式。M3 因独立 GPU 持续
验收缺失仍未完成；提前设计 M4 的模型不改变该状态，也不宣称具备并发安全分析。

## 替代方案

- 继续让 checker 同时填指令和全局列表：两个真相来源，特化/变换容易失配，拒绝。
- 仅存 Read/Write 集合：无法表达出现点、mask、早退和循环，拒绝作为分析输入。
- 将整套 path/loop 快照永久塞进节点：移动节点后易陈旧，采用局部元数据加派生上下文。
- 先实现完整 SSA 或全功能 effect token：本阶段代价过大，先提供必要定义引用和
  保守未知；不因此授权优化重排。后续内存顺序建模需另立决策。
- 用用户 assume、bounds Safe 或不同 RegionId 消除 effect：混淆信任、边界和
  alias 语义，拒绝。
