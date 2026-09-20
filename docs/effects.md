# Tila 效应系统、竞争检测与 Uniformity

状态：设计基线与后续方向；当前已有局部 effect 元数据、TIR 派生的 kernel
Read/Write 汇总及 where 急切读取检查，并发检查尚未完成。
2026-09-19 [ADR-016](adr/016-instruction-effect-ir.md)已 Accepted；
[M4-01b](effect-ir.md)实现局部访问身份、Read/Write/RegionId、定义引用与 verifier。
[M4-01c](effect-summary.md)已实现 mask/path/loop 与 kernel summary 派生。
[M4-02 / ADR-010](adr/010-effect-diagnostic-policy.md)已实现独立 effects 策略。
M4-03b/c 已实现 atomic_add；M4-04b/c 已接入 Race 精确子集与启动门禁；
M4-05b/c 已实现内部 uniformity 分析与可选详细输出，尚无同步消费。
前置阅读：`design-principles.md` §2（第四支柱）、`type-system.md` §8–§9
（Region 概念）。

效应系统在 v0 对用户**完全不可见**（不出现任何效应语法），但它是
v1 三类诊断——`where` 陷阱、inter-program race、divergent barrier——
的基础设施。本文同时定义 v0 就要落库的效应**记录**与 v1 才启用的
效应**检查**。

---

## 1. 效应语法

```text
Effect :=
    Read[R]         读取区域 R
  | Write[R]        写入区域 R
  | Atomic[R]       原子读-改-写区域 R
  | Barrier[Scope]  屏障（CTA / GPU；v2 引入时启用）
  | Pure            无效应（算术、cast、where 纯值侧…）

R := Region 标签
   | Shared
```

**RegionId** 是编译器生成的结构化身份：Buffer 参数使用 `BufferRegion`，
裸 Ptr 参数使用 `ParamRegion`，内部独立分配使用 `InternalRegion`，来源未知时
使用 `UnknownRegion`。RegionId 不出现在公共 Ptr 参数中，也不携带长度；bounds
只使用独立的 Extent。完整模型见 ADR-002。

函数（kernel 与后续的 @tila.func 工具函数）签名由此升级：

```text
(A, B) -> C  !  { Read[A], Write[B] }
```

以下仅是粗粒度 may-effect 集合投影，不能作为逐访问控制流算法；条件求值、
提前 return、零次循环、mask 与循环携带定义见 ADR-016：

```text
顺序组合:  eff(s1; s2) = eff(s1) ∪ eff(s2)
分支:      eff(if c: s1 else: s2) = eff(s1) ∪ eff(s2)
循环:      eff(for …: s) = eff(s)
纯表达式:  eff(e) = Pure
```

---

## 2. v0 范围：只记录，不检查

v0 落库的内容：

1. 内存操作的效应记录为 **kernel 级聚合列表**（携带结构化 RegionId 的
   `TEffect`，
   进入 `--explain` 输出与诊断报告；per-instruction 效应字段留待
   M4 effect 检查启用时升级）；
2. kernel 级效应汇总进入 `--explain` 输出与诊断报告；
3. **别名关系独立建模**：`MustAlias | MayAlias | NoAlias` 不编码进 RegionId。
   不同外部参数静态默认 `MayAlias`；launch 根据实际存储对象/区间将其提升为
   `MustAlias` 或 `NoAlias`，重叠视图保持 `MayAlias`。显式别名声明
   `tila.alias(x, y)` 仍推迟到 v1。

M4-01b 已为 TLoad/TStore 增加不可变的局部 effect 元数据；M4-01c 已派生路径和
循环上下文，生成只读 kernel 汇总并移除 checker 的平行追加列表。
未知条件/地址仍保留可能访问，unsafe 只豁免 bounds，assume 不是真实控制分支。
现有实现状态不因 ADR 文件存在而升级。

v1 启用检查后，用户仍然不需要写任何效应——所有效应来自推导。

---

## 3. `where` 与急切求值陷阱

Triton 的 `where` 两侧都会求值，包括其中的 load：

<!-- tila-example: diagnostic -->
```python
x = tila.where(mask, tila.load(a), tila.load(b))   # 两个 load 都发生
```

Tila 规则：

- **值侧**：`where` 是纯函数（三分广播，签名见 intrinsics.md §3），
  dtype 必须两侧同型（无隐式提升）；
- **当前效应侧**：遍历已验证 TIR 的值操作数，内联读取（含 unsafe、深层表达式、
  load 的 coords/mask/other）产生 `TILA-EFFECT-007`。已计算值的定义引用不重新
  执行 load；变量复用、重绑定、分支合并和循环携带引用不会因此误报。
- 条件操作数的读取不单独告警；嵌套 where 的读取只归属最近的值分支，避免重复。
- 这是静态急切求值提示，不是越界或必然访存结论。false mask、Const 未选路径
  不消除结构提示；每条 load 仍受其自己的 mask 约束。
- `TILA_EFFECTS=off|warn|error`，默认 warn；CLI `--effects` 覆盖环境配置。
  off 不关闭 bounds/verifier；warn 在 report/explain 展示；error 在装饰、特化和
  每次启动前拒绝，包含缓存命中与零 grid。与 `TILA_SAFETY` 完全独立。

```text
warning[TILA-EFFECT-007]: eager read in where 'then' operand
    at: line 3
    phase: check, launch, specialize
    where site: body/0/value
    Both value operands are evaluated; each load retains its own mask.
    Read site=body/0/value/a region=BufferRegion[0:x] line=3
```

修复按意图区分两侧：`load(a, mask=cond, other=pure_value)` 和
`load(b, mask=~cond, other=pure_value)`；标量 bool 用 `not cond`。仅在形状兼容
时组合原有 bounds mask；other 中放 load 仍会急切读取。不会自动改写用户代码。

Python 示例：`TILA_EFFECTS=error python example.py`；CLI 示例：
`tila check example.py --effects error`。策略不进入编译缓存键，因为不改变生成代码，
但入口会逐次重算诊断。原始候选保留在 `tk.warnings`；report/explain 按当前策略
展示。详细设计与边界见 ADR-010，专项及 golden 在 `tests/test_where_effects.py`。

---

## 4. Race 检查（精确子集与启动门禁已实现）

详细契约见 [ADR-018](adr/018-minimal-race-analysis.md)（Accepted）；M4-04b/c 已实现
[精确子集和启动门禁](race-launch-audit.md)。`TILA_RACE`/`--race` 独立控制策略，
默认 warn；`--show-races` 展示稳定详细输出，模型/查询/缓存遥测另行显式开启。
分析必须覆盖 Read/Write 与 Atomic/普通访问对，包括单 site 自配对；不同 RegionId
不代表 NoAlias。跨 program 通过不等于同 program 多 lane 已无竞争。

SPMD 语义下，不同 program instance 并发执行同一 kernel。对每个
`Write[R] / Atomic[R]` 效应，检查其地址表达式在 program 维度上的
相交性：

```text
write 地址 A(pid)：
    ∀ pid1 ≠ pid2:  A(pid1) ∩ A(pid2) = ∅        → 通过
    ∃ 共同可达、有效且无排序的冲突访问对           → error TILA-RACE-001
    无法判定                                       → warning TILA-RACE-002
```

### 4.1 可证安全

<!-- tila-example: current; mode=syntax -->
```python
tila.store(out, pid * BLOCK + arange(0, BLOCK), v)
```

在 BLOCK 为正、lane 范围正确、中间运算无溢出且布局受支持时，预期可证明
不同 pid 的区间 `[pid*B, pid*B+B)` 互斥；M4-04b 内部分析覆盖上述一维精确子集。

### 4.2 可证冲突

```text
error[TILA-RACE-001]:
multiple program instances may write the same location
    write address:  out + 0
    depends on program_id:  no
consider: partition addresses by pid; use atomic_add only for additive updates
```

### 4.3 判定域与保守性

复用整数/谓词编码与独立 alias 事实，比较实际字节区间。精确子集以外保持 Unknown；
SMT sat 但共同可达性未确认时只是候选。ADR-018 默认 warn：确认冲突报错，
Unknown 告警；error 模式也拒绝 Unknown，off 明确记录未检查。兼容原子之间允许
同址更新，Atomic 与普通访存不能自动豁免。`pid == 0` 可形成单写者条件，但
不同 program 上的互补分支不构成全局互斥。同次普通 store 的重复 lane 已检查；
不同 site 或迭代的顺序无法确认时仍 Unknown，不从相同 lane 编号推出安全。

---

## 5. Atomic 的严格类型化

M4-03a 已冻结 [ADR-017](adr/017-minimal-atomic-add.md)，M4-03b 的
[前端/IR/CPU](atomic-cpu.md)已实现，M4-03c 已完成[限定 GPU 验收](atomic-gpu.md)。它限定首版
为 atomic_add、Global ReadWrite、i32/u32/f32、Relaxed/GPU；以下其他枚举和操作
仍是长期设计。AtomicInfo 保留 op/order/scope，不能只当普通 Write；接入现有
verifier/summary/where 与 explain v2 的具体迁移、CPU 顺序和 GPU 验收见 ADR。
原子标签不意味着与普通访存混用自动无竞争，后续 race 分析必须单独处理。

<!-- tila-example: future; milestone=M4 -->
```python
tila.atomic_add(ptr, v, order=tila.Relaxed, scope=tila.GPU)
```

- **值类型必须精确匹配指针元素类型**：`Ptr[f32]` 配 `f64` →
  TILA-TYPE（无隐式 narrowing，哪怕 Triton 容忍）；
- 操作数 dtype ∈ `AtomicTarget` 能力类（type-system.md §2）；
- `MemoryOrder / MemoryScope` 是**类型化枚举**，不是字符串：

```text
MemoryOrder := Relaxed | Acquire | Release | AcqRel
MemoryScope := CTA | GPU | Sys
```

- `atomic_cas: Ptr[T] × T(expected) × T(new) → T`；
- `atomic_*` 的 mask 形态：mask 也必须是精确同形 Mask；
- 每个原子操作在效应层记为 `Atomic[R]` 并进入报告汇总；仅兼容的原子更新
  之间允许同址竞争，不能因此忽略与普通访存的冲突。

---

## 6. 最小 Uniformity（Partial，内部分析）

[ADR-019](adr/019-minimal-uniformity.md) 已在 M4-05b 冻结为 Accepted，替换本节原先
混用 Program/CTA/Warp 的草案。首版比较逻辑 tile 元素，不推断物理 thread 布局。
已有[内部派生分析与 verifier](uniformity-analysis.md) 和[可选详细输出](uniformity-audit.md)。

| 层级 | 保证/来源 |
| --- | --- |
| LaunchUniform | 同一 launch 一致：Const、宿主按值 scalar、num_programs |
| ProgramUniform | 单个 program 内一致：program_id；不同 program 可不同 |
| Varying | 允许逐逻辑元素变化：arange；不是已经证明分歧 |
| Unknown | 尚无保证：首版 load/atomic 旧值、未知数据流/布局/预算耗尽 |

纯逐元素操作取依赖 join；值一致性与控制参与/退出状态分开记录。归约得到单一值
也不能证明所有参与者到达归约点。分支 merge 必须包含 selector，循环出口保留
零次路径，return 不得丢失剩余参与域；assume/Race Safe 不能升级一致性。

未来同步消费者必须同时满足所需值和控制保证，并有 target 参与者映射契约；
不能仅凭 ProgramUniform 就批准 CTA barrier。V/Unknown 不满足硬性门禁时拒绝，
可选审计的 off/warn/error 不得绕过合法性检查。本阶段没有 barrier/shared memory
API 或 UNIFORM 活动错误码；CLI 只提供 `--show-uniformity`，无策略 env。
规则、正反例及实施拆分以 ADR-019 为准。

---

## 7. 错误码一览（本文引入）

| 码 | 场景 | 阶段 |
|---|---|---|
| TILA-EFFECT-007 | where 分支内内存效应 | v1（默认 warning） |
| TILA-RACE-001 | 共同可达的无排序冲突访问对 | M4-04c 已实现 |
| TILA-RACE-002 | 无法证明访问对无冲突 | M4-04c 已实现，按 race 策略 warning/error |
| TILA-RACE-003 | 非法 race 策略/访问对预算 | M4-04c 已实现 |
| TILA-UNIFORM-001/002/003 | 所需保证不满足/Unknown/未来配置非法 | ADR-019 仅设计，未注册 |
| TILA-TYPE-030 | atomic 值类型不精确匹配 | v0（随 atomic 一起落） |

## 8. M4-02 本地验收（2026-09-20）

18 项专项覆盖求值位置、定义复用/重绑定/合并/循环携带、深层读取、条件读取、
嵌套 where 去重、false mask 与 other、静态不可达告警、策略切换、缓存及空启动
门禁、bounds 独立性、元数据损坏、CLI 环境继承与覆盖。warning/error 使用同一
稳定诊断 golden；官方示例的既有 TIR/Triton/explain golden 未改变。

完整 CPU **1002 passed**，本地 RTX 3090 严格 GPU **300 节点／444 案例通过**，
均零跳过。产物：`artifacts/cpu/m4-02-results.xml` 与
`artifacts/ci-gpu/20260919T165747Z-de6zmzkp/report.json`（本地忽略目录）。
当前工作区尚无独立远端 CPU CI 记录；无隔离 GPU 持续验收，M3 仍未完成。
