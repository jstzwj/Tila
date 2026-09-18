# ADR-006：Intrinsic registry 架构与完整性门禁

- 状态：**Accepted**
- 日期：2026-09-18
- 实施状态：M1-05 已实现（2026-09-18）
- 关联：[ADR-005](005-const-type-domain.md)、[状态表](../status.md)

## 背景

当前 `PUBLIC_INTRINSIC_NAMES`/`METHOD_INTRINSIC_NAMES` 已消除了公共导出、frontend
allowlist 和 checker 名字集合的部分漂移，但签名、表面形式、状态、effect、target 与后端覆盖
仍散落在文档和多个 `if/getattr` 分派中。`docs/intrinsics.md` 曾声称“声明式签名表驱动”，实际
复杂规则仍由 `_in_load`、`_in_dot` 等手写 handler 完成。

完全声明式的类型规则 DSL 会把 shape、bounds、alias、staging 和 target 规则重新发明一遍；
只保留名字 tuple 又无法回答“这个 intrinsic 在什么表面形式出现、由谁检查、会发射哪些 TIR、
哪些后端应该覆盖”。本 ADR 在两者之间划定 registry 的职责。

## 决策

### 单一 catalog，专用 handler

`INTRINSICS` 是表面 intrinsic 的唯一机器可读 catalog。每个表项是不可变的
`IntrinsicSpec`，至少包含：

```text
IntrinsicSpec {
  name: IntrinsicName
  surface_forms: set[PublicCall | SubscriptCall | MethodCall | LoopForm]
  public_export: bool
  availability: Implemented | Partial | Designed | Deferred
  status_key: StatusKey
  arity: Arity
  keywords: tuple[KeywordSpec, ...]
  stage_rule: StageRuleId
  checker_handler: HandlerId
  effect_rule: Pure | EffectRuleId
  tir_ops: tuple[TirOpId, ...]
  backend_expectation: set[Check | Interpreter | Triton | Launch]
  target_requirement: TargetRequirementId
  docs_anchor: DocsAnchor
}
```

registry 管理“存在什么、如何进入、必须由哪些组件覆盖”；专用 checker handler 仍管理复杂
类型推导、shape、facts、bounds obligation 和诊断。`load/store/dot/reshape` 不需要被压成
脆弱的声明式类型 DSL。简单同构 intrinsic 可以共享通用 handler，但共享是实现选择，不是
registry 的前提。

`HandlerId`、`EffectRuleId`、`TirOpId` 与 `TargetRequirementId` 使用稳定符号标识，不在
catalog 模块直接保存跨模块 Python callable，避免 import cycle 和导入时副作用。启动/测试时由
各组件的显式 handler map 解析；未知或重复 ID 立即失败。

### 表面形式是显式数据

- `ti.exp(x)` 等为 `PublicCall`，需要公共占位符并进入 frontend。
- `ti.cast[dtype](x)` 同时声明 `SubscriptCall`，不能靠 frontend 特判后忘记 registry。
- `mask.any()`/`mask.all()` 是 `MethodCall`，不得导出为 `ti.any`/`ti.all`。
- `ti.range(...)` 是 `LoopForm`，只允许出现在 `for` header；普通 call 必须给定向诊断。
- `byte_offset` 当前可保留公共名字，但 availability 为 Deferred，checker handler 必须给稳定的
  unavailable 诊断。是否导出与是否完整实现是两个独立字段。

`PUBLIC_INTRINSIC_NAMES`、`METHOD_INTRINSIC_NAMES`、frontend allowlist 和公共占位符列表都
由 catalog 派生，只保留只读兼容 view，不再手工维护平行 tuple。

### 状态与文档的所有权

- registry 是 intrinsic 机器事实的来源：名字、形式、handler、覆盖、target、effect 分类。
- `docs/status.md` 仍是所有公开语言能力（不仅 intrinsic）的唯一用户状态表；每个 spec 的
  `status_key` 必须精确指向其中一行，availability 必须与该行一致。
- `docs/intrinsics.md` 保存人类可读的完整签名和约束；`docs_anchor` 必须存在。M1-05 可从
  registry 生成索引/覆盖摘要，但不从 Python 表项生成复杂自然语言规则。
- ADR 记录稳定设计，不作为当前实现状态来源。

这样避免在 registry 和 Markdown 中复制两份自由文本签名，同时仍能用测试发现名字、状态和
覆盖漂移。

### Checker、TIR 与 backend 的边界

frontend 按 `surface_forms` 构造统一 HIR Call/loop form。checker 通过 spec 的
`checker_handler` 查找显式 handler map，不再依赖 `getattr(self, f"_in_{name}")` 约定。

typed TIR 是 backend 边界。一个表面 intrinsic 可能发射多个 TIR op，同一 TIR op 也可能被
多个表面 intrinsic 复用，因此 registry 记录 `tir_ops` 作为可达性声明，但 interpreter/Triton
真正的执行 handler 由独立的 TIR capability registry 管理。完整性测试沿以下关系检查：

```text
surface intrinsic -> checker handler -> declared reachable TIR ops
                                      -> interpreter/Triton TIR handlers
```

不能伪造“一项 surface intrinsic 对应一个 lowering handler”的一一映射。若 intrinsic 只在
check/launch 阶段生效（例如编译期断言），`tir_ops` 可为空，但 backend expectation 必须解释为
不适用，而不是缺失。

### Effect、bounds、stage 与 target

- 纯函数显式标为 `Pure`；有内存/事实副作用的 intrinsic 必须给 `EffectRuleId`，不能用缺省值
  偷偷视为 pure。
- bounds obligation 不是 effect。load/store 的专用 handler 分别构造 obligation 与
  RegionId effect，遵守 ADR-002。
- `stage_rule` 描述参数/结果是 Stage 1、specialization-known 或 runtime 的约束；
  `static_assert` 使用 ADR-005 的 `StagedBool`，不能写成不存在的 `Const[bool]`。
- `target_requirement` 默认为明确的 `TargetNeutral`；需要 dtype/atomic/GPU 能力时引用受控
  requirement ID。空字符串和随意 predicate 不得进入 spec。

### 完整性门禁

M1-05 必须建立以下导入期或测试期校验：

1. intrinsic 名字和 status key 唯一；docs anchor 与各类 handler ID 均可解析；
2. public export、frontend 接受集合和 surface form 精确一致；method-only 名字不泄漏到公共模块；
3. arity/keyword 在 frontend/checker 前置统一拒绝，专用 handler 不再各自漂移；
4. Implemented/Partial 项拥有 checker handler；Deferred/Designed 但仍公开的项拥有定向拒绝 handler；
5. spec 声明可达且后端 expectation 要求的每个 TIR op 都有对应 interpreter/Triton handler；
6. 非 Pure 项有 effect rule；内存访问项明确 bounds policy，且两者不能共用一个字段；
7. availability 与 `docs/status.md` 对应行一致，docs anchor 存在；
8. registry 的 schema/semantic revision 进入编译缓存版本，影响生成代码的元数据变化不会误命中旧缓存。

新增 intrinsic 若缺任何必需信息，CI 必须失败；不能等到用户调用后才由 `getattr` 或 backend
`else` 暴露遗漏。

## 被拒绝的方案

1. **继续维护名字 tuple + `_in_<name>` 反射**：只能检查拼写，无法覆盖表面形式、状态、effect、
   target 和 backend。
2. **把所有类型规则写成声明式签名 DSL**：复杂内存和 shape 规则会变成第二套 checker，诊断和
   proof facts 反而更难维护。
3. **在各模块使用装饰器自注册**：导入顺序决定 catalog，难以获得稳定快照、避免循环依赖并做
   缺项检查。
4. **把 Python callable 直接放进中央 catalog**：intrinsics 模块会反向导入 checker/lowering，
   产生循环依赖和导入副作用。
5. **用 surface 名字直接代表 backend handler**：表面调用与 typed TIR 不是一一对应，会漏掉
   checker 重写和共享 op。
6. **从 registry 生成全部规范正文**：表项适合机器事实，不适合承载 shape 等价、bounds 证明和
   诊断解释等长篇语义。

## 迁移与验收

1. 先定义 enums、`IntrinsicSpec`、不可变 catalog 和 validator，不改变语义。
2. 将现有 public/method/checker 名字 tuple 改成 catalog 派生 view，保持兼容导入。
3. frontend 改为查询 surface form；checker 改用显式 handler map。
4. 为 typed TIR 建立 backend capability view，连接 `tir_ops` 覆盖校验。
5. 把 effect/stage/target/status/docs 元数据逐项补齐，再删除旧的平行 allowlist。
6. 运行全量测试，要求零 skipped；新增一个故意缺 handler/状态/后端覆盖的坏 spec 单元测试，
   证明每类门禁都会失败。

M1-05 已完成不可变 catalog、派生 public/method view、显式 checker handler map、统一
call-shape 预检、effect/bounds 分栏、typed-TIR backend capability map、状态/文档关联和缓存
semantic revision。复杂签名仍由专用 handler 与本文共同表达，不宣称存在声明式类型 DSL。
