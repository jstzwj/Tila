# ADR-011：SMT 默认证明引擎与信任来源

- 状态：**Accepted**
- 日期：2026-09-19
- 实施状态：**Partial**；M2-02/03 已完成 DAG、信任来源、默认 Z3、Int/BitVec、预算与缓存；一般数据流可达性、性质测试和完整审计尚待后续
- 归属：M2；整数语义由已接受的 [ADR-007](007-integer-semantics.md) 固定，GPU 验证衔接 M3

## 背景与决定

将 Z3 提前为默认通用 bounds 证明引擎，替代原计划的“继续扩建手写
fast path，再接可选 SMT slow path”。保留常量折叠、直接事实匹配等小型、
可验证的快速路径，不维护两套长期并行的复杂求解器。

接入 SMT 不等于扩大语言表达式子集，也不替代整数语义、控制流分析、
信任来源或资源限制。实现切换时将受支持的 Z3 版本纳入标准开发/运行依赖
和锁文件；版本与预算值通过回归和性能测量确定，本 ADR 不虚构已验证组合。
缺少求解器应明确报配置错误，不得静默换成较弱的默认验证模式。
M2-03 已固定 `z3-solver==4.16.0.0` 并更新锁文件；预算、基准与实际编码边界见
[默认 Z3 证明器](../smt-prover.md)。

## 谓词与整数编码

- 用共享子表达式 DAG 保存 `And/Or/Not`，不强制转换为 DNF；保留 lane、
  broadcast 关系、路径条件和源位置。未知布尔值仍需保留稳定身份。
- Const/shape 的数学整数与 kernel 有限位宽整数分开建模。后者按 dtype、
  signedness、cast 和实际操作语义编码为 BitVec；转换到数学整数必须有
  明确的表示范围及无溢出依据，不能无条件混用 Int 和 BitVec。
- `pid * BLOCK + lane` 等推理及整除/contiguous hint 必须计入中间运算溢出。
  显式 i64 可以扩大范围，但不能替代证明。shape/grid 到索引类型的窄化也要检查。
- ADR-007 必须先固定溢出、负数除法/取模、除零、移位和转换边界；在对应
  语义尚未定义或编码未覆盖时返回 Unknown，不调用宿主算术猜测结果。
- Stage 1 保留符号义务；需要 Const 系数或实际 launch 事实的证明延迟到
  Stage 2，避免把两个未绑定符号的乘积假称为 Presburger 常量乘。

## 查询与结果

对每个访问查询：`facts ∧ path ∧ mask ∧ ¬in_bounds`。

| 求解结果 | Tila 解释 |
|---|---|
| unsat | 在记录的前提下没有越界；产生 ProvenSafe 及依赖 |
| sat，且反例对具体程序可达 | ProvenUnsafe；给出索引、输入约束和源位置 |
| sat，但涉及循环/数据流保守近似且未确认可达 | Unknown；仅报告候选反例 |
| unknown、预算耗尽或不支持的编码 | Unknown；记录明确原因 |

路径不可达也可能使查询 unsat；报告应区分不可达访问与有效访问的边界证明。
矛盾的用户假设不得被隐去，至少保留全部使用到的假设及位置；不能把依赖
错误 assume 的条件证明描述为无条件安全。SMT unsat core 是相关前提子集，
不承诺最小证明或完整的人类可读推导。

## 结论与信任来源正交

```text
ProofResult {
  verdict: ProvenSafe | ProvenUnsafe | Unknown | Exempted
  dependencies: StaticFact | CheckedLaunchContract | UserAssumption
  trace, source_locations, reason, candidate_counterexample
}
```

dependencies 是集合，允许多种来源共存。`SafeUnderContract` 保留为有已检查
契约依赖的显示摘要；同时显示 UserAssumption，不能覆盖或隐藏它。

- `unsafe_load/store` 返回 Exempted，只豁免本次义务，不产生安全事实。
- `assume` 是用户承担真实性的前提；派生事实和 hint 继承其来源。
- launch contract 必须每次验证后才能激活；缓存命中不能绕过验证。
- 快速路径和 SMT 使用同一结果结构、信任规则、strict/warn 策略。
- Unknown 在 strict 下拒绝；已确认 ProvenUnsafe 在任何模式都拒绝。

## 资源、缓存与审计

- 设单查询 timeout/rlimit、每个 kernel 检查或特化的累计预算，以及公式
  构建大小限制；构建阶段也不得发生无界展开。
- 预算不足返回 Unknown，不能删除析取分支后宣称安全。预算是执行可预测性
  的保障，不宣称 SMT 或整个 fast path 对所有输入线性复杂度。
- 缓存包含规范化义务、路径和 mask、facts/信任依赖、Const/launch 绑定、
  整数编码语义版本、求解器版本/相关配置。Unknown 不得跨更高预算永久复用。
- explain 区分已证明、依赖用户假设、已豁免、缺事实、不支持、超时/资源耗尽；
  将候选反例和证明依据映射到 Tila 源码，保留可复现查询。

## 迁移和验收

M2-02 实施记录：`predicates.py` 保存共享 And/Or/Not、未知值身份、比较位置及
lane/broadcast 映射；`facts.py` 的不可变 ProofResult 由 checker、launch、explain
共用。unsafe 是 Exempted；assume 以 UserAssumption 传播，实际 grid/assume_launch
校验后才登记 CheckedLaunchContract。无 launch 的符号 grid 只显示 pending，
不伪装成已检查契约。当前 hints 仅依赖结构性 StaticFact，不从 assume 推导。

旧 DNF 构建已移除。M2-03 已接入 Boolean/Int/BitVec SMT 编码、定义域与前提一致性
查询、候选模型及可重放查询、预算和来源敏感 LRU。常量无条件越界可确认为
ProvenUnsafe，其余 SAT 保守标记 Unknown 候选；循环/数据流反例可达性的扩大仍待后续。
迁移还修复了仅有上界也判安全、后置/分支/循环假设污染及行列 lane 混同。

1. 完成 ADR-007 与整数边界测试，建立统一谓词/ProofResult 接口。
2. 接入 Z3，在测试中对新旧路径做差异审计；重点检查“旧拒绝、新判安全”及
   “旧判安全、新给反例”。旧结果不是正确性 oracle，差异需核查具体语义。
3. 覆盖 unsat、可达 sat、近似伪反例、unknown、timeout、总预算耗尽、假设污染、
   矛盾前提、unsafe 局部性、缓存契约重验、嵌套布尔组合和有限位宽溢出。
4. 切换默认引擎、依赖、CLI/explain 和状态表；退役手写复杂 DNF 推理，避免
   永久双实现。文件如需移除，按仓库规则移动到带时间戳的 trash 目录。
5. 尽早在固定 GPU 环境核对整数/cast/mask 小用例；无 GPU 时明确未验证，
   不以 CPU 回归代替 M3 退出条件。

## 后续接口设计边界

以下是独立后续任务，不随 SMT 接入直接开放公共 API：

- Mask：拟允许 shape 合法的布尔 tile 用于 mask，比较产生的谓词与加载的未知
  布尔值分开；`valid & (offsets < N)` 的上界依据来自比较，valid 本身不给边界。
  保留 scalar-if 限制，先定义组合、broadcast 和后端规则，再实施。
- Const bool：优先设计独立参数域与带类型标签缓存键；启用前更新 ADR-005 的
  版本边界，当前 0.2.x 仍仅支持 ExactInt。
- NumPy integer：较低优先级，评估白名单接收、范围检查和规范化；不接受任意
  `__int__` 对象，不因易用性绕过 bool/int 区分。
- 浮点字面量：保留当前严格规则，先设计指定 dtype/舍入的常量构造方式，再以
  真实 kernel 评估是否放宽默认行为；具体语法尚未决定。

## 被拒绝的方案

- 继续扩建复杂手写求解器后才加可选 SMT：重复维护语义，且无法消除 DNF 展开问题。
- 无预算 SMT：可判定不代表能在可接受编译时间内结束。
- 将全部索引简单编码成 Int，或统一换成 i64：都不能自动保证有限位宽执行一致。
- 将所有 sat 当成真实 bug、所有 unsafe 当成 ProvenSafe：分别混淆近似与可达性、
  豁免与证明。

## 求解器参考

- [Z3 Bitvectors](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/)
- [Z3 资源参数](https://microsoft.github.io/z3guide/programming/Parameters/)
- [Z3 查询结果](https://microsoft.github.io/z3guide/programming/Z3%20Python/Introduction/)
