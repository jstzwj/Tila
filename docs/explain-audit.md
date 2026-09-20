# M2-06：audit explain v1 与 golden 契约

`kernel.explain()`、`tila explain` 与 bounds 错误共享义务审计渲染器。
标题标记 `audit explain v1`。这是可读文本格式契约，不是 JSON/SMT 序列化协议。
测试入口 `tests/test_audit_golden.py`，快照位于 `tests/golden/m2_explain/`。

## 默认章节与顺序

kernel 标题、Const 绑定和 launch 前提提示之后，顺序固定为：

1. `parameters`：含隐式参数、标量精化及 Const 精化，按名字排序。
2. `types`：推导类型，按名字排序。
3. `facts`：符号区间、Const、非负性及已记录的 launch 对齐信息。
4. `obligations`：按访问源码顺序。
5. `hints`、`effects`、`aliases`、`warnings`、`notes`。
6. 有延迟约束时附加 `deferred constraints`。

每条义务依次展示 path、mask、conclusion、兼容旧消费者的 state、源位置、
trust sources、proof route、counterexample、unknown/pending、cache 边界、
SMT-LIB replay 可用性、fix。空节写明 `(none)`。

四态 verdict 与信任来源保持分离：SafeUnderContract 只是 ProvenSafe 的摘要，
UserAssumption 单独标记，不可达访问和豁免也保留原因。来源与位置排序去重。
位置沿用 frontend 的 kernel 源片段行号，不是文件绝对行号。

path/mask 不使用 Python 对象地址或 Z3 内部变量名；不同未知布尔身份使用不同
编号，映射后的数值符号保留轴身份。共享公式使用局部引用限制展开长度，超过
256 个 DAG 节点显示明确的 display-budget 标记；这只省略显示，不裁剪证明公式。

## 稳定边界与可选附件

M4-02 的 where 诊断沿用 warnings 节：`effects=off` 隐藏该类诊断，warn 展示
warning，error 对已有 kernel 的 report/explain 展示 error；装饰、特化、启动
入口则拒绝。每条记录带 where site、读取 site/RegionId、源片段行号和修复建议。
独立策略不改变 bounds 结论，见 [ADR-010](adr/010-effect-diagnostic-policy.md)。

M4-01d 增加 `--show-effects` / `show_effects=True`，在 effects 后附加
`tila.effect-details.v1` JSON；默认输出不变。字段、谓词 DAG 引用、Const 特化
和运行时绑定隔离契约见 [Effect 明细审计](effect-audit.md)。

默认输出不显示缓存命中/未命中，也不显示求解器任意选择的模型数值。
反例分类仍明确：ProvenUnsafe 为 `confirmed reachable witness`，Unknown 为
`candidate only; reachability unverified`。这避免具体 SAT 模型选择影响稳定快照。
Z3 unknown 的底层文字归一化，完整原因仍保存在 ProofResult.reason。

| 选项 | 内容与稳定性 |
|---|---|
| `--show-witness` | 按符号名排序的具体见证；数值可能随 Z3 版本或求解顺序变化 |
| `--show-cache` | 本次调用的 hit/miss/not-stored/disabled/not-applicable；不属于默认稳定文本 |
| `--show-query` | 末尾附加原始 SMT-LIB；不固定 Z3 内部命名或 let 排序 |

Python API 对应 `explain(show_witness=True, show_cache=True, show_query=True)`。
CLI 可用于 `explain`、`check --explain`，也可在 `check` 错误时请求这些附件。
缓存状态不参与 ProofResult 相等性；缓存不能改变 verdict、来源或契约要求。
`not-applicable` 包括快速路径、豁免、缺失坐标/extent 及在查询前耗尽预算的情况。

默认文本的稳定性以同一输入、证明配置、历史 launch 元数据和同一语义结论为前提。
真实墙钟超时可能改变 verdict；不会为了通过 golden 将 Unknown 改写成 ProvenSafe。
保留的 trace 是 Tila 的证明路线摘要，不是完整 Z3 proof 或最小 unsat core。

## 错误与重放

BOUNDS-001/002 的 Unknown 和 BOUNDS-003 的已确认越界均附带审计字段。
预算耗尽使用预算修复建议，不再误说成缺 mask。每个错误只保留一个外层 fix 节。
有查询时，`TilaError.proof_result.query` 保存完整重放文本；查询前预算失败、豁免
或纯快速证明可能没有查询，输出明确写 `not available`。

```sh
PYTHONPATH=src python -m tila explain examples/add_kernel.py
PYTHONPATH=src python -m tila explain examples/add_kernel.py --show-query --show-cache
```

重放 API：捕获错误的 `proof_result.query`，交给 `z3.Solver().from_string(...)`；
CLI 文本中逐条取 `SMT-LIB replay queries` 附件，去除显示缩进后同样可以重放。
原始查询用于核对 SMT 问题；可达性分类仍依赖 Tila 的精确/抽象执行上下文，
不能仅凭重放得到 sat 就把候选升级为已确认越界。

## Hint 依据与只读性

explain 用私有 Lowering 实例收集其实际会发射的 hint，不编译或执行生成代码，
不改 TKernel、last_report 或 launch 契约。max_contiguous 与 multiple_of 都带
StaticFact、源位置、结构依据以及整数 launch guards 的依赖提示；不从用户 assume
推导 hint。explain 本身仍允许温热进程内 proof 缓存。

## 验证

新增 24 项测试、15 份 golden：九类结果字段快照、安全/Unknown 的 CLI explain、
Unknown/ProvenUnsafe/预算错误，以及 hint 来源。固定 ProofResult 的字段快照
用于验证词汇与顺序；真实 CLI、solver 缓存、预算、SMT 重放和跨 hash seed 子进程
用于验证集成。实际模型数值只在固定见证 fixture 中做 golden，不把任意 Z3 模型当 ABI。

完整 CPU 基线：628 passed、零 skipped。修改默认字段或顺序时必须评审 golden，
必要时提升格式版本；更新不会通过自动覆盖快照掩盖语义差异。
