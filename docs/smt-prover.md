# 默认 Z3 证明器（M2-03）

实现入口为 `src/tila/solver.py`，通过 `SolverProtocol.prove` 返回不可变
ProofResult。`z3-solver==4.16.0.0` 是标准运行依赖，已进入 `uv.lock`；缺失依赖或
非法预算配置报 `TILA-PROOF-001`，不会静默改用弱检查器。

## 编码与判定

Const、shape、经过 ADR-007 整数门禁的索引使用数学 Int；普通整数参数、加载值
使用带 signedness 的 BitVec。checker 为回绕运算、整数窄化、位运算和移位保存
不可变定义，solver 用 Int2BV/BV2Int 表达实际位宽；这两个转换是显式的，不能把
普通有限位宽运算直接当成无界整数。商和余数遵循 floor 规则，包括负除数；
MIN/-1 经目标位宽回绕。右移区分算术和逻辑移位。

每次 launch 的数值定义域/中间溢出门禁仍先执行。SMT 中除零和越界移位的定义域
也必须在前提下可证，不能把未定义运算当作 false 前提制造 vacuous safety。
浮点运算、加载内容及尚未精确建模的数据流以未知值保守近似；不新增浮点 SMT。

无用户假设且无待证定义域时，可用小型直接匹配/区间/grid 快速证明；其余默认查询
`facts ∧ path ∧ mask ∧ ¬in_bounds`。DAG 直接编码 And/Or/Not，未知布尔身份保持共享。
包含 UserAssumption 的查询同时检查前提一致性；矛盾假设返回 Unknown 并显示来源。
无用户假设的不可满足前提报告不可达访问。
未知布尔向量的不同 lane 视图使用独立布尔身份；同一视图重复展开保留共享身份，
避免将行条件与列条件误当同一布尔值而产生虚假的不可达证明。

unsat 返回 ProvenSafe。SAT 模型带索引、extent 和输入符号值。M2-04 对首个访问的
精确标量输入、常量与有限位宽定义确认可达，包括可编码的条件路径；其余 SAT
仍是 Unknown 候选，不把加载值、lane、phi 或循环抽象模型声称为具体可达程序。
具体限制和 CPU 重放证据见 [数据流与解释器](dataflow-interpreter.md)。
结果依赖不承诺最小集合：SMT 路径保守记录全部查询来源，小型捷径记录使用的来源。
`ProofResult.query` 保存可交给 Z3 重放的 SMT-LIB 越界查询；explain/错误信息展示
原因、来源和候选模型，不默认打印整个查询。

## 预算

每个 kernel 的装饰期检查、materialize、explain 和 launch 分别建立一个 session，
访问之间共享累计预算。所有默认值可用 `TILA_PROOF_<字段大写>` 环境变量覆盖，
或内部 `ProofConfig` 指定；0 表示该项预算为零，绝不向 Z3 传入代表无限制的 0。

| 字段 | 默认值 | 约束 |
|---|---:|---|
| `timeout_ms` | 100 | 单次 solver.check 毫秒上限 |
| `rlimit` | 200000 | 单次 Z3 确定性资源计数上限 |
| `total_ms` | 2000 | session 累计构建、序列化、求解时间 |
| `max_queries` | 128 | session 总 check 次数，包含定义域与可达性查询 |
| `max_nodes` | 8000 | 单义务构建的 DAG/算术/事实节点预算 |
| `max_depth` | 128 | 算术表达式/定义递归深度；Boolean DAG 使用迭代遍历 |
| `max_integer_bits` | 4096 | 常量及绑定的位数上限 |
| `max_query_bytes` | 2000000 | 重放查询与规范化键表达式的总 UTF-8 字节数 |
| `cache_entries` | 256 | 进程内 LRU 最大条目数；0 禁用缓存 |

预算耗尽、Z3 unknown、timeout 或不支持的编码均返回带原因的 Unknown；不删掉
Or 分支以“完成证明”。这些限制约束证明阶段，不是整个 Python frontend 的硬实时保证。
构建与序列化采用计数/截止时间检查，单次底层操作结束后也可能超过墙钟目标。

2026-09-19 本机 100 次无缓存的否定 mask 查询：中位 1.34 ms、P99 1.58 ms、
最大 10.76 ms。全量回归包含大共享 DAG、真实 rlimit unknown、累计预算、
零预算及模拟 solver timeout 传播。默认值是该开发基线的保守初值，不承诺所有
kernel 都能在预算内完成；复杂非线性/混合 Int-BV 查询可能仍需更高预算。

## 缓存与每次 launch 校验

LRU 键包含规范化查询、定义域、来源及位置、访问身份、Const/实际 launch 绑定、
grid 检查状态、可达性输入/上下文精确性、编码版本、Z3 版本和所有预算配置。Unknown 不缓存，较低预算失败
不会污染较高预算。缓存只保存纯 ProofResult，不复用有可变断言的 solver。

`assume_launch`、shape/grid、refinement 和整数定义域校验发生在 proof cache
查询之前，每次 launch 重验。符号 grid 的 materialize 结果仍只能标 pending，
不能成为 CheckedLaunchContract。当前缓存为有界进程内缓存，不提供磁盘持久化。
编译器通过锁串行使用 Z3 默认 context 和进程缓存，避免并发 launch 共享可变求解状态。

## 新旧行为核对

- `~(offs >= N)`：旧快速路径 Unknown，Z3 证明上界；CPU 实际尾块执行对照通过。
- `(i < 4) | (i == 7)`：没有共同上界原子，Z3 仍能证明 `i < 8`。
- `index & 7`、`cast[u8](index)`：依靠实际位宽证明范围，配合 CPU 执行校验。
- `i32_max + 1`：按位宽得到 i32_min；不得借用无界整数的非负结论。
- 用户假设冲突、模型可能不可达、求解器预算不足：保持 Unknown 和审计信息。

M2-05 已完成有界系统性质测试与结论差异审计，覆盖范围与重放方式见
[证明审计](m2-proof-audit.md)。M2-06 已固定 [audit explain v1](explain-audit.md)
和 CLI/关键错误 golden；模型、缓存遥测和原始查询是可选附件。
一般循环不变量和可达性不在当前精确子集内。

参考：[Z3 Bitvectors](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/)、
[资源参数](https://microsoft.github.io/z3guide/programming/Parameters/)。
