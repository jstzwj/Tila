# M2-05：证明结论差异与性质审计

日期：2026-09-19。基于 M2-04 提交 `6290122`，新增 67 项参数化测试；完整
CPU 基线为 604 passed、零 skipped。本批未发现需要修正的新执行语义错误，
未修改生产证明器、整数语义或默认预算。

## 验证方法与范围

入口为 `tests/test_m2_properties.py`；独立 Python 整数参考、失败记录及重放工具为
`tests/m2_audit_support.py`。不调用生产 `numeric.wrap/integer_binary` 计算期望值，
避免解释器与 oracle 共享同一错误。SMT 从真实输入 BitVec 绑定原始位模式，
不是绕过 signedness 的 BV2Int 结果替换。

| 检查 | 本次范围 |
|---|---|
| CPU 整数运算 | i8/u8 的 `+ - * // % & \| ^ << >>` 全部合法输入对，共 1,055,744 对；除数排除零，移位计数为 0…7 |
| SMT 整数运算 | 每种运算遍历全部 256 个左操作数，右操作数取符号/极值/小常量切片；移位覆盖全部计数，共 33,792 点 |
| cast | i8/u8 到 i8/u8/i16/u16/i32/u32/i64/u64 的 16 组全域对照，共 4,096 点 |
| 一元与中间溢出 | i8/u8 `~` 全域；先回绕再除法，禁止消除中间溢出；i8_max+1 的负索引反例 |
| 安全判定 | 8 组 guarded/unguarded 义务与 256 输入真值表对照；确认反例从 SMT 模型取输入，在 CPU 内存检查上重放 |
| 布尔变形 | 种子 2052026，24 组三原子、4 条等价律：德摩根、分配、双重否定、交换；每条核对 i8 全域并证明差异路径不可达 |
| 共享 DAG 压力 | 160 层共享吸收式，节点线性增长；充分预算证明安全，小构建预算必须 Unknown |
| 控制流 | i8 全域 × 6 种循环次数，共 1,536 次 TIR 执行；30 个边界组合再走公开 launch，覆盖零次、提前 return 和回绕 |
| 广播与视图 | 固定种子的 40 组形状/mask；负 stride 切片、转置等价、masked load/store 与底层存储对照 |
| 缓存 | 14 组来源、extent/输入绑定、signedness、可达性上下文；随机顺序写入、逆序读取，与无缓存结果及来源对照 |
| 预算/假设 | 7 类零预算与预热缓存隔离；真实 kernel 的前置、后置、分支和零次循环 assume，核查快照与 debug 执行顺序 |

除零、非法移位、宽整数边界、用户矛盾假设、lane 视图隔离及 launch 契约逐次重验，
同时由既有 `test_integer_semantics.py`、`test_proof_model.py` 和
`test_smt_solver.py` 覆盖。穷举范围只限表中小位宽子集，不声称穷举 i64/u64 输入空间。

## 结论差异台账

本次直接双跑的是保留的小型 `fast_obligation` 与当前 SMT，不是重新部署历史版本
完整编译器。该界限避免将快速战术的局限误称为某个历史版本的全部行为。

| 案例 | 快速路径/对照抽象 | 当前结论 | 执行证据 |
|---|---|---|---|
| 已知非负，mask 为 `not (x >= 8)` | Unknown | ProvenSafe | i8 全域中每个 active 索引均合法 |
| 已知非负，mask 为 `x < 4 or x == 7` | Unknown | ProvenSafe | i8 全域逐点核对，析取不需要共同上界原子 |
| 精确标量路径 `x < 0`，访问长度 8 | Unknown | ProvenUnsafe | active 负索引实际触发 CPU bounds 断言 |
| 同一义务标为抽象执行上下文 | Unknown | Unknown | 不能仅凭 SAT 提升可达性；即使测试可造出实例，也不替证明器扩大模型承诺 |
| 将非负 i8 的 `x+1` 错当数学整数 | 抽象模型可得到 ProvenSafe | 正确位宽模型为 ProvenUnsafe | x=127 时实际结果 -128；只有上界 mask 仍会越界 |

最后一项是有意构造的错误抽象对照，不表示当前快速路径会忽略已有的位宽定义。
未发现“当前 ProvenSafe、合法前提下 CPU 实际越界”的案例。该结论受上述有限覆盖范围约束。

`src/tila` 中不存在复杂 DNF 展开/分支截断的执行入口：只有 `predicates.guaranteed`
的小型必然原子提取；SMT 直接消费共享 DAG。无需删除或恢复历史 DNF 实现。

## 预算与失败重放

性质测试显式使用审计预算：单查询 10 秒、rlimit 2,000,000、session 30 秒，默认关闭
缓存；缓存测试单独启用容量 128/16。生产默认值保持不变，既有默认预算测试继续运行。
初始审计使用 2 秒/10 秒；后续 GitHub 托管复验的一项 i8 加法查询耗尽 wall-clock
预算返回 Unknown，因此为语义对照提高测试专用时间预算，不接受 Unknown 充当通过，
也不重试或跳过失败。结论差异记录包含 ProofConfig；CI 上传 M2 失败 JSON/SMT 查询。
固定种子只控制生成器，不保证不同机器的求解时间或任意 SAT 模型一致；重放保存具体模型/输入。

运行专项：

```sh
PYTHONPATH=src python -m pytest -q tests/test_m2_properties.py
```

整数、布尔与执行对照失败时，记录保存到 `artifacts/m2-proof-audit/failure-*/`；可用
`TILA_AUDIT_FAILURE_DIR` 指定其他目录。每次创建新目录，不覆盖之前的失败。
`case.json` 保存种子、操作/类型或生成案例编号、期望/实际结果与精确 pytest 重跑入口；
逐点差异仅保留第一个失败输入，而非整批数组。SMT 数值差异另存该输入的
`query.smt2`；结论差异保存原义务查询。这是最小单输入见证，不承诺全局最小 AST。
缓存/预算等普通 pytest 断言由确定性测试节点直接重跑，不额外生成数值见证文件。

```sh
PYTHONPATH=src:tests python tests/m2_audit_support.py artifacts/m2-proof-audit/failure-XXX/case.json
```

工具重新计算单操作的 Python/CPU 结果，存在 SMT 查询时以有限预算重放，并打印
完整测试重跑命令；组合流水线、控制流和广播案例使用该命令恢复原固定种子场景。
记录与重放链路本身有一个故意注入错误的回归，在 pytest 临时目录完成，不污染工作区。

## 后续

M2-05 已完成本批验收。下一步 M2-06 固定 explain/诊断的审计输出，包括结论与
信任来源分离、候选和确认反例、不可达路径、Unknown 原因、hint 依据及稳定 golden。
一般循环递推、加载相关精确可达性和更广 GPU 语义对照仍保持原有边界。
