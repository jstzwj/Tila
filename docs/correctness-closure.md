# 正确性收口与 Safe 规则审计

日期：2026-09-20。状态：**PASS（本轮限定收口）**。
决策见 [ADR-020](adr/020-correctness-closure.md)。
本轮暂停下一阶段扩展，按错误 Safe → 实际 ABI 精化 → 静态类型/地址域 → 规则审计实施。

## 已复现问题与修复

| 反例 | 原行为 | 修复后 |
|---|---|---|
| 长度 4，写 `2*i`，mask 为 `-2*i < 4` | canon 丢负号，错误 ProvenSafe，实际访问 0/2/4/6 | 保留符号；check/launch 拒绝，GPU 后端不会被调用 |
| `f32 | Positive` 接收 `1e-50` | 宿主检查通过，CPU 实际值为零 | 先按实际 ABI 舍入，精化拒绝，输出不变 |
| Const 两分支分别生成长度 4/8，分支外消费 | 使用 then 代表类型，特化后才暴露广播错误 | 不同形状标成 variant，使用点 TYPE-020；消费移入分支仍可运行 |
| i32 tile 指针偏移累计 `i + MAX + MAX + 2` | CPU 回绕为 i，数学偏移为 `2**32+i` | CPU/GPU 使用 checked i64 地址域；大偏移再用 i64 抵消后读回原元素 |
| 嵌套 else 分支携带 variant | 外层可能丢掉标记，或 canon(None) 崩溃 | 两侧标记合并，hint 仅在两侧都有且相等时保留 |

精确整数到浮点另覆盖双重舍入：`2**80 + 2**56 + 1` 直接转换 f32，
不能先丢失 binary64 精度。地址检查同时覆盖每步及累计字节位移，包含中间溢出后抵消。

## Safe 出口清单

本表穷举 `facts.fast_obligation/_prove_clause_detail` 与 `solver.ProofSession`
中的 bounds Safe 返回分支；不是整个语言的形式化证明。所有行都依赖 ADR-020 的
合法绑定、数值定义域、值身份/作用域前提。Exempted/Unknown 不纳入 Safe。

| 规则 | 必须成立的蕴含/额外前提 | 对照证据 |
|---|---|---|
| 直接谓词上界 | 可用原子确实蕴含 `coord < extent`，另证非负 | signed mask 反例；450 组符号/偏置/关系小域枚举；原始 SMT 蕴含 |
| 数值区间 | coord 下界非负，coord 上界严格小于 extent 下界；区间须包住真实值 | interval 规则原始 SMT；枚举；已有整数溢出测试 |
| 互补/常假原子不可达 | 活跃 path/mask/事实确实矛盾；不得因规范键碰撞制造矛盾 | complement/constant-false 查询；异号“伪矛盾”反例；用户假设隔离 |
| 精确 grid | 已验证 `grid == extent`、合法 pid 和非负下界 | exact-grid 查询；`test_proof_model` 未验证契约 Unknown/逐次重验 |
| cdiv/grid | 正 step、lane 范围、对应 grid 与整除契约，以及 coord 非负 | cdiv 原始 SMT 域检查与蕴含；原 bounds/launch 测试 |
| SMT 不可达 | 运算定义域先被证明；premises UNSAT；矛盾用户假设不得算无条件 Safe | smt-unreachable；`test_smt_solver`/`test_proof_model` |
| SMT bounds UNSAT | 有效编码上的 `premises ∧ ¬goal` UNSAT，Unknown/SAT 不得提升 | smt-boolean；既有小位宽枚举、布尔/控制流变形测试 |
| 缓存命中 | 同编码版本、公式、来源、绑定、预算/域和可达性语义；启动门禁仍执行 | signed-predicate cache 隔离；既有 solver/race/launch 缓存反例 |

快速规则保留。新增测试直接构建 Encoder 的原始 premises/goal，再独立发起 Z3 查询，
不会让同一 fast return 充当自己的 oracle。规范化另用 Python 整数求值检查 715 个
表达式、每个 81 组赋值，共 57,915 次求值；比较同键表达式的真实结果，包含负系数、
交换、非线性不透明项。此枚举限于定义良好的表达式，原始定义域另由数值/SMT 门禁测试。

已有 `test_smt_solver.py` 覆盖超时、构建/查询预算与 Unknown 不缓存；
`test_proof_model.py` 覆盖来源与 assume 作用域；`test_integer_semantics.py` 和
M2 审计覆盖普通机器运算、cast 及中间溢出。新增用例在 `test_soundness_closure.py`，
GPU ABI/地址与失败不执行用例在 `gpu_soundness.py`，已加入严格审计入口。
新增 `golden/soundness-explain.txt` 固定异号 mask 的 Unknown/候选反例边界，
不将尚未确认可达的 SMT 候选改成 ProvenUnsafe；默认 strict 仍拒绝启动。

## 复现与验收

```bash
PYTHONPATH=src python -m pytest -q tests/test_soundness_closure.py tests/test_fix_staticif.py tests/test_smt_solver.py tests/test_proof_model.py
PYTHONPATH=src python -m pytest -q --junitxml=artifacts/cpu/correctness-closure-results.xml
PYTHONPATH=src python tools/gpu_audit.py --output artifacts/correctness-gpu
```

最终全量验收：CPU **1217 passed、97 warnings、零 skipped**，包含新增 26 项
正确性专项与全部既有 golden。记录：`artifacts/cpu/correctness-closure-results.xml`。
RTX 3090 固定环境 **402 节点/546 语义案例通过、218 warnings、零 skipped**，
`accepted=true`、环境无漂移。完整 GPU 记录：
`artifacts/correctness-gpu/20260920T134805Z-7pk50mz_/report.json`。
四个聚合节点内部计 148 案例，因此语义案例数为 `402 - 4 + 148 = 546`。

GPU warnings 仍为限定 Race 域的 Unknown，不代表全部 kernel 无竞争。
审计产物包含源码、环境、失败材料及 replay.json；保留首次专项的测试夹具错误、
首次全量中 numpy.float64 标量兼容性回归的失败记录。修复后重新完整复验，未跳过失败项。
这些是本地工作区证据；本轮尚无对应提交的远端 CPU CI，隔离 GPU CI 仍未建立。

## 剩余边界

本轮只收口已确认的反例、类型/数值接口及 bounds Safe 规则。受限 Race 和内部
uniformity 的既有测试仍回归，但不据此推导一般无竞争或同步正确。CPU/GPU 对照
只覆盖列出的固定组合和用例，有限枚举不能证明所有程序。继续保留完整编译器
正确性、复杂控制流/别名、并发同步和独立 GPU CI 的未完成边界。
