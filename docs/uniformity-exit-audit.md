# M4-05d：Uniformity 小域与限定退出审计

日期：2026-09-23。范围：ADR-019 的**内部逻辑值与控制分析**。
结论：**限定退出**；本地全量 CPU 与对应提交的托管 CPU 记录见下文。
不代表 GPU 同步、安全 barrier、物理 CTA/Warp 一致或整个 M4 完成。

## 审计对象与独立性

`tests/uniformity_domain_support.py` 固定种子 `20260923`，对 17 类合法程序模板各用
4 组绑定，共 **68 组 case、3 个 program/case**。绑定包括 i8 边界 -128/-1/0/127、
Const Bool 两值、Const 循环 0/1/3 次及宿主 bool。`tests/test_uniformity_domain.py`
是 pytest 入口；不依赖随机机器状态。模板覆盖：

| 主题 | 具体观察 |
| --- | --- |
| 值来源与分期 | program_id、arange、宿主 i8 回绕、Const bool、load 返回 Unknown |
| 值传播 | rebind、相同定义 where、不同值 where、布尔等价条件、reshape/broadcast、完整与部分归约 |
| 控制 | pid/宿主/Const 分支、selector 合并、提前 return、零次与非零次循环、pid 控制循环、嵌套循环、内存驱动分支 |
| 急切读取 | false 条件 `where(False, load(...), lane)` 仍有一次读取/program；普通 load 与内存驱动分支也记录读取次数 |

独立 Python 参考按模板直接计算整数、布尔、tile 元素和分支/循环结果，不调用
TIR transfer、`uniformity.py`、SMT 或 `Interp`。真实观察来自现有 CPU `Interp`，
仅通过子类记录赋值、induction、merge/loop 合并点和实际读取；运算仍调用解释器。
因此参考可发现分析器错误保证及本域 CPU 求值差异，但观察器与解释器共享 TIR
身份、控制执行入口及部分运算代码，**不是整个编译器的独立形式化语义**。

每个事件记录定义 site、program、逻辑元素坐标、值形状、值和嵌套循环迭代序号。
整数/布尔以同 dtype 定义点上的规范数值比较；本域不含浮点位模式比较。
LaunchUniform 在相同动态实例比较所有实际求值的 program/逻辑元素；
ProgramUniform 在 program 内比较。控制检查与值分开：声明 Launch Full 的
执行实例必须覆盖本域全部 3 个 program；Program Full 必须覆盖其逻辑值形状的
元素。`Reachability=Unreachable` 不能有事件。Varying/Unknown 不要求产生实际
不同值，Unknown 不会因本域样本恰巧一致而升级。

这些断言只检查有限执行中的**必要条件**。逻辑 Full 的作用域、条件化可达性
及合法已定义执行前提仍依 ADR-019；不把某次没有发生的站点当作 Full 证明。
GPU 不支持的 runtime/loop 提前 return 只用于 CPU 内部语义观察，不转化为 GPU
正例。所有真实内存访问均在小域内有效；本轮不验证 bounds 或 Race。

## 发现与修复

发现：可能零次执行的循环，其 induction 或仅在体内定义的变量会产生缺失入口
的循环出口合并引用。旧分析已将该引用的**值**降为 Unknown，却保留循环后继的
Full **控制**，使只要求控制的逻辑消费入口可能错误得到 Satisfied。

修复于 `src/tila/uniformity.py` 的 `joined`：只对缺失入边的**合并引用事实**标记
控制 uncertain，值仍为 Unknown；合法的循环后继语句不被整体污染。
零次循环测试确认 `j`/体内定义的出口引用在 Launch/Program 均为 Unknown，
逻辑 requirement 不能 Satisfied，而后继独立常量仍可保留 Full。
这属于内部事实的保守收紧，不新增活动诊断或 launch 门禁。

## 变形、故障注入和隔离

布尔条件取反与 `where` 条件互补的模板在所有绑定上产生相同具体定义事件；
不要求保守的 Uniformity DAG 表达完全相同。i8 算术只按回绕后的设备整数对照，
不应用实数代数恒等式。零次循环要求保留入口 `pid` 值，嵌套循环分别使用 `(j,k)`
作为动态实例，避免跨迭代混合值。

十一类故障注入：将 pid 的 P 错标 L、lane/广播矩阵/部分归约的 V 错标 P、
load 的 U 错标 P、where 或分支合并省掉 selector、忽略零次循环入口、return 后错标 Launch Full、
pid 控制循环体错标 Launch Full、内存驱动分支错标 Launch Full。独立事件断言
逐一找到反例；重算 verifier 也拒绝伪造摘要，但不以它代替独立观察。
另将一条 tile 事件的逻辑元素参与者故意截短，检查 Program Full 的观察器
会报错。Const Bool/Int 域、跨绑定旧摘要、缺失摘要、预算耗尽与未建模 CTA scope
继续 fail closed；含缺失入边的逻辑消费结果为 Unknown。

## 可复现记录与本地结果

`artifacts/cpu/m4-05d-uniformity-summary.json` 包含固定域、逐 case 记录、完整性、
观察事件与层级/参与计数。选跑为 partial，不能当作全域验收。
失败自动保存在 `artifacts/uniformity-audit/failure-*/`，含 seed、case、Const/宿主
绑定、grid、预算、源码 SHA/源码/TIR、动态事件、逻辑元素参与者、参考结果、
被违反的摘要事实与问题清单。它保留**一组固定绑定，不做全局 AST 最小化**。
单案例重放检查源码与绑定未漂移，并要求事件、读取数和问题清单与记录一致：

```sh
PYTHONPATH=src python -m pytest -q tests/test_uniformity_domain.py
PYTHONPATH=src:tests python tests/uniformity_domain_support.py CASE.json
```

本地专项：**88 项通过，零失败/跳过**，加上 M4-05b/c 既有 64 项共 152 项。
完整审计摘要：**68 case，1570 条事件，36 次实际读取**；已观察定义层级计数
L=124、P=192、V=104、U=16；控制计数 Launch Full=384、Launch Conditional=40、
Launch Unknown=12、Program Full=424、Program Unknown=12。计数取自当前固定域，
不作为未来不同域的语义 golden。全部参考结果和强保证必要条件均通过。

全量本地 CPU 回归及 golden：**1500 passed、99 warnings、零失败/错误/跳过**；
JUnit 为 `artifacts/cpu/m4-05d-full-results.xml`，59 项测试名称/类名含 golden。
本提交的 GitHub 托管 CPU CI/JUnit/审计摘要：**待提交后取得**；旧 C2 run
只作为前置条件，不替代本轮验收。

## 退出边界

限定退出仅说明本固定有限域没有观察到错误的 L/P 或 Full 保证，十一类过强结论
及部分参与者故障能被审计发现；未证明任意程序的 Uniformity 正确性。
没有真实同步消费者、物理线程布局或 GPU 一致性消费对照；`consumers` 仍为
`not-installed`。内存值、浮点运算、dot 等按既有保守规则保持 Unknown，不因
本轮小域通过而放宽。没有新增 barrier/shared memory API、策略开关或活动错误码。
整体 Uniformity/M4 仍 Partial；M3 仍因缺少隔离 GPU CI 而 NOT READY。
