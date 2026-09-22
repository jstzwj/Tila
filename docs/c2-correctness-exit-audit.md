# C2：正确性收口退出验收

日期：2026-09-21。范围：C0/C1 后的有界组合验证，不增加语言功能或同步 API。
状态：**PASS（限定退出，托管 CPU 已复验）**。2026-09-22，提交 `7edd591` 的
[托管 CPU run](https://github.com/jstzwj/Tila/actions/runs/35686086849)已通过；
GPU 证据仍仅为下述本地固定环境验收，不将它称为持续 GPU CI。

## 固定的兼容性边界与迁移

| 原先可能被接受的写法 | 当前契约 | 迁移示例或方法 |
|---|---|---|
| 对 WriteOnly Buffer 执行 load | 普通、unsafe、false mask 下均禁止读取 | 同时读写的参数改为 `ReadWrite`；仅读取改为 `ReadOnly` |
| `cast[bool](mask)` 或 `cast[i32](mask)` | Mask 不属于可 cast 的值，不能因此变成 Scalar | 布尔 mask 直接组合；数值选择用 `where(mask, 1, 0)`，得到 Block 后再作合法 cast |
| Const 分支生成长度 4/8 的值，分支后统一消费 | 活跃前驱合并必须 dtype/rank/shape 一致 | 将对应 load/store 移入各自分支，或让两侧生成相同形状 |
| runtime if 或循环体内 return 后依赖后续语句 | GPU 在 verifier 阶段拒绝；CPU 保留退出语义 | 简单运行期条件可把完整后续操作放入 `if not done:`；必须保持分支类型合法。循环退出暂无通用自动改写 |
| `f32 \| Positive` 传入宿主正数 `1e-50` | 先按 f32 ABI 舍入，再检查 Positive，实际零值被拒绝 | 使用转换后仍满足精化的实参；不能用宿主正数替代设备值前提 |
| 依赖窄整数累计指针偏移回绕 | 每步及累计元素/字节位移遵守 checked i64 地址域 | 用可证明合法的偏移表达式；不能借中间溢出后抵消来访问原位置 |

顶层、仅受 Const 分支控制的 GPU return 保持支持。只给最终 store 补 mask，不能
替代保护前面的 load 或其他副作用；`where` 两侧仍急切求值。上述收紧不新增
Mask API，也不恢复跨分支不同形状的条件类型。细节见 [ADR-020](adr/020-correctness-closure.md)。

## 有界生成域与独立参考

`tests/c2_audit_support.py` 生成 **140 个程序模板、1120 组具体绑定**，种子
`20260921` 固定绑定执行顺序。它是固定有限组合，不是任意语法 fuzz 或全位宽枚举。

| 维度 | 覆盖 |
|---|---|
| 整数 | i8/u8，先窄化再加 3/乘 3，最后转 i32；含中间回绕 |
| mask | 重新计算、De Morgan 等价式、重绑定前旧 mask、异号表达式、常假 |
| 控制与形状 | 直线、Const 分支、宿主 bool 运行期分支、循环、Const 提前 return、4×4 广播、Ptr |
| 宿主绑定 | seed 为 -129/-128/-1/0/6/127/255/256，shift 含 -1/0/1/8；两侧分支，循环 0/1/3 次 |
| 布局与地址 | Buffer stride 1/2、非零 view offset、非连续输出及 backing sentinel；Ptr 先 +3 再 -3，之后使用累计索引 |

每模板使用 8 组预先固定的绑定，并非对宿主参数的完整笛卡尔积。模板之间对
family/dtype/op/mask 做完整笛卡尔积。独立 Python 参考使用 `% 256` 和有符号解释
计算中间值，不调用 Tila 的 numeric、DimExpr 或解释器来计算预期索引。

参考记录每个 lane 的坐标和 mask，包括不活跃 lane。CPU trace 使用实际 TIR
表达式与控制流求值，但拦截输入 load：越界地址只记录，不解引用。对照访问与
输出值后，再运行正常 CPU launch；非法候选用禁止进入执行后端的钩子验证门禁。
合法绑定还检查真实 load、非连续写回和 backing sentinel。

每条 load 证明分别与参考比较：Safe 不得包含活跃越界访问；已确认 Unsafe 必须
存在具体越界访问。运行期 bool 的证明覆盖两个取值，不能误当作只承诺当前绑定；
因此参考另枚举另一个 bool 取值。数值定义域先经过 numeric gate，未将未定义运算
当作 bounds 前提。这里只有一处输入 load；正常输出 store 的坐标由固定 lane 决定。

测试专用预算为每查询 2000ms、rlimit 200,000、session 10000ms，proof cache 关闭。
除缓存容量外，各上限均不小于生产默认值，并有断言防止以后生产预算增加而审计漏跟；
不通过更小的工作预算回避生产中可能得到的确定结论。
该批用于发现错误结论，允许资源或建模限制导致 Unknown，并逐项记录原因和
正常启动是否成功；不强求布尔等价式具有相同求解速度或相同完备度。
直线、广播、Ptr 的 fresh/false mask 正例必须实际启动成功，防止“全部拒绝”通过审计。
生产默认预算与策略没有修改。绑定复用同一份已检查 TIR；通用证明缓存隔离仍由 C0/M2 专项覆盖。

## 故障注入与重放

四类注入分别测试审计本身的敏感性：伪造 Safe、伪造已确认 Unsafe、CPU cast 丢失
tile 形状、CPU 指针读取起点偏移。它们验证 oracle 能拒绝错误结果，不宣称重新
执行了所有历史缺陷版本。另有布尔等价的参考对照和单案例捕获/子进程重放测试。

失败保存到 `artifacts/c2-audit/failure-*/`：case JSON、固定种子、生成源码、TIR、
shape/stride/view offset/值、Const 与标量绑定、预算、证明查询、失败原因和首个
非法事件。保留一组失败绑定；**不声称完成全局 AST 最小化**。生成器或预算变化
会明确拒绝混用重放，应使用审计时保留的源码。失败不通过 retry、skip/xfail 消除。

```sh
PYTHONPATH=src python -m pytest -q tests/test_c2_exit.py
PYTHONPATH=src:tests python tests/c2_audit_support.py CASE.json
PYTHONPATH=src python tools/gpu_audit.py --output artifacts/correctness-gpu
```

CPU 汇总为 `artifacts/cpu/c2-summary.json`，含 complete、结论计数、实际启动、
门禁检查和逐绑定记录；选跑只有 partial，不能替代完整验收。GitHub 托管 CPU
工作流已加入该摘要及 C2 失败目录。GPU 审计将 C2 失败材料保留在对应 run 内，
查询、源码快照、pytest node 和生成 Triton 源码一同保留；CUDA case 重放执行
其原始 GPU 测试节点，需要 GPU 环境。

同时修复 `tools/replay_gpu_audit.py` 漏掉 `gpu_soundness.py` 的问题，并接入
`gpu_c2.py`。存在这些文件的新快照必须重放它们；没有这些文件的旧快照继续兼容。
新增正反夹具检查两类快照的命令与产物目录，不安装依赖或伪造 GPU 执行证据。

## 验收结果

新增 GPU 集合为 **56 项正常 CPU/GPU 对照＋28 项启动门禁负例**，覆盖上面的
7 类程序、两种 dtype 及回绕边界；GPU 正例全部使用 fresh mask，不将其他 mask
的 Unknown 自动解释成已支持的 GPU 正例。严格入口总计应为 **501 节点 / 645 语义案例**，
计算为既有 417/561 加本轮 84 项；4 个历史聚合节点内部有 148 个案例。

最终全量 CPU：**1412 passed、99 warnings、零失败/错误/跳过**，包含新增 148 项
及全部既有 golden（59 项测试名称/类名含 golden）。JUnit：`artifacts/cpu/c2-results.xml`。
最终完整 GPU：**501 passed、219 warnings、零失败/错误/跳过**，固定环境无漂移，
`accepted=true`。记录：`artifacts/correctness-gpu/20260920T231345Z-m3egpwlv/report.json`。
已有 GPU warnings 不等于一般 Race 证明；C2 单 program 专项显式 race=off。

最终 `artifacts/cpu/c2-summary.json` 的 complete=true、status=passed：

| 指标 | 本次结果 | 含义 |
|---|---|---|
| 程序模板/绑定 | 140 / 1120 | 全部预先固定的组合均检查 |
| 独立 load 义务查询 | 677 ProvenSafe、443 Unknown、0 ProvenUnsafe | 未发现错误 Safe 或错误已确认 Unsafe；不把 Unknown 算成 Safe |
| Unknown 原因 | 407 个 Z3 canceled、36 个未确认可达的候选 | 保留原始 reason，不把 canceled 细分为尚未记录的 timeout/rlimit 类别 |
| 独立参考中的实际越界绑定 | 67 | 正常 CPU 启动全部在进入执行后端前拒绝 |
| 正常 CPU 启动 | 676 成功、444 拒绝 | 成功值/布局/sentinel 与参考一致；拒绝均为 bounds 诊断 |
| GPU 新增 | 56 正例、28 负例 | 前者真实执行与 CPU/参考一致，后者未进入执行后端 |

独立查询与正常启动各自重新证明。22 个独立 Safe 在启动时仍被保守拒绝，
21 个独立 Unknown 在正常启动重新取得证明后成功；这不表示 Unknown 可以直接
放行。预算与事实上下文会影响完备度，具体结论计数是本次记录，不作为稳定 golden。
**1120 组审计通过，指对结论、trace、值和门禁的断言通过，不是 1120 组均判 Safe。**

第一批不含 runtime 分支的 GPU 子集 72 项也通过，但未计作完整验收。一次 CPU
专项因磁盘空间耗尽导致报告写入失败；中断的测试文件已恢复，已有产物保留并放回
原路径，之后重新完整验收。初始较低资源预算的完整记录不作为最终退出依据；
最终将工作上限提高到不小于生产配置后，再次通过上述 CPU/GPU 全量。

前置提交 `0906bc6` 的 [CPU 托管验收](https://github.com/jstzwj/Tila/actions/runs/35521460584)
已通过 1264 项，golden、JUnit 与 Race 枚举附件已核对；不把该旧提交的结果当作 C2 的托管验收。

2026-09-22，C2 提交 `7edd591` 的上述托管 run 通过 **1412 项、零失败/错误/跳过**。
job labels 为 `ubuntu-24.04`、runner group 为 GitHub Actions。下载的 `cpu-test-report`
已核对 JUnit、59 项名称/类名含 golden 的记录、Race 摘要与 C2 完整摘要，副本在
`artifacts/cpu/hosted-7edd591/`。远端 C2 为 140 模板/1120 绑定、complete=true、
status=passed：675 ProvenSafe、445 Unknown（409 canceled、36 未确认可达候选），
672 次正常启动；67 个实际越界绑定全部在执行前拒绝。远端与本地的预算内结论
计数不同，均未把 Unknown 当作 Safe；这些计数不是必须逐次相等的 golden。

## 退出边界

本轮退出要求：约定域内无错误 Safe/已确认 Unsafe，CPU trace 与独立参考一致，
GPU 正例无未解释的编译或值差异，负例在启动前拒绝，故障注入和失败重放通过。
Unknown 不作为 Safe，也不要求消灭所有 Unknown。

不覆盖一般加载内容参与控制流、任意循环、全部 shape/dtype、别名/并发执行或
全编译器形式化正确性。runtime/loop return 的 GPU 限制仍保留。测试只在本地
固定 RTX 3090/SM86 环境提供 GPU 证据，独立 GPU CI 缺失，M3 继续 NOT READY。
2026-09-22 的 [恢复评审](m4-05d-resumption-review.md)已通过，M4-05d 可启动，
范围仅为内部值/控制分析的小域与退出审计；本轮没有实施该阶段或恢复其他功能扩展。
