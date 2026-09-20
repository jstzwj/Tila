# M4-04d：小域枚举、变形测试与 Race 退出审计

日期：2026-09-20。结论：**M4-04 的限定覆盖验收完成**；整体 race 能力仍为
Partial，不代表一般 GPU 程序无竞争，也不关闭 M4 的 uniformity 或 M3 的持续 GPU CI 缺口。
契约见 [ADR-018](adr/018-minimal-race-analysis.md)，启动语义见
[M4-04c](race-launch-audit.md)。

## 独立事件枚举

`tests/test_race_audit.py` / `tests/race_audit_support.py` 固定种子 **2040426**。
种子只改变仿射案例顺序；表中的笛卡尔积全部枚举，不做随机抽样承诺。
先用 Python 任意精度整数生成活跃事件，再用 CPU 解释器求值并截获 store；
截获器不写内存。按实际 view 的相对基地址、stride 和 dtype 字节宽度检查重叠，
逐对比较 InterProgram/IntraProgram 结论，而非仅比较 kernel 总结论。

| 家族 | 有限域 | 分析次数 |
| --- | --- | ---: |
| 仿射及等价变换 | grid 0..4；STEP -2/0/1/4/7；SCALE 0/1/2；BIAS -1/0/14；原式与反转 lane、交换加乘操作数、德摩根 mask | 450 |
| u8 narrowing | grid 1/2/4；STEP 0/1/255/256；SCALE 0/1；BIAS i32 MIN/-1/0/255/256/MAX | 144 |
| 控制流 | grid 0/1/3；STEP 0/1/4；END 0/1/3；提前 return 的 LIMIT 0/1/3 | 81 |
| alias/部分字节重叠 | grid 1/2/4；i8 view offset 0/15/16/17/20/32、stride 1/2/4/8；与 offset=16 的 i32 view 比较；每例新分配重跑 | 144 |
| 合计 | 747 个不同归一化输入，包含 72 次重复分配验证 | **819** |

检查 Safe 没有遗漏的事件对，Unsafe 的 witness 确实包含活跃的 program/lane、
重叠字节和有效循环迭代。相同写入值不豁免竞争。scalar 顺序访问不被当作
同 program 竞争。非循环精确域要求结论一致；窄整数家族允许明确的
`numeric-gate-not-confirmed` Unknown，绝不要求绕过独立数值门禁去确认 SAT。
这里的循环仅为固定次数、无循环内提前 return/携带地址的标量访问。

审计预算固定为单查询 2000ms、总计 10000ms、rlimit 2000000、禁用 proof 缓存，
其他上限沿用 ProofConfig。生产默认预算未放宽。现有预算/缓存专项另行验证生产边界。
三项故障注入分别伪造 Safe、错误字节地址和同 program 的跨 program witness，
确认审计器能发现这些错误。

## 发现与修复

德摩根变换发现 bool mask 的 `~` 在 Race 表达式编码器中未处理，原先会从确认冲突
退化成 `candidate-path-not-confirmed`。现补充仅 bool/Mask 类型的逻辑取反，
整数位取反不会误用逻辑规则；缓存分析版本由 2 升为 3。450 次变形分析覆盖修复。
没有发现枚举范围内的错误 Safe 或无效确认 witness。

本次 819 次分析中：**466 ProvenSafe、343 ProvenUnsafe、10 Unknown**。
10 次 Unknown 均为 `numeric-gate-not-confirmed`，没有将它们计作安全证明。

i32 边界再 narrowing 的部分案例仍受数值门禁限制，明确返回 Unknown；这是
精确覆盖限制，不作为“全部反例已确认”的证据。CPU 求值和事件枚举仍核验这些输入。

## 失败与重放

失败保存到 `artifacts/race-audit/failure-*/case.json`，包含种子、单个归一化输入、
至多一个冲突事件对、各对结论、candidate/confirmed witness 及原始 SMT 查询。
这里“最小”指单个输入/事件对，不声称全局最小 AST。路径可通过
`TILA_RACE_AUDIT_FAILURE_DIR` 更改。记录不含真实进程地址或 tensor 内容。

```bash
PYTHONPATH=src:tests python tests/race_audit_support.py CASE.json
PYTHONPATH=src python -m pytest -q tests/test_race.py tests/test_race_policy.py tests/test_race_audit.py
```

重放重新构造同一 view 布局和 kernel，重新比较事件及当前分析结果；不把保存的
旧 verdict 当作真值。已通过故障记录/子进程重放测试。正常运行保存
`artifacts/cpu/m4-04d-race-summary.json`，逐例记录结论与原因；该文件只描述
实际执行的输入，运行测试子集时不是完整覆盖报告。GitHub 托管 CPU 工作流已配置
上传该记录及失败目录；本轮尚未推送，不能宣称已取得远端成功记录。

## ADR-018 退出核对

| 要求 | 证据与结论 |
| --- | --- |
| program/lane 身份、尾 mask、有限位宽 | 新增枚举及原 `test_race.py` 的中间回绕、三轴 grid；限定域通过 |
| alias、stride、view、不同 dtype 字节重叠 | 新增混合 i8/i32 枚举；原读写/stride-hole/Ptr 反例；通过 |
| 分支、return、Const、循环、急切操作数、assume | 新增循环/return 枚举；原 where/false-mask 内层 atomic、Const 死分支、assume 不隐藏冲突；通过 |
| atomic 同址兼容、普通访存混用 | 原 atomic/普通访问对与 atomic CPU/GPU 验收；通过；不证明算法原子性 |
| 候选与确认、未知可达性 | 原间接地址/opaque guard/loop-carried/循环内 return 仍 Unknown；通过 |
| 策略、门禁、预算、缓存、绑定隔离 | `test_race_policy.py` 31 项及原预算反例、3 份 golden；通过 |
| GPU 不启动已知竞争 kernel | `gpu_race.py` 7 项真实 CUDA 门禁/缓存测试；通过；不执行冲突 kernel 观察偶然错误值 |
| 可复现审计 | 新增 18 个 pytest 节点，819 次枚举/变形分析、故障注入和单输入重放；通过 |

本地 CPU 全量 **1128 passed，零跳过**，包括 88 个 Race pytest 节点；记录为
`artifacts/cpu/m4-04d-results.xml`。新增枚举摘要状态 passed，共 819 条观察记录。
RTX 3090 固定环境严格复验 **395 节点/539 案例通过，零跳过**，报告为
`artifacts/ci-gpu/20260920T100156Z-66hvg4so/report.json`。
GPU 仍以 race=warn 验收（本次 217 条 warning），允许明确 Unknown 告警，
不能将数值测试通过解释成无竞争证明。

## 尚未覆盖与交接

一般跨 site 向量顺序、间接地址/依赖内存内容的路径、复杂 merge/loop-carried、
多维/未支持布局、完整并发内存模型保持 Unknown 或不支持。枚举只覆盖表中有限域，
不能推广到所有 dtype/shape/grid/target。跨 kernel/stream/host、shared memory、
barrier 和 uniformity 不属于此次证明。

M4-04a/b/c/d 可按限定契约收口；整个 M4 仍缺 uniformity 独立契约与实现。
下一步宜先形成最小 uniformity 设计 ADR，明确依赖层级和未来 barrier 的消费边界，
不直接增加 barrier/shared memory。M3 仍 **NOT READY**：缺隔离 GPU 持续验收。
