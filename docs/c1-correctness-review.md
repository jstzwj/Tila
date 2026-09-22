# C1：正确性收口的独立边界复核

日期：2026-09-20。范围：C0 后的邻接反例，暂不恢复 M4-05d 或新增语言功能。

## 提交与托管验收

先前 M4 Race/uniformity 成果整理为 `5ba6024`，C0 修复为 `4aa13e9`，均已推送 main。
`4aa13e9` 的 [CPU regression](https://github.com/jstzwj/Tila/actions/runs/35519623735)
已成功：GitHub 托管 `ubuntu-24.04`，1217 项、零失败/错误/跳过。
已下载 `cpu-test-report` 并核对 XML、59 项名称/类名包含 golden 的测试记录及
Race 枚举摘要；本地副本在 `artifacts/cpu/hosted-4aa13e9/`。不连接开发者机器。

本轮 C1 修复提交 `785df52` 已推送 main，其
[独立托管运行](https://github.com/jstzwj/Tila/actions/runs/35521016183)也已通过：
1264 项、零失败/错误/跳过，JUnit、59 项 golden 测试记录与 Race 枚举摘要已下载核对。
本地副本在 `artifacts/cpu/hosted-785df52/`；托管 runner 身份已核实。

## 复核结果及修复

| 边界 | 发现与处置 | 证据 |
|---|---|---|
| WriteOnly 读取 | Buffer 漏掉读权限检查，materialize 原可成功；新增前端拒绝及 verifier 独立检查 | 普通/unsafe/false mask/where 急切分支/other 嵌套读取、Ptr 对照；伪造新 metadata 的 TIR 也拒绝 |
| Mask cast | 原实现将 Mask 当 Scalar，甚至允许其进入 if；按 ADR-012 既定规则拒绝，不新增 Mask 转换 | Mask→bool/i32 均拒绝；verifier 拒绝 Mask cast 及布尔 tile cast 丢失 shape |
| 合法布尔 tile cast | Block[bool] 保持完整 rank/shape；bool→bool 保留布尔身份，数值→bool 不制造 bounds 事实 | 4×4 where 值选择再 cast、尾 mask、重绑定与加载值反例；CPU/GPU 对照 |
| Const 分支 return | CPU 正确，但 Triton 仍检查后续语句，使已返回分支的 shape 污染后续编译 | lowering 把 continuation 放到各活跃分支并在 return 截断；8 个 GPU 组合，source map/原 TIR 保持 |
| runtime if / loop 内 return | 实测 Triton 分支类型合并/SCF loop 编译失败；不把后端异常当作支持 | GPU verifier 提前 TARGET-009 拒绝，4 个真实 CUDA 绑定负测试确保未调用执行后端；CPU 正反例继续通过 |
| Ptr/view/负偏移 | 在已测组合内未发现新的错误接受 | view 起点重定位、先正后负位移、访问 view 前元素拒绝、masked offset 六组绑定、负 stride Ptr 拒绝、零次循环保持原指针 |
| 分支/循环类型 | C0 保守规则在新增组合中保持 | 12 组 Const/提前 return/零次或多次循环、循环内 shape variant 拒绝；运行期及循环内 return 的 CPU 语义单独验证 |

runtime if 或循环中出现 return，GPU 门禁采用结构性保守拒绝，包括实参恰使路径
不执行的情况；不由“碰巧不进入”推定生成程序可编译。顶层和仅受 Const 分支控制
的 return 仍支持。完整运行期退出转换需单独设计，不在本次审计中扩大范围。

一项旧 frontend alias 测试把同时读写的 Buffer 标为 WriteOnly，已改为 ReadWrite，
保留别名解析断言；新增权限负测试固定禁止行为。verifier 识别独立的 Const 参数
引用，避免将缺少 runtime dtype 的合法 staged cast 错判为非数值对象。
旧 source-map 测试中的循环内 return 改为嵌套 assume，保留内层语句及 else 的
源位置断言；return 的源位置映射由新的合法 Const 分支测试覆盖。

## 验收与重放

CPU 新测试：`tests/test_c1_review.py` 及 `test_fix_return.py` 的目标门禁反例。
GPU 新测试集成到 `tests/gpu_soundness.py`，严格入口总节点数增加至 417。

```bash
PYTHONPATH=src python -m pytest -q tests/test_c1_review.py tests/test_fix_return.py
PYTHONPATH=src python -m pytest -q --junitxml=artifacts/cpu/c1-results.xml
PYTHONPATH=src python tools/gpu_audit.py --output artifacts/correctness-gpu
```

RTX 3090/SM86 固定环境完整严格验收：**417 个节点 / 561 个语义案例通过，
零失败/跳过，accepted=true**。记录：
`artifacts/correctness-gpu/20260920T154500Z-34uua1ba/report.json`。
其中 11 项新 GPU 正例与 4 项新启动门禁负例分别验证支持行为和拒绝边界。
GPU 子集的失败与后续成功记录均保留；失败记录含实际生成源码、编译原因和
重放参数，没有对失败项作 skip/xfail。

CPU 全量：**1264 passed、99 warnings、零 skipped**，包含全部既有 golden，
较 C0 新增 47 项。记录：`artifacts/cpu/c1-results.xml`。
本轮代码提交后的托管 CPU 验收独立于前述 C0 run，另在 [CPU CI](cpu-ci.md) 登记。

后续文档提交 `26d0f45` 的[托管复验](https://github.com/jstzwj/Tila/actions/runs/35521201375)
出现 1 项失败、1263 项通过：M2 小位宽性质用例 `False-+-i8-1` 的 2 秒查询预算
耗尽，得到 `Unknown: canceled`，没有错误 Safe。失败 JUnit 保存在
`artifacts/cpu/hosted-26d0f45-failed/`。仅将该性质测试模块的审计时间预算调为
10 秒/查询、30 秒/session，rlimit 仍 2,000,000；确定结论断言、零预算反例和
生产默认预算均保留。另补 M2 失败 JSON/SMT 的 CI 上传路径，结论差异记录带预算。
修正提交 `0906bc6` 的[托管复验](https://github.com/jstzwj/Tila/actions/runs/35521460584)
已通过 1264 项且零跳过，附件已核对；没有把 Unknown 视为通过或隐藏失败。
后续组合覆盖与兼容性迁移见 [C2 退出审计](c2-correctness-exit-audit.md)。

## 结论边界

本轮覆盖上述具体权限、cast、控制流与地址组合，并收紧未兑现的 GPU return 声明。
不表示所有表达式或控制流已获得形式化证明，也不补齐同步、所有权或全域 Race。
恢复下一阶段前应先评审这些兼容性变化；独立 GPU CI 缺口仍在。
