# M4-04c：Race 诊断、启动门禁与绑定隔离

按 [ADR-018](adr/018-minimal-race-analysis.md) 接入 CPU/CUDA 启动前检查，复用
[M4-04b 精确子集](race-analysis.md)。只扩展已有分析的消费路径和同次 tile store
重复 lane 检查，不增加 atomic 操作、barrier 或 uniformity。

## 使用与策略

`TILA_RACE=off|warn|error`，默认 **warn**；CLI `--race` 显式值优先于环境。
它独立于 bounds 的 `TILA_SAFETY` 和 where 的 `TILA_EFFECTS`。

| 策略 | 已确认冲突 | Unknown |
|---|---|---|
| off | 不查询，记录 disabled/未检查 | 不查询，记录抑制来源 |
| warn | TILA-RACE-001，执行前拒绝 | TILA-RACE-002，RuntimeWarning 后允许执行 |
| error | TILA-RACE-001，执行前拒绝 | TILA-RACE-002，执行前拒绝 |

非法策略/访问对预算为 TILA-RACE-003；已有 proof 配置错误仍使用 TILA-PROOF-001。
off 不关闭 verifier、bounds、自然对齐、target 或其他启动契约。
检查位于参数绑定、Const/refinement、assume_launch、target、numeric 和 bounds
校验之后、`_execute` 之前；已有 Triton 编译缓存不能绕过检查。

没有具体绑定的 check/build 不把假想 grid/alias 当作真实冲突，Race 仍 pending；
`tila explain file.py --show-races` 或 `kernel.explain(show_races=True)` 展示符号报告。
默认 explain 章节不变，显式 Race 输出也不借用历史 launch 的绑定。

## 同 program 边界

启动分析同时检查 InterProgram 和 IntraProgram。相同 program 内同一次普通 tile
store 的不同 lane 使用独立 lane 变量，地址重复且共同可达时确认冲突，即使写相同值。
循环中的确认要求两个 lane 属于同次迭代；不同迭代的潜在重叠仍报告 Unknown。
同 program 的普通标量顺序访问不会误报成竞争；兼容 atomic 保留原语义。

不同 site 的向量访问没有完整执行顺序模型，即使逻辑 lane 编号相等也不据此豁免。
若能证明地址不交则通过，否则 `intra-program-order-not-modeled`；包括部分合法的
原地读写模式。rank>1、复杂数据流等仍保守 Unknown。error 模式因此比 warn 更严格，
并不保证现有全部 examples 在 error 模式可运行。

## 输出与重放

新增独立 JSON schema **tila.race-details.v1**，不迁移 effect-details.v2。它按稳定
site 对/分析域顺序记录结果、原因、源行、Read/Write/Atomic、region/dtype、独立的
path/mask/loop、依赖来源、候选/确认分类、未检查范围及显式关闭记录。
默认不打印模型选择、原始查询、缓存遥测或绝对指针；`--show-witness`、`--show-query`、
`--show-cache` 与 `--show-races` 可组合。TILA-RACE-001/002 均指向 Tila 源行；
构建/预算不足尚无访问对时展示首个未检查访问，不伪造一对已确认事件。

成功启动保存只读结果 `kernel.last_race_report` 和稳定 JSON 字符串
`kernel.last_race_details`。每次启动开始清空旧记录；失败不保留旧绑定。
Race 拒绝时异常附带 `race_report` 和详细的 `race_details`（含可选重放材料）；
不把失败候选写回长期 TIR。空 grid 在 host 契约检查后记录 empty-launch，不执行
kernel，不继承此前报告或 alias/alignment overlay；off 模式记录 disabled。

修复建议区分地址分区和累加算法；只有确为加法更新时才建议 atomic_add。
没有 atomic_store，也不能用 bounds 的 unsafe 豁免竞争。

## 缓存与预算

独立进程内 LRU 只保存完整、无 Unknown 的不可变结果（包含已确认冲突）；不保存
tensor 引用或 solver 可变断言。默认容量沿用 `TILA_PROOF_CACHE_ENTRIES=256`，0 禁用。
键包含完整 TIR/定义/effect、源码指纹、实际 dtype/shape/stride/地址/设备、带类型的
Const/标量、grid、target、num_warps/debug、assume_launch 契约、分析/语义/Z3 版本
和预算。地址仅用于私有键；不出现在稳定诊断。无法可靠构造键时禁用缓存。

策略不进入结论缓存键，每次按当前策略重新应用；off 不读写分析缓存。
Unknown/超时/截断不缓存，因此提高预算会重新分析。每次启动契约仍先重验，
同对象、不同 view、alias、Const、grid、源码或 target 改变不会复用错误结果。

`TILA_RACE_MAX_PAIRS` 默认 4096，同时计入两个域的 pair；单查询/总量/构建预算
沿用 `TILA_PROOF_*` 配置值，但 race 与 bounds 的 session 和计数独立。

## 本地验收

- M4-04b 的 39 项及新增 31 项 CPU 专项，共 **70 项通过**：策略、两个分析域、
  同次循环 lane、alias/view/Const/预算/策略切换、缓存容量与来源、失败/空启动、
  CLI 和三个 golden（详细 JSON、错误文本、状态边界）。
- CPU 全量 **1110 passed，零跳过**，`artifacts/cpu/m4-04c-results.xml`。
- RTX 3090 固定环境 **395 节点/539 语义案例通过，零跳过**，新增 7 项真实 CUDA
  门禁/缓存/策略测试。报告：`artifacts/ci-gpu/20260920T092234Z-7go3hchd/report.json`。
- 从前一轮完整验收 `20260920T091318Z-cv7r_dh9` 保存的源码/锁文件重建环境，
  重放 `cuda_alias_rebinding_on_warm_cache`：
  **1 passed、394 deselected**；诊断子集不代替完整验收。

GPU 严格入口固定 bounds=strict、effects=warn、race=warn，并保存环境及重放参数。
完整验收允许明确的 Race Unknown 告警（该 GPU 记录 217 条 warning），不表示所有
通过案例都已证明无竞争；确认冲突的负测试通过，是因为验证了拒绝且未启动 kernel。
未执行真实冲突 kernel 去等待偶发错误值。旧 388/532 基线仍是历史记录。

支持承诺仍仅 ADR-009 的 RTX 3090/SM86 组合。当前无隔离 GPU CI，M3 仍 NOT READY；
当前修改也尚无新的远端 CPU CI 记录。后续 [M4-04d](race-exit-audit.md)已完成
小域枚举、变形与限定覆盖退出审计，不能用阶段通过代替完整并发安全模型。
