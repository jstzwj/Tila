# M4-05c：可选 Uniformity 输出与绑定隔离

日期：2026-09-20。基于 [ADR-019](adr/019-minimal-uniformity.md) 和
[内部分析](uniformity-analysis.md)，新增只读可选审计。Uniformity 整体仍 Partial：
没有同步消费者、物理 CTA/Warp 映射或 barrier/shared memory API。

## 入口与策略

`tila explain FILE --show-uniformity`、`tila check FILE --explain --show-uniformity`
或 Python `kernel.explain(show_uniformity=True)` 在原输出末尾添加
`uniformity-details:` JSON 章节。可与 effects/races/query/witness/cache 选项组合；
后面三者不改变 Uniformity JSON。未开启时不执行 Uniformity 分析，默认 explain
和已有 golden 保持兼容。入口不编译、不执行 kernel、不修改 launch 报告。

本轮审计策略决定：以显式展示开关作为独立入口；没有消费者时，正常 Varying 或
Unknown 不产生告警或启动错误。因此不新增无实际用途的 `--uniformity off/warn/error`、
`TILA_UNIFORM` 环境变量或活动 UNIFORM 错误码。未来真实消费的硬性合法性检查
仍不能被可选审计策略关闭；相关诊断注册与物理 scope 契约在消费者落地时另行评审。

## 稳定 JSON 契约

schema 为 **`tila.uniformity-details.v1`**，由 `src/tila/uniformity_audit.py` 产生。
字段/枚举含义变更需要显式版本或兼容迁移，不能仅更新 golden 隐藏差异。

| 字段组 | 语义 |
| --- | --- |
| schema/kernel/stage/parameters/consts | 当前 TIR、参数类型和本次 Const；Bool/Int 显式区分 |
| scope/requires_defined_execution | 逻辑 Launch/Program；仍以合法且已定义执行为前提，不替代 bounds/数值检查 |
| provenance/assumptions/launch_bindings/cache | verified TIR and Const only；assume 不加强；launch 绑定排除；每次 recomputed |
| budget/complete/incomplete_reason | 确定性分析预算及是否完成；complete 只表示遍历/固定点完成，不等于所有事实满足消费者要求 |
| consumers/consumer_status | 当前固定为空和 not-installed，不能伪造“所有同步检查通过” |
| facts | 稳定 site 字典序；定义引用、源行号、值层级、输入依赖、规则、原因、来源及独立 control |
| accesses | TIR 访问顺序；kind/RegionId/dtype/源位置及独立 path、mask、loop；不含真实地址 |
| exit_control/predicates | 出口参与/可达性，以及共享 predicate DAG，按首次结构访问分配 p0/p1 等局部 ID |

control 固定包含 selection_level、Launch/Program participation、reachability、path、
selectors、loops、returned、iteration_sources、reason。语句没有值时 value_level=null，
不能将 null 理解为 LaunchUniform。Varying 是未建立 program 内一致保证，不是已确认分歧。

内部 `uniformity_details(kernel, consts=None, config=...)` 支持 symbolic/partial/specialized，
无缓存；低预算可通过内部 UniformityConfig 验收。公共 explain 沿用既有 Const 默认值
解析与精化检查，缺失必需 Const 仍报原错误，不擅自将历史 launch 参数带入。
因此普通 explain 的此章节通常为 specialized，绝不显示为已绑定 GPU launch。

构造预算耗尽可能只有部分事实或空表；complete=false，缺失事实须按 Unknown 处理。
本轮修正内部显示边界：预算未完成或循环退出/参与未建立时，出口可达性为 Unknown，
不能因初始化 path=true 就显示 Reachable。Unreachable 与 None participation 只用于
已确立的死路径，不作为已满足消费要求的证据。

## 消费契约测试边界

内部 `evaluate_requirement` 先重新验证摘要，再组合值约束和逻辑参与要求，返回
Satisfied、Unsatisfied、Unknown 或 NotApplicable；不可达点为 NotApplicable。
缺失事实、预算、未知保证和未建模物理 scope 都不能得到 Satisfied。
只有 Satisfied 允许测试 fixture 执行其模拟操作，且实际消费者仍须独立验证执行合法性。
这不是新增同步 API，也未接入 lowering/launch；生产 JSON 明确没有消费者。

## 隔离与验收

新增 `tests/test_uniformity_audit.py` **21 项专项**：

- 完整 JSON、explain 文本与状态边界三份 golden；覆盖四层级、symbolic/特化、
  零次循环、提前 return、预算耗尽和空事实表；
- CLI explain/check、Const Bool 精确域与失败调用、跨进程 hash seed、组合 flags；
- 同一 TIR 的不同 Const/预算、已修改的 TIR、冷/热编译缓存，不复用旧层级；
- 同址/分离/stride view、成功/空 launch，以及缺参/dtype/grid/执行失败后的审计；
- 延迟开启且只读；真实地址不进入输出；逻辑消费 fixture 拒绝 Varying/Unknown、
  未建模 CTA、预算和伪造摘要。

没有 `last_uniformity_report` 绑定快照；从当前 TIR/Const 重算，使 failed/empty launch
无需清理一个新的历史事实容器。不会把未建立的 GPU 绑定等价性伪装成缓存命中。

本地 CPU 全量 **1191 passed，零跳过**，含内部分析与审计共 64 项专项；记录为
`artifacts/cpu/m4-05c-results.xml`。原有默认 explain/golden 回归同时通过。

```bash
PYTHONPATH=src python -m pytest -q tests/test_uniformity.py tests/test_uniformity_audit.py
```

本轮没有新的 GPU 同步证据或远端 CPU CI 记录。既有 RTX 3090 数值/atomic/Race
验收不替代 uniformity 同步验收。下一步 M4-05d 做小域/变形、故障注入和退出审计；
整个 M4 及缺少隔离 GPU 持续验收的 M3 均不能因本阶段通过而宣称完成。

2026-09-22 的 [M4-05d 恢复评审](m4-05d-resumption-review.md)补充 C0/C1/C2 后的
实施边界与退出要求：优先独立观察值/控制保证，再做变形、故障注入及可重放审计。
现有重算 verifier 不构成独立 oracle；M4-05d 尚未实施，真实同步消费者仍未安装。
