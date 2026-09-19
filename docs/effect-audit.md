# M4-01d：详细输出与缓存／绑定隔离

2026-09-20。ADR-016 的详细输出与隔离验收已实现；不包含 atomic、race 或
uniformity，也不改变 M3 尚缺隔离 GPU 持续验收的状态。

## 输出契约

`kernel.explain(show_effects=True)`、`tila explain file.py --show-effects` 和
`tila check file.py --explain --show-effects` 在 effects 与 aliases 之间增加
`effect-details` 节。默认 explain 格式不变。

节内为 JSON，schema 固定为 `tila.effect-details.v1`。字段顺序稳定：阶段、
带类型的 Const 绑定、来源与排除边界、谓词表、按 TIR 遍历顺序的访问表。
`symbolic` / `partial` / `specialized` 描述特化阶段；公开 explain 使用解析后的
完整 Const 绑定。内部 `effect_details(tk, consts)` 也允许符号或部分特化。

每条访问记录 site、Read/Write、RegionId、元素 dtype、地址空间、源码片段行号、
may-access/excluded、独立的 path/mask 引用以及外到内的 loop 列表。循环含
归纳变量定义引用、起止步长 TIR 表达式与 may-enter 谓词。site 是当前 TIR 的
结构位置，不承诺源码编辑后保持不变；line 沿用 kernel 片段行号。

谓词按确定性首次访问顺序编号，共享 DAG 只输出一次，不作指数级展开。
unknown 保留条件 site、定义点和迭代上下文；代表无法静态确定，不能据此声称
实际必然访问。excluded 仅由 path 或 mask 的确定 false 得出；嵌套读取独立记录。
不输出对象地址、宿主内存地址、Z3 模型或缓存命中细节。未知谓词类型明确报错。

## 隔离与生命周期

每次明细请求都重新验证 TIR 并派生汇总，无 effect summary 缓存。编译缓存、
证明缓存、用户 assume、runtime alias 和 alignment 均不作为删除访问的事实。
Const[bool] 与 Const[int] 精确区分，不接受 bool/int 隐式互换。

运行时 alias/alignment 是成功且非空启动的诊断快照；每次启动先清除旧证据，
任何阶段失败或空 grid 返回均清除部分绑定证据。成功绑定的 RegionId 不变，
实际 alias 关系仍可随同一数组、独立数组、stride/view 而变化。该快照不是并发
线程隔离 API；不在本阶段承诺同一 JITFunction 的并发启动诊断隔离。

explain 不编译或启动 kernel，不改变上次启动报告、资源、hint 或编译缓存。
本阶段修复了失败/空启动可能遗留 runtime_aliases，以及中途失败遗留部分
runtime_alignments 的问题。已提交执行但随后报错时，清除诊断不意味着回滚内存。

## 验收

`tests/test_effect_audit.py` 覆盖版本 golden、默认输出兼容、Const 切换、冷热
编译缓存独立性、循环/false mask/unsafe/assume、两条 CLI 路径、hash seed
稳定性、同址/独立/stride-view 绑定、只读 explain，以及缺参、dtype、grid、
执行异常和空启动清理。前置控制流反例仍由 `test_effect_summary.py` 覆盖。

本地 CPU **984 passed**，RTX 3090 严格 GPU **300 节点／444 案例通过**，均零跳过。
原始产物：`artifacts/cpu/m4-01d-results.xml` 及
`artifacts/ci-gpu/20260919T164112Z-lgqa0l56/report.json`（本地忽略目录，不公开主机元数据）。
本次工作区尚无
独立远端 CPU CI 记录，不能复用前置提交的成功记录作当前提交的证明。
