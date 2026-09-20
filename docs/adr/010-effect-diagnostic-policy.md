# ADR-010：where 急切效应诊断与独立策略

- 状态：Accepted（2026-09-20，M4-02）
- 前置：ADR-016；不启用 atomic、race、uniformity。

## 决策

`where(cond, a, b)` 急切计算两个值操作数。对已验证 TIR 中值分支求值时
实际包含的 load（含 unsafe）生成 TILA-EFFECT-007。使用逐访问 effect 的 site、
RegionId 和源位置；穷尽遍历 OPERANDS，包括坐标、mask、other 等嵌套表达式。
TName 必须具有经过 verifier 验证的定义引用，引用已计算值不会重新执行定义，
因此不追溯该定义中的 load。赋值、重绑定、分支合并及循环携带值适用同一规则。

仅在 where 条件中读取不告警。嵌套 where 的读取归属最近的值分支，避免同一
访问被外层重复报告；内层条件的读取若位于外层值分支，仍归外层。每个
where 的每侧最多一条诊断，列出该侧所有访问 site。

本阶段采用静态结构诊断：Const 未选分支、不可达路径、false load mask 仍可
告警，不把它表述为必然发生的内存事务。load 自身 mask 仍有效，其他参数的
嵌套读取仍急切求值。不使用 SMT、assume 或 launch 事实消除这类提示；不判断
用户是否有意急切读取。安全的内联 masked load 也允许显式选择 off。

## 策略与边界

独立环境变量 `TILA_EFFECTS=off|warn|error`，缺省 warn；CLI `--effects` 显式
选项覆盖环境变量，未指定时继承环境变量。非法值报 TILA-EFFECT-008。

- off：不展示该类诊断，不实施 effect 门禁；仍验证元数据、检查类型及 bounds。
- warn：保留历史行为，在 report/explain/CLI check 中展示 warning；Python 启动
  不额外发送 Python warnings，原始候选仍可通过 tk.warnings 检查。
- error：装饰期、materialize 和每次 launch 拒绝存在该类诊断的 kernel，
  包括缓存命中和空 grid；不依赖 bounds strict/warn。

候选由 TIR 重算而非依赖旧 warning 列表。策略不影响代码生成或 proof 结论，
因此不进入编译缓存键；每次入口重新检查，不能通过先 warn 编译再 error 启动
绕过门禁。report/explain 对已存在的 kernel 只展示当前级别的诊断，不执行代码；
CLI 导入文件时若 error 已生效，装饰期可直接拒绝。

## 修复建议与验收

根据意图分别使用 `load(a, mask=cond, other=...)` 与
`load(b, mask=~cond, other=...)`；标量 bool 使用逻辑 not。保留原有 bounds mask，
只在形状兼容时组合。回退 other 应为纯值；其中的嵌套 load 仍会求值。
不能建议把两个分支都换成相同 cond，也不自动改写可能影响异常或结果的代码。

验收覆盖直接/深层/unsafe 读取、条件读取、定义复用及重绑定、循环/合并引用、
嵌套 where 去重、false mask 的 other 读取、策略切换及缓存门禁、CLI、稳定 golden。
race 严格度留待独立设计；本 ADR 不定义 race 算法或抑制语法。
