# M4-03b：atomic_add 前端、Effect IR 与 CPU 参考

本页保留 M4-03b 的历史实施记录；当前 GPU 支持和最新验收见 [M4-03c](atomic-gpu.md)。

2026-09-20；实现 [ADR-017](adr/017-minimal-atomic-add.md) 的 CPU 阶段。
registry 状态为 **Partial**：没有 GPU lowering，不提供真实 GPU atomic 支持。

## 可用接口

`ti.atomic_add(ptr, value, mask=..., order=ti.Relaxed, scope=ti.GPU)`；Buffer 可用
`ti.atomic_add(buf, coord0, ..., value, ...)`，也接受与 load/store 一致的坐标元组。
仅 Global ReadWrite 的 i32/u32/f32、标量或一维访问 tile。value 可以是同 dtype
标量或严格同形 tile，mask 可以是标量 bool 或严格同形布尔 tile。原子返回旧值；
inactive lane 不访问内存且返回同 dtype 零。无 other、unsafe 或字符串限定符。

MemoryOrder / MemoryScope 是不同的 enum 类型，目前仅导出 Relaxed / GPU。
限定符可直接引用或用捕获的模块级别名，不能用任意 Python 表达式构造。
CPU 的 scope=GPU 是参考该语言契约，不代表 CPU 实现 GPU 线程同步。

每个 active 元素逐次更新，支持重复地址及同存储的别名参数。CPU program
按 pid2/pid1/pid0 嵌套递增，tile 按逻辑顺序；这个顺序不约束未来 GPU 调度。
i32/u32 显式回绕；f32 更新执行 RNE 和 Global atomic 次正规数归零，旧值返回
保留更新前的内容。NaN payload 不承诺一致。

Buffer 正 stride/offset view 保留真实索引；Ptr 仍限定连续一维存储。每次绑定
检查四字节自然对齐、字节 stride 是元素大小的倍数、stride 为正；缓存和空 grid
不绕过检查。bounds 义务和 debug active 索引检查沿用现有机制，未增加 unsafe。

## IR、诊断与后端边界

TAtomicAdd 是唯一执行表达式；TAtomicStmt 保留未使用返回值的调用。AtomicInfo
记录 Add/Relaxed/GPU；verifier 重算局部 effect，拒绝缺失、错型或身份冲突。
变量定义引用不会重新执行 atomic。summary 保留独立 path/mask/loop 和参数中
嵌套访问，where 检查按最近值分支归属区分 Read 与 Atomic(Add)。

可选 `--show-effects` 迁移为 `tila.effect-details.v2`：每条访问新增 `atomic`
字段，普通 Read/Write 为 null，Atomic 为 op/order/scope 对象。默认 explain
章节与现有 load/store 文本不变。相关 golden 已显式更新，未提供旧 v1 兼容开关。

CPU 通过 NumPy 参数正常 `kernel[grid](...)` 执行；`dump` / `explain` 可审计。
`materialize`、CLI check/build、直接 Triton source 生成以及 CUDA launch 暂以
`TILA-TARGET-012` 拒绝 atomic，包含死分支与零 grid。类型/权限/bounds 等更早
错误仍可先报告。不以 dummy GPU handler 伪装完成后端支持。

新增 TILA-MEM-008 表示 atomic 自然对齐/布局契约失败；TILA-TARGET-012 表示
未支持的 atomic 配置或 GPU 路径。bounds 修复建议不会建议不存在的 unsafe atomic。

## 验收与下一步

专项 `tests/test_atomic_cpu.py` 与 `tests/golden/atomic-cpu.txt` 覆盖签名、限定符、
重复地址/别名、返回值复用/丢弃、嵌套副作用、mask、Const/零循环、整数/浮点
边界、stride/view、bounds/debug、verifier 损坏、where 严格度与后端拒绝。
GPU 现有套件复验只证明既有功能无回归，不计为 atomic GPU 证据。

本次 **36 项 atomic 专项**，完整 CPU **1038 passed**；既有 RTX 3090 严格
GPU **300 节点／444 案例通过**，均零跳过。另在真实 CUDA tensor 上检查零/
非零 grid 的 atomic launch：均 TARGET-012 拒绝，数据未改变。这是拒绝证据，
不是 GPU atomic 执行证据。

本地产物：`artifacts/cpu/m4-03b-results.xml`、
`artifacts/ci-gpu/20260920T064528Z-ha9g_bn7/report.json`。产物保留在忽略目录，
不公开主机信息；当前改动尚未取得独立远端 CPU CI 记录。

以上是 M4-03b 时点的历史证据；GPU 拒绝边界已由 [M4-03c](atomic-gpu.md) 的限定支持替代。
当时下一步 M4-03c 计划实现 target capability 与 lowering，并按 ADR-017 检查重复地址
旧值多重集、浮点合法调度结果及执行次数，取得真实 GPU 证据后扩大支持声明。
没有隔离 GPU runner，M3 的持续 GPU 验收仍未完成。
