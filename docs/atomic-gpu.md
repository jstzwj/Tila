# M4-03c：atomic_add GPU lowering 与严格验收

日期：2026-09-20。按 [ADR-017](adr/017-minimal-atomic-add.md) 完成最小
atomic_add 的 CPU/GPU 路径，替代 M4-03b 时的 GPU 全面拒绝。

## 支持边界

只支持 ADR-009 固定 RTX 3090/SM86、Python 3.11.9、PyTorch 2.10.0+cu128、
Triton 3.6.0、CUDA runtime 12.8、driver 595.84。Global ReadWrite，i32/u32/f32，
Relaxed/GPU，标量或一维 tile，Buffer 或连续一维 Ptr；Buffer 可用正整元素 stride/view。
value 精确 dtype，mask 为标量 bool 或精确同形；自然四字节对齐每次 launch 检查。
其他原子、order/scope、dtype 和架构不在支持范围。

每个有效元素执行独立 RMW，返回旧值；inactive 元素不访问目标，返回同 dtype 的零。
嵌套操作数仍按求值规则执行，外层 false mask 不消除内层 atomic。
整数回绕；浮点允许合法的并发更新顺序，使用 ADR-017 的 f32 RNE/Global FTZ 规则。
CPU 的确定顺序不是 GPU 的唯一正确结果。混合普通读写的竞争安全尚未分析。

## Lowering 与反例修复

生成 helper 显式转型 mask/value，调用 `tl.atomic_add(..., sem="relaxed", scope="gpu")`，
再将 inactive 返回值归零。helper 名和临时变量避开用户标识符。
f32 输入及结果经过非纯寄存器 identity，保持零加数的 RMW 与未使用旧值时的逐元素更新。
常量零若退化成 load，会丢失 FTZ/有符号零更新；直接使用原子 inline PTX 又会绕过
Triton 的逻辑元素到执行线程映射，因此实际内存操作仍交给 `tl.atomic_add`。
锁定版本的相关优化见
[Triton 3.6.0 LoadStore lowering](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp)。

含 atomic 的表达式显式排列操作数求值，避免嵌套表达式重复更新。
运行期 and/or 转成控制流以保持短路；CPU store 同步按地址、value、mask 顺序求值。
临时语句保留 Tila 源位置，已有 backend 诊断/产物机制可重放；Triton source 有独立 golden。
能力矩阵新增 dtype/order/scope，参与既有 target 缓存指纹；语义 revision 升为 13。
热缓存、空 grid 与 effects=off 均不能绕过 verifier 或每次启动的布局/自然对齐校验。

## 验收方法

`tests/gpu_atomic.py` 共 88 个参数化节点，覆盖 4/8 warps：

- i32/u32/f32 重复地址票号：精确最终值、旧值多重集、尾部及全 false mask，重复三轮。
- 非碰撞 Buffer/Ptr、正 stride/view、整数边界回绕、返回值 CPU 对照。
- f32 受控四元素更新枚举 24 种合法顺序，联合核对旧值和最终值；丢弃返回值也验证。
- f32 signed zero、subnormal FTZ、NaN/Inf、舍入中点、字面零反例；NaN 不要求 payload 相等。
- 固定种子 208009 的有限随机输入，以高精度总和及 gamma 误差界验收；不要求并发逐位等于 CPU。
- 嵌套 atomic、alias、where/mask 求值、运行期短路、Const、循环和 return。
- 非法布局/对齐、target 禁用、热缓存/空 grid 拒绝且不启动，以及布局/Const/warp 缓存隔离。

统一入口 `tools/gpu_audit.py` 固定 `TILA_EFFECTS=warn`，包含 atomic 专项并保存
环境、种子、源码、生成内核、JUnit 与失败现场。无 GPU、版本漂移、跳过或数量不符均失败。
`tools/replay_gpu_audit.py` 支持新附件中的 atomic 测试，并兼容不含该文件的历史附件。

## 本地结果与限制

CPU 全量 **1040 passed**；RTX 3090 严格验收 **388 节点 / 532 语义案例全部通过**，
均零跳过，其中新增 atomic 88 节点。
本地证据：`artifacts/cpu/m4-03c-results.xml` 和
`artifacts/ci-gpu/20260920T071828Z-_e2pu1v7/report.json`。
原始附件含本地主机信息，不直接公开上传。

已从该附件保存的源码与锁文件重建环境，通过 `--case atomic_literal_zero` 重放
两个反例（返回旧值使用/丢弃）：**2 passed、386 deselected**。这是诊断子集重放，
不替代上述完整严格验收。

当前提交尚无新的 GitHub 托管 CPU CI 记录，不能沿用旧 run。
没有隔离 GPU runner，本地通过不等于持续 GPU CI；M3 仍 NOT READY。
未实现 race/uniformity，不宣称浮点原子跨调度可复现或任何性能提升。
