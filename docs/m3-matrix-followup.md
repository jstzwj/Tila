# M3 退出审计后续：CPU CI 与 dtype 矩阵补测

日期：2026-09-19；范围仍为 ADR-009 的 RTX 3090/SM86 固定组合。本次补已有能力
的证据与实现差异，不增加公共 intrinsic、泛型接口或其他 GPU 架构支持。

## 托管 CPU CI

首个 [GitHub 托管 CPU run](https://github.com/jstzwj/Tila/actions/runs/35448123253)
已成功，对应 `b269aba7c24b00028b5c1cd45e5da17c870a6905`：926 项全部通过，失败 0、
跳过 0。下载 `cpu-test-report` 附件后核对 JUnit，包含 15 个官方示例 golden。
这是提交 b269aba 的证据。后续矩阵提交 `4774194` 的
[托管 run](https://github.com/jstzwj/Tila/actions/runs/35448947430)已取得 **937 项
全通过、零跳过** 的独立记录，JUnit 附件已下载核对，包括新增 dot 回归与 15 个
官方示例 golden。
runner 为 GitHub 托管 Ubuntu；没有注册或连接开发者机器。附件有保存期限，流程
与本地重跑命令见 [CPU CI](cpu-ci.md)。

## 补测范围

统一入口新增 `tests/gpu_matrix_gaps.py` 的 140 个节点：

| 操作 | 新证据 | 判定规则 |
|---|---|---|
| zeros / reshape | bool、8 种整数、f16/bf16/f32/f64，各测 rank 1/2，共 52 节点 | 16 元素、(4,4) 变形和恢复；整数极值、NaN/Inf/±0，按字节检查零值或位模式保持 |
| 官方 add 的 dtype 专门化 | 12 arithmetic dtype × N=127/128/129 × 4/8 warps，共 72 节点 | 从官方源函数仅替换 dtype；整数使用 Python 任意精度加法再按位宽回绕，浮点用选定值的 f64 参考再舍入；tail mask 与溢出边界 |
| dot | f16/f32 acc × 舍入/随机 × 连续/转置 stride × 4/8 warps，共 16 节点 | f16 输入，16×32 乘 32×16；f64 参考与累加/输出舍入误差界，RNE 边界要求精确一致 |

随机种子固定为 306；其他数据固定枚举。add 的公开示例仍是 f32 API，测试专门化
不是新泛型能力。bool 不属于 arithmetic add；FP8 不进入正向执行矩阵。
dtype 覆盖不等于所有 shape/rank/stride/控制流组合已认证。

## 发现并修复的 dot 差异

旧 checker 接受 f16 accumulator，CPU 以 f32 计算并返回 f16；旧 lowering 直接
生成 `tl.dot(a, b, acc)`，而 Triton 默认结果为 f32，导致 f16 acc 在真实编译时
类型断言失败。现仅对此路径先显式提升 acc 到 f32，再在 dot 结果处转回 f16。
f32 路径不变，既有示例 golden 不需要重生成。

测试在 dot 后立即 cast 到 f32，再存储 f32，验证舍入发生在中间结果处。选定精确
输入覆盖 f16 RNE 中点的向下/向上舍入，不依赖最终 store 自动变窄。随机案例按
f32 累加误差及 f16 输出舍入计算误差界，不要求不同累加顺序按位一致。
registry semantic revision 升至 8，防止旧生成语义缓存复用。

CPU 新增 11 项：中间舍入回归、6 种不支持 dot 输入的拒绝、两个 FP8 dtype 的
zeros/reshape build 拒绝。原有 FP8 参数/中间 cast launch 门禁与负测试保留。

## 当前验收与剩余边界

当前 GPU gate 要求 **300 个测试节点全部通过**，覆盖 **444 个语义案例**，零跳过；
不能把 300/444 理解为通过比例。四个聚合节点包含 148 案例，因此 300−4+148=444。
完整本地 CPU 基线为 937 项。严格报告、源码、生成 kernel、失败诊断和重放命令
由 `tools/gpu_audit.py` 保存，不公开上传本地机器附件。
本次 accepted 报告位于 `artifacts/ci-gpu/20260919T142007Z-15c5lo1h/report.json`，
基于 b269aba 加工作区修改；报告附带源码，不能只用该 commit 重放新用例。
重放入口已纳入新矩阵文件，同时保留对不含该文件的历史附件的兼容。

用户确认目前没有独立 GPU runner/服务器，因此自动 GPU CI 保持未完成，M3 仍为
IN_PROGRESS。原计划中 FP8 执行和 bf16/f32 dot 输入不在本阶段支持承诺：当前明确
拒绝，完整能力归后续 target/FP8 设计；不通过修改测试跳过它们来制造通过。
当前支持矩阵已补齐上述有限范围的缺口；其他形状、架构和版本仍未验证。
