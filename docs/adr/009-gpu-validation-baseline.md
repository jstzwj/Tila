# ADR-009：固定 GPU 语义验收环境

状态：**Accepted**

日期：2026-09-19；M2-08 本地验证基线，非正式 GPU 支持承诺。

## 固定组合

| 项目 | 版本/设备 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，compute capability 8.6 |
| NVIDIA driver | 595.84 |
| Python | 3.11.9 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| PyTorch CUDA runtime | 12.8 |
| NumPy | 1.24.3 |
| ml_dtypes | 0.5.4 |

这是已实际运行的 Linux 本地环境。CUDA 版本指 PyTorch wheel 的 runtime，
不是 `nvcc` 或 nvidia-smi 显示的最高可支持版本。审计记录 OS、git revision、
工作区状态、源码副本及 CUDA_VISIBLE_DEVICES；通过活动设备 UUID 查询 driver，
不假设可见 device 0 就是物理 device 0。不要求机器只有一张卡，仅单设备验收。

## 验收边界

统一命令：`PYTHONPATH=src python tools/gpu_audit.py`。
环境必须精确匹配上述组合，全部 300 个 pytest 节点成功且无 skipped 才能
标记 `accepted: true`。四个既有专项是聚合节点，其内部含 148 个固定案例；
M2-08 另有 66 个新增节点，共 214 个语义案例；M3 至 alignment 阶段增加 90 个节点，
退出审计后 dtype 矩阵再增加 140 个节点，当前共 444 个案例。
无 CUDA、依赖缺失、环境不匹配、测试失败或被跳过均返回非零，并保留失败记录。
`--exploratory` 允许其他组合做探索，`--case` 允许定位子集，二者无论成功与否
都不能作为完整基线验收。CPU pytest 与 GPU 门禁分离，不以 CPU 通过充当 GPU 证据。

M2-08 覆盖整数/cast、布尔 mask、Const bool、浮点常量、正 stride/offset view、
零 stride 只读广播、tail mask、12 种 arithmetic dtype 的 sum/max 与特殊浮点值。
PyTorch 不提供负 stride tensor，因此负 stride 仍只有 CPU 证据。
不承诺 FP8/dot 的完整矩阵、并发/race、跨版本/跨架构、性能或所有示例的 GPU 一致性。

## 尚未建立的支持

M3-01 保留全量依赖锁与官方示例 differential，详见
[支持矩阵与操作手册](../gpu-support.md)。远端 GPU 自动验收已撤下，公开项目的隔离
GPU CI 待建立；[托管 CPU CI](../cpu-ci.md) 单独运行回归/golden，不提供 GPU 证据。
初始验证范围仅覆盖本表 RTX 3090/SM86 的组合及已列举操作。
M3-02 的结构与静态 target verifier 见 [launch/target](../launch-target.md)。其他架构/版本、
完整性能分析仍未验证；M3-03 的编译后硬资源门禁与语句 source map 见
[后端诊断](../backend-diagnostics.md)。不能因此将整个后端声明为全域稳定。
升级依赖必须新建/更新基线并完整重跑，不能直接覆盖旧审计证据。

M3-04 逐 intrinsic 的 dtype/shape 证据与拒绝范围见 [操作审计](../gpu-operation-audit.md)。
FP8 storage/cast（包括中间值）没有执行支持；Stage 1 类型设计不等于本 target 支持。
