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
环境必须精确匹配上述组合，全部 70 个 pytest 节点成功且无 skipped 才能
标记 `accepted: true`。四个既有专项是聚合节点，其内部含 148 个固定案例；
另有 66 个新增节点，共 214 个语义案例。
无 CUDA、依赖缺失、环境不匹配、测试失败或被跳过均返回非零，并保留失败记录。
`--exploratory` 允许其他组合做探索，`--case` 允许定位子集，二者无论成功与否
都不能作为完整基线验收。CPU pytest 与 GPU 门禁分离，不以 CPU 通过充当 GPU 证据。

M2-08 覆盖整数/cast、布尔 mask、Const bool、浮点常量、正 stride/offset view、
零 stride 只读广播、tail mask、12 种 arithmetic dtype 的 sum/max 与特殊浮点值。
PyTorch 不提供负 stride tensor，因此负 stride 仍只有 CPU 证据。
不承诺 FP8/dot 的完整矩阵、并发/race、跨版本/跨架构、性能或所有示例的 GPU 一致性。

## 尚未建立的支持

目前没有专用持续 GPU CI runner，也没有环境安装/镜像的持续重建验证。
M3-01 仍需确定 runner 与正式支持矩阵，随后建立持续门禁、示例 differential。
本 ADR 接受的是可复现实验组合及验收规则；不能因此将项目整体 GPU 后端升级为
稳定支持。升级依赖必须新建/更新基线并完整重跑，不能直接覆盖旧审计证据。
