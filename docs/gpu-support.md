# GPU 支持与持续验收（M3-01）

初始验证范围只包含 **Linux x86_64、NVIDIA GeForce RTX 3090 / SM86**，
Python 3.11.9、PyTorch 2.10.0+cu128、Triton 3.6.0、CUDA runtime 12.8、
driver 595.84。其他设备、架构、驱动/软件版本均未验证，不能由本矩阵推导支持。
M3-02 已增加结构与静态 target verifier；M3-03 已增加语句 source map 和编译后资源门禁，
详见 [后端诊断与重放](backend-diagnostics.md)。其他架构与完整性能分析仍未验证。

## 从空环境重建

前提：Linux x86_64/glibc >= 2.28、git、C 编译器、上述 NVIDIA driver、可见 RTX 3090。
CUDA 用户态库由 PyTorch 的已锁定 wheel 提供，不要求另行安装 nvcc。
安装 uv **0.8.14** 后，从仓库根运行：

```sh
uv sync --project ci/gpu --frozen --python 3.11.9
PYTHONPATH=src ci/gpu/.venv/bin/python -m pytest -q
PYTHONPATH=src ci/gpu/.venv/bin/python tools/gpu_audit.py --output artifacts/ci-gpu
```

`ci/gpu/uv.lock` 锁定全部传递依赖、下载来源和 hash；只从显式 PyTorch cu128
index 解析 torch，其余来自 PyPI。`.python-version` 固定 Python；uv 可下载缺失的
Python 3.11.9。不要用 `uv lock` 替代 `--frozen` 来运行验收；升级依赖应提交锁文件
变更并重新验证。根项目 `triton` extra 也固定 3.6.0。

## 持续验收状态

公开仓库不连接开发者本地机器。远端 GPU workflow 与机器注册已移除，当前不提供
自动 GPU CI 或每日调度；保留依赖锁、验收工具和本地执行方式。持续验收需要另行
设计适合公开项目的隔离环境，不能将本地通过视为持续 CI 已完成。

M3-06 新增独立的 [GitHub 托管 CPU CI 配置](cpu-ci.md)，运行回归与 golden，
不连接本地机器、不运行 CUDA。它不补足 GPU 持续验收缺口。
[M3 退出审计](m3-exit-audit.md)与[矩阵补测](m3-matrix-followup.md)记录复验与剩余阻塞项。

## 附件与重放

本地审计目录包含环境、完整包版本、原始源码和锁文件、生成内核、pytest XML/日志
与局部变量。这些记录可能包含主机路径和设备标识，不应直接作为公开附件上传。
无 GPU、核心版本漂移、任何 skip、测试数量不符都失败；子集/探索不能标记 accepted。

下载解压后定位含 report.json/source 的目录：

```sh
python tools/replay_gpu_audit.py /path/to/extracted/run --case test_matmul
```

该命令在附件内的锁定源码重建环境，不依赖原始绝对路径或当前工作区。
诊断重放不产生完整验收承诺；正式验收仍走 gpu_audit.py。

## 已执行范围与下一步

原 M2 的 70 个节点/214 个语义案例保留；增加 32 个官方示例配置、同设备负测试及
hint 开关对照；M3-02 再增加 8 项 launch/cache 负测试，当前共
112 节点/256 案例；M3-03 增加 3 项，M3-04 再增加 36 项，当前为
151 节点/295 案例；M3-05 增加 9 项达到 160 节点/304 案例；退出审计后补测
增加 140 项，当前为 **300 个 pytest 节点全部通过，覆盖 444 个语义案例**。
CPU 全量基线为 937 passed，零 skipped。M3-02/03/04/05 及矩阵补测已在本地固定环境通过；
[launch/target 规则与边界](launch-target.md)详述前置门禁范围。
逐 intrinsic 的 dtype/shape 与证据见 [操作审计](gpu-operation-audit.md)，不是全组合认证。
对齐契约的单位、hint 来源与缓存规则见 [alignment hints](alignment-hints.md)。

既有验证结果仍为工程证据；包含部署细节的远端日志和附件不再公开提供。
此前无 CUDA 的负向验收返回非零并保存 failed report。

| 操作 | 已验证范围 |
|---|---|
| add | f32，127/128/129，tail mask，4/8 warps |
| add dtype 专门化 | 官方源函数仅替换 dtype，12 arithmetic dtype；127/128/129、4/8 warps、整数回绕与浮点特殊值；不增加泛型 API |
| softmax | f32，31/32/33，f64 reference，rtol=2e-6/atol=2e-7 |
| matmul | f16 输入/f32 累加，完整块与 M/N/K 尾块、转置 stride；f64 参考及按 K 的误差界 |
| self-attention | causal，长度 32/65，head dim 32，f16 probability 舍入容差 2e-3 |
| fused-attention | causal/noncausal，2 heads，长度 32/65，两段 KV、exp2；同上独立 f64 参考 |
| sum/max、整数、mask、常量 | 延续 ADR-007/008/012–015 和 M2 审计范围 |
| exp/exp2、浮点 cast、bf16 广播 | 四种浮点指数函数与窄中间舍入、16 个 cast 组合、bf16 外积广播；具体误差标准见操作审计 |
| FP8 storage/cast | Stage 1 类型设计保留；固定 target build/launch 拒绝，包含中间 FP8 值 |
| zeros / reshape | bool + 12 arithmetic dtype，16 元素、一维/二维变形，按位检查 |
| dot f16/f32 acc 与输出 | f16 输入，16×32×16、连续/转置 stride、4/8 warps、RNE 中点和固定随机；f16 输出在 dot 处立即舍入 |

launch 选项为 `kernel[grid].with_options(num_warps=4或8)(...)`；不抢占 kernel 的
同名标量/Const 参数空间。拒绝 bool、NumPy integer、浮点或未验证的 warp 数。
缓存区分设备 UUID、capability、软件版本、warp 数、源码和实际 dtype/布局/alignment。
CPU/CUDA 混用、跨 CUDA device 的张量在执行前拒绝（TILA-TARGET-008）。
CUDA 运行时目前仅放行 RTX 3090/SM86 与 Triton 3.6.0（TILA-TARGET-007）；
完整驱动/Python/PyTorch 组合由严格验收入口检查，普通 launch 不声称完成全部版本认证。
hint 检查覆盖静态/已检查契约来源和带/不带 hint 的真实 GPU 结果一致；新增一维
连续 Buffer/Ptr 的基地址 alignment 发射。更一般布局保守不发射，不作性能提升承诺。
