# GitHub 托管 CPU 回归

M3-06 新增 [.github/workflows/cpu.yml](../.github/workflows/cpu.yml)，仅使用 GitHub
托管的 `ubuntu-24.04`，Python 3.11.9。触发入口为 main push、pull_request 和手动
workflow_dispatch。[首次托管运行](https://github.com/jstzwj/Tila/actions/runs/35448123253)
已通过：提交 b269aba 的 926 项回归、零 skipped，JUnit 附件已下载核对，包含 15 个
官方示例 golden。后续修改必须取得各自的验收证据，不能沿用旧提交的成功状态。

矩阵补测提交 `47741941cc9fd5603b41883f7b4801f126eae408` 的
[托管运行](https://github.com/jstzwj/Tila/actions/runs/35448947430)也已成功：**937 项
全部通过，失败/错误/跳过均为 0**。`cpu-test-report` 附件已下载核实，包含新增
f16 dot 舍入回归与全部 15 个官方示例 golden。本轮 ADR 文档不改变这些测试行为。

后续 M4-01b 提交 `f4add240605366a6da2e8ae3bb9253a9a438cc67` 的
[托管运行](https://github.com/jstzwj/Tila/actions/runs/35454545217)通过 **953 项、零跳过**，
JUnit 附件已下载核实。M4-05c 工作区本地 1191 项通过，另见 [Uniformity 审计](uniformity-audit.md)，
不将旧提交的 run 视为当前修改的远端验收。

C0 [正确性收口](correctness-closure.md)当前本地通过 1217 项（新增 26 项专项），
包括异号 mask 诊断 golden；本轮尚未取得对应的托管 CPU run。

工作流已增加 Race 枚举记录和失败重放 JSON 的附件路径；本轮配置尚待远端运行验证。

工作流不连接开发者机器，不需要 GPU、仓库 secret 或 Triton。PyTorch 使用官方
CPU-only wheel `2.10.0+cpu`，用于 tensor view/bf16 写回和宿主转换拒绝测试。仓库权限为
contents:read，checkout 不保留凭据；第三方 action 固定到 commit。PR 使用普通
pull_request，不能通过 pull_request_target 在特权上下文执行提交代码。

[ci/cpu/requirements.txt](../ci/cpu/requirements.txt) 固定 CPU 直接/传递依赖和构建工具
版本；它不是 wheel hash 锁，也不锁定 GitHub runner OS 镜像的补丁版本。使用这些依赖
安装本项目后运行完整 pytest，包含 checker、解释器、SMT/信任/缓存回归，以及 TIR、
Triton source、explain、CLI 和后端诊断 golden。生成 Triton 文本不等于实际 GPU 编译。

pytest 失败、空测试集和任何 skipped 均不能通过。工作流只上传托管 runner 生成的
JUnit XML（保存 14 天），不上传本地 GPU 日志、源码快照或机器环境附件。失败时从
Actions 日志/JUnit 取得测试 node id，在相同依赖环境用 `python -m pytest node-id -q`
重跑；不要自动覆盖 golden 来消除失败。

## 本地复现

```sh
conda create -n tila python=3.11 pip -y
conda activate tila
python -m pip install -r ci/cpu/requirements.txt
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTEST_ADDOPTS='' TILA_DEBUG=0 TILA_SAFETY=strict \
  python -m pytest -q --junitxml=artifacts/cpu/results.xml
```

已有 tila 环境时跳过 create。2026-09-19 本地 conda 实际解析为 Python 3.11.16；它是
CPU 回归环境，不冒充 Python 3.11.9 的固定 GPU 基线。GPU 严格验收继续使用
[ci/gpu/uv.lock](../ci/gpu/uv.lock) 重建的环境。

该 conda 环境已通过 926 项测试、零 skipped，且确认 torch 为 CPU-only、未安装
Triton、`pip check` 无错误。后续矩阵补测的本地基线见 [补测记录](m3-matrix-followup.md)。

当前只配置 Linux/Python 3.11 的单个 CPU job。最低 Python 3.10、更新 Python、
Windows CI、独立 lint/type job 尚未建立。自动 GPU CI 已撤下，本工作流不能补足它。
