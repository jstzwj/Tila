# GitHub 托管 CPU 回归

M3-06 新增 [.github/workflows/cpu.yml](../.github/workflows/cpu.yml)，仅使用 GitHub
托管的 `ubuntu-24.04`，Python 3.11.9。触发入口为 main push、pull_request 和手动
workflow_dispatch。配置提交到远端后才会生效；没有远端成功 run 就不宣称已通过托管验收。

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
Triton、`pip check` 无错误。GitHub 托管首个成功 run 仍待提交推送后取得。

当前只配置 Linux/Python 3.11 的单个 CPU job。最低 Python 3.10、更新 Python、
Windows CI、独立 lint/type job 尚未建立。自动 GPU CI 已撤下，本工作流不能补足它。
