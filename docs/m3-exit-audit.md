# M3-06：M3 退出审计

日期：2026-09-19。审计结论：**NOT READY，整个 M3 仍 IN_PROGRESS**。
M3-06 的审计交付已完成，不等于 M3 的退出条件全部满足。本次不增加语言或后端功能。

## 审计基线与证据

审计对象为 `0.3.0.dev0`，实现基线 commit
`4d62300bf7fcb992b41eb5411117b3af5abbec07` 加本次文档/CPU CI 配置变更。
GPU 复验没有修改 src、tests 或 GPU 工具；报告保存了当时工作区状态和源码。
审计结合实际源码、测试断言与复验结果，不把矩阵中存在一个操作名称视为全域认证。

固定 GPU：RTX 3090 / SM86、driver 595.84、Python 3.11.9、PyTorch 2.10.0+cu128、
Triton 3.6.0、CUDA runtime 12.8、NumPy 1.24.3、ml_dtypes 0.5.4。
本次严格本地验收 **160 passed，零 skipped，304 个语义案例，accepted=true**。
四个聚合节点含 148 个案例，所以节点数和案例数不能相加冒充测试数量。
本地报告在 `artifacts/ci-gpu/20260919T140341Z-pbzeoyhc/report.json`，与 results.xml、
source 和重放材料一起保留；目录被 git 忽略，不是公开下载承诺。原始附件包含机器
环境，不直接发布。已撤下的远端 GPU Actions 记录不作为可用的公开 CI 证据。

新建 conda `tila` 环境（Python 3.11.16、PyTorch 2.10.0+cpu，无 CUDA/Triton）
已完成 **926 passed，零 skipped** 的全量 CPU 回归/golden，`pip check` 通过。
首次独立运行发现 3 项测试依赖 torch；现固定安装 CPU-only wheel，未跳过或削减测试。
JUnit 保留在 `artifacts/cpu/results.xml`。CPU workflow 已检查 YAML/命令语法及
正常、空集合、skip、failure、error 五种门禁场景；这属于本地验证，不是托管 run。

## 退出条件逐项核对

| 条件 | 结论 | 证据及范围 |
|---|---|---|
| 至少一个固定真实 CUDA 环境全绿 | 本地通过 | `tools/gpu_audit.py` 严格 gate，本次 160 节点；没有自动 GPU CI |
| 四个核心示例 CPU/GPU differential | 限定范围通过 | `tests/gpu_examples.py`：add、matmul、self-attention、fused-attention，另含 softmax；尾块、stride、4/8 warps 等具体参数见 GPU 支持文档 |
| Implemented intrinsic lowering 可编译 | 逐操作有证据，全组合未满足 | 22 个 Implemented/Partial intrinsic 的索引由 `test_gpu_evidence_inventory_tracks_registry_and_real_tests` 对账；static_assert 是 checker-only，byte_offset Deferred；不能据此认证所有可接受 dtype/shape |
| 非法 device/target/launch 有明确诊断 | 通过已测范围 | `test_m3_launch.py`、`gpu_launch.py`、`gpu_examples.py`；无启动断言及输出不变检查，零 grid 仍先验证绑定 |
| 优化 hint 有可靠来源 | 通过已发射范围 | lowering 的三个发射点逐项见下表；CPU 负测试、explain golden、真实 GPU 开/关对照 |
| source map、编译错误与资源门禁 | 通过已测范围 | `test_m3_backend.py`、`gpu_backend.py`：稳定诊断、cause/附件、真实编译故障注入、shared-memory 拒绝且不启动、重放和缓存命中 |
| 官方示例 golden | 通过 | `test_example_goldens.py`，五个示例 × TIR/source/explain，共 15 份；全部随 CPU 回归运行 |
| 持续 GPU 验收（M3-01/计划迭代 E） | 未满足，阻止退出 | 自动 GPU workflow 与机器注册已撤下；只有本地验收，CPU CI 不替代 GPU CI |

完整操作范围见 [gpu-capabilities.json](gpu-capabilities.json) 和
[操作审计](gpu-operation-audit.md)。支持承诺仅限 [GPU 支持矩阵](gpu-support.md)
列举的 RTX 3090 固定组合，不推广至 A800、其他 SM86 卡或不同软件版本。

## Hint 来源核查

| 发射点 | 可靠事实 | 拒绝/隔离证据 |
|---|---|---|
| `multiple_of(pid * step, step)` | STATIC：结构性乘法；还需整数 launch guard，step 为可发射编译期值 | 无约束 step 不发射，pid 分支事实不泄漏；`test_m3_launch.py` |
| `max_contiguous` | STATIC，匹配当前变量定义行的连续性事实 | 不能用其他定义、用户 assume 或陈旧事实授权；`test_hint_requires_definition_specific_static_evidence` |
| 基地址字节 `multiple_of` | CHECKED：显式 Aligned、本次实际 view data_ptr 已验证、一维 stride=1；元素大小单独记录 | `test_alignment_hints.py` 和 `gpu_alignment.py`：无声明/非连续不发射、offset 校验、失败/空 grid 清空、缓存区分事实、开关结果一致 |

用户 assume 不自动变为静态/launch 证明。基地址字节对齐不等于元素偏移整除，
更一般 stride/多维/派生地址保守不发射。寄存器与 spill 只是性能信息；本审计不
宣称性能提升。source map 只保证可确认的语句行，辅助函数/全局资源错误允许 unknown。

## 尚未关闭的缺口

1. **GPU 持续验收缺失。** 需要适合公开项目、与开发者机器隔离的 GPU 执行环境，
   恢复严格 gate、失败/版本漂移/skip 非零退出、可重放且可公开的附件，并取得实际
   远端运行证据。此项仍归 M3-01；没有时间表就不能写成“每日调度待启用”。
2. **计划范围与已测矩阵仍不一致。** 计划 §8 M3.3/3.4 提到 FP8 编译、add 多 dtype、
   更广 matmul 精度配置；现有示例 add 只有 f32，dot 只有 f16 输入/f32 累加输出，
   FP8 参数和中间 cast 在 Tila 层拒绝。zeros GPU 仅 f32、reshape 仅 i32，dot 的
   f16 输出也未取得本矩阵 GPU 证据。需逐项补证据，或明确评审缩减退出范围及拒绝
   契约；本次审计不自行修改原退出标准来宣告通过。
3. **CPU 托管运行与更广平台证据分开记录。** 本次新增 GitHub 托管 Linux CPU CI
   配置与固定依赖，覆盖回归/golden；配置尚未提交推送时没有远端 run 证据。最低
   Python、Windows 和其他 Python CI 尚未建立，见 [CPU CI](cpu-ci.md)。

其他架构/版本、完整 FP8、race/atomic/uniformity、泛型和性能层保持未验证/后续阶段，
不因本次通过扩大支持范围。`--exploratory` 只放宽审计环境门禁，不绕过 runtime 的
RTX 3090 target 门禁，也永远不能生成完整 accepted 结论。

## 复验入口与后续顺序

CPU 安装与命令见 [CPU CI](cpu-ci.md)；GPU 从锁定环境执行：

```sh
PYTHONPATH=src ci/gpu/.venv/bin/python tools/gpu_audit.py --output artifacts/ci-gpu
```

多卡机器可用 CUDA_VISIBLE_DEVICES 指定一张空闲卡；不记录主机地址/UUID 到公开文档。
后续先取得托管 CPU CI 首次成功记录，再评审 M3 退出范围与证据缺口，最后解决隔离
GPU 持续验收。M3-02..05 保留限定范围 DONE；M3-01 和整个 M3 不能标记完成。
