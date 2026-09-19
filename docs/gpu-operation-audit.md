# M3-04：操作／dtype 支持矩阵与编译覆盖审计

日期：2026-09-19；版本：0.3.0.dev0。Target 仍限 [ADR-009](adr/009-gpu-validation-baseline.md)
的 RTX 3090/SM86 固定组合；本阶段没有新增架构、dtype 或 dot 输入域。

逐 intrinsic 的机器可读清单为 [gpu-capabilities.json](gpu-capabilities.json)。它覆盖
registry 中全部 22 个 Implemented/Partial intrinsic，包括非公开方法 any/all；
byte_offset 仍 Deferred。每行记录已测 dtype、shape、测试函数与限制。CPU 测试
检查清单没有遗漏/过期名字，并检查引用的测试函数存在。这是证据索引，不替代真实
GPU 验收，也不是所有 dtype × shape × stride × 控制流组合的完整认证。

## 当前证据范围

| 操作 | dtype／shape 证据 | 边界与主要测试 |
|---|---|---|
| program_id、arange | i32，尾块、1D/2D 坐标 | grid/index 门禁；gpu_examples.py |
| num_programs、range | i32，两个 program、固定循环及 matmul K 循环 | gpu_capabilities.py 的 structural；不声称穷尽循环形式 |
| load/store | bool、8 个整数、f16/bf16/f32/f64；mask、stride/view、广播 | gpu_semantics.py、gpu_boolean_smoke.py；负 stride 仅 CPU |
| unsafe_load/store | i32，16 元素、两个 program | structural 用实际合法地址验证执行；豁免 bounds 不豁免 target |
| cast | 四种浮点的 16 个组合；既有整数/布尔案例 | 有限边界、RNE 中点、signed zero、NaN/Inf；不是所有整数/浮点交叉组合 |
| constant | f16/bf16/f32/f64 | ADR-015 46 个按位案例；仅有限源值，无 FP8 |
| where、广播 | bool/i32 的既有案例，bf16 外积广播，attention 的 f32 | eager 两分支；不能用来隐藏不安全内存访问 |
| dot | f16 输入、f32 累加/输出；二维、K 尾块、转置 stride | bf16/f32/FP8 输入由 checker 拒绝；f16 输出尚未列入本行 GPU 承诺 |
| zeros | f32 二维累加 tile | matmul/attention；其他 dtype 仍只有各自已有前端/CPU 覆盖，不扩张 GPU 承诺 |
| sum/max | 12 arithmetic dtype，二维双轴、带 mask 一维 | ADR-008，NaN/Inf/signed zero、窄输出、整数回绕；bool/FP8 拒绝 |
| exp/exp2 | f16/bf16/f32/f64，一维尾块、特殊值、窄中间结果 | 本阶段新增；下述计算/舍入与容差规则 |
| reshape | i32，8 → (2,4) → 8 | 元素数保持；未认证全部 dtype/rank 组合 |
| any/all | bool lane reduction | 既有 68 组布尔专项；不是数值归约 |
| assume | scalar 符号谓词，debug 成功断言路径 | 用户信任来源不变；不在共享 runner 上故意触发 device assert |
| static_assert | 编译期 bool | checker-only，通过后擦除，无需存在 Triton runtime opcode |

FP8（f8e4m3fn/f8e5m2）仅保留 Stage 1 注解、storage/cast 类型规则。本环境尚未建立
完整 FP8 绑定、舍入和 CPU/GPU 执行契约；build 的 target verifier 明确拒绝 FP8
参数或中间值，CUDA launch 也在执行前报 `TILA-TARGET-009`。这不是断言 SM86
硬件绝不可能实现某种 FP8 转换，而是当前 Tila 不提供该组合的执行支持。
仅把 f32 临时转换为 FP8 后再转回 f32 也不能绕过门禁。
CPU 的该中间 cast 路径同样在解释前拒绝，避免落入 NumPy 的 storage-only 异常。

## 本次发现与修复

新增真实 GPU 测试首先复现了四个编译失败：exp/exp2 × f16/bf16。Tila checker
接受并保留输入类型，但旧 lowering 直接调用仅接受 f32/f64 的 Triton math 接口。
现将窄输入提升为 f32 计算，在该操作结果处立即舍入回 f16/bf16；CPU 同样在此处
恢复声明 dtype。后续显式 cast 到 f32 不能恢复舍入位。旧 CPU 路径直到 store 才
隐式变窄，会让中间 cast 观察到额外精度，新增回归专门覆盖这个问题。
scalar f64 也按声明类型使用 f64 计算，不再因非 ndarray 走 f32。registry semantic
revision 升为 7，避免复用旧语义缓存。

测试输入固定，无随机采样；包含 -4、-1、±0、0.125、0.5、1、4、±Inf、NaN 和尾块。
对这些输入用 NumPy f64 参考，rtol 分别为 f16=2^-9、bf16=2^-6、f32=2e-6、
f64=2e-14，atol=0；NaN 位置、Inf 符号参与比较。这是该案例集的容差，不是整个
函数域的误差上界。窄结果再转 f32 的选定非中点案例要求 CPU/GPU 与显式舍入参考
逐值一致。浮点 cast 矩阵要求结果与目标 dtype 转换一致，并单独检查 signed zero。

## 官方示例 golden

`tests/golden/examples/` 保存 add、softmax、matmul、self-attention、fused-attention
各自的 canonical TIR、Triton source、audit explain，共 15 份。测试使用原始示例的
默认 Const、全新 JIT 对象、debug=0、strict；保留 explain 中真实 Unknown 和 launch
契约，不把尚未 launch 的状态改写为 ProvenSafe。更新 snapshot 时必须审查语义变化，
不能因测试失败直接重新生成接受。CPU output 与 GPU differential 沿用已有示例测试。

## M3 退出条件核查

| 条件 | 本阶段结论 |
|---|---|
| 一个固定真实 CUDA 环境全绿 | 本地严格验收；新的远端 CI 证据待提交后获取 |
| 核心示例 CPU/GPU differential | 五个官方示例已有证据，15 份 golden 补齐 |
| Implemented intrinsic 有编译证据 | 每项有索引；static_assert 为 checker-only；证据限清单标注的 dtype/shape |
| unsupported target、device、launch 诊断 | M3-02/03 已覆盖，本次增加 FP8 storage/中间 cast 门禁 |
| optimization hint 有来源 | 既有 hint 有来源与负测试；alignment 契约到新增 hint 的发射仍待下一阶段 |
| 持续验收 | 专用 runner 已建立，M3-01 的默认分支每日调度仍待合并启用 |

本阶段不宣布整个 M3 完成。下一步先完成 alignment 契约到 hint 的来源记录、
发射与真实 GPU 对照，再做 M3 最终退出审计；主分支调度是独立的未完成项。

## 运行入口

```sh
PYTHONPATH=src ci/gpu/.venv/bin/python -m pytest -q
PYTHONPATH=src ci/gpu/.venv/bin/python tools/gpu_audit.py --output artifacts/ci-gpu
```

CPU 新增 26 项（11 项能力/回归与 15 项 example golden）；GPU 新增 36 个节点，
纳入严格 gate 与附件重放。全量基线为 CPU 910 passed、GPU 151 节点/295 案例，
均零 skipped。源文件、失败数据和测试节点仍由统一审计入口保存。
