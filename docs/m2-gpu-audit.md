# M2-08 CPU/GPU 语义审计

本文保留 M2-08 当时的 70 节点/214 案例验收证据。统一入口在 M3 扩展为
160 节点/304 案例，当前固定环境、CI 与重放方式见 [GPU 支持](gpu-support.md)。

日期：2026-09-19。契约见 [ADR-008](adr/008-reduction-precision.md)，固定环境见
[ADR-009](adr/009-gpu-validation-baseline.md)。

## 运行与重放

```sh
PYTHONPATH=src python tools/gpu_audit.py
PYTHONPATH=src python tools/gpu_audit.py --case 'test_reduction and u64'
PYTHONPATH=src python tools/gpu_audit.py --exploratory
```

每次创建独立 `artifacts/m2-gpu-audit/<UTC时间>-<唯一后缀>/`，不覆盖旧目录。
`report.json` 记录基线、实际环境、包版本、版本差异、seed、revision、dirty 状态、验收结论；
`results.xml` 与 `pytest.log` 记录每个稳定参数化节点、断言、局部变量和源码位置；
`source/` 保存当时 tracked/untracked 源文件，`tracked.patch` 保存相对 HEAD 的差异；
`kernels/` 和 `tmp/` 保留生成内核，`replay.json` 保存命令与必要环境。

重放当前代码可用统一入口的 `--case`。重放旧失败时进入记录的 `source/`，按
ADR-009 准备环境，使用 `replay.json` 的执行设置，再运行
`PYTHONPATH=src python -m pytest 'tests/gpu_semantics.py::<节点ID>' -v --showlocals`。
`replay.json` 中绝对路径代表原始运行，复制目录后需更新路径。
测试输入是固定边界集合与独立 `np.random.default_rng(208009)`；复现不依赖运行顺序。
入口固定 strict、关闭 Tila/Triton interpreter 强制开关、关闭 debug 和 pytest 外部
插件自动加载，防止宿主环境把 GPU 验收变成解释执行或改变测试选择。
旧四项均为固定枚举，不使用全局 RNG；常量 oracle 为独立有理数舍入参考。
归约失败直接保留小型 4×8 输入，未宣称具有通用自动最小化器。

## 覆盖

| 专项 | 语义案例 | 比较标准 |
|---|---:|---|
| 整数既有专项 | 23 | 精确值；回绕、除模、移位、cast、循环边界 |
| 布尔 tile/mask | 68 | 穷举布尔输入、广播、尾部 mask、any/all |
| Const bool | 11 | 分支、缓存重复绑定、短路 |
| 显式浮点常量 | 46 | f16/bf16/f32/f64 位模式、独立 oracle |
| sum/max | 48 | 12 dtype × 2 算子 × 2 轴；整数独立参考、浮点误差界 |
| 归约特殊值 | 8 | 4 浮点 dtype × 2 算子；NaN、无穷、零分类 |
| masked reduction | 3 | 全负输入与补齐 lane 的 neutral value |
| 非连续 view/广播 | 2 | offset、转置、正 stride、零 stride、完整 backing sentinel |
| tail cast | 5 | 多 program、尾部长度、i32→i16 回绕边界 |

70 个 pytest 节点（四项既有专项为聚合节点），总计 214 个语义案例。

本地验收记录：`artifacts/m2-gpu-audit/20260919T101034Z-xevl10cx/report.json`，
70 passed、零 skipped，`accepted: true`；环境完全匹配 ADR-009。
CPU 全量回归为 832 passed、零 skipped。该 GPU 记录的基准 commit 为
`c955ee6`，本次未提交改动已保存在记录的 `source/` 和 `tracked.patch`，
不能将结果误记为未修改的 c955ee6 已具备 M2-08。审计附件本地保留，不纳入 git。

## 本次发现与修复

- 宽无符号 PyTorch dtype 未绑定：补齐 uint16/uint32/uint64 映射，验证高位值与归约；
  未支持 dtype 的错误路径也改为结构化 dtype mismatch，避免访问 None.name。
- 窄整数 max：lowering 显式恢复输入 dtype，防止后续算术继承 Triton 默认提升。
- max NaN：CPU np.max 传播 NaN，Triton 默认 max 不传播；改成显式传播 combine。
- bool/FP8 numeric reduction 与 bool axis 曾缺少能力域门禁，现与既有算术域一致拒绝。
- load 浮点 tile 直接转整数仍无法证明域，保持 TILA-NUM-001；不为通过对照放松门禁。

这些对照不扩大安全证明器范围。负 stride GPU、一般次正规归约、完整 dot/FP8、
跨架构及持续 CI 仍未验证；其余限制见 ADR-009。
