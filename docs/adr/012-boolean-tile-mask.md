# ADR-012：布尔 tile 与 Mask 的受限互操作

- 状态：**Accepted**（M2-07a 已实现）
- 日期：2026-09-19
- 阶段：M2-07a；优先实施，作为加法式接口变更评审
- 关联：ADR-011；不改变 ADR-007 的整数规则

## 问题与不变量

比较产生 Mask，加载 bool Buffer 产生 Block[bool]。后者需要用于执行掩码，但其
内容未知，不能凭 dtype 获得任何边界事实。当前 checker 的 mask/where/布尔组合
入口原有类型限制；本决策不把两套类型整体合并，也不让 tile 隐式进入 scalar-if。

## 推荐语法与类型规则

不新增 `as_mask` API；仅在布尔消费位置做受控转换。

<!-- tila-example: current; mode=syntax -->
```python
enabled = ti.load(flags, offsets, mask=offsets < N, other=False)
active = enabled & (offsets < N)
value = ti.load(data, offsets, mask=active, other=0.0)
```

| 消费位置 | 提案规则 |
|---|---|
| load/store 的 mask、where 的条件 | 接受 Scalar[bool]、Mask[S]、Block[bool,S]；沿用既有广播检查 |
| `&`、`\|`、`~` | tile 布尔组合返回 Mask[广播后的 shape]；纯标量返回 Scalar[bool] |
| `and/or/not` | 仅标量，保持短路；tile 使用按 lane 的 `&/\|/~` |
| `.any()`、`.all()` | Mask/Block[bool] 显式归约为 Scalar[bool]；不把结果当作逐 lane 边界事实 |
| scalar-if、static_assert | 不新增 tile 接受规则；标量的 staging 仍需独立检查 |
| 数值算术、cast、bool Buffer store 的 value | 不因本提案开放 Mask→数值或 Mask→可存储 bool 的转换 |

整数的位运算保持原规则。禁止把 i8/u8 的 0/1 当成布尔 tile。where 两个数值
分支仍遵守相同 dtype 规则；shape 失败在 checker/特化期报现有 shape 诊断。

## 内部表示与证明

继续使用 MaskT、BlockT(ScalarT(bool)) 和共享 Predicate DAG，不增加运行时转换
指令。新增集中式布尔消费者适配函数，返回 `(shape, bool operand, predicate)`，
供 mask、where、布尔组合、归约复用，避免入口各自放宽。

比较保留符号谓词；加载布尔值默认生成未知身份。复制同一个 SSA 值保留身份，
不同 load（包括同地址的重复 load）不自动等同；重赋值、循环 widening、分支选择
沿用数据流规则。展开/广播保留相对 lane 轴映射，不能把行条件当列条件。

`enabled & in_bounds` 可由 in_bounds 证明安全；`enabled | in_bounds`、单独
enabled 通常不能。safe load flags 不证明其内容与 data 索引有关。
bool load 的 other=False 在 CPU/GPU 执行中生效，第一版证明仍可对加载值保守
使用未知谓词；不得凭“masked load”捏造 enabled⇒其加载 mask 的关系。
显式 assume 仍标 UserAssumption；unsafe 不产生事实。

## 兼容性、实现与验收

已有合法程序类型和行为不变；原本拒绝的一部分布尔消费者获得支持。错误路径
沿用 TYPE/SHAPE 族并显示 found/expected。更新 registry、状态表、类型规范，
同步 CPU/Triton、explain golden 与 M2-05 性质矩阵后才标 Implemented。

验收覆盖：加载/比较混合、标量与行列广播、错误 dtype/shape、未知条件 OR 的
反例、重复值/独立加载身份、分支/循环事实隔离、debug assume、masked-off lane，
以及 CPU/GPU 相同输入的布尔执行对照。没有 GPU 证据时明确标注未验证。

## 未选择的方案

- 全面合并 Mask 与 Block[bool]：会连带改变 cast、存储、归约和公开类型打印。
- 任意 bool 都携带边界事实：混淆执行谓词与逻辑证明，可能误判安全。
- 仅增加显式 as_mask：同样需要完整的身份/广播规则，却给常见组合增加语法负担。

## 实现证据

共享布尔消费者适配已接入 checker 的 mask、where、组合和归约入口；复用既有
TIR，修正 CPU/Triton 标量 bool 取反，registry semantic revision 升为 3。
mask 允许向访问形状广播，但不能增加访问 rank。assume 的符号合取限制保持。

`tests/test_boolean_tiles.py` 的 39 项专项覆盖执行、身份/轴隔离、分支/循环、Ptr、
debug assume 与审计 golden。`PYTHONPATH=src python tests/gpu_boolean_smoke.py`
的 68 组 CPU/GPU 对照通过（RTX 3090、PyTorch 2.10.0+cu128、Triton 3.6.0、CUDA 12.8）；
这不是更广 target 支持矩阵或持续 GPU CI 承诺。
