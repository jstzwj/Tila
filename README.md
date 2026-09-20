# Tila

Tila 是一门以 Python 语法承载、面向 GPU kernel、生成 Triton 源码的静态
强类型 DSL。`@tila.jit` 不执行函数体，而是读取 Python AST，建立 Tila
自己的 HIR/TIR，并在运行前检查 dtype、shape、控制流、内存访问能力和
bounds obligation。

Tila 的 checker、CPU reference interpreter 与 GPU 后端仍持续接受正确性审计。同一份 typed TIR
可以生成 Triton 源码，也可以交给 NumPy reference interpreter 执行。Triton
和 CUDA 路径已有 RTX 3090 / SM86 固定验证组合。公开仓库不连接开发者本地机器，
当前通过本地工具进行 GPU 验收，不提供自动 GPU CI。已实现 [launch/target 门禁](docs/launch-target.md)、
[编译诊断与资源检查](docs/backend-diagnostics.md)、[操作/dtype 审计](docs/gpu-operation-audit.md)
及 [alignment hint 来源与缓存隔离](docs/alignment-hints.md)。跨架构验证仍未闭环，
因此后端整体状态仍为 `Partial`。

当前暂缓功能扩展，优先[正确性收口](docs/correctness-closure.md)：修复已确认的
错误 Safe 与跨层语义差异。边界结论依赖记录的前提及启动门禁，不代表一般无竞争、
同步正确或结果确定性；静态分支不同 shape 的分支外使用已保守拒绝。

> 类型不只描述“值是什么”，还描述 tile shape、编译期常量、读写能力和
> 内存访问成立的条件。

当前能力的唯一状态清单见 [docs/status.md](docs/status.md)。设计目标与实施顺序
分别见 [docs/design-principles.md](docs/design-principles.md) 和
[plan.md](plan.md)。

当前开发版本为 `0.3.0.dev0`：按 [ADR-013](docs/adr/013-const-bool-domain.md)
新增 `Const[bool]`（仅原生 Python True/False，CLI 使用 `--const FLAG=true/false`）。
`Const[int]` 的 ExactInt 门禁保持不变；该扩展不回移至 0.2.x，也不代表正式 0.3.0 已发布。
宿主侧可用 `ti.host_int(np.int64(128))` 显式生成 Python int，详见
[ADR-014](docs/adr/014-host-integer-normalization.md)；该函数不能在 kernel 内调用。
kernel 内可用 `ti.constant[ti.f16](0.1)` 显式构造 RNE 舍入常量；支持
f16/bf16/f32/f64，按位保存结果，详见 [ADR-015](docs/adr/015-rounded-typed-constants.md)。

---

## 最小示例

<!-- tila-example: current; mode=exec -->
```python
import tila as ti

N = ti.Dim("N")


@ti.jit
def add_kernel(
    x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    y: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
    out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
    BLOCK: ti.Const[int, ti.PowerOfTwo] = 128,
):
    pid = ti.program_id(0)
    offsets = pid * BLOCK + ti.arange(0, BLOCK)
    mask = offsets < N

    a = ti.load(x, offsets, mask=mask)
    b = ti.load(y, offsets, mask=mask)
    ti.store(out, offsets, a + b, mask=mask)
```

这里 checker 可以静态确认：

- `a`、`b` 和输出元素都是 `f32`，store 不发生隐式转换；
- `offsets` 和 `mask` 的 shape 一致；
- `out` 具有写能力，`x/y` 只有读能力；
- mask 谓词可以证明三个内存访问均不越界。

完整可执行版本位于 [examples/add_kernel.py](examples/add_kernel.py)。

---

## 当前已验证能力

以下只列 `docs/status.md` 中标为 `Implemented` 的核心能力。

- **静态前端**：`inspect.getsource → ast.parse → HIR → typed TIR`，kernel
  函数体不会作为普通 Python 执行；
- **数值类型检查**：signed/unsigned/float 域分离，只允许安全 widening，
  narrowing 与 int/float 混算需要显式 `ti.cast[...]`；
- **Shape 检查**：符号维、受限 DimExpr、broadcast、reshape numel、dot
  内维和 load/store 坐标 shape；
- **值类别**：Scalar、Block、Mask、Unit 与 `Const[int]`；
- **当前 Buffer 用法**：dtype/shape/access/alignment 绑定，Buffer 坐标形式的
  masked load/store 和读写 capability 检查；
- **控制流**：runtime-if、模块常量 constexpr-if、Const 参数 static-if、
  `ti.range` 循环、loop-carried 类型稳定性和裸 `return`；
- **核心内建**：`program_id`、`num_programs`、`arange`、`range`、`zeros`、
  `load/store`、`unsafe_load/store`、`cast`、`where`、f16 `dot`、`sum/max`、
  `exp/exp2`、`reshape`、`assume`、单参数 `static_assert` 和
  `mask.any()/all()`；
- **Bounds safety**：逐访问 obligation、区间/非负事实、谓词 DAG、cdiv/grid
  contract、`launch_auto`、ProofResult/信任来源以及 strict/warn 策略；
- **审计**：`check`、`dump`、`build`、`explain`，可查看类型、事实、effect
  汇总、obligation 和 proof trace；
- **CPU reference interpreter**：官方 add、softmax、matmul、self-attention 和
  fused-attention 示例都有跨平台 subprocess smoke test；
- **事实驱动 lowering**：已证明的 contiguous 和特定 `pid*STEP` 模式会生成
  `tl.max_contiguous` 与 `tl.multiple_of`。

这份摘要不替代逐项状态表。Ptr 的 ADR-001 公共语法与 ADR-002
RegionId/Extent/alias 拆分已经实现；多维 `buf.ptr` 在 v0 明确拒绝，FP8、
alignment 优化反馈和 Triton 后端仍是 `Partial`，因此没有放进上述完整能力承诺。
单参数 `static_assert` 支持 Stage 1 谓词和 Const specialization 谓词；
`Const[int]` 的默认值、CLI、materialize/explain 与 launch override 均采用
ExactInt 门禁，不接受 bool 或可转换对象。

---

## 编译与执行管线

```text
Python 子集 + ti.* 命名空间
    ↓  @ti.jit：读取源码并建立 HIR，不执行 kernel 函数体
    ↓  Stage 1：子集、dtype、shape、capability、控制流、obligation
    ↓  Stage 2：Const、延迟 shape 约束、grid/launch facts、bounds 结论
    ↓  Typed TIR
    ├─ NumPy reference interpreter（当前持续验证路径）
    └─ Triton source lowering（GPU 支持矩阵与 CUDA CI 尚待完成）
```

Bounds 当前采用默认 Z3 通用证明器，保留区间、谓词和 grid-contract 小型捷径。
[ADR-011](docs/adr/011-smt-proof-and-trust.md) 已决定在 M2 将 Z3 作为默认通用
证明引擎；M2-02 已用布尔 DAG 替代强制 DNF，并分离证明结论与信任来源。
M2-03 已接入锁定版本的 Z3、Int/BitVec 编码、资源预算和来源敏感缓存；无法证明
的访问保持 `Unknown`，在默认 strict 模式下拒绝。

M2-01 已按 [ADR-007](docs/adr/007-integer-semantics.md) 实现整数回绕、floor
除法/取模、转换与移位定义域，并在每次 launch 前验证符号索引的中间溢出和
shape/grid/标量范围。复杂数据依赖定义域目前保守拒绝，完整 SMT 尚待实施。
新增整数 GPU 对照可运行 `PYTHONPATH=src python tests/gpu_integer_smoke.py`；
已有本地验证，不构成完整 GPU CI 或跨版本支持承诺。

M2-08 统一验收入口：`PYTHONPATH=src python tools/gpu_audit.py`，固定环境、
归约契约、覆盖范围及失败重放见 [GPU 语义审计](docs/m2-gpu-audit.md)。
完整本地通过仍不代表已有持续 GPU CI。

M3 的固定依赖重建、本地 GPU 验收、官方示例覆盖与 launch 选项见
[GPU 支持矩阵](docs/gpu-support.md)；初始组合仅 RTX 3090 / SM86。

---

## 安装与验证

推荐使用 uv 同步开发环境：

```bash
uv sync --extra dev
uv run pytest -q
```

也可以使用 pip：

```bash
python -m pip install -e ".[dev]"
pytest -q
```

开发依赖包含 bf16 interpreter 测试所需的 `ml_dtypes`。标准开发环境要求
全量测试零 skipped；README 不硬编码测试数量，当前验证数字记录在
[docs/status.md](docs/status.md) 中。

运行 CPU 示例，无需 GPU：

```bash
python examples/add_kernel.py
python examples/softmax.py
python examples/matmul.py
python examples/self_attention.py
python examples/fused_attention.py
```

---

## CLI

```bash
python -m tila check examples/add_kernel.py
python -m tila check examples/add_kernel.py --explain
python -m tila dump examples/matmul.py
python -m tila build examples/add_kernel.py -o build/
python -m tila run examples/add_kernel.py
```

- `check`：运行 Stage 1/2 检查并输出 obligation 摘要；
- `explain`：输出类型、事实、hint、effect 和 proof trace；
- `dump`：输出 canonical TIR；
- `build`：生成 `.triton.py` 与 `.tir.txt`；
- `run`：执行示例文件自己的 `main()`。

CLI 在 Windows 窄编码终端下会主动配置 UTF-8，相关 cp1252 场景有回归测试。

---

## 当前限制

以下能力不是当前完整承诺。精确状态和边界见 [docs/status.md](docs/status.md)：

- Triton/CUDA 目前只验证 RTX 3090 / SM86 固定组合；本地验收的环境锁与重放方式见
  [GPU 支持](docs/gpu-support.md)。
  编译资源门禁和语句级 source map 已实现，其他架构及完整性能分析仍未验证；
- Ptr 公共语法与 RegionId/Extent/alias 模型已定型；Buffer v0 不公开
  Strides/AddressSpace，`buf.ptr` 只允许 rank-1、stride-1；显式多维 flatten
  和更丰富 pointer arithmetic 尚未设计；
- FP8 仅保留前端 storage/cast 类型规则，固定 GPU target 的 build/launch 明确拒绝；
- alignment 声明逐次校验；一维连续 Buffer/Ptr 已反馈为基地址 hint，其他布局仍保守；
- effect 已有逐访问记录、控制流汇总和独立 `where` 策略；最小 atomic_add 与
  限定 Race 子集已实现，未知顺序/数据流保持 Unknown；uniformity 已有内部分析和
  可选 `--show-uniformity` 输出，尚无同步消费；
- TypeVar、公开 capability 集合、target database 和性能诊断尚未实现；
- 默认 SMT 证明器为 Z3，尚无公开可插拔 solver 接口；已发射 hint 记录来源，五个官方
  示例已有 TIR、Triton source 和 explain golden，M3 退出审计已完成但仍有阻塞项；
- `full/trans/cat/min/log/sqrt/rsqrt/abs/floor/ceil` 等内建尚未实现。

这些限制是显式的工程状态，不会被默认为“由 Triton 自动支持”。

---

## 文档

| 文档 | 用途 |
|---|---|
| [docs/status.md](docs/status.md) | **当前能力的唯一状态清单** |
| [docs/m1-exit-audit.md](docs/m1-exit-audit.md) | M1 冻结项、退出条件与自动化证据 |
| [docs/m3-exit-audit.md](docs/m3-exit-audit.md) | M3 退出审计、已通过范围与剩余阻塞项 |
| [docs/cpu-ci.md](docs/cpu-ci.md) | GitHub 托管 CPU 回归/golden 配置与本地复现 |
| [docs/gpu-support.md](docs/gpu-support.md) | 固定 GPU 环境、本地验收与重放 |
| [docs/gpu-operation-audit.md](docs/gpu-operation-audit.md) | 操作/dtype 证据、支持边界与 M3 退出条件核查 |
| [plan.md](plan.md) | M0–M6 实施计划、ADR、任务台账和退出标准 |
| [docs/adr/README.md](docs/adr/README.md) | 已接受的公共语法与核心 IR 架构决定 |
| [docs/design-principles.md](docs/design-principles.md) | 语言定位、设计目标与非目标 |
| [docs/surface-language.md](docs/surface-language.md) | Python 子集、staging 和 launch 协议 |
| [docs/type-system.md](docs/type-system.md) | 类型系统完整设计；含部分未来语法，使用前对照状态表 |
| [docs/refinements.md](docs/refinements.md) | refinement、事实传播、assume/unsafe 与 hints |
| [docs/bounds-safety.md](docs/bounds-safety.md) | proof obligation、grid contract 和证明结论 |
| [docs/intrinsics.md](docs/intrinsics.md) | 内建签名设计与阶段标注 |
| [docs/effects.md](docs/effects.md) | effect/race/atomic/uniformity 设计；多数属于未来阶段 |
| [docs/roadmap.md](docs/roadmap.md) | M0–M6 里程碑摘要与 FP8/Effect/Target/TypeVar 阶段归属 |

---

## 项目状态

当前工作区是 2026-09 的语言重置实现。旧设计和旧实现已按仓库规则移动到
`trash/`，不再代表当前 Tila。

M0/M1/M2 已完成；M3-02 至 M3-05 已完成 launch/target 门禁、后端诊断、操作/dtype
审计及限定范围的 alignment hint 发射。M3-06 退出审计已完成，结论为 **NOT READY**。
固定环境支持本地 GPU 验收；GitHub 托管 CPU CI 已验证矩阵提交 `4774194`，回归/golden
附件已核实。[矩阵补测](docs/m3-matrix-followup.md)已覆盖限定形状的 dot f16 输出、
zeros/reshape 与 add 多 dtype；自动 GPU CI 已撤下，隔离 GPU 持续验收仍缺失。

近期工作优先级是：

1. 后续修改持续运行托管 CPU CI 与本地 GPU gate，保留矩阵边界及失败重放；
2. 有独立 GPU 资源后建立隔离 GPU CI；目前没有 runner，不连接开发者机器补位；
3. [M4-01c](docs/effect-summary.md)已从 TIR 派生 path/mask/loop 与只读 effect 汇总；
   [M4-01d](docs/effect-audit.md)已固定可选详细输出与缓存／绑定隔离验收。
   [M4-02](docs/adr/010-effect-diagnostic-policy.md)已实现 where 急切读取检查与独立
   `TILA_EFFECTS=off|warn|error`／`--effects` 策略（默认 warn）。
   [M4-03b](docs/atomic-cpu.md)已实现最小 atomic_add 前端、Effect IR 和 CPU 参考；
   [M4-03c](docs/atomic-gpu.md)已完成固定 RTX 3090 GPU lowering 与本地严格验收。
   [M4-04b/c](docs/race-launch-audit.md)已接入 Race 精确子集与启动门禁：
   `TILA_RACE=off|warn|error`，默认 warn；确认冲突拒绝，Unknown 告警；
   [M4-04d](docs/race-exit-audit.md)已完成小域枚举、变形与限定覆盖退出审计；
   [M4-05b](docs/uniformity-analysis.md)已冻结 ADR-019，实现内部值/控制分析与 verifier；
   [M4-05c](docs/uniformity-audit.md)已接入可选详细输出、golden 与绑定隔离验收；
   barrier/shared memory 不在本轮范围，隔离 GPU CI 缺失，M3 仍未完成。

实施进度以 [plan.md](plan.md) 的任务台账为准。
