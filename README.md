# Tila

Tila 是一个 **strongly-typed、shape-safe、layout-aware** 的 tile-based GPU 编程 DSL，定位为 **Triton 之上的类型化 frontend**。它不重新发明 GPU 编程模型：API 采用与 Triton 同层级的 `tila` 命名空间（`tila.jit` / `tila.program_id` / `tila.arange` / `tila.load` / `tila.store` / `tila.cast`），把 Triton 隐含的 dtype / shape / layout 关系显式化并静态化。

```
Tila Source（Python 可判定子集 + tila 固有命名空间）
   │  ast.parse + 子集校验（E11–E15）
   ▼
tila_ast ── intrinsic resolution（tila.load → LoadIntrinsic）
   │  type check（dtype → shape → layout）＋ launch analysis（独立 phase）
   ▼
TIR ── typed straight-line SSA（语义核心，每个值带完整类型）
   │  total、确定、无新语义拒绝
   ▼
Triton Source ──► Triton 编译器 ──► TTIR/TTGIR ──► PTX ──► GPU
```

核心思想：

- **Triton-compatible programming model，stricter static semantics**。表面语法与 Triton 同层级：`import tila`、`@tila.jit`、`tila.program_id`、`tila.arange`、`tila.load(a + offs, mask=mask)`、`+`。Triton 留给运行期/JIT 的 dtype 提升、隐式广播、store 隐式转型，Tila 全部前移到编译期显式检查。Tila 不是 "prettier Triton"，而是"拥有自己静态语义、再编译到 Triton 的小型语言"。
- **内建 = compiler intrinsic，不是 Python 函数**。`tila.load(...)` 由 frontend 识别为 `IntrinsicRef("load")` 并做编译期签名检查（`docs/language-spec.md` §2）；`tila` 模块对象在 Python 层只是 placeholder。
- **类型分层**：`ScalarType(dtype)` / `TileType(dtype, shape, layout)`；`Buffer` / `Address` 是编译器内部类型——指针语义上存在、语法上隐藏（`a + offs` 被 check 为 `Address`）。layout 是**语义类型信息，不是用户语法**。
- **dtype universe 与 Triton 对齐、运算语义刻意收紧**：bool、i8–i64、u8–u64、FP16/BF16/FP32/FP64、四种 FP8（存储 dtype：算术一律 E16）；无隐式提升与隐式转换，跨 dtype 必须显式 `tila.cast(x, tila.float32)`，否则 E02。
- **shape 是迷你类型系统**：`Const` / `Symbol` / `Product`；广播规则收窄（相等或 size-1，其余 E03）；v0.1 无 shape 算术（引入即需要 Presburger 求解器，不做）。
- **Layout Algebra 是编译器内部的语义系统**：layout term（identity / broadcast / product）+ 化简律 L1–L5，等价判定 = normalize 后结构相等。用户看不到 layout（"layout 是语义，不是语法"）。
- **TIR 是项目真正的核心**：typed、SSA、canonical、layout-aware、backend-complete；TIR 不做类型推断（类型在 checker 已全部 resolve）。GPU codegen 全部交给 Triton：lowering 对有效 TIR 全函数（total）、确定、不做新的语义拒绝。
- **两类错误严格分开**：E01–E17 是 Tila 语言错误（编译期拒绝）；B01–Bxx 是 Triton 后端兼容性错误（`docs/triton-lowering.md` §8）。
- **桥接问题（开放）**：Tila 抽象 layout 代数 ↔ Triton concrete encoding（BlockedEncoding / LinearEncoding / CTA layout…）如何对接，是 v0.2+ 的核心研究问题，由 Phase 0 Triton oracle 与 differential testing 兜底（`docs/development-plan.md` §2/§8）。

## 规范示例

```python
import tila


@tila.jit
def add(
    a: tila.Tensor[tila.float32, N],
    b: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    pid = tila.program_id(0)

    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N

    x = tila.load(a + offs, mask=mask)
    y = tila.load(b + offs, mask=mask)

    z = x + y

    tila.store(c + offs, z, mask=mask)
```

启动（由生成的 launcher 完成，N 从 `a` 的形状自动读取，grid 由 launch analysis 推导）：

```python
add[grid = (triton.cdiv(N, BLOCK),)](a, b, c, BLOCK=128)
```

## 文档（v0.1 定稿，2026-08 评审修订）

| 文档 | 内容 |
|---|---|
| `docs/semantic-model.md` | 语义模型：值类别（Constexpr/Scalar/Tile/Buffer/Address）、执行模型、内存与指针模型、内建语义合同、constexpr 模型、翻译语义 |
| `docs/language-spec.md` | 表面语言：Python 子集 + `tila` 固有命名空间、文法 EBNF、内建函数表、名字/单赋值规则、符号维与 constexpr、启动语法、允许/禁止清单 |
| `docs/type-system.md` | 类型分层、dtype 域与能力表、shape 迷你系统与广播、layout 代数与化简律 L1–L5、typing rules R1–R12、错误示例 |
| `docs/ast.md` | tila_ast 节点定义（含 intrinsic resolution）、pyast→tila_ast 转换表、TIR（直线 SSA）定义与不变量、canonical dump 格式 |
| `docs/type-checker.md` | 规则驱动检查器、ARITH_RULES 机器可读分派矩阵、内建检查流程、normalize/equiv、诊断目录 E01–E17、add 走查、测试策略 |
| `docs/triton-lowering.md` | total/deterministic lowering 映射表、值命名与匿名内联、launcher 模板、compile_kernel/CLI、后端校验层 B01–Bxx、黄金验收 |
| `docs/v0.2-preview-2d.md` | 预览片段：符号维（batch 可变）+ 二维 tiling、`tila.expand_dim`/size-1 广播/`&`、`Product` 项与律 L5、batched_add 走查 |
| `docs/v0.2-matmul-fragment.md` | matmul fragment：R16 dot、Mma layout term、R9' 内存边界、E18、桥接假说 H1 的检验结果 |
| `docs/layout-oracle-notes.md` | Stage 0 oracle 实测笔记：单 encoding 不变量、encoding 参数规律、MMA 观测（H1 强形式证伪/家族分离成立） |
| `docs/development-plan.md` | 开发路线 Stage 0–6（Triton oracle → minimal Tila → 强类型 → layout 代数 → 2D → 后端校验 → 差分测试）、测试体系、第一竖切清单 |

## 示例

```
examples/add.tila        第一个 kernel：向量加法（v0.1 规范内最大程序）
examples/add.lowered.py  期望的 Triton 输出（带注释参考版；字节级黄金文件见 lowering 文档 §2）
examples/fp8_add.tila    FP8 存储 dtype 范式：load → cast → 计算 → cast → store
examples/fp8_add.lowered.py  对应的期望 Triton 输出
examples/saxpy.tila      标量参数 + 标量 cast（alpha 经 tila.cast 进入浮点运算）
examples/masked_add.tila masked load 的 other 语义（other=0.0）
examples/matmul.tila     matmul fragment：tila.dot + MMA 布局（v0.2 片段二）
examples/batched_add.tila     符号维（batch=M、特征=N 运行时可变）+ 二维 tiling（v0.2 预览）
examples/batched_add.lowered.py  对应的期望 Triton 输出
```

## 快速上手

**方式一：像 Triton 一样在 Python 里直接用**（`pip install -e .` 后）：

```python
from __future__ import annotations   # 符号维 N 出现在注解里 → PEP 563 延迟求值
import tila

@tila.jit
def add(a: tila.Tensor[tila.float32, N], b: tila.Tensor[tila.float32, N],
        c: tila.Tensor[tila.float32, N], BLOCK: tila.constexpr = 128):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    x = tila.load(a + offs, mask=mask)
    y = tila.load(b + offs, mask=mask)
    z = x + y
    tila.store(c + offs, z, mask=mask)

add(a, b, c, BLOCK=64)   # torch.Tensor(cuda) → GPU；np.ndarray → CPU 解释器
```

`@tila.jit` 运行时与 CLI 走**同一套编译管线**（子集校验 / checker E 码 /
launch analysis / total lowering），按 (源码, constexpr overrides) 缓存——每个
override 组合一次完整编译（TIR 是特化产物，semantic-model.md §6）；grid 与
符号维自动推导。kernel 里的语言错误在调用时同样以 E01–E17 拒绝。

**方式二：CLI（.tila 源文件，黄金测试的规范路径）**：

```bash
python -m tila build examples/add.tila -o build/ --explain   # 生成 build/add.triton.py + add.tir.txt
python -m tila dump  examples/add.tila --constexpr BLOCK=64  # 打印特化后的 canonical TIR
python -m tila run   examples/add.tila                       # GPU smoke（无 triton 时走 interpreter）
pytest                                    # 全量测试（GPU 集成测试自动跳过）
scripts\run_gpu_tests.bat                 # Windows 下 GPU 差分（需 MSVC 环境）
```

Python API：`from tila import compile_kernel` → `compile_kernel(source, {"BLOCK": 128})` 返回
`(TKernel, Triton 源码, TIR dump, explain)`；失败抛 `TilaError`（E01–E17）。

## 实现结构

```
src/tila/
  api.py                 tila 命名空间：jit / constexpr / 内建 placeholder（Python 层不可调用）
  intrinsic.py           INTRINSICS 静态表 + tila 表面名字白名单
  frontend/
    parser.py            ast.parse 包装（SyntaxError → E11）
    desugar.py           pyast → tila_ast（子集校验 + intrinsic resolution，E11–E15）
  ast/
    nodes.py             tila_ast 节点定义
  types/
    dtype.py             dtype 域 + 能力表（§1.1）
    shape.py             Shape 迷你系统（Const/Symbol/Product）+ 广播 ⊗
    layout.py            Layout term + normalize/equiv（律 L1–L5）
    type.py              ScalarType / TileType / BufferType / AddressType / UnitType
  checker/
    checker.py           check_kernel / Env / 推导（规则驱动）+ 匿名 id 终态化
    arithmetic.py        运算分派矩阵（ARITH_RULES）
    builtin.py           内建检查流程（program_id/arange/load/store/cast）
    broadcast.py         ⊗ 与 shape 规则（E03/E04）
    layout.py            layout 推导与 equiv 检查（E05）
  tir/
    ops.py               TIR 指令
    printer.py           canonical dump（LayoutNamer：L0/L1/… 按首现序命名）
  backend/
    triton/
      validator.py       后端校验层（B01–B03）
      lowering.py        TIR → Triton 源码 + launcher（total，无新语义拒绝）
      printer.py         源码渲染（命名、最小括号、匿名内联、use_count 决策）
  launch/
    analysis.py          launch analysis（E17；tiling 惯用法识别）
  interp.py              reference interpreter（TIR → NumPy，差分 oracle）
  diagnostics.py         TilaError / BackendError / 诊断渲染（E 码 / B 码）
  driver.py              compile_kernel / CLI（build、run、dump）
tests/
  golden/                add/fp8_add/saxpy/masked_add 的 .tir.txt 与 .triton.py（逐字节比对）
  test_golden.py、test_diagnostics.py（E01–E17 矩阵）、test_types.py（L1–L5）、
  test_checker.py、test_lowering.py、test_interp.py、test_fuzz.py、test_driver.py、
  test_gpu_integration.py（marker gpu）
tools/
  dump_triton_layout.py  Stage 0 Triton oracle（arange/load/add/mask 的 concrete encoding）
scripts/
  run_gpu_tests.bat      Windows 下 GPU 差分（triton 后端需 MSVC 环境 + CC=cl）
```

## 开发路线

**Stage 0 先写最小 Triton oracle，不写编译器**——手写 add 的 Triton kernel，对 BLOCK × num_warps × dtype 跑通并检视 TTIR/TTGIR，验证 Tila 的类型假设（arange/broadcast/load/add/mask 的实际 encoding 行为，即 "identity 抽象是否够用"）；随后 frontend → types → checker + TIR → lowering → GPU differential test。完整路线与验收见 `docs/development-plan.md`。

## 状态与里程碑

设计文档定稿（2026-08 按外部评审重写）；**v0.1 编译器已实现**：
frontend（子集校验 + intrinsic resolution）→ checker（R1–R12、E01–E17、launch
analysis）→ TIR（canonical dump）→ total Triton lowering + launcher，以及
reference interpreter（TIR→NumPy，CPU 差分 oracle）与后端校验层（B 码）。

**v0.2 matmul fragment（`docs/v0.2-matmul-fragment.md`）已实现**：`tila.dot`（R16）、新的非擦除 layout term `Mma`（第一个不由 arange 推导的分布）、R9' 内存边界修订（store 布局无关，坐标语义）、E18；GPU 上与 torch.matmul 对拍一致。桥接假说 H1 完成首次检验：家族级分离成立、强形式证伪（详见 oracle 笔记）。

**v0.2 二维预览（`docs/v0.2-preview-2d.md`）已实现**：`tila.expand_dim`（R13）、
size-1 广播（R14，逐轴 Product 推导）、`&` 合取（R15）、律 L5、rank-2 注解与
`program_id(1)` 解禁、二维 grid launch analysis；`batched_add` 全链路走通
（TIR dump 与文档 §4 走查逐行一致、GPU 与 torch 对拍通过）。

**Stage 0 Triton oracle 已完成**（`docs/layout-oracle-notes.md`）：48 组
BLOCK × num_warps × dtype 实测——1D/2D kernel 内全部张量值共享唯一 concrete
encoding，`identity`/`Product` 抽象在实测范围内成立；encoding 参数是
(BLOCK, num_warps, dtype) 的函数（元素守恒 + 16B 向量化上限 + order 于 2D 生效）。

验收状态：

- **黄金测试**：`add.tila` → `add.tir.txt` / `add.triton.py` 与文档规范内容
  **逐字节一致**；fp8_add / saxpy / masked_add / batched_add 同样纳入黄金。
- **GPU 正确性**：生成的 Triton 源码经真实 Triton 编译器在 GPU 上运行——
  `add` 与 `torch.add` 在 N ∈ {1, 127, 128, 129, 1000} × BLOCK ∈ {32, 128}
  逐元素一致；`masked_add` / `batched_add` 通过同一 TIR 的 interpreter/GPU
  双路径对拍；`fp8_add` 在 SM < 8.9 的卡上由 B01 正确拦截
  （interpreter + ml_dtypes 覆盖语义）。
- **测试**：190 用例——E01–E17 诊断矩阵、layout 律 L1–L5、R13–R15 与
  Product 推导、typing 规则、lowering 单元（最小括号/匿名内联/dtype 表）、
  interpreter 差分（1D/2D）、变异 fuzz（无原生异常泄漏）、GPU 集成。

下一步（按优先级）：**stride 化 MemoryLayout**（消灭手工线性化盲区——matmul 实现期间再次实证）→ **K 循环与累加**（打破单赋值，完整 matmul）→ 内建扩充与归约。