# Tila 表面语言与编译管线

状态：设计基线（2026-09-17 设计重置）。
前置阅读：`type-system.md`（类型规则）、`intrinsics.md`（内建签名）。

Tila 以 Python 语法子集为表面语言、以 `tila` 固有命名空间为唯一
内建入口，由 `@tila.jit` 触发独立编译管线。本文定义：语言子集、
两阶段检查、launch 协议，以及一个端到端走查。

---

## 1. 宿主原则：解析源码，不执行 Python

`@tila.jit` 在**装饰时**取回源码并建立 Tila 自己的 IR：

```text
inspect.getsource(fn) → ast.parse → Tila HIR（名字解析 + intrinsic 解析）
```

不使用运算符重载 tracing、不执行函数体。理由：

- 错误位置直接对应源码行（AST 带行列）；
- 控制流、作用域、单/多赋值语义由 Tila 自己定义，不受 Python
  动态语义污染；
- 定义期检查（Stage 1）在 import 时即可报错，不必等到调用。

代价是语言子集必须收窄（§2）——这是刻意交换：**静态可判定性优先**。

---

## 2. Python 子集

### 2.1 允许

| 构造 | 说明 |
|---|---|
| 函数定义 | `@tila.jit` 修饰，单返回类型路径（v0：**裸 `return`（提前退出）合法；值 return 拒绝——TILA-SYN-021，结果写入输出 buffer） |
| 语句 | 赋值、 augmented（仅 `+=` 于显式累加器，v1）、表达式语句、裸 return |
| `if / elif / else` | StaticIf（Const 条件，含 Const 参数的 TStaticIf 延迟到特化期）或 If（scalar bool），§4 |
| `for i in tila.range(...)` | 唯一循环形式；step 为 Const ≥ 1；循环体内定义的变量是循环局部的（循环后使用 → TILA-TYPE-023） |
| 元组 | 仅在内建明确定义处：坐标访问 `(rows, cols)`、launch 的 grid 元组 |
| 注解 | kernel 参数注解 = Tila 类型；`from __future__ import annotations` 支持 |
| 模块级 | `Dim/ConstDim` 声明、TypeVar 声明、被 jit 函数引用的 Const 标量 |

### 2.2 禁止（TILA-SYN）

闭包与嵌套函数、`while`、`break/continue`（v1 评估）、列表/字典/
集合推导、任意 Python 调用（`tila.*` 白名单之外）、`import` 于函数体、
全局可变状态读写、`try/except/with`、类定义、lambda、字符串/None
参与运算、`is/in`、动态属性访问。

清单式拒绝、错误指到具体构造：

```text
error[TILA-SYN-003]:
while-loops are not part of the Tila subset
    line 12:  while k < K:
use:  for k0 in tila.range(0, K, BK):
```

### 2.3 `tila` 命名空间

`import tila` 后，kernel 体内**只有** `tila.*`（内建表，
intrinsics.md）、kernel 参数、本函数局部变量、模块级 Const/Dim
符号可见。Python 内建（`len/max/abs/...`）一律不可见——防止
"看起来能用，实际语义不明"。

---

## 3. 两阶段检查

### 3.1 Stage 1：定义期（装饰时）

```text
Python AST → Tila HIR → 名字解析 → 符号化类型检查 → 约束生成
```

检查（全部不依赖具体特化值）：

- 子集合法性（TILA-SYN）；
- dtype 层：混算、字面量、cast、store 值类型（TILA-TYPE）；
- shape 层：broadcast、dot 内维、reshape numel、坐标 shape
  （TILA-SHAPE）；
- Const 层：runtime 值误入 Const 语境（TILA-CONST）；
- 内存层：写 ReadOnly、元素/字节 offset（TILA-MEM 的静态部分）；
- 控制流：分支合并同型、loop-carried 稳定；
- intrinsic 签名匹配（按约束表，而非逐个手写检查器）；
- bounds 义务生成与**符号可判部分**的预证明。

产出：`Typed Tila HIR + 约束池`（约束池在特化前保持符号形态）。

### 3.2 Stage 2：特化期（调用时）

launch 提供的事实代入约束池：

```text
T      ← 实参 dtype（TypeVar 绑定）
BLOCK  ← constexpr 实参/override
N, M   ← 张量实际 shape（符号绑定）
对齐   ← data_ptr() 实测（校验 Aligned[k]）
target ← backend/arch/triton_version
```

求解剩余约束：

- Const 精化（PowerOfTwo / Range / MultipleOf）；
- shape 等价约束的数值判定；
- 硬件能力（dtype 支持的 dot 组合、shared memory 上限等，
  TILA-TARGET，intrinsics.md §4）；
- 对齐声明校验（TILA-MEM-003）。

通过后：`Typed HIR → Tila TIR（typed SSA）→ Triton 源码生成 →
Triton 编译 → cubin`，并按（源码哈希, 特化键）缓存。

任一阶段失败的错误都带：码、位置、已知类型/事实、修复建议。

---

## 4. Staging 规则汇总

| 语境 | 要求 | IR 节点 |
|---|---|---|
| `tila.arange` 界、`range` 的 step、shape 里的 Const 维 | `Const[int]` | 折叠 |
| Stage 1 `if` 分支消除 | 模块级已知 bool | 直接折叠（未选分支不进 IR） |
| Const 参数 static-if | 只依赖 `Const[int]` 的 staged bool | TStaticIf（特化期选择） |
| `if` 运行分支 | scalar bool | If |
| Block 级条件执行 | 不存在——masked load/store + `where` | — |

混合形式 `if BLOCK >= 128 and n > 0:` 含 runtime 标量，因此整体是 runtime-if，
不能享受静态分支消除；需要按 Const 特化裁掉外层分支时，应拆成嵌套两层。

---

## 5. Kernel 签名

<!-- tila-example: future; milestone=M5 -->
```python
M = tila.Dim("M");  N = tila.Dim("N")
T = tila.TypeVar("T", bound=tila.Float)

@tila.jit                      # 或 @tila.assume_launch(N % BLOCK == 0) 叠加
def kernel(
    x:    tila.Buffer[f16, (M, N), tila.ReadOnly],   # Buffer/Ptr/标量/Const
    y:    tila.Buffer[T,  (M, N), tila.WriteOnly],   # TypeVar
    alpha: tila.f32,                                  # 运行期标量（可带精化）
    N:    tila.i32,                                   # 维符号的运行期形态
    BLOCK: tila.Const[int, tila.PowerOfTwo] = 128,    # Const 默认值
):
    ...
```

- 参数顺序自由；维符号（M/N）在注解中使用、由 launcher 绑定；
- 标量参数的精化注解 = launch 契约（type-system.md §6.4）；
- 默认值仅 Const 参数可有。

---

## 6. Launch 协议

<!-- tila-example: current; mode=syntax -->
```python
kernel[(grid,)](x, y, alpha=2.0, N=N, BLOCK=128)
kernel.launch_auto(x, y, alpha=2.0)   # grid 由 launch analysis 推导
```

`launch_auto` 的推导规则：对每个被使用的 pid 轴，从义务中找
`pid*STEP + lane`（或裸 pid 标量坐标）模式，`grid_ax = ceildiv(维,
STEP)`，并把推导结果直接登记为 grid 契约（bounds-safety.md §5.3）；
无法推导时报 TILA-TYPE-105。

launcher 职责（顺序固定）：

1. **绑定**：dtype/shape/strides/data_ptr/device 从张量提取；
   Const 参数取实参或默认；
2. **契约检查**：标量精化、`assume_launch` 谓词、别名检查（可选）；
   失败抛 `TilaLaunchContractError`（显式异常，非 `assert`）；
3. **特化缓存查询/生成**：键 =（源码哈希, dtype 集, Const 集,
   target, 对齐类）；
4. **grid 契约登记**：`cdiv` 模式识别 → pid 事实注入
   （bounds-safety.md §5）；
5. **发射**：Triton kernel launch（Buffer 展开为 ptr+strides+dims）。

---

## 7. 端到端走查：add_kernel

### 7.1 源码

<!-- tila-example: current; mode=exec -->
```python
import tila

N = tila.Dim("N")


@tila.jit
def add_kernel(
    x:   tila.Buffer[tila.f32, (N,), tila.ReadOnly],
    y:   tila.Buffer[tila.f32, (N,), tila.ReadOnly],
    out: tila.Buffer[tila.f32, (N,), tila.WriteOnly],
    BLOCK: tila.Const[int, tila.PowerOfTwo],
):
    pid = tila.program_id(0)
    offs = pid * BLOCK + tila.arange(0, BLOCK)
    mask = offs < N
    a = tila.load(x, offs, mask=mask)
    b = tila.load(y, offs, mask=mask)
    tila.store(out, offs, a + b, mask=mask)
```

### 7.2 Stage 1：类型与事实

```text
pid  : i32                facts: 0 <= pid < grid0
offs : Block[i32, (BLOCK,)]
       facts: pid*BLOCK <= offs < pid*BLOCK + BLOCK,
              0 <= offs, contiguous(offs, BLOCK),
              基址 pid*BLOCK : MultipleOf[BLOCK]
mask : Mask[(BLOCK,)]     pred: offs < N
a, b : Block[f32, (BLOCK,)]
a+b  : Block[f32, (BLOCK,)]        同 dtype 同形，无提升 ✓
```

义务（三条，逐条）：

```text
load x:  mask ⇒ 0 <= offs < N
         ⇒ 由 pred 给上界、facts 给下界 → ProvenSafe ✓
load y:  同上 ✓
store out: 同上 ✓；Access=WriteOnly 允许 store ✓；值类型 f32 精确匹配 ✓
```

race（v1 启用）：store 地址 = `pid*BLOCK + lane`，跨 pid 仿射
不相交 ✓。

### 7.3 Stage 2：特化（`BLOCK=256`，一维 `cdiv(N, BLOCK)` grid）

```text
Const 精化: 256 是 PowerOfTwo ✓
pid 事实: 0 <= pid < ceildiv(N, 256)          （grid 契约注入）
对齐:     data_ptr() % 16 == 0 → Aligned[16]  （实测通过）
```

### 7.4 Lowered Triton（期望输出）

<!-- tila-example: generated -->
```python
@triton.jit
def add_kernel_lowered(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)   # ← 事实自动发射
    mask = offs < N
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)
```

（`max_contiguous/multiple_of` 的发射形态以 refinements.md §4 的
规则与 Triton API 适配为准；示例展示意图。）

launcher 侧生成：张量绑定、`N`/grid 推导、契约（本例无）、
`(ptr, N, BLOCK=256)` 参数展开。

---

## 8. 诊断渲染规范

M2-06 的稳定文本契约见 [audit explain v1](explain-audit.md)：默认固定章节与
信任/反例分类；`--show-witness`、`--show-cache`、`--show-query` 分别开启具体
见证、缓存遥测和原始 SMT-LIB。关键 bounds 错误同样支持这些附件。

M4-04c 增加独立 `--race off|warn|error`（优先于 `TILA_RACE`，默认 warn）。
确认冲突在执行前拒绝，Unknown 默认告警，error 模式也拒绝 Unknown；off 明确记录
未检查，不关闭 bounds/target/atomic 合法性门禁。`--show-races` / Python
`explain(show_races=True)` 显示 `tila.race-details.v1`，可组合上述三个附件选项。
无实际 launch 绑定时保持 pending；详细范围见 [Race 启动验收](race-launch-audit.md)。

M4-05c 增加 `--show-uniformity` / Python `explain(show_uniformity=True)`，
用于 explain 或 check --explain 的可选 `tila.uniformity-details.v1` 输出。
从当前 TIR/Const 重算，不使用 launch 绑定，无缓存、无同步消费；默认输出不变。
值层级、控制参与、Unknown 与预算边界见 [Uniformity 审计](uniformity-audit.md)。

完整活动错误码、phase/severity 元数据和维护门禁见
[`diagnostics.md`](diagnostics.md)。机器事实来源是 `DIAGNOSTIC_REGISTRY`。

所有 Tila 错误/警告统一结构：

```text
error[TILA-<FAMILY>-<NNN>]: <一句话>
    <源码行与脱字符标注>
<为什么错：涉及的类型/事实，各自完整形态>
required: <规则>
known:    <已知事实>
fix:      <至少一条可执行建议>
```

实现要求：

- E 码（TILA-*）是语言层错误，与后端/目标错误（TILA-TARGET）、
  launch 错误（`TilaLaunchContractError`）三类分开；
- 同一诊断在 CLI（`python -m tila check`）与运行时（调用触发）
  两个入口共享同一渲染路径；
- 非 debug CLI 不泄漏 Python traceback；未知内部故障包装为
  `TILA-INTERNAL-001`，仅 `TILA_DEBUG=1` 原样抛出；
- `--explain` 输出：类型环境、事实集、义务证明链（四态 + 证明路径：
  命中的谓词及其来源 / 数值区间 / cdiv 战术与契约）、发射的 hint、
  效应汇总——全部可审计。入口：`python -m tila explain <file>` 或
  `python -m tila check <file> --explain`（grid 契约为 launch 期事实，
  explain 模式下相关义务显示为 Unknown/待 launch）。
