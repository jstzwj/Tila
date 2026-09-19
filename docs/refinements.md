# Tila 精化类型与编译器事实

状态：设计基线（2026-09-17 设计重置）。
前置阅读：`type-system.md`（尤其 §4–§6、§8–§9）。

---

## 1. 定位

Refinement 是 Tila 的第二根支柱（design-principles.md §2）：它把
"这个值在什么条件下有效"编码进类型，同时服务两个方向：

1. **安全方向**：为 bounds 证明、alignment 校验提供事实来源
   （bounds-safety.md）；
2. **优化方向**：把 Triton 的手工 hint（`multiple_of` /
   `max_contiguous`）从用户接口升级为编译器自动推导的静态事实。

当前可提取的符号谓词域主要为 **Presburger 算术**（线性 + 整除 + 常量乘），
外加有限的一组类型级谓词。M2 按 [ADR-011](adr/011-smt-proof-and-trust.md)
引入默认 SMT 与有限位宽整数编码；不会因为 Z3 支持某种理论就自动扩展公共语法。

---

## 2. 谓词目录

### 2.1 作用于 Const

```text
PowerOfTwo                 BLOCK: Const[int, PowerOfTwo]
Range[lo, hi]              闭区间（v0 端点为整数字面量，lo <= hi）
MultipleOf[k]              k 为正整数字面量
Positive / NonNegative     Range 的常用简写
```

### 2.2 作用于标量 / Block[i32]

```text
0 <= x < N                 区间精化（launch 契约形式，type-system.md §6.4）
x % 16 == 0                整除事实
```

运行期值的区间精化只能以 launch 契约或 `tila.assume` 引入，
不能凭空声明后未经校验就使用（§5）。

### 2.3 作用于 Ptr / Buffer

```text
Aligned[k]                 指针基地址按 k 字节对齐（正的 2 次幂，launch 实测）
Contiguous                 内部推导的连续布局事实
StrideEq[axis, value]      内部推导的轴 stride 事实（单位：元素）
```

`Contiguous` 和 `StrideEq` 在 v0 不作为用户可写注解；它们来自 launch 绑定，
详见 [ADR-003](adr/003-buffer-stride-address-space.md)。带参数 refinement 的唯一
构造语法与适用位置见 [ADR-004](adr/004-refinement-construction-syntax.md)。

### 2.4 语法

注解位置使用 Tila 自己的方括号语法（不依赖 `typing.Annotated`，
但 v0 实现可以接受 `Annotated` 作为同义糖）：

<!-- tila-example: current; mode=exec -->
```python
import tila as ti

X = ti.Dim("X")


@ti.jit
def constrained(
    p: ti.Ptr[ti.f16, ti.Global, ti.ReadOnly, X, ti.Aligned[16]],
    n: ti.i32 | ti.Positive,
    tile: ti.Const[int, ti.Range[32, 1024], ti.PowerOfTwo,
                   ti.MultipleOf[16]],
):
    pass
```

---

## 3. 事实传播

checker 在类型环境 Γ 中为每个值同时维护**类型**与**事实集** Φ：

```text
Γ, Φ ⊢ e ⇒ T ⊳ Φ'
```

### 3.1 来源规则（事实如何产生）

| 来源 | 产生的事实 |
|---|---|
| `tila.program_id(ax)` | `0 <= pid_ax < grid[ax]`（grid 契约，bounds-safety.md §5） |
| `tila.arange(0, B)` | `0 <= lane < B`，且 `contiguous(lane, B)` |
| `tila.range` 归纳变量 | `0 <= i < end` |
| launch 契约注解 | `Positive/NonNegative/Range` 注入区间事实；`MultipleOf` 注入模等式事实；声明均由 launcher 先验证 |
| `tila.assume(pred)` | `pred` 本身 |
| Const 精化 | `BLOCK` 的 PowerOfTwo / Range |
| 比较产生的 Mask | 谓词以共享 DAG 携带（`Mask[S] { offs < N }`），保留源位置、lane 与 broadcast 映射 |
| 加载/计算的 int 值 | 每个值获得符号身份（`__vN`），使 `assume` 与义务证明能作用于数据依赖值（gather 工作流） |

### 3.2 传播规则（事实如何流动）

以下为数学整数传播规则；用于 kernel 有限位宽值时，M2 必须先证明相关中间
运算不溢出，或按 BitVec 语义求解，不能直接套用：

```text
0 <= a < A,  0 <= b < B        ⇒  0 <= a+b < A+B
0 <= a < A,  c >= 0            ⇒  0 <= c*a <= c*(A−1)      （c 为 Const）
x % k == 0                     ⇒  x 是 k 的倍数
x ≡ y (mod k), y % k == 0      ⇒  x % k == 0
0 <= lane < B, contiguous      ⇒  idx = base + lane 是长度 B 的连续段
```

事实绑定值身份和作用域；赋值产生的新值不得继承旧值的不适用事实。分支合并
处取**交**（只保留共同可推出的部分），循环使用保守不变量。M2 派生事实必须
传递静态来源、已检查契约及用户假设依赖，避免作用域或缓存中的假设污染。

### 3.3 示例

<!-- tila-example: current; mode=syntax -->
```python
offs = pid * BLOCK + tila.arange(0, BLOCK)
```

推导产物：

```text
offs : Block[i32, (BLOCK,)]
facts:
    pid*BLOCK <= offs < pid*BLOCK + BLOCK
    contiguous(offs, BLOCK)          # 继承自 arange
    0 <= offs   （由 0<=pid、BLOCK>0、0<=lane）
```

## 4. 事实 → Triton hint：类型系统反哺优化

lowering 阶段把已证明的事实自动发射为 Triton 原语：

| Tila 事实 | 发射 |
|---|---|
| `contiguous(offs, B)`（offs 以 arange 为基） | `tl.max_contiguous(offs, B)` |
| `offs = pid*STEP + arange(0,B)` 且 STEP 为 Const/字面量 | 拆分基址后发射：`offs_base = pid * STEP; offs = offs_base + arange(...); tl.multiple_of(offs_base, STEP)`（soundness：`pid*STEP` 恒被 STEP 整除；TIR 不变，纯 lowering 变换） |
| `Aligned[k]`（Buffer/Ptr 声明并实测通过） | 地址计算的对齐假设；`k` 参与 load/store 向量化宽度推导 |
| 内部 `StrideEq[axis, 1]` | coalescing 分析的事实输入（性能诊断，v2） |

规范要点：

- **只有 Proven 的事实可以发射 hint**——Unknown/Exempted 永远不产生 hint
  （错误 hint 比没有 hint 危险）；
- 发射的 hint 集合在 `--explain` 输出中完整列出（哪个值、哪条事实、
  发射成什么），可审计；
- 当前没有公开 `tila.hint.*`；未来接口仍需独立设计。M2 起 hint 的依据需记录
  用户 assume 和契约依赖，不能将受信任前提包装成无条件静态事实；完整优化
  消费点覆盖仍归 M3/M6。

---

## 5. `assume` 与 `unsafe`：两个显式逃逸舱

### 5.1 `tila.assume(pred)`

<!-- tila-example: current; mode=syntax -->
```python
idx = tila.load(idx_ptr, mask=m)          # 数据依赖索引，静态证不出
tila.assume((idx >= 0) & (idx < N))       # 注入事实
x = tila.load(data, idx)                  # 现在 bounds 可证 ✓
```

语义：

- **checker**：`pred` 携带 UserAssumption 和源位置加入当前作用域 Φ，之后的访问点保存快照；
- **debug 构建**：lower 为 `tl.device_assert(pred)`——错误假设在
  运行期被抓；
- **release 构建**：仅作为编译器假设，零运行代价。

约束：`pred` 必须是 Presburger 域内谓词；assume 的谓词本身不检查
真伪（那是 device_assert 的事），但**每个 assume 都出现在诊断汇总里**。

M2-02 已使派生证明继承 UserAssumption 依赖，debug 执行检查不使它自动成为
release 的无条件事实。M2-03 已用 Z3 检查前提一致性；矛盾的假设返回 Unknown
并显示来源。当前 hint 仅从结构事实生成，记录 StaticFact 与源位置，不从 assume
生成优化 hint；未来新增这类推导也必须传播依赖。

### 5.2 `tila.unsafe_load / tila.unsafe_store`

放弃单次访问的 proof obligation：

<!-- tila-example: current; mode=syntax -->
```python
x = tila.unsafe_load(data, idx)     # 义务转移给程序员，报告汇总可见
```

与 assume 的区别：assume 给出**正向事实**（可复用于后续证明），
unsafe 只是**局部豁免**（不产生任何事实）。两者都不可嵌套、不可
全局开启。

M2-02 已使 unsafe 返回 Exempted，不再用 ProvenSafe 表示；不注入后续事实，
见 ADR-011。

### 5.3 严格度模式

```text
--safety=strict（默认）: Unknown 的 bounds obligation → error
--safety=warn          : → warning（kernel 可编译，风险自担）
```

实现机制：CLI 的 `--safety {strict,warn}` 设置环境变量
`TILA_SAFETY`，launch 与 `check`/`build`/`materialize` 共读同一开关；
降级仅作用于 `TILA-BOUNDS-001/002`，`TILA-BOUNDS-003`（可证越界）
任何模式都无条件 error。

模式下钻不影响其他类别的错误（race/uniformity 各自独立）。

---

## 6. 与其他文档的关系

- 事实如何被消费成边界证明：`bounds-safety.md`；
- Alignment 的特化期实测：`type-system.md` §9.4；
- PowerOfTwo 等 Const 精化如何进 specialization 约束池：
  `surface-language.md` §5（两阶段检查）；
- 效应/Region 精化（符号区域标签）：`effects.md`。
