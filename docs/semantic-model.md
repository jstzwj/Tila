# Tila 语义模型

状态：定稿（2026-08 评审修订）。本文回答"一个 Tila 程序意味着什么"：值类别、执行模型、内存与指针模型、内建语义合同、constexpr 模型、翻译语义与正确性合约。类型规则见 `type-system.md`，表面文法见 `language-spec.md`，IR 见 `ast.md`。

---

## 1. 值类别

Tila 的全部值分为六类，逻辑层次如下：

```
Value
  ├── Constexpr           编译期整数（形如 BLOCK；arange 边界、shape、grid 的事实来源）
  ├── Scalar              运行期标量（pid、符号维 N、字面量）
  ├── Tile                运行期块（寄存器驻留：load 的结果、算术的对象）
  ├── Buffer              全局数据对象（kernel 参数；指针语义上存在、语法上隐藏）
  ├── Address             地址值（Buffer + Tile[i32] 偏移，仅存在于 TIR 与内部检查）
  └── unit                store 的返回类型
```

| 类别 | 出现层 | 编译期可见性 | 示例 |
|---|---|---|---|
| `Constexpr[int]` | 参数注解、kernel 体 | 全部可见（值被特化进产物） | `BLOCK` |
| `Scalar[i32]` / `Scalar[f32]` | kernel 体 | 类型可见，值不可见 | `pid`、`N`、`2`、`0.0` |
| `Tile[dt, Σ, L]` | kernel 体 | 类型（dtype/shape/layout）可见 | `offs`、`x`、`mask` |
| `Buffer[dt, Σ]` | kernel 参数 | dtype 与 shape 可见（注解） | `a` |
| `Address[dt, Σ, L]` | 内部（`a + offs` 的结果） | 全部可见 | — |
| `()` | 语句 | — | `store` 的返回 |

要点：

- **标量值域只有 `i32` / `f32`**：宽 dtype（f16、fp8…）只存在于 Tile/Buffer 元素层面，不给标量配 17 种类型。整数字面量默认 `Scalar(i32)`、浮点字面量默认 `Scalar(f32)`；在类别匹配的上下文中按该位置的 dtype 重新解释（规则 R12）。
- **没有独立的 IndexTile 类别**：索引块就是 `Tile[i32, Σ, L]`。数据/索引的区分由**用法**建立（索引 tile 进入 `a + offs` 与 mask 谓词），不再由类型类别区分——这是相对初稿的简化。
- `Buffer` 只有 dtype 与 shape（注解给出），没有 layout：它不参与分布；layout 属于下一次 load 产生的 Tile。

---

## 2. 执行模型

v0.1 完整采用 Triton 的 program model：

```
launch
  │
  ├── program_id(0)
  ├── tile（arange + 偏移构造索引块）
  ├── load（按 mask 取全局数据进寄存器）
  ├── compute（tile 逐元素运算）
  └── store（按 mask 写回全局）
```

- 一次 launch 启动 `grid = (ceil_div(N, BLOCK),)` 个 program；`program_id(0)` 即 program 索引。
- **Tila 是 tile-programming DSL，不是 CUDA-thread DSL**：用户永远不写 threadIdx / blockIdx / warp / lane / register。v0.1 的 kernel 是直线型（straight-line）：无循环、无分支、无自定义函数；每个 program 顺序执行函数体，单赋值保证每个名字恰好被求值一次。
- program 之间无通信、无共享内存、无同步（v0.1）。
- 调用方契约：多个张量参数共享同一符号维（如 `a`、`b`、`c` 都是 `N`）时，launcher 在运行期断言这些张量的对应维长度相等（§7）。

---

## 3. 内存模型

- **全局内存，row-major，无 stride 注解**（v0.1 唯一 MemoryLayout；保留这一层的命名空间为将来 stride/swizzle/TMA 用）。
- **随机访问是乱的（scatter/gather）与否由用户表达，编译器不做静态边界证明**：越界与否由 mask 表达（与 Triton 一致）。
- **masked load**：mask 为假的通道跳过读；`other` 给出时该通道的值为 `other`，未给出时未定义。
- **masked store**：mask 为假的通道跳过写。
- 因此 `add` 中越界通道上的未定义值永远不会被写入越界通道——程序整体正确。

---

## 4. 指针模型（typed abstract memory）

**指针语义上存在、语法上隐藏**：Tila 表面语言没有指针类型、没有 `ptr_add` 之类的指针内建；用户只写最自然的 `a + offs`。

```
Buffer[dt, Σ]                        # kernel 参数（注解给出 dtype/shape）
  +  Tile[i32, Σ', L]                # 索引块
  ↓
Address[dt, Σ', L]                   # 内部类型：load/store 的消费对象
  ↓ load
Tile[dt, Σ', L]                      # 数据进寄存器，dtype 来自 Buffer
```

检查器把 `a + offs` 判定为 `Address`（规则 R7），`tila.load(addr, ...)` / `tila.store(addr, value, ...)` 消费它；**指针算术只存在于生成的 Triton 代码**（`tl.load(a + offs)` 中的 `a + offs`），TIR 中的 `addptr` 指令是唯一显式指针构造点。

---

## 5. 内建语义合同

内建是 **compiler intrinsic**，不是 Python 函数：`tila.load(...)` 在 Python 层的实现体只是 `raise RuntimeError("tila.load can only be used inside @tila.jit")`，永远不会被执行。frontend 对 `tila.<name>` 做 intrinsic resolution（`language-spec.md` §2/§4），checker 按合同静态检查。v0.1 五个内建的语义合同：

### 5.1 `tila.program_id(axis)`

```
axis : Constexpr[int] ∈ {0}          # v0.1 收紧；v0.2 预览扩至 {0,1}
─── tila.program_id(axis) : Scalar(i32)
```

返回当前 program 在 axis 方向上的索引。语义与 Triton `tl.program_id` 一致。轴必须编译期确定——否则无法推导 grid（E13）。

### 5.2 `tila.arange(start, end)`

```
start = 0（字面量）
end   : Constexpr[int]，特化值 = 2^k（1 ≤ k ≤ 20）
─── tila.arange(0, end) : Tile[i32, (endᵛ,), identity(endᵛ)]
```

构造长度为 `end`（取 build 期特化值 `endᵛ`）的稠密索引块，元素为 `0, 1, …, end-1`。语义与 Triton `tl.arange` 一致；**约束（起点 0、2 的幂、≤ 2^20）是 Tila 的主动收紧（检查口径 E06），不随 Triton 版本摆动**。

### 5.3 `tila.load(ptr, mask=?, other=?)`

```
ptr   : Address[dt, Σ, L]
mask  : Tile[bool, Σ', L']，Σ' ⊗ Σ 良式且 L' ~ L（v0.1：Σ' ≡ Σ）（可选）
other : 与 dt 类别匹配的字面量（可选；给出时必须同时给出 mask，否则 E13）
─── tila.load(ptr, mask=m, other=o) : Tile[dt, Σ, L]
```

每个 program 从 `ptr`（= Buffer + 索引）读取 `Σ` 个元素进寄存器。mask 为假的通道按 §3 处理。v0.1 要求掩码形状与索引块相同（无广播收窄差异）；`other` 限定字面量是 v0.1 的表面语言边界——Triton 的 `other` 可为任意块并广播，将来放开时按 R5 的前提扩展即可。

### 5.4 `tila.store(ptr, value, mask=?)`

```
ptr   : Address[dt, Σ, L]
value : Tile[dt', Σ', L']，dt' = dt（严格相等，无隐式转型），Σ' ⊗ Σ 良式且 L' ~ L（v0.1：Σ' ≡ Σ）
mask  : Tile[bool, Σ, L]（可选）
─── tila.store(ptr, value, mask) : ()
```

逐元素写回。Triton 的 `tl.store` 允许 value 的隐式广播与类型转换；**Tila 刻意移除这两个隐式**（dtype 严格相等 E02、shape 严格 E03）——显式是卖点，不是缺口。store 的返回类型是 `()`，只能作为表达式语句出现（E08）。

### 5.5 `tila.cast(x, dtype)`

```
x     : Tile[dt₁, Σ, L] 或 Scalar[dt₁]
dtype : tila.<dt₂>（DTypeRef）
─── tila.cast(x, dtype) : Tile[dt₂, Σ, L] / Scalar[dt₂]
```

数值 dtype 之间任意互转（cast 矩阵全开，含 bool 与 FP8 两个端点；FP8 作为 cast 的源与目标都合法，限制只在算术侧）。语义 ≡ Triton `.to(dtype)`。**隐式转换不存在**——一切跨 dtype 都必须显式 cast，这既是静态可判性的来源也是诊断建议的标准答案（`use tila.cast(rhs, tila.float32)`）。

---

## 6. constexpr 模型

`BLOCK: tila.constexpr = 128` 按调用点特化：

- **检查逐值特化**：每个 override 值对应一次完整的 check → TIR → lowering；TIR 是该值的**特化产物，跨值不复用**（按 BLOCK=128 检查过的 TIR 不允许拿去跑 BLOCK=256——host 需要其他值时以新 override 重新编译，编译是廉价纯函数，driver 按 (源码, overrides) 缓存产物）。
- **发射保留名字**：生成的 Triton 里仍是 `tl.arange(0, BLOCK)`、形参仍是 `BLOCK: tl.constexpr`、不折叠——保持与源码 1:1 的同构和稳定的黄金输出；Triton 对每个 launch 值做自己的 jit 特化。Triton 自身的编译期校验因此只是纵深防御，不承担 Tila 的健全性。
- constexpr 支持整数运算（`BLOCK // 2` 是合法 arange 边界）：**检查期**求值（求值失败 → E10），**发射期**按表达式原样渲染。
- constexpr 在需要 `Scalar(i32)` 的位置自动提升（R3）；整数字面量同理。

---

## 7. 翻译语义与正确性合约

**翻译语义（translational semantics）**：Tila 的 v0.1 不独立给出执行语义——一个 Tila kernel 的含义 ≡ 它 lowering 出的 Triton kernel 的含义。由此得到两条硬性合约：

1. **lowering 对有效 TIR 是 total 且确定的**：所有可判定的属性（dtype、shape、layout、capability）已被 checker 消耗；lowering 只查表发射，同一输入必得同一输出（黄金测试逐字节比对）。lowering 中的失败只可能来自 Triton 侧（§8 后端校验层），不算 lowering 的失败。
2. **运行期契约由 launcher 的断言执行**：rank 断言（每张量注解 rank）、静态维断言（注解为字面量的维）、符号维相等断言（同一符号绑定多个张量时为其 shape 对应维）。

## 8. 编译期 / 运行期分界

| 层 | 决定什么 | 失败时 |
|---|---|---|
| Tila checker | dtype、shape、layout 等价、capability、launch plan | E01–E17（编译期，不产出任何 Triton 代码） |
| Tila lowering | 逐 op 的 Triton 文本（total，无失败） | — |
| 后端校验层 | Triton 接受产物：fp8 架构可用性、tile numel 上限、encoding 可实现性 | B01–Bxx（"backend validation"，不走 E 码） |
| launcher | 运行期断言：rank、静态维、符号维相等；grid 计算 | Python AssertionError |
| Triton 编译器 | TTIR → TTGIR → PTX（layout inference / conversion） | Triton 自身诊断 |

本表是"强类型"的工程含义：**凡可判定属性，全部在 checker 前移消化；Tila 侧永远不产生半路失败。**