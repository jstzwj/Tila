# Tila 设计原则与定位

状态：设计基线（2026-09-17 设计重置）。本文件是全部 Tila 文档的总纲。
旧设计文档（2026-08 基线及其 v0.x 系列）已整体归档至 `trash/`，不再有效。

---

## 1. Tila 是什么

Tila 是一门 **Python 语法承载的、面向 GPU kernel 的静态强类型 DSL**，
最终编译到 Triton。它不是 "给 Triton 加 type hints"，而是一门拥有自己
静态语义的小型语言，只是复用 Python 的表面语法与 Triton 的执行模型：

```text
Tila Source（Python 可判定子集 + tila 固有命名空间）
    ↓  @tila.jit：ast.parse → Tila HIR（不走 Python 执行/运算符重载建图）
    ↓  Stage 1 定义期检查：名字解析 + 符号化类型检查 + 约束生成
    ↓  Stage 2 特化期检查：调用时求解剩余约束（dtype/Const/target/alignment）
    ↓  Typed Tila IR（typed SSA）
    ↓  Triton lowering → Triton 编译器 → GPU
```

一句话定位：

> **Triton 把 GPU 编程从 CUDA 里解放出来；Tila 把 GPU kernel 的正确性
> 从 "跑起来才知道" 里解放出来。**

Tila 与 Triton 的分工：

| | Triton | Tila |
|---|---|---|
| 编程模型 | SPMD tile 编程 | 相同（同层级 `tila.*` 命名空间） |
| dtype 提升/转换 | 运行期/隐式 | 编译期显式，禁止隐式混算 |
| shape/broadcast | 动态检查或静默广播 | 静态类型检查（shape 是类型的一部分） |
| 内存安全 | 越界即 UB | 编译期证明，或显式 `unsafe` 逃逸 |
| 写只读内存 | 运行期才知道 | 编译期拒绝 |
| 优化 hint（multiple_of 等） | 用户手工 | 类型系统推导并自动发射 |

**编译目标锁定 Triton**：Tila 不直接生成 PTX，Triton 是唯一后端。
这保证 Tila kernel 的性能下限等于手写 Triton 的性能下限。

---

## 2. 类型系统的灵魂：四根支柱

> **Tila 的类型不只描述"一个值是什么"，还描述"这个值有多少个、它在哪里、
> 它能不能被写、它索引哪里、它在什么条件下有效"。**

对应四根支柱，缺一不可：

1. **Shape（有多少个）**——shape 是类型的一部分：
   `Block[f16, (M, K)]`；`dot` 的内维匹配、broadcast 合法性、reshape
   numel 守恒全部是类型检查，不是运行期检查。

2. **Refinement（在什么条件下有效）**——精化谓词进入类型：
   `Const[int, PowerOfTwo]`、`i32 where 0 <= x < N`、`Aligned[16]`。
   由此派生两大能力：**静态边界证明**（bounds-safety）与
   **自动优化事实**（alignment / multiple_of / contiguous）。

3. **Pointer capability（它能不能被写、指向哪里）**——指针类型携带
   Access（ReadOnly/WriteOnly/RW）、AddressSpace、Region、Alignment；
   `store` 到 `ReadPtr` 是编译错误。

4. **Effect（它索引/读写哪里）**——内存操作携带 `Read[X]/Write[Y]/Atomic[Z]`
   效应；函数签名由 `(A, B) -> C` 升级为 `(A, B) -> C ! {Read[A], Write[B]}`。
   效应是 race 检测、`where` 陷阱诊断、uniformity 检查的基础。

这四层叠加在一个普通的 scalar 语义层（严格数值转换、字面量多态）之上，
构成 Tila 的完整类型语法（见 `type-system.md`）。

---

## 3. 受限 dependent typing：刻意不上完整依赖类型

完整 dependent types（任意表达式进入类型）最终会变成 theorem prover
项目，既难实现也难给出好的错误信息。Tila 刻意收窄为
**restricted dependent typing**：

- **Shape types**：`Block[T, (M, N)]`，shape 表达式是符号仿射式；
- **Refinements**：`0 <= x < N`、`x % 16 == 0` 等线性/仿射谓词；
- **Compile-time values**：`Const[int]` 及其精化；
- **Type-level predicates**：`PowerOfTwo / Aligned[k] / MultipleOf[k]` 等
  有限公共谓词；`Contiguous / StrideEq[axis, value]` 是 launch 推导的内部事实。

判定性来自两个限制：**谓词限定为 Presburger（线性 + 整除）算术**，
**求解分层（区间抽象解释 fast path + 可插拔 Presburger/SMT slow path）**。
GPU kernel 的索引计算 90% 是 `a*x + b`、比较、取模、整除——这个子集
足够覆盖，且可判定、可给出人话错误信息。

---

## 4. 静态化原则：Triton 留给运行期的，Tila 前移到编译期

Triton 的若干动态语义是 GPU kernel bug 的主要来源。Tila 逐项收紧：

| Triton 行为 | Tila 规则 |
|---|---|
| 混合 dtype 自动 promotion（signed/unsigned 同宽取宽者等） | 禁止隐式混算；只允许安全 widening（见 type-system.md §6）；其余显式 `tila.cast` |
| `store` 把 value 静默转换到指针元素类型 | dtype 必须精确匹配，否则 TILA-TYPE 错误 |
| load/store 的 mask/value 静默广播 | 广播合法性静态验证（相等或 size-1） |
| 越界访问是 UB | 每个 load/store 产生 proof obligation；证不出则 TILA-BOUNDS |
| `where` 两侧都会求值（含 load） | 效应系统识别并诊断（TILA-EFFECT） |
| `multiple_of/max_contiguous` 手工 hint | 类型精化自动推导、lowering 自动发射 |
| runtime 值传给 constexpr 参数 | TILA-CONST 编译错误 |

**收紧是刻意的**：Tila 的规则独立于 Triton 版本，是 Tila 自己的语义；
Triton 是它的编译目标，不是语义来源。

---

## 5. Tila 要静态抓住的 bug

这张表是整个项目的验收矩阵（标注 v0 = MVP 范围，见 roadmap.md）：

| Bug | Tila 结论 | 阶段 |
|---|---|---|
| f16 值写进 i32 指针 | error（TILA-TYPE） | v0 |
| signed/unsigned 混算 | error（TILA-TYPE） | v0 |
| narrowing cast 无显式标注 | error（TILA-TYPE） | v0 |
| 字面量超出目标 dtype 表示范围 | error（TILA-TYPE） | v0 |
| dot 内维不匹配 | error（TILA-SHAPE） | v0 |
| broadcast 不合法 | error（TILA-SHAPE） | v0 |
| reshape numel 不守恒 | error（TILA-SHAPE） | v0 |
| mask shape 与指针/值不匹配 | error（TILA-SHAPE） | v0 |
| runtime 值用于 constexpr | error（TILA-CONST） | v0 |
| block mask 当作 Python bool（`if mask:`） | error（TILA-TYPE） | v0 |
| store 到 ReadOnly buffer/指针 | error（TILA-MEM） | v0 |
| load/store 坐标 shape 与 buffer 不匹配 | error（TILA-SHAPE） | v0 |
| 指针元素类型错误 | error（TILA-MEM） | v0 |
| 元素 offset 与字节 offset 混用 | error（实现注：byte_offset 未排期，调用即拒 TILA-SYN-050；元素 offset 的类型规则已生效） | v0 |
| 显式越界（可证） | error（TILA-BOUNDS） | v0 |
| masked load 证不出安全 | error / 需 assume 或 unsafe（TILA-BOUNDS） | v0 |
| alignment 精化与实际不符 | error（TILA-MEM，特化期） | v0 |
| 目标硬件不支持该操作 | error（TILA-TARGET，特化期） | v0 |
| `where` 分支中隐藏内存效应 | warning / strict 模式 error（TILA-EFFECT） | v1 |
| 明显的 inter-program 写冲突 | error（TILA-RACE） | v1 |
| 疑似 race（证不出不相交） | warning（TILA-RACE） | v1 |
| divergent barrier | error（TILA-UNIFORM） | v1 |
| uncoalesced 访问等性能问题 | warning（性能诊断，永不作为类型错误） | v2 |

原则：**正确性问题用 error，可实现性问题用 error（特化期），性能问题
只用 warning**。性能偏好不得伪装成类型错误。

---

## 6. 逃逸舱：`unsafe` 与 `assume`

静态分析一定会遇到无法证明的合法 kernel（gather/scatter 的数据依赖
索引就是典型）。Tila 提供两个显式逃逸舱，而不是静默放行：

- **`tila.unsafe_load / tila.unsafe_store`**：放弃该访问的边界证明义务，
  义务转移给程序员。unsafe 操作在诊断报告中显式汇总，永远可见。
- **`tila.assume(pred)`**：向 checker 注入一条 refinement fact。
  debug 构建下 lower 成 `device_assert`，release 构建下作为编译器假设。

设计要求：**逃逸必须刺眼**——不能是全局开关，只能逐访问/逐谓词声明；
`--safety=warn` 可以把证不出的 bounds 从 error 降为 warning，但默认
严格。

---

## 7. 类型事实即优化事实

Tila 的类型推导不只防 bug，还反向喂给 Triton 优化器：

```text
checker 推导的事实                         lowering 自动发射
─────────────────────────────────────    ─────────────────────────
idx = pid*BLOCK + arange(0, BLOCK)  ⇒    tl.max_contiguous(idx, BLOCK)
                                            tl.multiple_of(idx, BLOCK)  （当 BLOCK 是 MultipleOf[k]）
p : Ptr[f16, ..., Aligned[16]]      ⇒    相应对齐假设进入地址计算
Const[int, PowerOfTwo] 精化          ⇒    向量化/掩码优化的静态前提
```

长期方向是把 `multiple_of / max_contiguous` 这类手工 hint 彻底从用户
接口里去掉，由 refinement 系统接管（见 refinements.md §4）。

---

## 8. 非目标（v0 明确不做）

- 完整 dependent types / 通用 theorem proving；
- borrow checker、复杂 alias analysis、shared-memory typestate；
- warp-level ownership / warp specialization 的类型化；
- 复杂 layout 类型系统（v0 只保留 layout 作为内部优化事实层，
  不进用户类型语法，见 type-system.md §11）;
- 多后端（CUDA 之外的目标在 v2 之后）；
- 性能的精确预测（v2 只做结构性性能警告）。

---

## 9. 文档地图

| 文档 | 内容 |
|---|---|
| `design-principles.md` | 本文：定位、四支柱、验收矩阵、非目标 |
| `type-system.md` | 类型语法全量：Scalar/Block/Mask/Const/Ptr/Buffer、DimExpr、数值转换与字面量、broadcast、控制流类型、泛型、双向类型推导 |
| `refinements.md` | 精化谓词体系、事实传播、assume/unsafe、类型事实 → Triton hint |
| `bounds-safety.md` | 边界证明：proof obligation、求解器架构、grid 契约、四态结论 |
| `effects.md` | 效应系统、race 检测、atomic 类型化、uniformity |
| `surface-language.md` | Python 子集、@tila.jit 两阶段检查、staging、完整走查 |
| `intrinsics.md` | 类型化内建签名表、硬件能力约束 |
| `roadmap.md` | MVP 8 项、阶段规划、与现有代码的关系、测试策略 |
