# Tila 类型化内建签名表

状态：规范目标（2026-09-18 对齐 ADR-005/006）。
前置阅读：`type-system.md`。本文是内建的人类可读签名来源；当前可用性唯一以
[`status.md`](status.md) 为准。

---

## 1. 原则：机器 catalog + 专用语义 handler

按 [ADR-006](adr/006-intrinsic-registry.md)，每个内建在机器可读 registry 中登记名字、
表面形式、状态键、arity/keywords、checker handler、effect、可达 TIR、后端覆盖和 target
要求。复杂的 shape、bounds、staging 和类型推导继续由专用 handler 实现，不虚构一套已经
存在的声明式类型 DSL。M1-05 已实现完整 registry、显式 checker handler map 和 typed-TIR
backend capability 门禁；`status.md` §8.0 提供逐 spec 状态关联。

记号：

```text
T, U ∈ DType；S, Si ∈ Shape（DimExpr 元组）
Mask[S] 为谓词类型；?x = 可选参数
e1 ~ e2 = DimExpr 等价（type-system.md §4.3）
cap(X) = 能力类约束（type-system.md §2）
!{…}   = 效应（effects.md）
```

---

## 2. 核心内建签名

> **状态标注**：本文中的签名是规范目标，未标注不等于已经实现。每项当前状态只查
> `status.md`；`M4` 等标签仅表示 roadmap 归属，`未排期` 表示尚无近期实现承诺。

### 2.1 索引与程序上下文

```text
program_id(axis: Const[int])                    → i32
    facts: 0 <= pid_axis < grid[axis]
num_programs(axis: Const[int])                  → i32
arange(start: Const[int], end: Const[int])      → Block[i32, (end−start,)]
    要求: end > start；facts: 0 <= lane < end−start, contiguous
range(start: i32, end: i32, step: Const[int] ≥ 1)    → 循环归纳 i32
    start 为字面量 0（规范形态）或 int 标量（运行期起点，
    如两段 KV 循环的 lo = pid*BLOCK_M）
    facts: start <= i < end（i 取 start + k·step）；
    非负性事实仅当 start 结构非负时登记
```

### 2.2 构造

```text
zeros(S: ConstShape, T)                          → Block[T, S]
full(S: ConstShape, v: Const, T)                 → Block[T, S]   [未排期]
```

ConstShape = 每维 Const[int]（值形态才可构造）。

### 2.3 内存访问（两种一等形式）

```text
# Buffer 形式（推荐）——bounds 面向声明 shape
load[B](buf: Buffer[T, D, …, Access ∈ {ReadOnly, ReadWrite}],
        coords: (Block[i32, S0], …, Block[i32, Sk−1]),
        mask?: Mask[S], other?: T)
    → Block[T, S]
    约束: S = broadcast(S0, …, Sk−1) 逐轴对应 D
    !{Read[region_id(buf)]}
    义务: mask ⇒ ∀axis: 0 <= coords[axis] < D[axis]

# Ptr 形式——bounds 面向 Extent，effect 面向编译器生成的 RegionId
load[P](p: Block[Ptr[T, …, Extent=E, …], S] | Ptr[T,…,E,…],
        mask?: Mask[S], other?: T)
    → Block[T, S]
    义务: mask ⇒ 0 <= idx < E（E 为 UnknownExtent 时结论为 Unknown）
    !{Read[region_id(p)]}

store[B](buf: Buffer[T, D, …, Access ∈ {WriteOnly, ReadWrite}],
         coords, value: Block[T, S], mask?: Mask[S])   → Unit
    约束: value 与 S 精确同 dtype 同形（无隐式转换/广播放宽）
    !{Write[region_id(buf)]}

store[P](p: Block[Ptr[T, …, Access 写, Extent=E, …], S],
         value: Block[T, S], mask?: Mask[S])           → Unit
    !{Write[region_id(p)]}

unsafe_load / unsafe_store                          同上，义务豁免
byte_offset(p: Ptr[T,…] | Block[Ptr, S], b: i32 | Block[i32, S])
                                                    → 同构指针值（字节单位）[未排期：v0 调用即拒（TILA-SYN-050）]
buf.ptr : Buffer[T, (N,), Access, Alignment] → Ptr  仅 rank-1、launch stride[0]==1；
                                                    继承能力、Extent=N 与 BufferRegion
```

`other` 默认为该 dtype 的零值语境常量；dtype 必须与 T 精确一致。

### 2.4 类型转换

```text
cast[U](x: T | Block[T, S] | Mask?—否)            → U | Block[U, S]
    T ≠ U 必须显式；FP8 只能经此进出 Float
constant[U](value: exact Python int/float source) → Scalar[U] (Stage1Known)
    U ∈ {f16, bf16, f32, f64}；源仅字面量、模块常量及一元负号
    直接 RNE/ties-even，按目标位模式保存；非有限/溢出拒绝
```

`constant` 是 kernel 内显式舍入构造，不是 Const 参数，也不是运行时 cast。
原生 float 按已解析的 binary64 值舍入，不承诺源码十进制实数的精确值。
构造/存储保留带符号零与次正规数；后续算术的次正规数行为由相应运算/target 决定。

### 2.5 选择与逻辑

ADR-012：下文和 load/store 签名中的 mask 位置接受 `Mask[S]`、`Block[bool,S]`
或 scalar bool。布尔 tile 支持 `& | ~`，可混合标量 bool；结果为 Mask，
纯标量组合仍为 bool。mask 只能广播到既定访问形状，不能扩大访问 rank。
加载的 bool 内容默认是未知谓词；执行 mask 与可证明边界事实分别处理。

```text
where(m: Mask[S], a: Block[T, Sa], b: Block[T, Sb])   → Block[T, S]
    约束: a/b 同 dtype（无提升）；S = broadcast(S, Sa, Sb)
    值侧纯函数；效应侧见 effects.md §3
any(m: Mask[S] | Block[bool,S]) → bool  # 仅 .any() 方法，归约到标量
all(m: Mask[S] | Block[bool,S]) → bool  # 仅 .all() 方法，不生成逐 lane 事实
```

### 2.6 点积

```text
dot(a: Block[A, (M, K)], b: Block[B, (K, N)],
    acc?: Block[C, (M, N)])
    → Block[C, (M, N)]
    约束:
        K1 ~ K2（内维等价，TILA-SHAPE-004）
        cap(A), cap(B) ∈ DotInput；C ∈ {f32, f16（target 允许时）}
        M, N, K > 0
        hardware: 组合 ∈ target 支持表（TILA-TARGET）
```

### 2.7 归约与逐元素

```text
sum / max / min(x: Block[T, S], axis: Const[int])      → Block[T, S \ axis]
    dtype 精确保持（窄 dtype 需显式 cast——不继承 Triton 的默认提升）
    facts: 结果继承幸存轴的整除/区间事实（可用时）
    注：min 尚未实现 [未排期]
exp / exp2（均已实现）/ log / sqrt / rsqrt / abs / floor / ceil [未排期]
    (x: Block[T, S]) → Block[T, S]                     cap(T) ∈ Float
    abs 另接受 Int
neg / +,−,*,/,%(整除语义按 dtype)                       逐元素，§6 转换规则
```

### 2.8 形状操作

```text
trans(x: Block[T, (M, N)])                    → Block[T, (N, M)]   [未排期]
reshape(x: Block[T, S1], S2)                  → Block[T, S2]
    约束: numel(S1) ~ numel(S2)（可判定；证不出 = TILA-SHAPE-005）
    实现语义: 目标形状每维必须编译期可知（Const/字面量；运行期维报
    TILA-CONST-001）；全常量 Stage 1 即判，符号积 canon 相等直接通过，
    其余 Const 符号情形延迟到特化期数值判定
expand_dims(x: Block[T, S], axis)             → Block[T, S 插入 size-1]
    表面语法糖: x[:, None] / x[None, :] 等价于 expand_dims（唯一允许的下标形态）
cat(a: Block[T, S], b: Block[T, S])          → Block[T, (2·S0, S1..)]  [未排期]
    reorder 参数未开放；Const bool 参数 ABI 已由 ADR-013 固定，reorder 仍需独立设计
```

### 2.9 原子 [M4]

```text
atomic_add/max/min/and/or/xor(p: Ptr[T,…], v: T | Block[T,S], …) → T | Block[T,S]
atomic_cas(p: Ptr[T,…], expected: T, new: T)  → T
    共同约束: v 类型与指针元素类型精确一致（TILA-TYPE-030）；
              cap(T) ∈ AtomicTarget
    order?: MemoryOrder = Relaxed；scope?: MemoryScope = GPU   （类型化枚举）
    !{Atomic[region(p)]}
```

### 2.10 逃逸与断言

```text
assume(pred: Block-level 谓词的标量/Presburger 形态)     → Unit
    fact 注入；debug → device_assert（refinements.md §5.1）
static_assert(pred: StagedBool)                          → Unit
    编译期断言：Stage 1 立即检查，Const specialization 条件延迟检查
    消息参数未开放；StagedBool 表示阶段属性，可来自 Const int 比较或 0.3.x Const bool
hint.multiple_of / hint.max_contiguous                   调试用；常规路径由编译器自动发射
```

---

## 3. 运算符到内建的映射

| Python 运算符 | Tila 语义 |
|---|---|
| `+ − *` | 逐元素，转换规则 type-system.md §6 |
| `/` | 浮点除法（Float 域）；Int 域不存在 `/`（用 `//`） |
| `// %` | 整除/取模（Int 域；符号语义按 dtype） |
| `< <= > >= == !=` | 标量→bool；Block→Mask[S]（谓词入类型） |
| `& \| ~` | Mask/Block[bool] 可混合 scalar bool；tile 结果为 Mask，纯标量结果为 bool |
| `<< >> & \| ^ ~`（Int） | 位运算，同 dtype 要求 |
| `p + offs` | 指针元素 offset 算术（type-system.md §8.3）；`p - offs` 不属于 v0 |
| 一元 `-` | neg |

---

## 4. 硬件能力约束（特化期）

内建/签名可携带 target 谓词，特化期统一求解：

```text
requires(target):
    backend == CUDA ∧ sm >= 89            # 如 f8 dot 组合
    dtype 组合 ∈ 支持表
    shared memory / 寄存器预算            # tile 超限 → TILA-TARGET-002
```

kernel 级显式要求：

<!-- tila-example: future; milestone=M5 -->
```python
@tila.requires(tila.cuda.sm >= 90)
@tila.jit
def kernel(): ...
```

```text
error[TILA-TARGET-003]:
operation requires CUDA SM >= 90
compiling for: CUDA SM 86
    at: tila.dot(a, b)   # 或 kernel 级 requires
```

原则：**能力缺口在 Tila 特化期报出**，绝不留给 Triton 编译器深处
炸出不可读错误。target 结构（backend/arch/triton_version）是显式
数据，不散落字符串判断。

---

## 5. 诊断码族总目录

具体活动编号、阶段、严重度与维护门禁以 [`diagnostics.md`](diagnostics.md) 为准；
下表只提供 family 导航。

| 族 | 含义 | 主要文档 |
|---|---|---|
| TILA-SYN-0xx | Python 子集违规 | surface-language.md §2 |
| TILA-TYPE-0xx / 1xx | dtype/字面量/控制流类型；launch 类型 | type-system.md |
| TILA-SHAPE-00x | shape/broadcast/dot/reshape | type-system.md §4/§7 |
| TILA-CONST-00x | Const 语境误用 | type-system.md §5 |
| TILA-MEM-00x | 指针能力/offset 类别/对齐 | type-system.md §8–§9 |
| TILA-BOUNDS-00x | 边界义务四态 | bounds-safety.md |
| TILA-EFFECT-0xx | 效应诊断（where 007 等） | effects.md |
| TILA-RACE-00x | 竞争 | effects.md §4 |
| TILA-UNIFORM-0xx | uniformity | effects.md §6 |
| TILA-TARGET-00x | 硬件能力 | 本文 §4 |

---

## 6. 新增内建的流程

1. 在 `INTRINSICS` 增加完整 `IntrinsicSpec`，不得增加平行名字 allowlist；
2. 在本文增加人类可读签名，在 `status.md` §8.0 增加唯一 status key；
3. 增加显式 checker handler；若产生 typed TIR，登记 `tir_ops` 和 backend expectation；
4. 事实产出（若给推导环境添事实，需给传播规则）；内存访问分别声明 effect 与 bounds rule；
5. lowering/interpreter 增加相应 TIR capability handler，target 要求使用受控 ID；
6. 诊断（新增码或复用族内语义）；
7. 测试：签名正/反例、bounds 义务（若涉内存）、golden lowering、
   差分（interpreter vs GPU）。

完整性 validator 或状态/后端关联测试不通过的内建不允许进入 `tila.*` 命名空间。
