# Tila 类型系统

状态：M1 冻结基线（2026-09-19 退出审计通过）。
前置阅读：`design-principles.md`。本文是类型语法的规范来源。

---

## 1. 类型语法总览

```text
Type
├── Scalar                        运行期标量
│   ├── bool
│   ├── i8  i16 i32 i64
│   ├── u8  u16 u32 u64
│   ├── f16 bf16 f32 f64
│   └── f8e4m3fn f8e5m2 …          仅存储 dtype（见 §2）
│
├── Const[int]                    0.2.x 唯一可声明的编译期参数域
│
├── Block[T, Shape]               寄存器驻留的 tile；Layout 为保留扩展轴（§11）
│
├── Mask[Shape]                   谓词块（≠ Block[bool]，见 §3.3）
│
├── Ptr[
│     T,                           元素类型
│     AddressSpace,                Global | Shared | Local  （v0: Global）
│     Access,                      ReadOnly | WriteOnly | ReadWrite
│     Extent,                      从基址可访问的元素数量（bounds 用）
│     Alignment                    字节对齐：正的 2 次幂整数 | Unknown
│   ]
│
├── Buffer[
│     T,
│     Shape,                       符号形状（每维一个 DimExpr）
│     Access,                      ReadOnly | WriteOnly | ReadWrite
│     Alignment                    字节对齐：正的 2 次幂整数 | Unknown
│   ]
│
├── Unit                          store 等的返回类型，只能作表达式语句
│
└── Token[...]                    async/barrier 保留（v2+，v0 不出现）
```

参数顺序固定，语法上支持简写（§8.2、§9.2）。

---

## 2. DType 域与能力分类

dtype 与 Triton 对齐，但**运算语义收紧**。能力分类是 intrinsic 签名
约束的词汇表（`intrinsics.md`）：

| 能力类 | 成员 | 用途 |
|---|---|---|
| `Bool` | bool | 谓词 |
| `Int` | i8 i16 i32 i64 | 有符号整数 |
| `UInt` | u8 u16 u32 u64 | 无符号整数 |
| `Float` | f16 bf16 f32 f64 | 可算术浮点 |
| `FloatStorage` | 四种 FP8 | 只能 load/store/cast，不参与算术 |
| `DotInput` | f16 bf16（及 f8 双集，target 允许时） | `dot` 操作数 |
| `AtomicTarget` | i32 i64 u32 u64 f32 f64（target 相关） | atomic 操作数 |

FP8 是**存储 dtype**：`load` 得到 `Block[f8e4m3fn, S]` 后必须显式
`tila.cast` 到 Float 才能算术——这本身是一条类型规则，不是风格建议。

---

## 3. 值类别：Scalar、Block、Mask 严格三分

### 3.1 Scalar 与 Block 不共用类型名

`x: tila.f32` 只表示一个标量；`[128]` 个 f32 必须写成
`Block[f32, (128,)]`。局部变量全部靠推导，用户只在 kernel 边界
写类型：

<!-- tila-example: current; mode=syntax -->
```python
offs = tila.arange(0, BLOCK)      # Block[i32, (BLOCK,)]
xs   = tila.load(x, offs, mask=m) # Block[f16, (BLOCK,)]
```

### 3.2 Block 的元素类型可以是 Ptr

```text
p    : Ptr[f16, Global, ReadWrite, N, 16]
offs : Block[i32, (BLOCK,)]
q    = p + offs                    # Block[Ptr[f16, ...], (BLOCK,)]
```

这是 `Ptr` 形式 load/store 的类型基础（§8.3）。

### 3.3 Mask[Shape] 是独立类型

`Mask[Shape]` **不是** `Block[bool, Shape]`：

- **构造**：只来自同形（可广播）Block 的比较、以及 `& / | / ~`
  的 Mask 组合；比较的操作数可以是符号表达式或字面量（字面量比较数
  同样产生谓词）；
- **当前谓词按 DNF 子句组织**：`&` 对两侧子句做笛卡尔积、`|` 拼接子句、
  `~` 保守地产生空子句（不携带可证信息）；bounds 证明要求目标谓词
  在**每个**子句下都可证（bounds-safety.md §3.2）；
- **流动**：只允许流入 `load/store` 的 `mask=`、`where` 的谓词位、
  `mask.any()/mask.all()`；
- **禁止**：算术、位运算之外的任何运算、作为 `if` 条件（见 §10.4）、
  显式 cast。

Mask 在 IR 内部携带谓词信息（`Mask[(BLOCK,)] { offs < N }`），
供 bounds 证明消费（`bounds-safety.md` §3）。

`if mask:` → `TILA-TYPE-0xx: expected scalar bool, found Mask[(BLOCK,)]`。

M2 按 [ADR-011](adr/011-smt-proof-and-trust.md) 将谓词迁移到保留 And/Or/Not
的 DAG，由 SMT 求解。随后独立设计允许布尔 tile 用于 `mask=` 和布尔组合：
比较 mask 可以携带边界谓词，内存加载的布尔值通常只有未知谓词；可作为执行
掩码不意味着能证明边界。该接口尚未实现，不改变当前 Mask/Block bool 边界，
也不允许 `if mask` 隐式归约。

### 3.4 Unit

`tila.store(...): Unit`。Unit 只能出现在表达式语句位置；
赋值给变量、传参、参与运算都是 `TILA-TYPE` 错误。

---

## 4. Shape 是类型的一部分

### 4.1 DimExpr 语法

Shape 的每个维度是一个符号表达式，而不只是整数：

```text
DimExpr :=
    Const[int]
  | Symbol                      维符号（N、M、BLOCK…）
  | DimExpr + DimExpr
  | DimExpr − DimExpr
  | DimExpr * Const
  | DimExpr // Const            floordiv
  | ceildiv(DimExpr, Const)
  | DimExpr % Const
  | min(DimExpr, DimExpr)  |  max(DimExpr, DimExpr)
```

除法只允许除数为编译期常量（保证 Presburger 可判定）。
`/`（真除）在 DimExpr 中不存在。

### 4.2 符号维的两类

- **运行期维** `N = tila.Dim("N")`：特化时绑定实际值，但类型层面
  永远是符号；进入 kernel 是 `i32` 标量参数（带 `0 < N` 事实，
  见 §6.4 与 bounds-safety.md §2）。
- **编译期维** `Const[int]` / `ConstDim`：特化时必须已知（值或
  override），可带精化（`Const[int, PowerOfTwo]`，见 refinements.md）。

### 4.3 Shape 相等判定

分两层：

1. **语法层**：normalize（常量折叠、交换律排序）后结构相等；
2. **约束层**：语法不等时生成等价约束 `S1 == S2`，进入约束池，
   在定义期（可判定部分）或特化期（符号代入后）求解。
   该机制覆盖：dot 内维、broadcast、store 值形状、`other=` 形状与
   mask 形状；特化期失败统一报 `TILA-SHAPE-004`（附 `what` 说明）；
   纯常量之间的不等**立即**报错（可判定，不延迟）；含运行期维符号的
   不等同样立即报错（v0 无法延迟验证）。

例如 `dot` 要求 `K1 == K2`；若 `K1 = BLOCK_K`、`K2 = 64` 且
`BLOCK: Const[int] = 64` 已知，特化期直接通过；若 `BLOCK` 精化含
`Range[32, 64]` 但无唯一值，则 **无法证明** → `TILA-SHAPE-004` 报告
"known facts 不足以证明 K1 == K2"。

错误信息规范（示例）：

```text
error[TILA-SHAPE-004]:
    tila.dot(a, b)
inner dimensions do not match
    a: Block[f16, (M, K1)]
    b: Block[f16, (K2, N)]
required: K1 == K2
known:    K1 = 32, K2 = 64
```

---

## 5. Const[int]：编译期整数参数与 staging

### 5.1 语法与推导

<!-- tila-example: current; mode=syntax -->
```python
BLOCK: tila.Const[int]                      # kernel 参数：编译期
N:     tila.i32                             # kernel 参数：运行期
TILE:  tila.Const[int, tila.PowerOfTwo]     # 带精化的编译期参数
```

Const 上的算术在 checker 内**常量折叠**，结果仍是 Const：

<!-- tila-example: current; mode=syntax -->
```python
HALF: Const[int] = 64
x = tila.arange(0, HALF * 2)     # 要求 Const[int]：HALF*2 折叠为 128 ✓
```

运行期值出现在 Const 语境：

```text
error[TILA-CONST-001]:
    tila.arange(0, n)
upper bound must be compile-time known
    found:    n : i32 (runtime scalar)
    required: Const[int]
```

### 5.2 Staging：constexpr-if 与 runtime-if 彻底分离

<!-- tila-example: current; mode=syntax -->
```python
if BLOCK >= 128:        # BLOCK : Const → constexpr if
    ...                 # 未选中分支不进入 IR（死分支消除发生在 HIR）
else:
    ...

if pid == 0:            # pid : i32 → runtime if
    ...                 # 两分支都进入 IR，走控制流类型规则（§10）
```

- 0.2.x 不提供可声明的 `Const[bool]`。constexpr-if 的条件是模块级可折叠
  bool，或只依赖 `Const[int]` 比较的 staged bool predicate；其 staging
  属性独立于值类型（[ADR-005](adr/005-const-type-domain.md)）。
- runtime-if 的条件必须是**标量 bool**（§3.3 的 Mask 拒绝规则）。
- **三路分类（实现语义）**：
  1. 条件仅含模块级常量 → Stage 1 折叠，死分支不进 IR；
  2. 条件仅含 Const 参数（如 `if BLOCK >= 128:`）→ **TStaticIf**：
     两分支都进 IR、各自独立检查，特化期（Const 代入后）决出生效
     分支；lowering 发射普通 python `if`（Triton 的 constexpr 参数
     使其在 trace 期即为静态）；
  3. 条件引用任何运行期值 → runtime If（两分支进 IR，走 §10 合并规则）。
- **static-if 的合并规则**：两分支同名变量类型相同 → 正常合并；
  类型不同 → 变量成为 static-variant，**使用点**报 TILA-TYPE-020
  （类型依赖 Const 特化值）；仅单分支定义 → 使用点报 TILA-TYPE-023。
- 两者在 Tila HIR 中是**不同节点**（StaticIf / If），lowering 路径
  不同，不存在"碰巧都行"的灰色地带。

---

## 6. 数值转换：比 Triton 严格得多

M2-01 已按 [ADR-007](adr/007-integer-semantics.md) 固定整数语义：runtime
加减乘/取负回绕，`//` 向下取整，`%` 与除数同号，MIN/-1 的商回绕为 MIN、
余数为 0。移位量须在位宽范围内；float→int 必须有限且向零截断后可表示。
非法或无法验证的运算定义域报 `TILA-NUM-001`。shape/grid/stride 当前仍有
i32 范围限制，Buffer 地址线性化与循环内部步进使用 i64。

索引的数学推理必须逐个通过中间运算无溢出门禁；普通数据算术保留回绕语义，
不能继承未经证明的数学等价/contiguous 事实。当前使用保守区间和每次 launch
验证，完整 BitVec/SMT 仍待 M2-03。

易用性后续任务：先设计明确 dtype/舍入的常量构造，不直接放宽当前字面量
规则；Const bool 与 NumPy integer 白名单需要独立 ADR/版本评审，当前
0.2.x 的 ExactInt 和隐式转换规则不因这次设计更新而改变。

### 6.1 隐式转换只允许安全 widening

```text
i8 → i16 → i32 → i64
u8 → u16 → u32 → u64
f16 → f32 → f64
bf16 → f32
```

二元运算的推导规则（标量与 Block 同形场景同规则，逐元素适用）：

```text
binop(T1, T2):
    T1 == T2                      → T1
    T1 可 widening 到 T2          → T2
    T2 可 widening 到 T1          → T1
    otherwise                     → error TILA-TYPE-012
```

据此被禁止的隐式组合（非穷举，按规则判定）：

```text
i32 + u32        signed/unsigned 转换有歧义
i8  + u8         同上（不存在 i8→u16 式跨符号链）
i32 + f32        int/float 语义不同，必须显式
f32 → f16        narrowing
i64 → i32        narrowing
u32 → i32        跨符号
f16 + bf16       互不 widening（公共目标 f32 也不是任一方的 widening 目标）
```

错误信息规范：

```text
error[TILA-TYPE-012]:
    z = x + y
cannot add i32 and u32: signed/unsigned conversion is ambiguous
use one of:
    x + tila.cast[i32](y)
    tila.cast[u32](x) + y
```

### 6.2 `tila.cast` 是唯一显式转换

<!-- tila-example: current; mode=syntax -->
```python
tila.cast[tila.f32](x)
```

函数形式 `tila.cast(x, tila.f32)` 是尚未实现的设计候选：

<!-- tila-example: future; milestone=M1 -->
```python
tila.cast(x, tila.f32)
```

- narrowing cast 合法但**必须显式**——责任显式化即目的；
- FP8 与 Float 之间只能经 cast；
- bool 与数值之间只能经 cast；
- cast 不改变 shape（`Block[T, S] → Block[U, S]`）。

### 6.3 字面量：语境多态

字面量有自己的内部类型 `IntLiteral[n] / FloatLiteral[v]`，在
**checking 方向**按语境实例化：

<!-- tila-example: diagnostic -->
```python
x: f16
x + 1            # 1 实例化为 f16 ✓
x + 1.5          # 1.5 实例化为 f16 ✓（若可精确/可表示）

y: u8
y + 300          # error: 300 cannot be represented as u8 (TILA-TYPE-013)
```

- 字面量实例化要求**值可被目标 dtype 表示**（f16 语境下不可表示的
  十进制字面量同样拒绝，不允许静默舍入——需要舍入时写显式常量并 cast）；
- 无语境的裸字面量默认 `i32`（整数）/ `f32`（小数）；
- 字面量不参与 widening 链的"跳变"：`u8 + 1` 中 1 直接实例化为 u8，
  而不是先变 i32 再报错。

### 6.4 运行期标量参数的精化注解（launch 契约）

<!-- tila-example: current; mode=syntax -->
```python
import tila as ti

N = ti.Dim("N")

@ti.jit
def kernel(x: ti.Buffer[ti.f32, (N,)], n: ti.i32 | ti.Positive):
    ...
```

运行期标量上的值精化是 **launch 契约**：launcher 在启动前检查
（失败抛 `TilaLaunchContractError`），通过后作为事实进入 kernel 内
的 bounds/refinement 求解环境。这是静态证明与运行期代价之间最便宜
的桥（详见 bounds-safety.md §5）。`Positive/NonNegative/Range` 注入区间事实，
`MultipleOf[k]` 注入模等式事实，且都由 launcher 先验证。

---

## 7. Broadcast：显式且可判定

规则收窄为两条，满足其一即可广播：

1. 对应维**相等**（DimExpr 等价，§4.3）；
2. 一方为 **size-1**。

标量与 Block 运算 = 标量按 size-1 广播。其余全部 `TILA-SHAPE-003`。

```text
a: Block[f32, (128, 64)]
b: Block[f32, (128, 1)]      # ✓ → (128, 64)
c: Block[f32, (64,)]         # ✗ (64,) vs (128, 64)：最后一维 64✓ 但 rank 对齐后第一维 64 vs 128 ✗
```

广播发生在：二元算术、比较（→ Mask）、`where` 三方、load/store 的
`other` 与坐标轴。**广播必须能静态判定**；判定不了就不是广播，是错误。

---

## 8. Ptr：指针携带能力

### 8.1 完整形式与推导

Ptr 主要由系统内部从 Buffer 推导产生，也可作为 kernel 参数（裸指针
接口，见 §9.3）。公共五参数由 ADR-001 固定为
`Element/AddressSpace/Access/Extent/Alignment`：`Extent` 参与 bounds，
`Access` 与 `Alignment` 参与**操作合法性**，`AddressSpace` v0 恒为 Global。
效应与 race 使用编译器生成、不可由用户声明的 `RegionId`，见 ADR-002。

```text
p: tila.Ptr[f16, tila.Global, tila.ReadOnly, N, 16]

tila.store(p + offs, v)
error[TILA-MEM-001]:
    cannot store through ReadOnly pointer
```

### 8.2 简写

```text
tila.ReadPtr[f16]    ≡ Ptr[f16, Global, ReadOnly, UnknownExtent, UnknownAlignment]
tila.WritePtr[f16]   ≡ Ptr[f16, Global, WriteOnly, UnknownExtent, UnknownAlignment]
tila.RWPtr[f16]      ≡ Ptr[f16, Global, ReadWrite, UnknownExtent, UnknownAlignment]
```

简写可追加 `Extent`、`Alignment`；省略时对应信息为 unknown（bounds 需
Buffer、证明或 unsafe）。**kernel 参数推荐 Buffer**。规范形式和兼容迁移规则见
[ADR-001](adr/001-ptr-public-syntax.md)。

### 8.3 指针算术：元素 offset 与字节 offset 分离

```text
Ptr[T, ...]  +  Block[i32, S] | i32        → Block[Ptr[T,...], S] | Ptr[T,...]
tila.byte_offset(ptr, bytes)               → Ptr / Block[Ptr]     # 唯一的字节寻址入口
```

`ptr + 16` 中的 16 是 **16 个元素**。想按字节移动必须
`tila.byte_offset`——元素/字节混淆这一类 C/CUDA 传统 bug 在类型层
被堵死（`TILA-MEM-002`）。

指针加法的结果继承源指针的全部能力参数。`p - offset`、指针比较和指针差
都不是 v0 语法；未来开放减法时必须先定义负 offset 与 bounds 规范化规则。

---

## 9. Buffer：kernel 参数的一等形态

### 9.1 类型与声明

<!-- tila-example: current; mode=syntax -->
```python
M = tila.Dim("M")
N = tila.Dim("N")

@tila.jit
def kernel(
    x: tila.Buffer[tila.f16, (M, N), tila.ReadOnly],
    y: tila.Buffer[tila.f16, (M, N), tila.WriteOnly],
    out: tila.Buffer[tila.f32, (M, N), tila.ReadWrite],
):
    ...
```

Strides 不属于公共类型参数：每维元素 stride 在内部符号化，特化期由 launcher
从实际张量提取绑定；AddressSpace v0 隐式且只允许 Global。完整决定见
[ADR-003](adr/003-buffer-stride-address-space.md)。

### 9.2 launch 时的自动绑定

<!-- tila-example: current; mode=syntax -->
```python
kernel[(grid,)](x, y, out)
```

launcher 从张量对象自动提取并绑定：

```text
dtype  ← x.dtype            并与声明的 f16 校验（不符 → TILA-TYPE-101 launch 错误）
shape  ← x.shape            与 (M, N) 符号绑定
strides← x.stride()         单位统一为元素
ptr    ← x.data_ptr()       对齐实测与 Aligned[k] 声明校验（§9.4）
device ← x.device           与 target 校验
```

**类型层看到的是 Buffer；lowering 层展开为 Triton 的
`ptr + strides + dims` 参数组。** 语言层永远不出现裸指针参数拆解。

### 9.3 访问形式

两种一等形式，推荐 Buffer 形式：

<!-- tila-example: current; mode=syntax -->
```python
# Buffer 形式：坐标访问（推荐——bounds 面向声明的 shape 证明）
a = tila.load(x, (rows, cols), mask=(rm & cm), other=0.0)
tila.store(out, (rows, cols), v, mask=(rm & cm))

# Ptr 形式：指针算术（bounds 面向 Extent/精化证明）
p = x.ptr                         # Buffer → Ptr（继承全部能力）
q = p + offs                      # Block[Ptr[f16,…], (BLOCK,)]
b = tila.load(q, mask=m)
```

v0 规则：Buffer 形式的 load/store **必须**可证明（或 mask 证明、或
launch 契约、或 unsafe）；Ptr 形式若无 Extent 信息，则要求精化或
走 unsafe（bounds-safety.md §1）。两条路的类型规则都在
`intrinsics.md` §2。

实现注：**Ptr 形式在 interpreter 上仅支持连续 1D buffer**（flat 偏移
语义）；Buffer 形式无此限制，GPU/Triton 路径亦无此约束。

### 9.4 Alignment 精化的特化期校验

`Buffer[...]` 与 `Ptr[..., Aligned[k]]` 的对齐声明在特化期用
`data_ptr() % k == 0` 实测校验，不符 → `TILA-MEM-003`。声明了的对齐
立即成为地址计算与向量化的事实（refinements.md §4）。

---

## 10. 控制流类型规则

### 10.1 分支合并必须同型

<!-- tila-example: diagnostic -->
```python
if c:
    x = tila.zeros((128,), tila.f16)
else:
    x = tila.zeros((64,),  tila.f16)
use(x)
```

```text
error[TILA-TYPE-020]:
branch result types differ
    then: Block[f16, (128,)]
    else: Block[f16, (64,)]
```

v0 **不支持运行期 union/sum 类型**——GPU kernel 里不需要，类型系统
不为它付复杂度。需要异构结果时用两个变量 + 各自 mask 语义表达。

补充规则（实现语义）：
- 分支**单侧**定义的变量：分支后视为可能未定义，使用点报
  TILA-TYPE-023（静态分支折叠加 static-if 变体同此规则）；
- 循环体内定义的变量是**循环局部**的：循环后使用报 TILA-TYPE-023
  （比 Python 语义更严格，保证类型环境的确定性）。

### 10.2 Loop-carried 类型不得漂移

循环变量在每次迭代的类型必须与进入时相同（dtype、shape 全等）；
改变 loop-carried dtype/shape 是 `TILA-TYPE-021`。

### 10.3 循环归纳变量

`for i in tila.range(0, K, BK)` 中 `i : i32`，checker 附带事实
`0 <= i < K`（bounds-safety.md §2）。step 必须是 Const 且 ≥ 1。

### 10.4 条件的类型

| 条件种类 | 判定 | 节点 |
|---|---|---|
| 模块级/Stage 1 已知 bool | 装饰期分支消除 | 无（直接折叠） |
| 只依赖 `Const[int]` 的 staged bool | 特化期分支选择 | TStaticIf |
| scalar bool | 运行期分支 | If |
| Mask[Shape] / Block[bool] | **error** TILA-TYPE-022 | — |

Block 级别的"条件执行"只有一条路：masked load/store 与 `where`。

---

## 11. Layout：v0 不进用户类型语法

`Block[T, Shape]` 预留第三个参数 `Layout`（Contiguous / Strided /
Blocked / DotOperand / …），但 v0 **不对用户暴露**：

- **正确性层**（dtype、shape、bounds、memory capability）与
  **优化事实层**（alignment、contiguous、后续 layout）分离
  （design-principles.md §2、refinements.md）；
- contiguous/alignment 以 refinement fact 形式存在并自动发射 Triton
  hint；
- 完整 layout 类型系统（含 dot operand family 的联合约束）是
  v2 主题，届时以 `Block[T, Shape, Layout]` 第三参数进入语法，
  现有语法不变。

---

## 12. 泛型 kernel

<!-- tila-example: future; milestone=M5 -->
```python
T = tila.TypeVar("T", bound=tila.Float)

@tila.jit
def relu(x: tila.ReadPtr[T], y: tila.WritePtr[T], n: tila.i32, BLOCK: tila.Const[int, tila.PowerOfTwo]):
    ...
```

- TypeVar 带 bound（能力类或 dtype 集合）；
- 定义期按 bound 检查（签名内只允许该能力类的运算）；
- 首次以具体 dtype 调用时**特化** `T = f16`，产生并缓存一份特化
  kernel；`T = bf16` 再来一次就是另一个特化——与 Triton JIT 的
  specialization 缓存一一对应；
- 泛型 + shape 符号可以组合（`Buffer[T, (N,)]`）。

---

## 13. 双向类型推导

Tila 采用 bidirectional typing，不做全局 inference：

```text
Γ ⊢ e ⇒ T     synthesis：从操作数综合类型（x = tila.load(...)）
Γ ⊢ e ⇐ T     checking：按期望类型检查（store 的 value、字面量语境）
```

典型分工：

- `load/dot/arange` ⇒ 结果类型；
- `store(p, v)` 中 `v ⇐ Block[T_p, S_p]`（T_p/S_p 来自指针）——
  这就是"store 隐式转型"被消灭的位置；
- 字面量只在 checking 方向实例化（§6.3）；
- TypeVar 实例化在 ⇐ 方向传播；
- `tila.cast[T](x)` 把任意 T 作为期望方向锚点。

推导失败的错误必须给出：表达式位置、涉及各方的完整类型
（含 shape 符号的已知值）、以及至少一个修复建议。

---

## 14. 错误码一览（本文引入）

| 码 | 场景 |
|---|---|
| TILA-TYPE-0xx | dtype 混算（012）、字面量不可表示（013）、分支不同型（020）、loop-carried 漂移（021）、Mask 当条件（022）、Unit 误用等 |
| TILA-SHAPE-00x | broadcast 不合法（003）、dot 内维（004）、reshape numel、坐标 shape 不匹配 |
| TILA-CONST-00x | runtime 值用于 Const 语境 |
| TILA-MEM-00x | 写 ReadOnly（001）、元素/字节 offset 混用（002）、对齐声明不符（003，特化期） |
| TILA-TYPE-1xx | launch 期参数类型不匹配（101 等） |

完整目录与诊断渲染规范见 `intrinsics.md` §5 与 `surface-language.md` §8。
