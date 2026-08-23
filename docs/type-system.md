# Tila 类型系统设计（v0.1）

状态：定稿（2026-08 评审修订）。对应实现目标：

```
parse → AST → type check（typing + launch analysis，产出 typed TIR 与 LaunchPlan）→ Triton lowering
```

---

## 0. 设计原则

1. **表面简单，内部丰富**。用户语言停留在 Triton 的抽象层级（tile 编程，无 thread/warp/lane）；丰富的类型信息（shape、layout）由编译器推导并挂在内部 IR 上。
2. **Layout 是语义，不是语法，也不是 Triton encoding 的转述**。用户永远不写 layout；layout 项只在编译器内部出现，用于证明操作合法。Tila 的 layout 是**等价/来源描述符**（provenance / equivalence descriptor），不是 GPU 具体分布的描述（§3.2）；"Tila 等价 ⇒ Triton 分配相同 concrete encoding"是桥接问题（§5），由差分测试兜底，不写成规范承诺。
3. **可判定的错误一律编译期拒绝**。lowering 阶段是纯语法映射、**对有效 TIR 全函数（total）、不做新的语义拒绝**——所有可判定的拒绝都在 typing 阶段完成。
4. **每条 typing rule 对应一条确定的 Triton lowering**。约束在 Triton 实际约束之上**主动收紧**并全部前移（如 `arange(0, 2^k)`，见 R2 注）——收紧口径属于 Tila，不随 Triton 版本摆动。
5. **没有隐式转换，没有隐式提升**。Triton 有混算提升表与 store 的隐式广播/转型；Tila 一律显式 `tila.cast`（否则 E02）、shape 显式一致（否则 E03/E05）。"Triton flexible, Tila explicit"。

---

## 1. 类型分层

类型按三个层次组织，逐层检查（顺序固定：dtype → shape → layout，§7）：

```
Type
  ├── ScalarType(dtype)                 标量值：pid、N、字面量
  ├── TileType(dtype, shape, layout)    计算值：一次 kernel 迭代操作的块（寄存器驻留）
  ├── BufferType(dtype, shape)          kernel 参数：全局数据对象（无 layout 字段——它不参与分布）
  ├── AddressType(dtype, shape, layout) Buffer + Tile[i32] 偏移的内部地址值（R7）
  └── UnitType                          store 的返回类型，只能作表达式语句（E08）
```

- 一个 Tila 类型可读作四元组 `Kind × DType × Shape × Layout`：Scalar 无 Shape/Layout、Buffer 无 Layout、Tile/Address 四者俱全、Unit 全无。
- **Buffer / Tile / Address 的分工**：Buffer 是"数据是什么"（kernel API，指针语义上存在、语法上隐藏）；Address 是"这次要读写哪一小块"（v0.3 起由 checker 在 load/store 内部按坐标构造，用户不可见、不可命名）；Tile 是"读进来/算出来的块"。数据流固定为：

  ```
  Buffer ──load(buf, coords)──► [Address（内部）] ──► Tile ──compute──► Tile ──store(buf, coords)──► Buffer
  ```

### 1.1 dtype 域：universe 与 Triton 对齐、运算语义刻意收紧

Tila 的 dtype 表覆盖 Triton `triton.language` 的全部标量类型（**universe 对齐**：名字与语义一一对应），并按**能力**分级——能力表是 typing rule 的前提，违反 → E16：

| 类别 | dtype | 算术 | 比较 | `& \|` | Tensor 元素 | mask | cast 源/目标 |
|---|---|---|---|---|---|---|---|
| 布尔 | `bool` | ✗ | `== !=` | ✓ | ✗ | ✓ | ✓ |
| 有符号整数 | `i8 i16 i32 i64` | `+ − ×` | 全部 | ✗ | ✓ | ✗ | ✓ |
| 无符号整数 | `u8 u16 u32 u64` | `+ − ×` | 全部 | ✗ | ✓ | ✗ | ✓ |
| 浮点 | `f16 bf16 f32 f64` | `+ − × /` | 全部 | ✗ | ✓ | ✗ | ✓ |
| FP8（存储 dtype） | `fp8e4m3 fp8e5m2 fp8e4m3fn fp8e4m3b15` | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |

- **FP8 定位为存储 dtype（Tila 的语言设计选择）**：只允许 load/store/cast；`x + y`（fp8）→ E16，错误信息直接给出正确写法 `tila.cast(x, tila.float16) + tila.cast(y, tila.float16)`。Triton 的 FP8 并非绝对不可算（如 `dot` 接受 e5m2 输入）——Tila 把"计算先升精度"的纪律类型化。
- **无隐式提升**：`x + y`（f32 + i32）→ E02，要求显式 `tila.cast(y, tila.float32)`。整数与浮点是不同类别；`u32` 与 `i32` 也是不同 dtype（比较也一样报 E02）。
- **cast 矩阵全开**：数值 dtype 之间任意互转（含 bool、FP8 两个端点），语义 ≡ Triton `.to`（翻译语义）。`bool` 目标 = 数值→掩码转换（非零为真），不是逻辑谓词的构造——mask 的语义来源始终是比较（R6/R10）或这一次显式数值判定，soundness 不依赖"bool tile = 真谓词"。
- **标量值域只有 `i32` / `f32`**：宽 dtype（u32、f16、fp8…）只存在于 Buffer/Tile 的元素层面。（表示层预留全部 17 种；v0.1 表面只产生 i32/f32。）
- **op ∉ dtype 类别的运算集 ⇒ 一律 E16**（bool 算术、整数 `/`、FP8 的一切算术与比较均落此处）——规则统一，不逐条枚举。`//`、`%` 与整数整除在 v0.1 **不属于运算符集**（文法层面即不存在，`language-spec.md` §3），归入 v0.2。
- **字面量上下文规则（R12）**：INT 字面量默认 `Scalar(i32)`、FLOAT 字面量默认 `Scalar(f32)`；在要求 `Scalar(dt)` 的位置（R4/R10 的标量侧、比较的标量侧、`other`），**类别一致**（整↔整、浮↔浮）的字面量按该 dt 解释，类别不一致 → E02。因此 `f16_tile + 1.0`、`f16_tile > 0.0`、`other=0`（u32 元素）合法，而 `f16_tile + 1` 报错。
- FP8 具体格式的可用性取决于目标架构（Triton 编译期校验，B01），Tila 原样透传；Triton 未来新增格式时 dtype 表零成本跟进。
- **指针类型刻意不进入表面语言**：Triton 的 pointer 在 Tila 中表现为 `Buffer` 参数（lowering 后成为 Triton 的指针参数与线性化表达式，`triton-lowering.md` §5）。v0.3 定稿起地址算术整体退场：Address 由 checker 在 load/store 内部按坐标构造（§5 R7 内部化），用户不可见。这是设计决策而非缺口：类型化坐标 + 禁止指针运算正是静态可判性的来源。

---

## 2. Shape 迷你系统

Shape 是独立的小型类型系统（"变量在 shape 与 mask，不在分布"，`v0.2-preview-2d.md` §1）：

```
Shape ::= ()                      # 标量（仅内部预留）
        | (Dim, …)                # rank ≥ 1（v0.1）
Dim   ::= Const(n)                # 编译期整数字面量（如 128）
        | Symbol(name)            # 运行期符号（如 N；自动绑定为 Scalar(i32) 参数）
        | Product(Dim, Dim)       # constexpr 积（如 2*BLOCK；代数上预留，v0.2 随 constexpr 积启用）
```

- **相等 `Σ₁ ≡ Σ₂`**：逐维比较。静态维比值（`128 ≡ 128`）；符号维比符号名（`N ≡ N`，`N ≢ M`）。没有子类型、没有松弛。
- v0.1 中 **Tile 的 shape 必须全静态**（由 arange 与 constexpr 运算构成）；符号维只允许出现在 Buffer 参数的注解 shape 里。
- **v0.1 没有 shape 算术**：`N - N % BLOCK`、`ceil_div(N, BLOCK)` 这类表达式不出现在 kernel 体（grid 计算在主机侧 launcher 完成）；`M + N`、`M - 1`、`M / 2` 一旦放开就变成约束求解（Presburger arithmetic）问题，不做。
- **广播（broadcasting）**：`Σ₁ ⊗ Σ₂` 右对齐逐轴：`(a, b) ↦ a（a=b）/ b（a=1）/ a（b=1）/ ✗ E03`。v0.1 的实际使用是**标量 → 任意 shape**（R4/R10）；size-1 维广播是 v0.2 预览（R13）。**除此之外（两轴均 >1 且不等）→ E03**。

### 2.1 DimExpr：shape 表达式的目标代数（承诺，v0.3+ 起；2026-08-24 两轮评审定案）

v0.1 冻结为 `Const | Symbol | Product(constexpr)`、无算术；"将来放开到什么程度、以什么判定语义放开"现在定死。按第二轮裁定：**shape 要支持全部整数运算**——但**支持全集 ≠ 全部等式可证**，这是两条独立的承诺（见"四职责"）。

**文法（目标全集）**：

```
DimExpr ::= Const(n) | Symbol(X) | Neg(e)
          | Add(e, e) | Sub(e, e) | Mul(e, e)
          | FloorDiv(e, e) | CeilDiv(e, e) | Mod(e, e)
          | Max(e, e) | Min(e, e)
Shape   ::= (DimExpr, …)
```

- **`/` 明确不进入 shape 域**：整数域上它有 floor 还是 real division 的歧义；一律写 `FloorDiv` / `CeilDiv`。`ceildiv` 是一等公民——grid 数学（`⌈N/B⌉`）的刚需，不让用户手写 `(N + B - 1) // B`。
- **分阶段启用**：阶段一只放行线性片段（`k + Σ cᵢ·Nᵢ`——重写产出平坦的系数向量，`N + N` 与 `2*N` 同规范化）；阶段二放行非线性项（`N*M`、`FloorDiv`/`CeilDiv`/`Mod`/`Max`/`Min`）——它们作为 symbolic 节点**保留原样，不强行化简**：`(N // 32) * 32 ≢ N` 永远不自动划等号，除非约束可证。
- **表示：保留 AST，不追求单一平坦 canonical form。** 非线性项存在后"唯一平坦规范形"不再成立也不必要；canonical = 确定性重写后的规范树（实现上 hash-consing 成 DAG，等价子式共享）。线性片段的系数向量是重写系统的一个出口，不是整个系统的终点。

**两层判定架构**：

```
DimExpr
  │
  ├─ Layer 1  无条件重写（cheap / sound / deterministic）
  │     常量折叠、交换律排序、结合律扁平化、
  │     x+0、x*1、x*0、x−x、x%1、max(x,x)、min(x,x)、…
  │
  └─ Layer 2  约束感知证明
        prove(a ≡ b, Γ.constraints) : ProofResult

ProofResult ::= ProvenEqual | ProvenNotEqual | Unknown
```

- **判定语义是 must_equal（保守 soundness）**：只有 `ProvenEqual` 放行；`Unknown` ⇒ 拒绝（E03）。`N` vs `M` 不是"断言不等"，而是**不可证**；运行期碰巧相等（N = K = 128）不放行——强类型 DSL 不赌运行期。`ProvenNotEqual` 用于诊断升级：错误信息说"两维可证不等"，而不是"无法证明相等"。
- **四职责（表达式完整性 ≠ 化简完整性）**：shape 系统只需 (1) 表示任意 DimExpr，(2) 需要时求值（build 期特化值），(3) 尽力证明等价，(4) 证不出就保守拒绝。不承诺把一切数学等价式归一。
- **约束（阶段二随 assume 引入）**：维度自动携带正性约束（`N ≥ 1`）；提案 `tila.assume(N % 128 == 0)`——**编译期 shape 约束、非运行期断言**；Γ 扩展为 `Values + ShapeSymbols + Constraints`。收益例：`reshape(x, [ceildiv(N,128), 128, M])` 需证 `ceildiv(N,128) * 128 ≡ N`——无条件不可证 ⇒ 拒绝，有 `assume(N % 128 == 0)` 则可证 ⇒ 放行，且错误信息直接报出缺失的约束。
- **fast/slow path**：Layer 1 + 规范树结构相等是主路径（一切线性等式与绝大多数程序内比较在此解决）；约束证明是第二层；SMT 求解器（Z3 类）只是**可插拔 fallback，永不进入主执行路径**——未接入时等价结论就是 Unknown ⇒ 拒绝，**健全性不依赖 solver 的存在**。
- 首批消费者：`cat`（§5 R-cat，`N + N → 2N`）与 `reshape`（numel 守恒证明）。stride 化 MemoryLayout（路线图下一项）仍只消费 `Symbol`（stride 符号按现有 sym 机制绑定），不提前引入算术。
- **注解表达式随阶段放开**：签名文法从 `dim ::= INT | IDENT`（`language-spec.md` §4 文法、§7 符号维绑定）按阶段扩到 DimExpr 子集；届时 frontend 在注解读取处把表达式 elaborate 成 `DimExpr`，checker 依旧永远不碰 Python AST（现有 pipeline 不变）。

---

## 3. Layout 系统（编译器内部）

Layout 分两层，用户两层都看不到：

> 记法约定：本文档中的 `L₀`、`L₁`、`L₂`… 是 **layout 元变量**（某次推导中某个具体 term 的名字，随推导命名，如 §8 的 `L₀ = identity(128)`），不是预定义的布局种类。`L₁ ≠ L₂` 读作"两个元变量可以指代不同分布"，与 RowMajor/ColMajor 之类的名词无关。

> **术语规则（两套 layout 严格分名）**：**DistLayout**（Tile/Address 侧——分布的等价/来源代数，§3.2）与 **MemoryLayout**（Buffer 侧——地址函数，§3.1 与 `v0.3-strides.md` §1.3）完全正交：前者有 normalize/equiv、后者是 per-Buffer 的事实（`Address(b, c) = base(b) + L_b(c)`）。文档与代码注释不使用泛化的"layout"指代二者之一。

### 3.1 MemoryLayout（Buffer 侧）

```
M ::= RowMajor          # 默认：不物化 stride 表，launcher 断言张量实际连续
    | Strided(strides)  # 显式声明：每轴 INT 或符号（v0.3 起，docs/v0.3-strides.md）
```

记录数据在显存中的摆放。v0.1 只有 RowMajor；v0.3 引入 `Strided`（尾随 strides
元组注解）与坐标寻址 R7'，stride 化设计定稿见 `docs/v0.3-strides.md`。**M 不参与
等价判定**：它是 per-Buffer 的事实，两个 Buffer 类型从不互相 unify（与 DistLayer
的 normalize/equiv 不对称——语义侧是代数，内存侧是事实）。后续支持 swizzle、
TMA descriptor 时继续在此层扩展。

### 3.2 DistLayout（Tile / Address 侧）

**定位（先于一切定义）**：DistLayout term 是**等价/来源描述符**，**不是 GPU 具体分布的描述**。`identity(128)` 的含义不是"128 个元素具体怎么摆到 warp/lane/寄存器"——它不携带 sizePerThread、threadsPerWarp、order 等 Triton concrete encoding（BlockedEncoding / LinearEncoding / CTA layout）的信息，而是"同一 128 元素分布族的来源标记"。Tila 证明的始终是：**Tila 语义下的分布等价被哪些操作保持**；"Tila 等价 ⇒ Triton 分配相同的 concrete encoding"是另一件事——那是 Tila 抽象与 Triton backend 的**桥接问题**（v0.2+ 的核心开放问题，§5），在桥接被精确化之前由 differential 测试兜底（`triton-lowering.md` §9），不写成规范承诺。

**策略：追踪等价类，不追踪具体映射。** Tila 只需要回答"两个 tile 的分布是否相同"，具体哪个线程拿到哪个元素由 Triton 在它自己的 layout inference 中决定。因此 DistLayout 是一个带来源（provenance）的符号项：

```
L ::= Identity(shape)            # arange 种子：元素 0..n-1 的规范分布
    | Mma(m, n, k)               # dot 种子（v0.2 起）：MMA 指令规定的分布
    | Zeros(shape)               # distinguished 种子态（v0.4 起）：分布未定——
                                #  全同元素的分布不可观测；equiv 排除、join 单位元
                                #  （L7）。不是普通 DistLayout term（v0.4-kloop §2.1）
    | Load(M, L)                 # 经 load 产生（M 是来源 MemoryLayout）
    | BcastScalar(L)             # 标量广播产生
    | Cast(L)                    # dtype 转换产生
    | Join(L, L)                 # 两个等价分布做逐元素运算
    | Broadcast(L, axis)         # size-1 轴广播（v0.2 起；见预览文档）
    | Product(L, L)              # 独立坐标分布的笛卡尔复合（v0.2 起；见预览文档）
    | Reshape(L, shape)          # 预留（v0.2）
    | Transpose(L, perm)         # 预留（v0.2）
```

`Zeros(Σ)`（v0.4，`docs/v0.4-kloop.md` §2.1）是 **distinguished 种子态，不是
普通 DistLayout term**：全同元素的分布不可观测，`Zeros` 记录"分布未定"
（ZeroSeed）而非一种具体分布——因此它按定义被排除在 equiv 之外（两个"未定"
相等是唯一的 equiv 成立情形），却以 **join 单位元**的身份参与合并（L7 = 种子
物化规则：未定 ⊕ 具体 = 具体）。checker 的累加器检查以状态机表述
（Seed → Materialized(L) → same L）。

### 3.3 化简律（normalize 到不动点）

```
(L1) load(M, L)      ≡ L        # load 结果的分布 = 索引的分布
                                #（线程 t 持有 idx 中分配给它的那些元素，
                                #  加载后仍在 t 手里）
(L2) bcast_scalar(L) ≡ L        # 标量广播不改变分布
(L3) cast(L)         ≡ L        # dtype 转换不改变分布
(L4) join(L, L)      ≡ L        # 等价分布逐元素运算，结果分布不变
(L5) Product 同余 + 单位元律      # v0.2 起：norm(Product(a,b)) = Product(norm(a),norm(b))；
                                #  Product(L, Identity(())) = L
(L6) Broadcast 幂等              # v0.2 起：Broadcast(Broadcast(L,i),i) = Broadcast(L,i)
(L7) join(Zeros(Σ), L) ≡ L      # v0.4 起（对称）：种子物化——distinguished
                                #  种子态 ⊕ 具体分布 = 具体分布。累加器 φ 类型化
                                #  的支柱（R19；仅在 join 侧落律，R14 广播路径
                                #  与 Zeros 的组合不落律）
```

L1 是整个系统最重要的不变量：**数据搬运不改变"谁持有哪个元素"**。MemoryLayout `M` 影响的是 load 的代价（coalescing、向量化），不影响结果的分布——所以它在等价判定中被抽象掉，仅作为来源记录保留。

### 3.4 等价与转换

- **等价 `L₁ ~ L₂`**：v0.1 判定 = `normalize(L₁) == normalize(L₂)`（结构相等）。
- **转换 `L → L'`**（带代价的 layout convert，未来用于 shuffle / shared memory round-trip）：v0.1 不实现，检查失败直接报错，错误信息给出修复建议（E05）。
- v0.1 事实：唯一种子构造点是 `arange`，L1–L4 全部是擦除性的，因此**正规形式恒为 `Identity(n)`**——等价判定退化为种子相等。算法保持通用签名，v0.2 引入非擦除项后无需改动调用方。

---

## 4. Kernel 签名与类型环境

```python
import tila


@tila.jit
def add(
    a: tila.Tensor[tila.float32, N],
    b: tila.Tensor[tila.float32, N],
    c: tila.Tensor[tila.float32, N],
    BLOCK: tila.constexpr = 128,
):
    ...
```

- 出现在 shape 里的符号（`N`）**自动绑定**为一个运行时 i32 标量参数（launcher 从张量形状读取并传入）。
- `tila.constexpr` 参数按调用点特化（对应 Triton 的 `BLOCK: tl.constexpr`）。**检查逐值特化、发射保留名字**（模型 B，`semantic-model.md` §6）：每个 override 值对应一次完整的 check → TIR → lowering，**TIR 是该值的特化产物、跨值不复用**；`tila.arange(0, BLOCK)` 的 shape 取 build 期值（如 128），生成的 Triton 里仍是 `BLOCK`。
- 类型环境 Γ 记录：参数类型、constexpr 的值与默认、符号维绑定。

## 5. Typing Rules

每条规则右侧给出对应的 Triton lowering。`c` 表示编译期常量。判读形式为 typing judgment `Γ ⊢ e : τ`。

### R1 program_id

```
c ∈ {0}        （v0.1；v0.2 预览片段扩至 {0,1}）
────────────────────────────
Γ ⊢ tila.program_id(c) : Scalar(i32)
```
⇝ `tl.program_id(c)`

轴取值集合与 launcher 的 grid 维数绑定：v0.1 launcher 只生成一维 grid（`triton-lowering.md` §7），允许 1/2 会让 `program_id(1)` 恒为 0，成为静默错误源——故收紧。

### R2 arange

```
c₁ = 0（字面量）    c₂ 为编译期常量表达式（字面量 / constexpr 名 / 其整数运算）
c₂ᵛ = 2^k（1 ≤ k ≤ 20）        其中 c₂ᵛ = build 期特化值；2^20 = TRITON_MAX_TENSOR_NUMEL
─────────────────────────────────────────────────────────────────────────────
Γ ⊢ tila.arange(0, c₂) : Tile[i32, (c₂ᵛ,), Identity((c₂ᵛ,))]
```
⇝ `tl.arange(0, ⟨c₂⟩)`——**发射保留 constexpr 名字/表达式，不折叠**（模型 B）

注：`c₁ = 0`、`c₂ᵛ = 2^k`、`≤ 2^20` 是 **Tila 的主动收紧**（检查口径 E06），不是对 Triton 约束的转述——Triton 当前实现的校验口径（end > start、差为 2 的幂、int32 范围）与文档口径（start/end 为 2 的幂、差 ≤ TRITON_MAX_TENSOR_NUMEL）之间存在漂移；Tila 取惯用法 `arange(0, BLOCK)` 收紧出自己的规则并全部前移。类型推导用 build 期特化值——**shape 保持全静态**；Triton 自身的编译期校验只作纵深防御，不承担 Tila 的健全性。

### R3 标量整数/浮点运算

```
Γ ⊢ e₁ : Scalar(i32)   Γ ⊢ e₂ : Scalar(i32)   ⊙ ∈ {+, −, ×}          （R3a）
Γ ⊢ e₁ : Scalar(f32)   Γ ⊢ e₂ : Scalar(f32)   ⊙ ∈ {+, −, ×, /}      （R3b）
────────────────────────────────────────────────────────────────────────────
Γ ⊢ e₁ ⊙ e₂ : Scalar(同一 dtype)
```
constexpr 整数与整数字面量自动提升为 i32 标量参与 R3a。跨类别（i32 ⊕ f32）→ E02；op 不在类别运算集 → E16（§1.1 统一原则）。标量值域只有 i32/f32。
⇝ 同形 Python 表达式

### R4 标量广播到 Tile

```
Γ ⊢ e : Scalar(dt)      Γ ⊢ t : Tile[dt, Σ, L]      ⊙ ∈ {+, −, ×, / 按 dt 类别}（整数配 i32 标量）
────────────────────────────────────────────────────────────────────────────────────────
Γ ⊢ e ⊙ t : Tile[dt, Σ, L]          （layout 由律 L2 保持；亦可 t ⊙ e）
```
⇝ 同形表达式（Triton 原生支持标量广播）

### R5 Tile ⊕ Tile

```
Γ ⊢ t₁ : Tile[dt, Σ₁, L₁]    Γ ⊢ t₂ : Tile[dt, Σ₂, L₂]    Σ₁ ⊗ Σ₂ 良式    L₁ ~ L₂
───────────────────────────────────────────────────────────────────────────────────
Γ ⊢ t₁ ⊙ t₂ : Tile[dt, Σ₁ ⊗ Σ₂, L₁]
```
v0.1 的实际形状要求是**严格相等**（`Σ₁ ≡ Σ₂`；两轴均 >1 且不等 → E03）；`⊗` 的 size-1 广播在 v0.2 预览文档 §3（R13）放宽。layout 由律 L4 保持。
⇝ 同形表达式

### R6 比较产生 mask

```
Γ ⊢ t : Tile[i32, Σ, L]（或任意数值 dt）    Γ ⊢ e : Scalar(dt) 或 Tile[dt, Σ', L']（Σ' ⊗ Σ 良式、L' ~ L）
◁ ∈ {<, ≤, >, ≥, ==, !=}     dt 支持 ◁（bool 只支持 == !=）
──────────────────────────────────────────────────────────────
Γ ⊢ t ◁ e : Tile[bool, Σ, L]
```
⇝ 同形表达式。字面量按 R12 解释类别。

### R7（历史；v0.3 定稿起内部化）

v0.1 的表面规则 `Γ ⊢ b + t : Address`（平面偏移）与初版 v0.3 的坐标扩展
`b + (t₀, …)` 均已**从表面语言移除**——地址算术整体退场，Address 成为
checker/TIR 内部类型：由 load/store 检查中的 `buffer_address` 构造
（`tila.load(buf, coords)` / `tila.store(buf, coords, value)`，坐标元组每轴
一个 i32 tile、长度 = buffer rank）。寻址前提并入 R8/R9，规则全文与设计裁决
见 `docs/v0.3-strides.md` §2.3。旧写法 → E07（独立算术）/ E13（load/store
实参形态，附迁移提示）。

**R8/R9 的 v0.3 寻址前提**（替代原 R7 前提行）：

```
Γ ⊢ b : Buffer[dt, Σ_B, M]     Γ ⊢ tᵢ : Tile[i32, Σᵢ, Lᵢ]（i = 0..r−1）   r = rank(Σ_B)
坐标元组长度 = r     Σ = Σ₀ ⊗ ⋯ ⊗ Σ_{r−1} 良式
───────────────────────────────────────────────────────────────────────────────────
内部 Address[dt, Σ, 坐标广播推导（Product/Join，同 R14 机制）]
```

线性化（`c₀*s₀ + c₁*s₁`）由 lowering 按 b 的 MemoryLayout 发射——手工线性化
盲区（两次"类型全对、结果全错"事故）从语言里根除。

### R8 load

```
Γ ⊢ b : Buffer[dt, Σ_B, M]   Γ ⊢ coords = (t₀,…,t_{r−1}) 寻址前提良式（R7 内部化）   mask/other 同左
Σ' ⊗ Σ 良式且 L' ~ L（v0.1 事实：Σ' ≡ Σ）    other 必须与 mask 同时给出（否则 E13）
────────────────────────────────────────────────────────────────────────────────────────────────
Γ ⊢ tila.load(p, mask, other) : Tile[dt, Σ, L]      （Load(M, L) 由律 L1 规范化为 L）
```
`other` 省略时 masked-out 通道的值未定义（与 Triton 一致）。边界安全由 mask 表达，编译器不做静态边界证明（与 Triton 一致）。
⇝ `tl.load(<p>, mask=<m>, other=<o>)`

### R9 store

```
Γ ⊢ b : Buffer[dt, Σ_B, M]   Γ ⊢ coords 寻址前提良式   Γ ⊢ v : Tile[dt', Σ', L']    mask（可选）
Σ' ⊗ Σ 良式且 L' ~ L（v0.1 事实：Σ' ≡ Σ）    dt' = dt（严格相等，无隐式收窄）
────────────────────────────────────────────────────────────────────────────────────────────────
Γ ⊢ tila.store(p, v, mask) : ()
```
store 的类型为 `()`，且**只能作为表达式语句**出现——赋值右侧或任何子表达式位置 → E08（unit 不可绑定、不可参与运算）。Triton 的 `tl.store` 允许 value 隐式广播与类型转换；Tila 刻意移除（E02/E03）——显式是卖点，不是缺口。
⇝ `tl.store(<p>, <v>, mask=<m>)`

### R10 布尔合取/析取

```
Γ ⊢ m₁ : Tile[bool, Σ₁, L₁]    Γ ⊢ m₂ : Tile[bool, Σ₂, L₂]    Σ₁ ⊗ Σ₂ 良式    L₁ ~ L₂
──────────────────────────────────────────────────────────────────────────────────────
Γ ⊢ m₁ & m₂ : Tile[bool, Σ₁ ⊗ Σ₂, L₁]          （| 同理）
```
操作数检查顺序：dtype 不一致（如 `bool & i32`）→ E02；dtype 一致但该 dtype 的能力集不含 `& |`（如 `i32 & i32`）→ E16。`and`/`or`（短路语义）不在语言中——块上短路无意义。⇝ `<m₁> & <m₂>`

### R11 显式 cast

```
Γ ⊢ x : Tile[dt₁, Σ, L]（或 Scalar[dt₁]）    dt₂ 为任意 dtype（17 种）
──────────────────────────────────────────────────────────────────
Γ ⊢ tila.cast(x, tila.<dt₂>) : Tile[dt₂, Σ, L]（Scalar 同理；layout 由律 L3 保持）
```
cast 矩阵全开（§1.1）：数值 dtype 任意互转，含 FP8 与 bool 两个端点。FP8 作为 cast 的**源和目标**都合法，限制只在算术侧。⇝ `<x>.to(tl.<dt₂>)`

### R12 字面量上下文规则

```
Γ ⊢ t : Tile[dt, Σ, L]    lit 为 INT 且 dt 属整数类（i*/u*）→ lit 按 dt 解释
                          lit 为 FLOAT 且 dt 属浮点类（f16…f64）→ lit 按 dt 解释
                          否则（类别不一致）→ E02
────────────────────────────────────────────────────────────────────────
Γ ⊢ t ◁ lit / t ⊙ lit / other=lit ：类型同 R6/R10/R8 的标量侧
```
`f16_tile + 1.0`、`f16_tile > 0.0`、`other=0.0` 合法；`f16_tile + 1` → E02（提示写 `1.0`）。

### R17 zeros（v0.4，`docs/v0.4-kloop.md` §2.1）

```
Γ ⊢ sᵢ 静态（INT 字面量 | constexpr 名，特化值 v(sᵢ)）    dt ≠ bool    rank ≤ 2
──────────────────────────────────────────────────────────────────────────────
Γ ⊢ tila.zeros((s₀,…,s_{r−1}), dt) : Tile[dt, (Const(v(sᵢ)),…), Zeros((v(sᵢ),…))]
```

常量分布种子：累加器的唯一合法播种方式。shape 必须静态（运行期符号维 → E20）——
累加器 shape 参与类型等价判定。⇝ `tl.zeros((BM, BN), dtype=tl.float32)`

### R18 for / tila.range（v0.4，`docs/v0.4-kloop.md` §2.2）

```
Γ ⊢ e : Scalar(i32)    step ∈ ConstExpr，特化值 ≥ 1    start ≡ 字面量 0
─────────────────────────────────────────────────────────────────────────
Γ ⊢ for x in tila.range(0, e, step): B    绑定  x : Scalar(i32)
```

循环引入的不是新算子而是新**作用域**（体内既有规则全部原样；三条围栏见
`v0.4-kloop.md` §2.4：外层名只读、累加器只 `+=`、循环局部不出循环）。
`program_id` 在体内引用合法（值跨迭代不变）。

### R19 累加与 φ（v0.4，`docs/v0.4-kloop.md` §2.3）

```
acc 在循环外由 R17 播种 : Tile[dt, Σ, Zeros(Σ)]        Γ, x ⊢ t : Tile[dt, Σ, L_t]
体内语句  acc += t ：dtype 相等（E02）· shape 严格同形（E03，不广播累加）
  L_acc ≡ Zeros → 结果 = normalize(L_t)（L7 单位元）· 否则 equiv(L_acc, L_t)（E05）
────────────────────────────────────────────────────────────────────────────────
φ(acc) : Tile[dt, Σ, L_exit]        循环后  acc : Tile[dt, Σ, L_exit]
```

φ 类型 = 体内最终类型（一行写完依赖两个围栏：zeros 种子使入口是 join 单位元；
体内禁读使 φ 无其它消费者）。`+=` 是普通 `add`（opcode 集不膨胀），累加语义由
TFor/TPhi 结构承载。⇝ 循环前 `acc = tl.zeros(...)`、体内 `acc = acc + …`
（loop-carried 由 Triton 接管）。

### 未来规则（v0.1 只预留 term 位置，不实现）

`reshape`/`transpose` 的化简律、`convert(L → L')` 带代价模型、`reduce`、`where`，以及 `cat`（R-cat，见下）。已定稿落地的：`expand_dim` 与 size-1 广播（R13/R14，`docs/v0.2-preview-2d.md`）、`dot` 的 MMA 约束（R16，`docs/v0.2-matmul-fragment.md`）、K 循环与累加器（R17/R18/R19，`docs/v0.4-kloop.md`）。比新规则更关键的是**桥接问题**：Tila 的等价/来源代数与 Triton concrete distributed layout（BlockedEncoding、LinearEncoding、CTALayout 等）如何对接——`Product(L₀, L₁)` 是 *Tila 语义分布积*，不声称等于 GPU 实际分布（`v0.2-preview-2d.md` §3）；`dot` 的 MMA 约束本质是桥接的第一个实例。这个问题决定 Tila 能否从 add demo 长成有研究价值的 DSL，实现前应先以 Triton oracle 实验钉死（`development-plan.md` §2）。

### R-cat（预留；shape 算术的首个消费者，§2.1）

```
Γ ⊢ a : Tile[dt, Σ, L₁]    Γ ⊢ b : Tile[dt, Σ', L₂]
Σ[i] ≡ Σ'[i]（∀ i ≠ axis）    L₁ ~ L₂
────────────────────────────────────────────────────────────────────
Γ ⊢ tila.cat(a, b, axis=k) : Tile[dt, Σ[k] ← Σ[k] + Σ'[k], L₁]
```

类型规则**先只管 shape**：结果维 `N + N` 经 §2.1 规范化为 `2N`——规则本身零特例，全部复杂度收进 shape 代数；结果的 layout 由独立的 layout rule 决定（shape 推导与 layout 推导分层书写，不混在一个 judgment 里）。

layout rule 采用**方案 A（前提 `L₁ ~ L₂`）**：两块分布等价时，沿拼接轴的延伸不改变分布族——结果 `~ L₁`，与律 L4 的擦除哲学同构。分布不等价的 cat（方案 B 的 `compatible` 偏序）不提供：那需要带代价的 layout convert（§3.4），等 convert 进入语言后再评估。若 Triton oracle 显示 `tl.cat` 的 realization 对结果分布另有要求，则升级为显式 `CatL` 项——引入前以 oracle 钉死（方法同 §5 桥接问题）。

---

## 6. 内部类型表示（Python 草案）

```python
# src/tila/types/
# type.py
from dataclasses import dataclass

Dim = Union["Const", "Sym"]            # v0.1
Shape = tuple[Dim, ...]

@dataclass(frozen=True)
class ScalarType:    dtype: str
@dataclass(frozen=True)
class BufferType:    dtype: str; shape: Shape; mem: "MemoryLayout"   # mem ∈ {RowMajor}
@dataclass(frozen=True)
class AddressType:   dtype: str; shape: Shape; layout: "LayoutTerm"
@dataclass(frozen=True)
class TileType:      dtype: str; shape: Shape; layout: "LayoutTerm"
@dataclass(frozen=True)
class UnitType:      ...

# TileType 中 shape 与 layout 是独立字段：shape 变换（expand_dim）与分布变换
# （Product 配对）互不隐含——Product 不携带 shape（v0.2 预览 §3 按此设计）。

# layout.py —— layout terms
@dataclass(frozen=True)
class Identity:      shape: Shape            # arange 种子
@dataclass(frozen=True)
class LoadL:         mem: "MemoryLayout"; idx: "LayoutTerm"
@dataclass(frozen=True)
class BcastScalarL:  of: "LayoutTerm"
@dataclass(frozen=True)
class CastL:         of: "LayoutTerm"
@dataclass(frozen=True)
class JoinL:         lhs: "LayoutTerm"; rhs: "LayoutTerm"
@dataclass(frozen=True)
class BroadcastL:    of: "LayoutTerm"; axis: int      # v0.2
@dataclass(frozen=True)
class ProductL:      lhs: "LayoutTerm"; rhs: "LayoutTerm"   # v0.2
# ReshapeL / TransposeL：v0.2 预留

def normalize(l: LayoutTerm) -> LayoutTerm:
    """应用律 L1–L4（v0.2 起含 L5/L6）到不动点；自底向上重写。"""

def equiv(l1: LayoutTerm, l2: LayoutTerm) -> bool:
    return normalize(l1) == normalize(l2)
```

---

## 7. 检查流程

```
parse → tila_ast（intrinsic resolution 完成）
  → 构建 Γ（kernel 签名、constexpr 特化、符号维绑定）
  → syntax-directed 自底向上推导，产出 typed TIR
       每个 binop 依次检查：dtype → shape → layout(normalize + equiv)
  → 任一检查失败 ⇒ 编译期错误（附两个操作数的完整类型与来源行号）
  → launch analysis（独立 phase，在 typed TIR 上推导 LaunchPlan；失败 ⇒ E17）
  → Triton lowering（逐 op 纯语法映射，total）
```

关键性质：**lowering 对有效 TIR 全函数（total）、无新语义拒绝**。所有可判定属性已被 typing 阶段消耗，这就是"强类型"在本项目中的工程含义。

---

## 8. add 例子的完整类型推导

源码见 `examples/add.tila`（tila 命名空间版）。

```
Γ:  a : Buffer[f32, (N,)]    b : Buffer[f32, (N,)]    c : Buffer[f32, (N,)]
    N : Scalar(i32)（符号维自动绑定）    BLOCK = 128（constexpr）
```

| 源码 | 规则 | 推导出的类型 | 规范化后 layout |
|---|---|---|---|
| `pid = tila.program_id(0)` | R1 | `Scalar(i32)` | — |
| `offs = pid * BLOCK + tila.arange(0, BLOCK)` | R3+R4+R2 | `Tile[i32, (128,), …]` | `L₀ = identity((128,))` |
| `mask = offs < N` | R6 | `Tile[bool, (128,), L₀]` | `L₀` |
| `x = tila.load(a, (offs,), mask=mask)` | R8（内部 addptr）+律 L1 | `Tile[f32, (128,), …]` | `L₀` |
| `y = tila.load(b + offs, mask=mask)` | R7+R8+律 L1 | `Tile[f32, (128,), L₀]` | `L₀` |
| `z = x + y` | R5：`f32=f32` ✓ `(128,)≡(128,)` ✓ `L₀~L₀` ✓ | `Tile[f32, (128,), L₀]` | `L₀` |
| `tila.store(c + offs, z, mask=mask)` | R7+R9：rank ✓ shape ✓ dtype ✓ layout ✓ | `()` | — |

注意核心一行 `z = x + y` 能通过，是因为两条独立的 load 经**律 L1** 规范化到同一个 `L₀ = identity((128,))`——这就是 layout algebra 在 v0.1 里干的第一件实事：证明两次独立加载的分布天然一致，无需任何转换。

### typed IR

canonical dump（黄金测试比对物）**唯一**定义于 `type-checker.md` §8，格式规则见 `ast.md` §5。此处不复刻第二份。

### Triton lowering

见 `examples/add.lowered.py`。逐行同形映射，`tila.load(a + offs)` → `tl.load(a + offs)`。

### 主机侧启动（launcher 职责，不在 kernel 类型系统内）

```
add[grid = (triton.cdiv(N, BLOCK),)](a, b, c, BLOCK=128)      # N 由 launcher 从 a 的形状读取
```

---

## 9. 编译期错误示例

强类型的价值在错误信息里表现最清楚。以下错误全部在编译期被拒绝，且不会生成任何 Triton 代码：

**① dtype 不匹配（E02）**

```
z = x + y      # x : Tile[f32, (128,)], y : Tile[i32, (128,)]

E02 TypeMismatch
cannot apply '+' to operands
  lhs: Tile[f32, (128,)]
  rhs: Tile[i32, (128,)]
expected: same arithmetic dtype
hint: use tila.cast(rhs, tila.float32)
```

**② shape 不匹配（E03）**

```
x = tila.load(a, (offs,), mask=mask)   # Tile[f32, (128,), L0]
y = tila.load(b + offs2, mask=mask)    # Tile[f32, (64,), L1]   offs2 = arange(0, 64)
z = x + y

E03 ShapeMismatch
cannot broadcast:
  lhs: (128,)
  rhs: (64,)
dimension 0: 128 vs 64
only equal or size-1 dimensions are broadcastable
```

**③ store 的 dtype 不收窄（E02）**

```
z : Tile[f32, (128,), L0]
tila.store(c + offs, z, mask=mask)      # c : Buffer[f16, (N,)]

E02 TypeMismatch
cannot store Tile[f32] into Buffer[f16]
implicit narrowing is not allowed; write tila.cast(z, tila.float16) if intended
```

**④ Triton 约束前移（E06）**

```
i = tila.arange(0, 100)

E06 ArangeShape
arange extent 100 is not a power of two (Tila requires 2^k, 1 ≤ k ≤ 20)
```

**⑤ FP8 是存储 dtype（E16）**

```
x = tila.load(a, (offs,), mask=mask)   # a : Buffer[fp8e4m3fn, N] → Tile[fp8e4m3fn, …]
y = tila.load(b + offs, mask=mask)
z = x + y

E16 StorageOnly
fp8 dtypes are storage-only: no arithmetic, no comparisons
hint: tila.cast(x, tila.float16) + tila.cast(y, tila.float16)
```

正确写法见 `examples/fp8_add.tila`。

---

## 10. v0.1 明确不做

推迟的东西与理由：

| 推迟项 | 理由 |
|---|---|
| 指针算术（表面语言中的任意指针运算） | v0.1 限于 `a + offs` 一种形态；v0.3 定稿整体移除——Address 内部化，坐标只在 load/store 实参出现 |
| shape 算术 / 符号求解 | 1D add 不需要；目标代数已定为全运算 DimExpr（§2.1，2026-08-24 裁定支持全部整数运算）——两层判定（确定性重写 + 约束感知证明）、must_equal 只认 ProvenEqual、SMT 仅可插拔 fallback 且不进主路径；引入时机随 `cat`（R-cat）/ reshape / 维度绑定索引评估 |
| 混合 dtype 提升格 | 显式 cast 更符合强类型立场，且实现极简 |
| 整数除法 `//`、`%` | v0.1 索引算术用不到；能力表收紧，随 2D/归约进入 v0.2 |
| layout convert + 代价模型 | 1D 全部布局同源于单一 arange，等价即可覆盖；转换是 2D/transpose 之后的事 |
| 2D tile、expand_dim、size-1 广播、Product | 已随 v0.2 预览与 matmul fragment 落地（`docs/v0.2-preview-2d.md`、`docs/v0.2-matmul-fragment.md`） |
| K 循环与累加器（已落地 R17/R18/R19 + 律 L7） | `docs/v0.4-kloop.md` 定稿落地：受限累加器（zeros 播种 + `+=` 围栏 + φ）、`tila.range`/`tila.zeros`、E20；嵌套循环 / 非 zeros 播种 / 体内读累加器仍推迟（`v0.4-kloop.md` §9） |
| dot / mma（已落地 R16） / reduction / where / 共享内存 | 归约在 K 循环累加之后；共享内存由桥接问题（§5）的结论决定 |
| mask 省略时的静态边界证明 | 与 Triton 语义对齐，mask 是用户表达边界安全的方式 |

---

## 11. 评审意见处置

原则：与定稿语义重合的吸收为显式条文；方向一致但时机不对的记为承诺；冲突的明确拒绝并给出理由。

### 第一轮（2026-08-24，symbolic shape 专题）

外部评审建议（"把类型系统做成小型 symbolic shape system，但不上完整 dependent type"）：

**已具备（建议与现状重合，条文不变）**：

- "Tensor⟨D,S,L⟩ 三元组" —— 即 §1 的 `Kind × DType × Shape × Layout` 四元组（Kind 同为推导对象）。
- "值分宇宙：Constexpr / ShapeSymbol / Scalar / Tile" —— `semantic-model.md` §1 的六类值分层（另含 Buffer/Address/unit 三类参数侧与内部值）；"三个名字宇宙互不相交"是结构保证（E12 名字冲突检查），见该文档 §1 要点末条。
- "符号维是隐式参数" —— §4：注解中的 `N` 自动绑定为运行期 `Scalar(i32)` 参数——类型层的符号 + 值层的运行期标量，一体两面（`offs < N` 的 mask 谓词由此无需投影函数）；launcher 断言一致性。
- "checker 不碰 Python AST" —— 现有 pipeline（pyast → tila_ast → checker，`ast.md`）。
- "检查应同时是推导" —— `type-checker.md` §2 的 `infer(expr) → (Type, TOp)`，每个 SSA 值带完整类型。

**采纳为承诺（本次新增条文）**：

- 线性规范形 LinearExpr：shape 算术的落地形式（§2.1）。**【第二轮修订：从"终点"降级为线性片段的规范形式 / 分阶段第一阶段，目标终点升级为全运算 DimExpr，见下】**
- must_equal 保守 soundness："不可证相等 ⇒ 拒绝"升为显式命名原则（§2.1）。
- 不透明项策略：非线性/未定项保留原样，等式只分可证/不可证（§2.1）。
- R-cat 草案：cat 的类型规则与 `N + N → 2N` 规范化（§5）。

**拒绝（与定稿设计冲突）**：

- **`Tensor[f32, [N, M]]` 方括号嵌套注解** —— 表面语法保持扁平 `Tensor[f32, N, M]`（单一 Subscript，`language-spec.md` §4）。嵌套 tuple 无信息增益，只与 PEP 563 字符串注解的再解析徒增纠缠。
- **`arange(0, N)` 符号边界提前放开** —— R2 前提（编译期 2^k）不动：Tile shape 全静态是 v0.1/v0.2 一切布局推导的地基；运行期长度由 launcher grid + mask 表达（符号边界 → E10）。
- **注解内 shape 表达式（`Tensor[f32, 2*N, M]`）现在放开** —— 时机问题而非方向问题：随线性代数一并进入（§2.1 末条），先开语法再补语义是反的。
- **v0.2 即引入 max/min/ceildiv shape 项** —— in-kernel shape 运算尚无消费者；grid 计算在主机侧 launcher（§2）。留待真实需求出现时按不透明项策略引入。
- **SMT / Presburger 求解** —— 同 §10：线性 canonical form 覆盖可预见需求，不预建求解器。**【第二轮修订：维持"不预建、不进主路径"，升级为"全运算代数下可作可插拔 fallback"，见下】**

### 第二轮（2026-08-24，全运算 shape 代数 + layout 语义）

裁定：shape 类型**支持全部整数运算**。评审建议逐条处置：

**采纳**：

- **DimExpr 全运算目标代数**：`+ − × floordiv ceildiv mod max min` 全集进入目标文法，分阶段启用（线性片段先行，非线性项随后、保留原样不强行化简）；`/` 以歧义为由拒绝（§2.1）。
- **ProofResult 三值**（ProvenEqual / ProvenNotEqual / Unknown）与 must_equal：只认 ProvenEqual，Unknown 拒绝，ProvenNotEqual 用于诊断升级——"可证不等"与"不可证"在错误信息中区分（§2.1）。
- **fast/slow path**：确定性重写为主路径；SMT（Z3 类）仅可插拔 fallback、永不进入主执行路径；未接入时结论即 Unknown ⇒ 拒绝，健全性不依赖 solver 存在（§2.1）。
- **shape 约束与 `tila.assume` 提案**：维度自动正性（N ≥ 1）；assume 是编译期 shape 约束、非运行期断言；Γ = Values + ShapeSymbols + Constraints（§2.1，阶段二）。
- **`ceildiv` 一等公民**（grid 数学刚需）；**`DimExpr` 命名**（Shape = DimExpr 列表，一维表达式与整体 shape 分名）；**hash-consing DAG** 表示（等价子式共享）。
- **cat 的 layout rule 定案为方案 A**：前提 `L₁ ~ L₂`、结果 `~ L₁`（擦除哲学同 L4）；方案 B（compatible 偏序）随 convert（§3.4）再评估；shape 推导与 layout 推导分层书写（§5 R-cat）。
- **"表达式完整性 ≠ 化简完整性"**：四职责——表示、求值、尽力证明、证不出保守拒绝（§2.1）。

**已具备（建议与现状重合）**：

- "layout 用户不可见、compiler-internal"——§0 原则 2。
- "logical（Tila 布局代数）与 physical（Triton encoding）分层，存在 realize 映射"——即 §3.2 定位与 §5 桥接问题；realize 的精化是开放问题，不写成规范承诺。
- "layout 只回答三问（compatible / infer / equiv）"——v0.1–v0.2 即 equiv + 构造点表（`type-checker.md` §3）；compatible 偏序随 convert 才有意义。
- "v0.1 布局可以非常弱"——事实：v0.1 正规形式恒为 `Identity(n)`。
- "每个 intrinsic 一条 typing rule、推导局部化"——R1–R16 + `infer` 规则表。

**拒绝**：

- **bidirectional typing（infer ⇐ check 双 judgment）**——v0.1/v0.2 表面语言没有值位置的类型注解（注解只在参数签名），纯 infer 对直线 SSA 已完备；待局部注解 / check 上下文进入语言时再评估。
- **`/` 进入 shape 域**——整数域歧义（floor vs real），`FloorDiv`/`CeilDiv` 显式消歧（§2.1）。
- **任意 Python `assert` 作为约束来源**——只接受未来的 `tila.assume`（语义明确：编译期约束，非运行期检查）。
- **solver 进入主路径 / 自造求解器**——fast path 优先，SMT 只作可插拔后端且默认不接入（§2.1）。