# Tila 类型检查器设计

状态：定稿（2026-08 评审修订）。checker 是唯一的 `tila_ast → TIR` 通道，实现 `type-system.md` 的规则 R1–R12 与化简律 L1–L4。本文给出：职责与接口、规则驱动结构、二元运算分派矩阵（机器可读）、内建检查流程、normalize/equiv 实现、诊断目录 E01–E17（含相对初稿的修订对照）、add 走查、测试策略与实现清单。

---

## 1. 职责与接口

```python
def check_kernel(k: tila_ast.KernelDef,
                 constexpr_overrides: dict[str, int]) -> tir.TKernel:
    """失败时抛 TilaError；成功返回可直接 lowering 的 TKernel。"""
```

- 输入：单个 KernelDef + 调用点的 constexpr 覆盖（编译期特化）。
- 输出：`TKernel`（参数表 + 指令序列 + launch plan）。
- 失败：`TilaError(loc, code, message, notes)`，绝不以 Python 原生异常泄漏（§6）。

内部固定两个 phase：**typing（§3–§5，产出 typed TIR 指令序列）→ launch analysis（§3 步骤 5，在 typed TIR 上推导 LaunchPlan）**。launch 分析是语义模式识别（识别 `pid * CE + arange` 的 tiling 惯用法），不是类型检查；独立成 phase 是刻意的——未来 autotune、多种 grid 策略、persistent kernel 都挂在这一层，不让 typing 膨胀成巨型分析器。

## 2. 规则驱动结构（不做 if/elif 泥潭）

检查器按"每类节点一个 judgment 函数 + 一张规则表"组织，不在 `infer` 里堆 `if isinstance(...)`：

```python
# src/tila/checker/checker.py
RULES: dict[type, Callable] = {
    IntLit:   check_lit,
    NameRef:  check_name,
    BinOp:    check_binop,     # 内部再查 ARITH_RULES 矩阵（§4.1）
    Call:     check_call,      # 内部按 intrinsic 分派到 builtin.py 的各流程（§4.2）
    Cast:     check_cast,
}

def infer(expr, env) -> (Type, TOp):
    rule = RULES[type(expr)]
    return rule(expr, env)
```

新增 intrinsic 或运算符 = 注册新条目，不触碰既有分派路径。`operator` 与 `intrinsic` 的签名表是数据（§4），文档从同一份数据渲染（§4 注）——spec / implementation / tests 不分叉。

## 3. 类型与 layout term 的实现

类型与 term 即 `type-system.md` §6 的 dataclass（`ScalarType / BufferType / AddressType / TileType / UnitType`；dist 侧的 `Identity / Mma / Product / Lift / Slice / Seed` 与标量面 `NoDist`；memory 侧 `RowMajor / Strided`）。v0.6a 起（评审 D）擦除包装 term（LoadL / BcastScalarL / CastL / JoinL）退役，**构造点直接产出正规形式**（loader 不变量：TIR 类型中的 dist 必须已是正规形式）；join 的带诊断规则在 `checker/layout.py`（strict_join / read_join / join_distributions / marginal）。补充：

```python
# src/tila/types/dtype.py
DTYPES = ("bool",
          "i8", "i16", "i32", "i64",
          "u8", "u16", "u32", "u64",
          "f16", "bf16", "f32", "f64",
          "fp8e4m3", "fp8e5m2", "fp8e4m3fn", "fp8e4m3b15")   # universe 与 triton.language 对齐

STORAGE_ONLY  = ("fp8e4m3", "fp8e5m2", "fp8e4m3fn", "fp8e4m3b15")   # 只能 load/store/cast
INT_KIND      = lambda dt: dt in DTYPES[1:9]                        # i*/u*：INT 字面量可配
FLOAT_KIND    = lambda dt: dt in ("f16", "bf16", "f32", "f64")      # FLOAT 字面量可配
BOOL_KIND     = lambda dt: dt == "bool"
```

layout term 的**构造点**（每条规则产出的 term，构造时即断言律的前提成立）：

| 规则 | 构造（v0.6a：直接正规形式） | 机制 |
|---|---|---|
| R2 `arange` | `Identity(shape)` | — |
| R13 `expand_dim` | `Lift(D, axis)` | 无所有权轴（v0.6a 更名自 Broadcast） |
| R4 标量⊕Tile | `read_join(D_tile, NoDist)` | 标量不携带分布（评审 §25） |
| R5 Tile⊕Tile | `join_distributions`（同形→strict_join / one-sided→read_join / 双广播→逐轴段联合） | L5/L6 |
| R6 比较 | 同 R4/R5 的 join（标量侧 NoDist） | |
| R7 `addptr` | `infer_coordinate_dist(t₀,…,t_{r−1})` | 逐轴段联合（n-ary Product） |
| R8 load | 结果 dist = 索引 dist（load 保序） | |
| R10 `& \|` | 同 R5 的 join | |
| R11 cast | 结果 dist = 操作数 dist（保序） | |
| R17 zeros | `Seed(shape)` | 种子状态载体（非 DistExpr） |
| R20 sum/max | `marginal(D, Σ, k)` | 分支六：Mma → Slice(Mma,k) |

**normalize / equiv**：

```python
# src/tila/types/dist.py（v0.6a 重构，评审 D）
def normalize_dist(d: TileDist) -> TileDist:
    """律 D1–D6（type-system.md §3.3）：Identity 原样 / Product 拍平 +
    逐分量 / Slice·Lift 内层规范化 / Lift 同轴幂等；无 speculative rewrite。"""

def equiv_dist(d1: TileDist, d2: TileDist) -> bool:
    return normalize_dist(d1) == normalize_dist(d2)
```

v0.6a 起正规形式恒为五 term + Seed 之一的结构树；构造点全部产出正规形式，
`TOp.origin_dist` 仅作诊断记录。dist 等价是语义代数，不由求解器判定
（`constraints` 层永不 import dist term——结构判定，无 SMT）。

## 4. 表达式推导

### 4.1 二元运算分派矩阵（机器可读）

`infer_binop` 先推导两侧得 `(τ_l, τ_r, op)`，再按类型类别查**同一份数据**（`src/tila/checker/arithmetic.py`；文档 §4.1 由此渲染）：

```python
# src/tila/checker/arithmetic.py —— v0.1 运算分派矩阵（唯一权威来源，文档由它生成）
ARITH_OPS   = {"+", "-", "*", "/"}
CMP_OPS     = {"<", "<=", ">", ">=", "==", "!="}
LOGIC_OPS   = {"&", "|"}

# 同 dtype 规则：(lhs_dtype, rhs_dtype) → 结果 dtype；表外组合 → E02。
# 全表 = 各能力类别内 dtype 的笛卡尔积（i*/u* 各 4×4、浮点 4×4；整数类别不含 "/"）。
ARITH_RULES = {
    (I32, I32): I32, (I64, I64): I64, (U32, U32): U32, (U64, U64): U64,
    (F16, F16): F16, (BF16, BF16): BF16, (F32, F32): F32, (F64, F64): F64,
    # …（完整表由 dtype.py 的能力表自动展开；混合组合不在表中 → E02。）
}
```

矩阵（人类可读视图，同一数据的渲染）：

**算术运算**（`+ - * /`；`/` 只对浮点类别开放，整数 `/` → E16）

| lhs ＼ rhs | `Scalar` | `Tile` |
|---|---|---|
| `Scalar` | R3 → `TArith`（i32 配 i32、f32 配 f32；跨类别 E02） | R4 → `TArith`（标量 = NoDist，dist 随 tile：read_join） |
| `Tile` | R4 → `TArith` | R5 → `TArith`（dtype 一致 → shape `⊗` → join_distributions） |

**比较运算**（结果一律 `Tile[bool, Σ_lhs ⊗ Σ_rhs, L]`；标量/字面量侧接受 `Scalar(dt)` 严格同 dtype，或**类别匹配的字面量**：`a > 0.0`、`a < s` 均合法）

| lhs ＼ rhs | 标量/字面量 | `Tile` |
|---|---|---|
| `Tile` | R6 → `TCmp` | R6 → `TCmp`（dtype 一致 → shape `⊗` → layout equiv） |
| `Scalar` | E07（无标量布尔类型，组合无定义） | — |

**`& |`**（R10）：两侧必须 `Tile[bool]`（否则 E02/E16，见 type-system §R10 注）→ shape `⊗` → layout equiv → `TLogic`。

检查顺序固定：**dtype → shape → dist**（错误报告优先级同序）。shape 相等/⊗ 用 `checker/broadcast.py` 的 `require_broadcast`；dist 用 `join_distributions` / `strict_join` / `read_join` / `require_equiv`。

**能力门（矩阵命中之后、规则前提的一部分）**：查能力表（`type-system.md` §1.1）——FP8 四种格式不做任何算术与比较（→ E16，消息附 `tila.cast(a, tila.float16) + …` 建议）；bool 不做算术、比较仅 `== !=`；整数不含 `/` 与 `& |`；`u*` 与 `i*` 是不同 dtype，混用 → E02。**统一原则：op ∉ 该 dtype 类别的运算集 ⇒ E16**，不逐条枚举。

### 4.2 内建调用流程

```
program_id(c):  args 恰 1 个且为字面量/constexpr 折叠值 ∈ {0}（v0.1；v0.2 预览片段内放宽为
                {0,1}），否则 E13 → R1
arange(a, b):   args 恰 2 个且为编译期常量表达式（ConstExpr）；a == 0（字面量）；b 的特化值 = 2^k
                （1 ≤ k ≤ 20），否则 E06；边界非常量表达式 → E10 → R2
                end 以 ConstExpr 进入 TIR，发射保留名字（模型 B）
load(buf, coords, mask=?, other=?):        # v0.3 定稿：一级坐标原语（两位置实参 + kwargs）
    1. buf 为 Buffer（NameRef → env.buffers，否则 E07）
    2. coords 为坐标元组（IndexTuple，只在 load/store 第 2 实参位由转换期构造；
       非元组 → E19 CoordinateForm）；长度 = rank(buf)（否则 E19 CoordinateArity）；
       每项 Tile[i32]（标量 → E19 CoordinateKind、非 i32 → E02）；
       坐标广播良式（E03）→ checker 内部 buffer_address 构造 Address 并发射 addptr
    3. mask（若给）：Tile[bool, Σ', L']；Σ' ⊗ Σ 良式（否则 E03）、
       dtype 必须 bool（E02）、L' ~ L（E05）
    4. other（若给）：字面量，类别与 dt 匹配（整型 dt 配 INT、浮点 dt 配 FLOAT），
       否则 E02；且要求 mask 同时给出（无 mask 的 other 无意义 → E13）
    5. 结果 dist = 索引 dist（load 保序）→ R8
store(buf, coords, value, mask=?):
    0. 语境检查：store 只能作为 ExprStmt；出现在 Assign 右侧或任何子表达式位置
       → E08（类型 () 不可绑定、不可参与运算）
    1. buf/coords 同 load 第 1–2 步（内部 Address）；value 为 Tile，否则 E07
    2. rank 相等（E04）→ shape Σ_v ⊗ Σ 良式（E03）→ dtype 严格相等，无隐式收窄（E02）
    3. layout：equiv(L_v, L)，mask 给出时再 equiv(L_mask, L)（E05）→ R9'（坐标语义）
cast(x, dt):
    args 恰 2 个；dt 为 DTypeRef（17 种任意，否则 E09）；x 为 Tile 或 Scalar（否则 E09，
    比如把 Buffer 当操作数 —— Tile/Address 也不可 cast，地址不是值）→ R11
sum/max(x, axis):                       # v0.5（docs/v0.5-reduce.md §2.1）
    1. args 恰 (tile, axis)——axis 位置实参或唯一 kwargs axis=（INT 字面量 /
       constexpr 特化值 ∈ {0,1}，否则 E13）
    2. x 为 Tile（E07）且 rank = 2（否则 E21 ReduceRank——rank-1 全归约与
       rank≥3 都推迟）
    3. 能力门同源：sum→'+' 算术能力、max→'<' 比较能力（fp8/bool → E16）
    4. dist：marginal(D, Σ, k)（checker/layout.py 的构造规则——Product →
       幸存分量 / size-1 轴 → 原样（Lift(D,k) 归约到无所有权轴 → D）/ 唯一
       非平凡轴被归约 → Identity(()) / Seed → Seed(Σ∖k) / Mma → Slice(Mma, k)
       （v0.6a 分支六）/ 其余 → E21 ReduceLayout——分支六后对合法输入
       为空位，保留码位防新 term）
    5. 结果 dtype = 输入 dtype（Σ∖k；TReduce 携带 op/axis 特化值）→ R20
exp/exp2/sqrt/abs(x):
    args 恰 1 个（E13）；x 为 Tile（E07）；dtype ∈ UNARY 能力表（E16——
    exp 族 float-only、abs int+float）；layout 保序（L8 记法，无新 term）→ R21
where(c, a, b):                         # v0.5（§2.3）；v0.6a R-PT 修订
    1. args 恰 3 个、无 kwargs（E13）
    2. c 为 Tile[bool]（E02）；a/b 各为 Tile 或字面量/neg_inf（标量名 → E13），
       不得同为字面量（E13）
    3. dtype：Tile 侧严格相等（E02）；字面量侧类别检查（R24，E02）；
       fp8 → E16（storage-only 无 select）
    4. shape：Σ = Σc ⊗ Σa ⊗ Σb 三方广播（E03，字面量侧不参与；cond 可
       撑大值侧结果——被撑大的轴在 dist 侧包 Lift）；
       dist：**R-PT——cond 是谓词不进入分布代数**；结果 = 值侧 pairwise
       join_distributions（同形双侧 → strict_join/E05；one-sided 广播 →
       read_join 取全形侧）→ R22
num_programs(axis):
    args 恰 1 个且为字面量/constexpr ∈ {0,1}（E13）→ Scalar(i32)；TNumPrograms
    不进入 launch analysis（observational）→ R23
```

- kwarg 未知键在转换期已被 E13 拒绝；位置实参多于形参要求 → E13。
- `DTypeRef` 出现在非 cast 第二实参/非注解位置 → E13。
- buffer 参数出现在表达式位置（除 `+` 的左操作数）→ E07（Buffer 只能作 R7 的左操作数）；未绑定 → E01。

### 4.3 Cast

`tila.cast(x, dt)`（R11）：operand 为 `Tile[dt₁, Σ, D]` 或 `Scalar[dt₁]`；目标 `dt` 为 DTYPES 任意成员；结果 `Tile[dt, Σ, D]`（v0.6a 起保序直接表达，CastL 包装随重构退役）/ `Scalar[dt]`。cast 矩阵全开——数值 dtype 任意互转，含 FP8 与 bool 两个端点（FP8 作为源和目标都合法，限制只在算术侧；bool 目标 = 数值→掩码转换，非零为真，不是逻辑谓词的构造）。字面量不可直接作 cast 操作数（配标量走字面量上下文规则 §4.1）。

### 4.4 叶子

- `IntLit` → `TConstInt`，需要 constexpr 处直接用值；类型 `Scalar(i32)`。
- `FloatLit` → `TConstFloat`，`Scalar(f32)`。
- `NameRef`：查 `locals → buffers → syms → constexprs`；buffer 出现在非 R7 位置 → E07（Buffer 只能作 `+` 左操作数）；未绑定 → E01。

---

## 5. 环境与签名检查

```python
@dataclass
class Env:
    buffers:   dict[str, BufferType]                 # 参数 Buffer（注解给出 dtype/shape/mem）
    syms:      dict[str, SymBinding]                 # 符号名 → (kind: dim|stride, 来源 buffer, axis)
    constexprs: dict[str, int]                       # 已特化的值
    locals:    dict[str, Binding]                    # name → (type, defining loc, tir id)

@dataclass(frozen=True)
class SymBinding:
    """v0.3：Dim 符号与 Memory(stride) 符号共用一张符号表（评审 §9/§11）——
    都是"运行期需物化的编译期符号"；kind 供诊断与将来的 assume/约束系统。"""
    kind: str                                        # "dim" | "stride"
    buffer: str
    axis: int

@dataclass(frozen=True)
class Binding:
    tila_type: Type
    loc: Loc                                         # 诊断回指"定义于第 N 行"
    id: str
```

签名检查顺序：

1. 逐参数解析注解：dtype ∈ DTYPES（否则 E12）；**bool 不得作 Buffer 元素 dtype → E12**（能力表规定 bool 只作 mask/比较结果——否则一个未被使用的 `Buffer[bool]` 参数会全程漏检）；**rank ≥ 2 → E12**（v0.1 内 2D 参数永远无法被 load/store，放行只是陷阱；`v0.2-preview-2d.md` 片段解禁 rank 2）；`dim` 为 `StaticDim(int)` 或 `SymDim(name)`；sym dim 按首次出现顺序收集，重复出现绑定同一符号。无注解参数 → `Scalar(i32)`。**尾随 strides 元组（v0.3）**：元素同 dim 文法（INT 且 ≥ 1，或 IDENT），长度必须等于 rank（否则 E12，转换期检查）；IDENT 按同款规则绑定（已绑定的符号（含维符号）→ 引用，新名 → 追加 sym 序），符号表条目为 `SymBinding(kind, buffer, axis)`（dim/stride 二分类共用一张表，`v0.3-strides.md` §2.2）；MemoryLayout 取 `Strided(...)`，省略元组则 `RowMajor`。坐标寻址的形态检查见 R7'/E19（`docs/v0.3-strides.md`）。
2. **名字冲突检查（E12）**：sym dim 名不得与任何参数名、constexpr 名相同，也不得是保留名——否则生成签名出现重复形参或保留名遮蔽。kernel 名、保留名（`tila`/`tl`/`triton`）不得作参数名/赋值目标（E12）。
3. `tila.constexpr` 参数：默认值为整数字面量（E15）；用调用点覆盖值特化；无默认且未覆盖 → E15。特化值绑定进产物：**每个 override 组合一次完整编译，TIR 跨值不复用**（模型见 `semantic-model.md` §6）。
4. **launch plan 推导（E17；独立的 launch analysis phase——typing 完成后在 typed TIR 上运行，见 §1）**：确定 grid 轴数（kernel 内实际使用的 `program_id` 最大轴 + 1）与每轴的 `(维, tiling constexpr)`（"tiling constexpr" = 在 `pid * CE` 中被引用的那个，每轴必须唯一确定；"维" = 该轴 mask 谓词（`offs < N`）覆盖的符号维，静态维直接代入字面量）；同时收集 rank/静态维/符号维断言。推导不出（如两个 constexpr 同时参与同一轴的 tiling）→ E17。结果存入 `TKernel.launch_plan`，lowering 只发射不推导——维持 lowering 对有效 TIR 全函数（total、无新语义拒绝）。
5. 函数体非空由转换期保证（空体/`pass` 在 frontend 已 E11 拒绝），checker 不再重复检查。

**constexpr 求值器**：对只含 `IntLit`、constexpr 引用、整数运算的子表达式求值（产出 `ConstExpr` 及其 build 期值）；**只用于检查**（arange 边界、pow2/上限、字面量提升），**发射侧不折叠**——ConstExpr 原样进入 TIR 与生成的 Triton（模型 B；检查口径逐值特化、TIR 跨值不复用）。求值失败（依赖运行时值）→ E10（仅 `arange` 边界会遇到）。

## 6. 语句检查

```
Assign(target, value):
    target 是保留名 → E12；已绑定名（含参数与符号维——N 是自动绑定参数，N = 5 同罪）→ E14
    infer(value)；若结果类型为 ()（即 value 是 store 调用）→ E08
    绑定 locals[target]，指令 src_name = target
    value 为 zeros 调用且 target 被 kernel 全体 '+='（预扫描）→ 登记为累加器（R19 前提）

AugAssign(target, value)（v0.4，docs/v0.4-kloop.md §2.3）:
    体内（不在任何循环 → E20 AccumForm）
    target 是外层 locals 且已登记累加器（否则 E20 AccumForm）
    infer(value)；value 非 Tile → E07；dtype ≠ 累加器 → E02；shape 严格同形 → E03
    layout：L_acc ≡ Zeros → 结果 = normalize(L_t)（L7）；否则 equiv → E05
    发射 add（id = {acc}.next{N}，src_name 不变）；当前绑定推进为链上下一环

For(target, iter=tila.range(0, end, step), body)（v0.4，§2.2/§2.4）:
    嵌套 → E20 NestedLoop；target 保留名/已绑定 → E12/E14
    range 形态：恰 3 实参、start 恒 0、end 为 Scalar(i32)、step ∈ ConstExpr ≥ 1（E20 RangeForm）
    作用域三围栏：外层 locals 只读（重绑定 → E14 既有）；累加器体内只可作 '+=' 左部
    （读 → E20 AccumRead）；循环局部（含 target）出循环不可见（引用 → E20 LoopScope）
    收尾：每个被累加名字构造 TPhi（pre = 入口 id、back = 最后 next id、类型 = 出口类型）
    前插 body 顶部；TFor(id = target) 携带嵌套 body 发射

ExprStmt(call):
    intrinsic != "store" → E08（v0.1 唯一合法表达式语句是 store；循环体内同样合法）
    按 §4.2 store 流程检查（含语境检查 0）
```

## 7. 诊断系统

```python
@dataclass(frozen=True)
class TilaError(Exception):
    loc: Loc
    code: str            # E01…E17
    message: str         # 英文，一行结论
    notes: tuple[Note, ...]   # 操作数完整类型回显 + 定义位置 + 修复建议
```

notes 的固定形态：每个相关操作数一行 `lhs : Tile[f32, (128,), L0]  (defined at line 4)`，末尾一行建议（如 `use tila.cast(rhs, tila.float32)`）。渲染格式与 `type-system.md` §9 的示例一致（`E02 TypeMismatch` 头 + 结论行 + 操作数行 + expected/hint 行）。

**诊断码目录（2026-08 修订版）**：

| 码 | 场景 | 触发点 |
|---|---|---|
| E01 | 未绑定名字（先使用后定义、拼写错误） | NameRef |
| E02 | dtype 不匹配（混 dtype 运算、store 收窄、字面量类别错配、`&` 两侧非 bool 等） | 各 dtype 检查 |
| E03 | shape 不匹配 / 不可广播（同轴均 >1 且不等） | R5/R6/R8/R9/R10 的 shape 检查 |
| E04 | rank 不匹配 | load/store 的 rank 检查 |
| E05 | layout 不等价 | R5/R6/R8/R9/R10 的 equiv |
| E06 | `arange` 起点非 0、长度非 2 的幂、≤ 1 或超 2^20 | R2 |
| E07 | 类型类别组合无定义（Buffer 参与非 R7 运算、Address 作值、标量间比较、cast 操作数非法等） | 矩阵、§4.2/4.4 |
| E08 | store/unit 语境：表达式语句不是 store；store 出现在值位置（含赋值右侧） | §4.2/§6 |
| E09 | cast 形态错误（操作数非 Tile/Scalar、`dt` 非 DTypeRef、实参个数） | §4.2/4.3 |
| E10 | `arange` 边界依赖运行时值（非常量表达式） | R2 / 折叠器 |
| E11 | 不在 Python 子集内（控制流、下标、from-import 等） | 转换期 |
| E12 | 非法注解与名字冲突（dtype 名、bool 作 Buffer 元素、rank ≥ 2（v0.1）、符号维名冲突、保留名遮蔽） | 转换期 / §5 |
| E13 | 内建调用形态错误（program_id 轴非常量、未知 `tila.` 属性、未知 kwarg、实参数量、无 mask 的 other、DTypeRef 位置非法） | 转换期 / §4.2 |
| E14 | 重赋值 / 参数遮蔽（含符号维） | §6 |
| E15 | constexpr 缺省值非法或调用点未提供 | §5 |
| E16 | op ∉ dtype 类别能力集（FP8 一切算术/比较、bool 算术、整数 `/`、整数 `& \|` 等） | 能力门（§4.1） |
| E17 | launch plan 不可推导（grid 轴/tiling constexpr/符号维对应不唯一） | §5 |
| E18 | dot 前提违反（dtype 非 f16/bf16、静态维 < 16、rank ≠ 2、收缩维不一致） | R16（v0.2 fragment） |
| E19 | 寻址形态错误（load/store 第 2 实参非坐标元组、元组长度 ≠ rank、标量坐标）。subcode：CoordinateForm / CoordinateArity / CoordinateKind / CoordinatePosition（渲染不变，`TilaError.subcode`） | R8/R9 寻址前提（v0.3；旧 `a + offs` 算术 → E07，旧 load/store 实参形态 → E13；注解侧 strides 错误归 E12） |
| E20 | 循环形态与围栏（iterable 非 tila.range、range 三件套形态、tila.range 越位、嵌套 for、zeros 形态、`+=` 纪律、体内读累加器、循环局部出循环）。subcode：NotRange / RangeForm / RangePosition / NestedLoop / ZerosForm / AccumForm / AccumRead / LoopScope | R17/R18/R19（v0.4，docs/v0.4-kloop.md §7）；dtype/shape/layout 链仍走 E02/E03/E05、单赋值冲突仍走 E14——E20 不做垃圾桶 |
| E21 | 归约前提违反（rank ≠ 2、边缘化未定义）。subcode：ReduceRank（rank-1 全归约/rank≥3 推迟）/ ReduceLayout（Mma 已由 v0.6a 分支六覆盖——`Slice(Mma, k)`；该 subcode 对合法输入成为空位，保留防未来原子 term 回归） | R20（v0.5，docs/v0.5-reduce.md §7；分支六见 docs/v0.6-attention.md §2.1）；axis 形态 → E13、dtype 能力 → E16、where 的 shape → E03——E21 不做垃圾桶 |

E01–E10、E16、E17 由 checker 抛出；E11–E15 由转换/签名阶段抛出（同一 TilaError 类型）。

**编号修订说明（相对 2026-08 前的初稿）**：为对齐评审意见的显式编号——重赋值从 E08 移至 **E14**（评审 §29 原文示例）、shape 不匹配从 E01 移至 **E03**（评审广播规则原文）、layout 不等价从 E03 移至 E05、未绑定从 E09 移至 E01、store 语境从 E10 移至 E08、arange 两码合流为 E06/E10、旧 E14（保留名误用）并入 E12/E13。旧编号 → 新编号对照：

| 初稿 | E01 | E02 | E03 | E04 | E05 | E06 | E07 | E08 | E09 | E10 | E11 | E12 | E13 | E14 | E15 | E16 | E17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 新稿 | E03 | E02 | E05 | E04 | E06 | E10 | E07 | E14 | E01 | E08 | E11 | E12 | E13 | E12/E13 | E15 | E16 | E17 |

**E07 的边界刻意收窄**：只表示"类型类别组合本身无定义"（Buffer/Address 误用、标量间比较等）；不属于类别问题的错误一律归 E02/E03/E05/E16（dtype/shape/layout/能力）——E07 不做兜底垃圾桶。

## 8. add.tila 检查走查

对 `examples/add.tila`（BLOCK=128 特化）执行 checker 的完整产出：

```
func @add(a: Buffer<f32, (N,)>, b: Buffer<f32, (N,)>, c: Buffer<f32, (N,)>, N: Scalar(i32), BLOCK: Constexpr(i32)=128)
L0 = identity(128)

%pid = program_id 0 : Scalar(i32) [#pid]
%BLOCK = constexpr_ref BLOCK : Scalar(i32)
%t0 = mul %pid %BLOCK : Scalar(i32)
%r0 = arange 0 BLOCK : Tile<i32, (128,), L0>
%offs = add %t0 %r0 : Tile<i32, (128,), L0> [#offs]
%N = sym_ref N : Scalar(i32)
%mask = lt %offs %N : Tile<bool, (128,), L0> [#mask]
%p0 = addptr %a %offs : Address<f32, (128,), L0>
%x = load %p0 mask=%mask : Tile<f32, (128,), L0> [#x]
%p1 = addptr %b %offs : Address<f32, (128,), L0>
%y = load %p1 mask=%mask : Tile<f32, (128,), L0> [#y]
%z = add %x %y : Tile<f32, (128,), L0> [#z]
%p2 = addptr %c %offs : Address<f32, (128,), L0>
store %p2 %z mask=%mask
return
```

本块即黄金比对物 `tests/golden/add.tir.txt` 的内容（`%BLOCK`/`%N` 按 `ast.md` §5 的合成引用命名规则取参数名原样、无 `[#…]` 尾注；dump 为单空格分隔 token，无列对齐；文档展示与生成器输出逐字节一致）。`arange 0 BLOCK`：边界按 ConstExpr 渲染参数名（**发射口径**）；类型里的 `(128,)` 仍取 build 期特化值（**检查口径**）。

checker 期间的关键判定记录（写入编译日志，`--explain` 打印；**非规范、仅供诊断**——黄金比对只认上方 TIR dump 与生成的 Triton 源码）：

```
line 11  z = x + y
    dtype : f32 == f32                            ok
    shape : (128,) ⊗ (128,) = (128,)              ok
    layout: load(M_rm, identity(128)) ~ load(M_rm, identity(128))
            normalize ⇒ identity(128) == identity(128)    ok          （律 L1 两次生效）
```

`Γ` 快照随语句增长：`pid → Scalar(i32)@10`、`offs → Tile<i32,(128,),L0>@10`、`mask/x/y/z …`。

## 9. 测试策略

1. **黄金 TIR**：`add.tila` → 与 `tests/golden/add.tir.txt` 逐字节比对（dump 格式见 `ast.md` §5）。
2. **诊断矩阵**：E01–E17 每码一个最小复现程序，断言 (code, loc)；再加"错误信息包含两个操作数类型与定义行号"的断言。
3. **单元**：`normalize/equiv` 直接构造 term 测律 L1–L4（E05 在 v0.1 表面语言中不可触发——1D 单种子使坏 layout 无从构造——必须走单元测试）。
4. **dtype 能力与 cast 矩阵**：FP8 算术/比较 → E16 且消息含建议；`u32_t + i32_t` → E02；`f16_t + 1.0` 通过、`f16_t + 1` → E02；cast 链合法程序（`fp8e4m3fn → f16 → u8 → bool`）产出正确 TIR；`other=0`（u32）/`other=0.0`（f16）通过、类别错配 → E02；重赋值（含 `N = 5`）→ E14。
5. **健壮性**：checker 对任意非法输入只抛 TilaError（fuzz 转换器产物 + 断言无原生异常）。
6. **差分（Stage 6）**：reference interpreter 与 Triton/GPU 结果互相对照（`development-plan.md` §8）。

## 10. 实现清单（函数级）

```
src/tila/types/dtype.py        dtype 域 + 能力表 + ARITH_RULES 数据源                （§3/§4.1）
src/tila/types/shape.py        Dim/Shape/broadcast(⊗)                                   （type-system §2）
src/tila/types/dist.py         DistExpr 五 term + Seed/NoDist + normalize_dist/equiv_dist（§3）
src/tila/types/memory.py       MemoryLayout（RowMajor/Strided + strides_of）        （type-system §3.1）
src/tila/types/join.py         纯结构谓词（axis_segments / is_proper_broadcast_projection） （type-system §3.5）
src/tila/types/type.py         ScalarType/BufferType/AddressType/TileType/UnitType      （type-system §6）
src/tila/frontend/desugar.py   pyast → tila_ast 转换 + 子集校验 + intrinsic resolution（E11–E15，ast.md §3）
src/tila/checker/checker.py    check_kernel / Env / infer / RULES 表                    （§2–§6）
src/tila/checker/arithmetic.py 运算分派矩阵（ARITH_RULES，唯一权威）                    （§4.1）
src/tila/checker/builtin.py    check_program_id/arange/load/store/cast                （§4.2）
src/tila/checker/broadcast.py  ⊗ 与 shape 规则                                          （§3）
src/tila/checker/layout.py     join/marginal 规则（strict_join/read_join/join_distributions/
marginal/grow_to_shape/infer_coordinate_dist + E05/E21）          （§3.5/§4）
src/tila/diagnostics.py        TilaError / Note / 渲染（E 码）                          （§7）
src/tila/tir/ops.py            节点 + printer（canonical dump）                         （ast.md §4–§5）
```

函数签名级 TODO（按实现顺序）：`build_env → check_signature（含名字冲突检查）→ infer_expr → check_binop → check_load/check_store/check_cast → emit ops → plan_launch（launch analysis phase，E17）→ dump`。