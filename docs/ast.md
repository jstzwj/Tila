# Tila AST 与 typed IR

状态：定稿（2026-08 评审修订）。本文定义编译器的三级中间表示及其转换：**Python AST → tila_ast（不带类型）→ TIR（带类型）**，以及 **intrinsic resolution**（`tila.load` → `IntrinsicRef("load")`）。节点定义精确到可直接抄成 `src/tila/ast/nodes.py` 与 `src/tila/tir/ops.py`。

---

## 1. 三级表示与职责划分

| 层 | 产物 | 职责 | 不做的事 |
|---|---|---|---|
| Python `ast` | 语法树 | 解析（`ast.parse`） | — |
| `tila_ast` | 表面子集的干净树 | **语法层**子集校验、脱糖（`tila.cast(x, tila.f32)` → `Cast`）、intrinsic resolution（`tila.load` → `Call(intrinsic="load")`） | 任何类型/语义判断 |
| `tir` | 带类型的直线 SSA | 携带推导出的类型与 layout term | — |

checker 是唯一的 `tila_ast → TIR` 通道；lowering 只消费 TIR。所有报错发生在前两级（E11–E15 转换/签名期）或 checker（E01–E10、E16、E17），lowering 对有效 TIR 全函数（total）。

**通用约定**：所有节点 `frozen dataclass`，均携带 `loc: Loc`；`tila_ast` 不含任何 Python 特有节点。

```python
@dataclass(frozen=True)
class Loc:
    line: int
    col: int
```

---

## 2. tila_ast 节点定义

```python
# src/tila/ast/nodes.py

# ---------- 类型注解 ----------
@dataclass(frozen=True)
class Ann(Node): ...                       # 基类，含 loc

@dataclass(frozen=True)
class TensorAnn(Ann):
    dtype: str                             # "f32" ...（tila.float32 解析后）
    dims: tuple[DimSpec, ...]              # rank ≥ 1

@dataclass(frozen=True)
class ConstexprAnn(Ann): ...

DimSpec = Union["StaticDim", "SymDim"]

@dataclass(frozen=True)
class StaticDim(Node):
    value: int

@dataclass(frozen=True)
class SymDim(Node):
    name: str                              # 自动绑定为运行时符号参数

# ---------- 顶层与语句 ----------
@dataclass(frozen=True)
class KernelDef(Node):
    name: str
    params: tuple[Param, ...]
    body: tuple[Stmt, ...]                 # ≥ 1 条

@dataclass(frozen=True)
class Param(Node):
    name: str
    ann: Optional[Ann]                     # None → Scalar(i32)（language-spec.md §4）
    default: Optional[int]                 # 仅 ConstexprAnn 允许非 None

Stmt = Union["Assign", "ExprStmt"]

@dataclass(frozen=True)
class Assign(Node):
    target: str                            # 单一名字（文法保证）
    value: Expr

@dataclass(frozen=True)
class ExprStmt(Node):
    call: Call                             # checker 强制 intrinsic == "store"（E08）

# ---------- 表达式 ----------
Expr = Union["IntLit", "FloatLit", "NameRef", "BinOp", "Call", "Cast", "DTypeRef"]

@dataclass(frozen=True)
class IntLit(Node):
    value: int                             # 可为负：-1 由转换期脱糖为负字面量

@dataclass(frozen=True)
class FloatLit(Node):
    value: float                           # 缺省 f32；类别匹配的上下文按该 dtype 解释（R12）

@dataclass(frozen=True)
class NameRef(Node):
    name: str

@dataclass(frozen=True)
class BinOp(Node):
    op: str                                # + - * / < <= > >= == != & |
    lhs: Expr
    rhs: Expr

@dataclass(frozen=True)
class DTypeRef(Node):
    name: str                              # "tila.float32" 解析后 → "f32"；只允许出现在
                                           # Cast 的第二实参和 TensorAnn 内（否则 E13）

@dataclass(frozen=True)
class Call(Node):
    intrinsic: str                         # INTRINSICS 表解析后的名字：
                                           # program_id | arange | load | store | cast
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...]   # (名字, 实参) 对，键 ∈ {mask, other}；tuple 保持可哈希

@dataclass(frozen=True)
class Cast(Node):
    dtype: str                             # tila.cast(x, tila.float32) 脱糖而来
    operand: Expr
```

---

## 3. pyast → tila_ast 转换规则

逐节点对照（接受项做机械映射；**拒绝项**抛 E11/E12/E13/E14，附"Python 语法 X 不属于 Tila v0.1 子集"）。**intrinsic resolution 在此阶段完成**：`tila.<name>` 属性访问被识别为 `IntrinsicRef`，查 `INTRINSICS` 静态表（`language-spec.md` §2）转成 `Call.intrinsic`；`tila.<dtype>` 转成 `DTypeRef`；`tila.Tensor[...]` 转成 `TensorAnn`；`tila.jit` / `tila.constexpr` 是装饰器与注解名。

| Python ast 节点 | tila_ast | 备注 |
|---|---|---|
| `Module(body=[Import, FunctionDef])` | `KernelDef` | import 必须恰好一个且是 `import tila`；`from tila import …` / `import tila as tl` / 额外语句 → E11 |
| `FunctionDef(decorator_list=[Attribute(tila, jit)])` | `KernelDef` | decorator 数 ≠ 1 或形式不符 → E11 |
| `arguments.args`（含 annotation） | `Param` 列表 | 出现 vararg/kwarg/kwonlyargs/非 constexpr 的默认值 → E11 |
| 注解 `Subscript(Attribute(tila, Tensor), Tuple(dtype_ref, dims…))` | `TensorAnn` | dtype_ref 非法 → E12；`tila.Tensor` 出现在别处 → E14；无注解 → 不产生 Ann（→ Scalar(i32)） |
| 注解 `Attribute(tila, constexpr)` | `ConstexprAnn` | 默认值非 `Constant(int)` → E15 |
| `Assign(targets=[Name], value=…)` | `Assign` | 多目标/非 Name 目标 → E11 |
| `Expr(Call)` | `ExprStmt` | `Expr` 的 value 非 Call → E11（E08 在 checker 期） |
| `BinOp(Add/Sub/Mult/Div)` | `BinOp` | 其余 BinOp（`// % **`、移位）→ E11 |
| `BitAnd/BitOr` | `BinOp` | `& \|` 合法；`^ ~ << >>` → E11 |
| `Compare`（单 op） | `BinOp` | 链式比较（多个 ops）→ E11 |
| `Call(func=Attribute(tila, name))` | `Call` 或 `Cast` | name ∈ INTRINSICS → `Call(intrinsic=name)`；name ∈ dtype 名且是 `cast` 的 `DTypeRef` 位置 → `DTypeRef`；`tila.cast(x, tila.<dt>)` 整体 → `Cast`；表外 name → E13；`func=Name` 或 `func=Attribute(非 tila)` → E11/E14（无用户函数） |
| `Name` | `NameRef` | 保留名（`tila`/`tl`/`triton`）出现在表达式位置 → E12 |
| `Constant(int)` / `Constant(float)` | `IntLit` / `FloatLit` | `str/bool/None/…` → E11 |
| keyword 实参 | `kwargs` | 键 ∉ {mask, other} → E13；positional-after-kwarg 由 Python 文法保证 |
| `UnaryOp(USub, Constant(int/float))` | 负字面量（`IntLit(-n)` / `FloatLit(-n)`） | 脱糖规则；其余一元用法仍拒绝 |
| 其余一切（`For/While/If/Return/Lambda/ comprehension/Subscript/Attribute 非 tila/其他 UnaryOp/BoolOp/…`） | — | E11 |

转换期**不做**任何类型判断（例如 `arange` 参数个数留给 checker），职责单一便于测试。

---

## 4. TIR：带类型的直线 SSA

TIR 是 kernel 体的扁平指令序列：每个值一个 `%id`，先定义后使用，单赋值。每条指令携带推导出的完整类型（含 layout term）与来源信息。

```python
# src/tila/tir/ops.py

@dataclass(frozen=True)
class TOp(Node):                           # 所有指令的基类
    id: str                                # "%pid"、"%t3"
    tila_type: Type                        # ScalarType / TileType / AddressType / UnitType
    src_name: Optional[str]                # 源码赋值目标名；匿名指令为 None
    origin_layout: Optional[LayoutTerm]    # 规范化前的原始 term（仅诊断用）

# ---- v0.1 指令 ----
@dataclass(frozen=True)
class TConstInt(TOp):      value: int          # opcode: const
@dataclass(frozen=True)
class TConstFloat(TOp):    value: float        # opcode: const
@dataclass(frozen=True)
class TSymRef(TOp):        name: str           # opcode: sym_ref     N → Scalar(i32)
@dataclass(frozen=True)
class TConstParamRef(TOp): name: str           # opcode: constexpr_ref  BLOCK：检查用值、发射用名
@dataclass(frozen=True)
class TProgramId(TOp):     axis: int           # opcode: program_id
ConstExpr = Union[int, str, "CBinOp"]          # 编译期常量表达式：字面量 | constexpr 名 | 整数运算

@dataclass(frozen=True)
class CBinOp:
    op: str; lhs: ConstExpr; rhs: ConstExpr

@dataclass(frozen=True)
class TArange(TOp):        start: int; end: ConstExpr   # opcode: arange
                                                       # start 恒 0；end 按表达式渲染发射（模型 B），
                                                       # typing 用 build 期特化值求值
@dataclass(frozen=True)
class TAddPtr(TOp):        base: str; offs: str        # opcode: addptr   R7：Buffer + Tile[i32] → Address
@dataclass(frozen=True)
class TArith(TOp):         op: str; lhs: str; rhs: str # opcode: add/sub/mul/div
                                                       # R3（Scalar⊕Scalar）/R4（Scalar⊕Tile）/R5（Tile⊕Tile）
                                                       # 操作数角色由值类型给出，opcode 不区分
@dataclass(frozen=True)
class TCmp(TOp):           op: str; lhs: str; rhs: str # opcode: lt/le/gt/ge/eq/ne   R6 → Tile[bool]
@dataclass(frozen=True)
class TLogic(TOp):         op: str; lhs: str; rhs: str # opcode: and/or   R10（Tile[bool]）
@dataclass(frozen=True)
class TLoad(TOp):          ptr: str; mask: Optional[str]; other: Optional[str]   # opcode: load
@dataclass(frozen=True)
class TStore(TOp):         ptr: str; value: str; mask: Optional[str]  # opcode: store（语句级，无值 id）
@dataclass(frozen=True)
class TCast(TOp):          dtype: str; operand: str    # opcode: cast   R11
@dataclass(frozen=True)
class TReturn(TOp):        ...                        # opcode: return（语句级）

# ---- v0.2 预览新增（docs/v0.2-preview-2d.md）----
# TExpandDim(TOp): tile: str; axis: int        # opcode: expand_dim   R12

@dataclass(frozen=True)
class TParam:
    name: str
    kind: str                              # "buffer" | "sym" | "constexpr" | "scalar"
    tila_type: Type
    default: Optional[int]                 # constexpr 用

@dataclass(frozen=True)
class LaunchPlan:                          # launch analysis phase 推导（typing 之后的独立 pass，
                                           # type-checker.md §1），lowering 只发射
    axes: tuple[GridAxis, ...]             # 每轴：维（符号维名 | 静态值）、tiling constexpr 名
    rank_asserts: tuple[tuple[str, int], ...]        # (buffer 名, 注解 rank)
    static_dim_asserts: tuple[tuple[str, int, int], ...]   # (buffer 名, 维下标, 静态值)
    sym_dim_asserts: tuple[tuple[str, int, str], ...]      # (buffer 名, 维下标, 符号维名)

@dataclass(frozen=True)
class TKernel:
    name: str
    params: tuple[TParam, ...]             # lowering 生成签名的唯一依据
    ops: tuple[TOp, ...]
    launch_plan: LaunchPlan                # grid 与 shape 断言的依据（E17 在 launch analysis 消化）
```

指令与 typing rules 的对应：`TProgramId`←R1、`TArange`←R2、`TArith`←R3/R4/R5、`TCmp`←R6、`TAddPtr`←R7、`TLoad`←R8、`TStore`←R9、`TLogic`←R10、`TCast`←R11、`TSymRef`/`TConstParamRef`/`TConstInt`/`TConstFloat`←叶子。

---

## 5. TIR 文本打印格式（canonical dump）

`tir.dump(kernel)` 的输出是黄金测试的比对物，格式必须确定。规则：

- **token 间以单个空格分隔，不做列对齐**（生成器以 `" ".join` 拼接）；赋值用 `%id = op …`。
- `%id`：有 `src_name` 的指令用 `%<src_name>`；匿名指令按出现顺序 `%t0, %t1, …`；**合成引用指令**（`TSymRef`/`TConstParamRef`）以被引用参数名原样作 id（`%N`、`%BLOCK`），且**不带** `[#src_name]` 尾注。
- 指令形如 `%id = opcode operand* : type [#src_name]`；语句级指令（`store`、`return`）不产生值，无 id、无类型。
- kwarg 形如 `mask=%mask`。
- **layout 打印规则**：layout 打印其**正规形式**；相同正规形式按首次出现顺序命名 `L0, L1, …`，并在 dump 首部列印 `L0 = identity(128)` 形式的定义行（shape 打印为 `identity((128,))`；v0.2 的 `product(L0,L1)` 同理）。
- 类型打印：`Scalar(i32)`、`Tile<i32, (128,), L0>`、`Tile<bool, (128,), L0>`、`Address<f32, (128,), L0>`、`()`；符号维打印名字（`(N,)`）；dtype 打印短名（`f32`、`fp8e4m3fn`）。
- 头部：`func @<name>(<param>: <type>, …)`，constexpr 参数带 `= <默认值>`（`BLOCK: Constexpr(i32)=128`）。
- dump **只含上述格式本身**：中文旁注、行号说明等不属于 dump 内容。opcode 拼写以 `type-checker.md` §8 的走查 dump 为参考实现。
- ConstExpr 操作数按表达式原样渲染（`128`、`BLOCK`、`BLOCK // 2`）；constexpr 参数名**不带** `%` 前缀——区别于操作数 id 引用（`%BLOCK`）。

`add.tila`（BLOCK=128 特化）的完整 dump：

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

本块即黄金比对物 `tests/golden/add.tir.txt` 的内容。

---

## 6. TIR 不变量（checker 保证，lowering 依赖）

1. 每个 `%id` 唯一，且先定义后使用（直线序）。
2. 每条指令操作数的类型满足其对应规则的全部前提（dtype、shape、layout 等价已检查）。
3. **匿名值在其唯一使用点发射**——v0.1 中表达式树展平且 checker 不做 CSE，匿名值恰被使用一次。但这是 **lowering 的发射策略依据，不是 TIR 必须永久维持的结构性质**：将来的 CSE/常量折叠/copy propagation 会产生"被多次使用的匿名值"或"单次使用的具名值"，届时发射器改按 use_count 决策（==1 内联、>1 物化为变量，`triton-lowering.md` §4），IR 本身不设此约束。
4. 所有 `Tile/Address` 类型中的 layout term 均可 normalize（无自由变量）。
5. `TLoad/TStore` 的 `ptr` 字段引用 `TAddPtr` 产生的 Address 值；`TAddPtr.base` 引用 kernel 的 buffer 参数名。
6. 每个值都有完全 resolve 的 dtype、shape、layout——**TIR 不做类型推断**（类型在 checker 已全部消耗；这是 lowering 能 total 的前提）。

---

## 7. 设计动机备注

- **为什么是 intrinsic resolution 而不是内建关键字节点**：`tila.load(...)` 是 Python 子集里的普通属性调用；识别为 intrinsic 后 checker 按合同检查，Tila 表面语言保持"Python 子集 + tila 固有命名空间"，不发明第二套语法。解析用静态表（`INTRINSICS`），表外属性直接 E13。
- **为什么不用 pyast 直接检查**：pyast 节点语义过宽（链式比较、多赋值目标、上下文表达式），子集校验后换成窄节点，checker 的分派永远不遇"不可能的形状"。
- **为什么 TIR 是扁平 SSA 而非类型化表达式树**：① 类型/layout 挂在**值**上而不是树上，诊断可以直接引用值的定义点；② 与 `type-system.md` §8 的文本 IR 一一对应；③ lowering 内联匿名指令后自然回到源码形态（`triton-lowering.md` §4）。
- **为什么 `origin_layout` 留在指令上**：normalize 后的信息足以判等，但不足以解释（"这个 layout 来自第 6 行那次 load"）；诊断需要原始 term。
- **为什么 `TArange` 保留 ConstExpr 而不折叠**：检查按 override 值特化（TIR 是特化产物、跨值不复用，`semantic-model.md` §6）；发射保留名字则维持与源码的 1:1 同构和稳定的黄金输出，生成的 kernel 形参仍是 `BLOCK: tl.constexpr`，Triton 对每个 launch 值做自己的 jit 特化。折叠成字面量在"逐值重编译"模型下虽不再产生覆盖空洞，但产物不再与源码同构，黄金输出也随 override 漂移——故发射口径恒保留名字（模型 B），检查口径恒用特化值。