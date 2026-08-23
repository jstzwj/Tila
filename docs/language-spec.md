# Tila 语言规范（v0.1）

状态：定稿（2026-08 评审修订）。本文定义 v0.1 的**全部**表面语言：词法、文法、内建 intrinsic、名字规则、符号维与 constexpr、启动语法、允许/禁止清单。语义模型（值类别、执行模型、内建语义合同）见 `semantic-model.md`，类型规则见 `type-system.md`，AST 与 IR 见 `ast.md`。

```
Tila source ──► tila_ast ──► type check ──► TIR ──► Triton source
```

---

## 1. 定位

表面语言 = **Python 3 的一个可判定子集 + `tila` 固有命名空间**。

- v0.1 只做**直线型（straight-line）一维 kernel**：无循环、无分支、无自定义函数。一个合法 v0.1 程序是恰好一个 `@tila.jit` 函数（+ 必需的 `import tila`），函数体由赋值语句和 `tila.store` 调用组成。向量加法（`examples/add.tila`）是规范内最大程序。
- 这不是永久限制，而是刻意的验收边界：v0.1 的全部价值在于把 类型系统 + layout 代数 + Triton lowering 这条链路走通。
- 源文件扩展名 `.tila`，内容必须是语法合法的 Python；解析直接使用标准库 `ast.parse`（`SyntaxError` 包装成 E11），随后子集校验并转换为 tila_ast（转换表见 `ast.md` §3）。

## 2. API 设计原则：内建是 intrinsic，不是 Python 函数

```
tila
  ├── jit                 kernel marker（装饰器）
  ├── constexpr           参数注解（编译期特化）
  ├── Tensor              参数注解（tila.Tensor[dt, dims]）
  │
  ├── program_id          执行上下文
  ├── arange              索引构造
  ├── load / store        内存
  ├── cast                dtype 转换
  │
  ├── float32 / int32 / … dtype 引用（17 种，见 §4）
  └── （v0.2+：num_programs、zeros、full、where、expand_dim、dot、abs/exp/log/sqrt、max/sum …）
```

- **`tila.<name>` 是唯一的调用形态**。v0.1 严格要求 `import tila`；**不支持** `from tila import load`、`from tila import *`、`import tila as tl`（别名是 API-level source compatibility 的未来目标，`triton-lowering.md` §10）。原因：`load(...)` 无法判定是不是 Tila intrinsic，而 `tila.load(...)` 一目了然。
- `tila.load` 等名字在 Python 层是 placeholder 对象（调用抛 `RuntimeError("… can only be used inside @tila.jit")`），**永远不会被执行**；frontend 把 `tila.load` 解析成 `IntrinsicRef("load")`（`ast.md` §3）。
- `@tila.jit` 只是 **kernel marker**：告诉编译器"这是一个 GPU kernel"，v0.1 不承担别的职责。
- intrinsic 的解析用**静态表**（`src/tila/intrinsic.py`）：

  ```python
  INTRINSICS = {
      "program_id": ProgramIdIntrinsic,
      "arange":     ArangeIntrinsic,
      "load":       LoadIntrinsic,
      "store":      StoreIntrinsic,
      "cast":       CastIntrinsic,
  }
  ```

  表外属性 → E13。

## 3. 词法元素

| 类别 | 内容 |
|---|---|
| 标识符 | `[A-Za-z_][A-Za-z0-9_]*`（同 Python） |
| 保留名 | `tila`（模块）、`triton`/`tl`（生成代码的模块名，防遮蔽）；dtype 名、内建名只在 `tila.` 后合法 |
| 整数字面量 | 十进制整数 `INT`；`-INT`/`-FLOAT` 为负字面量（转换期脱糖——负号是字面量前缀，不是一元运算符） |
| 浮点字面量 | `FLOAT`，如 `1.0`（缺省 f32；类别匹配的上下文按该位置的 dtype 解释，R12） |
| 运算符 | `+ - * / < <= > >= == != & \|`（无 `//`、`%`、`**`、移位、一元运算符） |
| 标点 | `( ) [ ] , = : @ .` |
| 注释 | `#` 至行尾 |

保留名不可作为 kernel 名、参数名或赋值目标（E12）——尤其 `tl`/`triton`：它们一旦进入上述任何位置，都会遮蔽生成代码里的模块导入，产物直接损坏。

## 4. 文法（EBNF）

```
module         ::= "import" "tila" NEWLINE kernel_def EOF
                                        # import 恰好一次、必须位于文件首（E11）
kernel_def     ::= "@" "tila" "." "jit" "def" IDENT "(" [params] ")" ":" NEWLINE
                   INDENT stmt+ DEDENT
params         ::= param ("," param)* [","]
param          ::= IDENT [":" ann] ["=" INT]
ann            ::= tensor_ann | "tila" "." "constexpr"
tensor_ann     ::= "tila" "." "Tensor" "[" dtype_ref ("," dim)* ["," strides] "]"
dtype_ref      ::= "tila" "." dtype_name
dtype_name     ::= "bool" | "i8" | "i16" | "i32" | "i64"
                 | "u8" | "u16" | "u32" | "u64"
                 | "f16" | "bf16" | "f32" | "f64"
                 | "fp8e4m3" | "fp8e5m2" | "fp8e4m3fn" | "fp8e4m3b15"
dim            ::= INT | IDENT                      # 静态维 | 符号维

stmt           ::= assign | aug_assign | for_stmt | expr_stmt
assign         ::= IDENT "=" expr NEWLINE
aug_assign     ::= IDENT "+=" expr NEWLINE          # v0.4：仅累加器；其它增强赋值 E20
for_stmt       ::= "for" IDENT "in" range_call ":" NEWLINE INDENT stmt+ DEDENT
                                                    # v0.4（docs/v0.4-kloop.md §1.1）：
                                                    # 只允许 kernel 体顶层（嵌套 → E20）；
                                                    # iterable 只能是 tila.range（E20）

expr           ::= or_expr
or_expr        ::= and_expr ("|" and_expr)*         # & | 仅 Tile[bool]（R10）
and_expr       ::= comparison ("&" comparison)*
comparison     ::= additive (comp_op additive)?     # 无链式比较
comp_op        ::= "<" | "<=" | ">" | ">=" | "==" | "!="
additive       ::= multiplicative (("+" | "-") multiplicative)*
multiplicative ::= atom (("*" | "/") atom)*
atom           ::= INT | FLOAT | IDENT | call | cast | "(" expr ")"

# v0.3（docs/v0.3-strides.md）：strides 尾随元组与坐标寻址
strides        ::= "(" stride ("," stride)* ")"     # 只允许作 tensor_ann 最后一项
stride         ::= INT | IDENT                      # INT ≥ 1；IDENT 按符号规则绑定
coord_tuple    ::= "(" expr ("," expr)+ ")"         # 只允许作 tila.load/tila.store
                                                # 的第 2 位置实参（其余位置的元组
                                                #  → E11；坐标须为 i32 tile，
                                                #  长度 = buffer rank，E19）
shape_tuple    ::= "(" static_dim ("," static_dim)* ")"  # v0.4：只允许作 tila.zeros 的
                                                # 第 1 位置实参；INT 字面量或
                                                # constexpr 名（静态量，E20）
range_call     ::= "tila" "." "range" "(" "0" "," expr "," const_expr ")"
                                                # v0.4：start 恒为字面量 0；end 为
                                                # 运行期 i32 标量；step 为 ConstExpr ≥ 1

call           ::= "tila" "." IDENT "(" [arg_list] ")"     # IDENT ∈ INTRINSICS（表外 E13）
arg_list       ::= expr ("," expr)* ["," kwarg ("," kwarg)*]
kwarg          ::= "mask" "=" expr | "other" "=" expr
cast           ::= "tila" "." "cast" "(" expr "," dtype_ref ")"
```

优先级（低→高）：`|` < `&` < 比较 < `+ -` < `* /` < 原子。与 Python 一致（因此 mask 合取需要括号：`(rows2 < M) & (cols2 < N)`）。

**注解规则**：

- **`tila.Tensor[dt, dims]`**：v0.1 的 tensor 参数**必须**注解（dtype 与 shape 是静态类型的信息锚点；无注解参数 → E12）。dims 中出现的 IDENT（如 `N`）自动绑定为符号维（§7）。rank ≥ 2 的注解（`tila.Tensor[tila.float32, M, N]`）在 v0.1 被 checker 直接拒绝（E12）——v0.1 里 2D 参数永远无法被 load/store（唯一索引种子是 1D arange），放行只是陷阱；文法形式为 v0.2 预览保留（`docs/v0.2-preview-2d.md`）。
- **`tila.constexpr`**：必须有整数字面量默认值（E15），调用点可覆盖（§7）。
- **无注解参数 → `Scalar(i32)`**（运行期标量；对齐 Triton 的 i32 推断惯例。v0.1 的标量值域只有 i32/f32）。

## 5. 内建 intrinsic（v0.1 全部）

v0.1 没有用户函数。全部可调用对象：

| 内建 | 形式 | 关键约束 | 结果类型 | 规则 | lowering |
|---|---|---|---|---|---|
| `program_id` | `tila.program_id(c)` | `c` 为字面量/constexpr 折叠值 ∈ {0}（v0.1；v0.2 预览扩至 {0,1}） | `Scalar(i32)` | R1 | `tl.program_id(c)` |
| `arange` | `tila.arange(0, c₂)` | `c₁ = 0`（字面量）；`c₂` 为编译期常量表达式且特化值 = 2^k（1 ≤ k ≤ 20） | `Tile[i32, (c₂ᵛ,), …]`（取 build 期特化值） | R2 | `tl.arange(0, c₂)`（保留名字） |
| `load` | `tila.load(buf, coords, mask=?, other=?)` | R8（§合同见语义模型 §5.3）：`buf` 为 Buffer、`coords` 为坐标元组（每轴一个 i32 tile，长度 = rank；v0.3，`docs/v0.3-strides.md` §1.2）、mask/layout 等价、`other` 为类别匹配字面量且必须与 mask 同时给出 | `Tile[dt, Σ, L]` | R8 | `tl.load(<buf + 线性化>, mask=…, other=…)` |
| `store` | `tila.store(buf, coords, value, mask=?)` | R9：dtype **严格相等**、shape 相等、layout 等价（R9' 放宽） | `()` | R9 | `tl.store(<buf + 线性化>, <v>, mask=…)` |
| `cast` | `tila.cast(x, dt)` | R11：x 为 Tile 或 Scalar，dt 为 dtype_ref（17 种任意） | `Tile[dt', Σ, L]` / `Scalar[dt']` | R11 | `<x>.to(tl.<dt>)` |
| `expand_dim` | `tila.expand_dim(t, axis)` | R13（v0.2 预览）：axis 编号结果张量轴 | `Tile[dt, Σ+1, L]` | R13 | `tl.expand_dims(<t>, axis)` |
| `dot` | `tila.dot(x, y)` | R16（v0.2 fragment）：rank-2、收缩维相等、dtype ∈ {f16,bf16}、各维 ≥ 16 | `Tile[f32, (M,N), Mma]` | R16 | `tl.dot(<x>, <y>)` |
| `zeros` | `tila.zeros(shape, dt)` | R17（v0.4）：shape 为静态元组（INT/constexpr 名，rank ≤ 2）、dt ≠ bool | `Tile[dt, Σ, Zeros(Σ)]` | R17 | `tl.zeros(<shape>, dtype=tl.<dt>)` |
| `range` | `for k in tila.range(0, e, s):` | R18（v0.4）：仅 for 的 iterable 位置（其它位置 E20）；start 恒 0、e 为 i32 标量、s 为 ConstExpr ≥ 1 | 绑定 `k : Scalar(i32)` | R18 | `for k in range(0, <e>, <s>):` |

运算符（作用于 Tile/Scalar 的二进制运算）：`+ - * / < <= > >= == != & |`。dtype 能力表见 `type-system.md` §1.1：bool 只能 `== != & |`；整数 `+ - *` 与全部比较；浮点另加 `/`；FP8 一切算术与比较 → E16。

`load` 的 kwargs：

- `mask`：省略时不做边界检查，越界行为未定义（与 Triton 一致）；kernel 作者负责用 `offs < N` 类的 mask 表达边界安全。
- `other`：省略时 masked-out 通道未定义；给出时必须是**与元素 dtype 类别匹配的字面量**（整型 dt 配 `INT`、浮点 dt 配 `FLOAT`，如 `other=0` / `other=0.0`），跨类别 → E02；**必须与 `mask` 同时给出**——无 mask 时不存在 masked-out 通道 → E13。

**指针类型与地址算术均不在表面语言中**（v0.3 定稿）：坐标只出现在
`tila.load(buf, coords)` / `tila.store(buf, coords, value)` 的实参位置；`Address`
是 checker/TIR 内部类型（在 load/store 检查中构造），用户看不到、不可命名。
旧形式 `a + offs` / `tila.load(a + offs)` 已移除（E07/E13，消息附迁移提示）。

## 6. 名字、作用域与单赋值

- 作用域只有一层：kernel 参数 + 函数体局部名。没有全局、没有闭包、没有嵌套作用域。
- **单赋值**：每个名字（含参数；**符号维是自动绑定的参数**，`N = 5` 同样算遮蔽）最多被赋值一次；重赋值、参数遮蔽均为 **E14**。要求 `x0 = load(...); x1 = x0 + y` 这类新名字——单赋值是 TIR 保持直线 SSA 的表面保证。
- 名字必须先定义后使用；未绑定名字 → **E01**。
- `tila.Tensor` 只出现在参数注解位置；函数体内流动的只有 `Scalar / Tile`（数据流 `Buffer ─load─► Tile ─compute─► Tile ─store─► Buffer`）。

## 7. 符号维度与 constexpr

**符号维（sym dim）自动绑定**：`tila.Tensor[dt, N]` 注解中出现的 IDENT（如 `N`）自动成为一个运行时 `Scalar(i32)` 参数：

- 收集顺序 = 在签名中首次出现的顺序（决定生成 Triton 签名中的参数顺序，`triton-lowering.md` §3）。**strides 元组里的 IDENT 同款规则（v0.3）**：已绑定的符号（含维符号，如 `Tensor[f32, K, N, (N, 1)]` 显式行主序）→ 引用，新名 → 追加入序（launcher 从 `tensor.stride(i)` 取值）。
- **符号维名不得与任何参数名、constexpr 名相同，也不得是保留名**（→ E12）——否则生成签名出现重复形参或保留名遮蔽。
- 同一符号可出现在多个参数注解里（`a: tila.Tensor[tila.float32, N], b: …`）；这是**运行时契约**：launcher 校验这些张量的对应维长度相等（`semantic-model.md` §7）。stride 符号跨张量共享同理（`b.stride(i)` 相等）。
- 函数体内 `N` 可当普通 `Scalar(i32)` 使用（如 `offs < N`）。
- **维符号的双重视图（v0.4 起正式规定）**：一个维符号有两个视角——
  type-level 是 **ShapeSymbol**（注解 `Tensor[f16, M, K]` 里声明形状结构），
  value-level 在 kernel 体内引用时**运行期物化**为 `Scalar(i32)`（其值 =
  launcher 从首个绑定 buffer 的 `shape[i]` 读取的运行期维长）。`K` 既能写进
  注解、又能作 `tila.range(0, K, BK)` 的 end 与 `rka < K` 的比较右部，依据是
  同一条 elaboration 规则：`DimSymbol → RuntimeDimValue → Scalar(i32)`。
  两个视角在 checker 的 Γ 中是同一绑定（type-system.md §4）。

**constexpr**：

- 注解为 `tila.constexpr` 的参数必须有整数字面量默认值（E15），调用点可覆盖；**每个覆盖组合一次完整编译**（check → TIR → lowering，TIR 是特化产物、跨值不复用；对应 Triton 的 `BLOCK: tl.constexpr`，Triton 对每个 launch 值自行 jit 特化）。模型细节见 `semantic-model.md` §6。
- constexpr 支持整数运算：**检查期**求值（`BLOCK // 2` 是合法 arange 边界；求值失败 → E10）；**发射期**保留名字与表达式、不折叠（模型 B）。
- constexpr 在需要 `Scalar(i32)` 的位置自动提升（R3）；整数字面量同理。

## 8. 语义模型（摘要）

完整语义见 `semantic-model.md`。要点：

- 值类别：Constexpr / Scalar / Tile / Buffer / Address / unit（§1）；执行模型 = Triton program model：program_id → tile → load → compute → store（§2）。
- 翻译语义：kernel 含义 ≡ 其 lowering 出的 Triton kernel 的含义（§7）；lowering total、确定、无新语义拒绝。
- 边界安全由 mask 表达：masked load/store 按掩码跳过通道（§3）。

## 9. 启动语法与 launcher

kernel 内不写 grid。启动有两种形态（等价，均为 v0.1 支持的目标；实现先由 launcher 承担）：

```python
# 形态 1：生成的 launcher（v0.1 的默认机制；断言 + grid 来自 checker 的 launch_plan）
add_launch(a, b, c, BLOCK=128)

# 形态 2：Triton 式启动表达式（frontend 识别 Subscript，构造 LaunchExpr）
grid = (triton.cdiv(N, BLOCK),)
add[grid](a, b, c, BLOCK=128)
```

launcher 模板与 grid 推导规则见 `triton-lowering.md` §7；launch analysis（独立 phase，失败 → E17）见 `type-checker.md` §3。

## 10. v0.1 允许 / 禁止清单

**允许**（完整清单）：1 个必需的 `import tila`；1 个 `@tila.jit` kernel；`tila.Tensor[...]`（含 v0.3 尾随 strides 元组）/ `tila.constexpr` 参数；赋值；`tila.store` 表达式语句；二元算术（`+ - * /`）、比较、`& |`；`tila.cast`；内建；int/float 字面量；`mask`/`other` kwarg；括号；**坐标元组（仅 `tila.load`/`tila.store` 第 2 位置实参，v0.3）**；**`for k in tila.range(0, e, step):` 循环与 `acc += tile` 受限累加、`tila.zeros` 播种（v0.4，`docs/v0.4-kloop.md`）**。

**禁止**（转换期直接拒绝，E11/E12/E13/E14）：`from tila import …` / `import tila as tl` / 多 import / 模块级其他语句；`return`、`if`、条件表达式；**循环的边界形态（v0.4）**：裸 `range(...)`/其它 iterable → E20 NotRange，嵌套 for、`while`、`break`/`continue`、`for…else` → E20/E11，`tila.range` 出现在非 iterable 位置 → E20，非 `+=` 增广赋值与体外 `+=` → E20，非 zeros 播种名作 `+=` 左部 → E20，体内读累加器 → E20，循环局部（含循环变量）出循环引用 → E20；用户函数与 lambda；元组/列表/字典/切片/下标 `a[i]`（**坐标元组的唯一豁免：load/store 的坐标实参位**）；属性访问（除 `tila.<name>` 形态）；一元运算符（负字面量除外）；链式比较；布尔 `and/or/not`；`// % **` 与移位；多目标赋值 `a = b = x`；字符串与 bool 字面量；f-string；class；`tila.<name>` 出现在非 call 位置或表外名字；dtype/内建名出现在 `tila.` 之外；非 store 的表达式语句；`store` 出现在表达式位置（含赋值右侧——它只能作为表达式语句，E08）；**地址算术 `buffer + …`（v0.3 定稿移除，E07/E13）**。

**rank 限制（已强制，非 de facto）**：`tila.arange` 是唯一的索引种子且为一维，而 load/store 要求 `rank(idx) = rank(buffer)`——因此 v0.1 在签名检查直接拒绝 rank ≥ 2 注解（E12），kernel 只能操作一维张量。二维（`expand_dim`、size-1 广播、`&`）与 rank-2 解禁见 `docs/v0.2-preview-2d.md`。

## 11. 参考程序与测试基准

- 规范内最大合法程序：`examples/add.tila`。
- 期望 lowering 产物：`examples/add.lowered.py`（带注释的参考版；字节级黄金文件见 `triton-lowering.md` §9）。
- 非法程序集（每个诊断码一个最小复现）：见 `type-checker.md` §9。