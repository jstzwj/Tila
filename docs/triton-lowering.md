# Tila → Triton Lowering 设计

状态：定稿（2026-08 评审修订）。lowering 消费 TIR（`ast.md` §4），产出 Triton Python 源码。本阶段是**纯语法映射**：不做任何 Tila 语义拒绝（类型/shape/layout 判断一概不发生），**对有效 TIR 全函数（total）且确定**（同输入必得同输出，黄金测试逐字节比对）；产物在 Triton 侧的合法性属于后端校验层（§8）与验收层（§9），不属于 lowering。

---

## 1. 原则与产物

| 原则 | 含义 |
|---|---|
| 无新语义拒绝 | 所有可判定属性已被 checker 消耗；lowering **对有效 TIR 是全函数（total）**——不做任何新的 Tila 语义拒绝，只查表发射。产物的 `exec` 与 Triton 编译期仍可能失败（缺 GPU、架构不支持 fp8 等），那属于 §8 后端校验层与 §9 验收层 |
| 确定性 | 值命名、参数顺序、括号、缩进全部由规则唯一决定 |
| 形态 | v0.1 产物 = 一个 Python 模块源码字符串；`exec` 后得到 `@triton.jit` 函数与 launcher |
| 演进 | v0.2+ 改为直接构造 Triton AST/TritonIR 对象（跳过源码与 exec），映射表不变 |

选择"生成源码"作 MVP 是刻意的：可读、可 diff、可人工送进 Triton 工具链调试——这正是一个 Triton frontend 最需要的性质。

## 2. 模块布局与精确格式

生成模块固定为三段（空行规则：段间两个空行，函数体 4 空格缩进）：

```python
import triton
import triton.language as tl


@triton.jit
def add(a, b, c, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(a + offs, mask=mask)
    y = tl.load(b + offs, mask=mask)
    z = x + y
    tl.store(c + offs, z, mask=mask)


def add_launch(a, b, c, BLOCK: int = 128):
    assert a.dim() == 1 and b.dim() == 1 and c.dim() == 1
    N = a.shape[0]
    assert b.shape[0] == N and c.shape[0] == N
    grid = (triton.cdiv(N, BLOCK),)
    add[grid](a, b, c, N, BLOCK=BLOCK)
```

此即黄金文件 `tests/golden/add.triton.py` 的全部内容（无注释、无尾随空白、LF、文件尾恰好一个换行）。`examples/add.lowered.py` 是同一代码的带注释参考版——**逐字节比对只认 `tests/golden/`**，`examples/*.lowered.py` 是人读版，不要拿去做 diff。

## 3. Kernel 签名映射

Triton 形参顺序（唯一规则）：

```
[buffer 参数（声明序）] → [符号维（注解中首次出现序）] → [constexpr 参数（声明序）]
```

- buffer/符号维参数：裸名字，无注解（Triton 从 launch 实参推断指针与 i32）。
- constexpr 参数：`名字: tl.constexpr`。
- add 例：`a, b, c` → `N` → `BLOCK: tl.constexpr`。

参数名与 kernel 名一律**透传源码名，无 mangling**：`tl`、`triton` 已进保留名清单（E12），从源头排除遮蔽模块导入的名字；checker 的名字冲突检查（E12）保证无重复形参。

**constexpr 模型：检查逐值特化，发射保留名字。** 编译入口 `compile_kernel(source, overrides)`（§8）每次都带着 override 值完整走 check → TIR → lowering——**TIR 是该 override 值下的特化产物**：shape 取特化值（全静态，2 的幂/上限等约束可查），检查结论只对该值有效。一个按 BLOCK=128 检查过的 TIR 不允许拿去跑 BLOCK=256；host 需要其他值时以新 override **重新编译**（编译是廉价纯函数，driver 按 (源码, overrides) 缓存产物），而不是复用已检查的 TIR。发射侧则保留 constexpr 名字与常量表达式（`tl.arange(0, BLOCK)`）、**不折叠**——这不是为了跨值复用产物（那已被上一句禁止），而是保持与源码 1:1 的同构映射和稳定的黄金输出。Triton 自身的编译期校验因此只是**纵深防御**，不承担 Tila 的健全性。

## 4. 值命名与语句发射

- **有 `src_name` 的指令**：发射为 `名字 = <表达式>` 的局部变量（黄金输出与源码同名，可读性来自这里）。
- **匿名指令**：不发射变量，在使用点**内联**其表达式。依据 v0.1 事实"匿名值恰被使用一次"（`ast.md` §6 不变量 3）；发射器实现为按 use_count 决策（==1 内联、>1 物化为变量）——将来引入 CSE/DCE 等优化产生多次使用的匿名值时，只改发射决策，IR 不设此约束。
- **`TStore`/`TReturn`**：无变量，直接发射调用行 / `return`。

发射顺序 = TIR 指令序。v0.4 起 TIR 一层可嵌套（`TFor.body`）：循环体递归发射，
每层缩进 +4 空格；`TPhi` 与合成引用一样**不发射**（loop-carried 由 Triton 的
for 语义接管——种子在循环前物化、体内以同名读写，跨迭代类型稳定由 R19 的
dtype/shape 严格检查保证）。函数体内的发射结果即"每条源码赋值一行 + store 一行
（循环体按层缩进）"。

## 5. 表达式映射总表

| TIR 指令 | Triton 表达式模板 |
|---|---|
| `TConstInt(v)` / `TConstFloat(v)` | `v` / 确定性 f32 格式化（先按 f32 舍入再 repr，避免 double 表示歧义） |
| `TSymRef(n)` | `n` |
| `TConstParamRef(n)` | `n`（body 内引用 constexpr **名字**，值特化交给 Triton） |
| `TProgramId(a)` | `tl.program_id(a)` |
| `TArange(0, e)` | `tl.arange(0, ⟨e⟩)`——⟨e⟩ 按 ConstExpr 渲染（`BLOCK`、`BLOCK // 2`、字面量），不折叠 |
| `TArith(op,l,r)` | `⟨l⟩ op ⟨r⟩`（按优先级加括号）；标量侧由 Triton 原生广播 |
| `TCmp(op,l,r)` | `⟨l⟩ op ⟨r⟩` |
| `TLogic(op,l,r)` | `⟨l⟩ op ⟨r⟩`（`&` / `\|`） |
| `TAddPtr(base, offs)` | `⟨base⟩ + ⟨offs⟩`——**指针算术只在这一层出现** |
| `TLoad(p,m,o)` | `tl.load(⟨p⟩[, mask=⟨m⟩][, other=⟨o⟩])` |
| `TStore(p,v,m)` | `tl.store(⟨p⟩, ⟨v⟩[, mask=⟨m⟩])` |
| `TCast(dt,x)` | `⟨x⟩.to(tl.<dt>)`（dtype 表见 §6） |
| `TExpandDim(t,a)` | `tl.expand_dims(⟨t⟩, a)`（v0.2 预览） |
| `TDot(l,r)` | `tl.dot(⟨l⟩, ⟨r⟩)`（v0.2 fragment） |
| `TZeros(shape,dt)` | `tl.zeros((⟨s₀⟩, …), dtype=tl.<dt>)`——shape 按 ConstExpr 渲染（名字/字面量）；单元素元组带尾逗号 `(64,)`（v0.4） |
| `TFor(var,e,s)` | `for var in range(0, ⟨e⟩, ⟨s⟩):` + 嵌套体缩进 +4（v0.4；φ 不发射） |
| `TPhi(pre,back)` | 不发射（Triton loop-carried 接管；种子/累加行以 src_name 同名渲染，如 `acc = acc + tl.dot(x, y)`）。**正确性映射**：`lower(LoopPhi) = Triton loop-carried assignment`——两侧不是逐字节对应而是语义等价，验收以同一 TIR 的 interpreter/GPU 差分为准（黄金只钉文本；v0.4-kloop §8.5） |

操作数字段是 TIR `%id`：查表得该 id 的**发射名**（有 src_name → 名字；匿名 → 递归内联其表达式）。`addptr` 是匿名指令的标准内联对象：`load %p0` → `tl.load(a + offs)`——表面语言的 `tila.load(a + offs)` 到 Triton 的展开点。

**括号规则（最小括号）**：运算符优先级沿用 Python 表：比较 = 6，`|` = 8，`&` = 9（Python 中位运算高于比较），`+ -` = 11，`* /` = 12，原子 = 最高。发射二元节点的子节点时，仅当子节点自身是二元节点且其优先级**低于**父节点上下文要求时加括号；同优先级左结合不加。确定性由此保证（例：`pid * BLOCK + tl.arange(0, BLOCK)` 无冗余括号）。

## 6. dtype 映射表

| 类别 | Tila | Triton |
|---|---|---|
| 布尔 | `bool` | `tl.int1` |
| 有符号整数 | `i8 / i16 / i32 / i64` | `tl.int8 / tl.int16 / tl.int32 / tl.int64` |
| 无符号整数 | `u8 / u16 / u32 / u64` | `tl.uint8 / tl.uint16 / tl.uint32 / tl.uint64` |
| 浮点 | `f16 / bf16 / f32 / f64` | `tl.float16 / tl.bfloat16 / tl.float32 / tl.float64` |
| FP8（存储 dtype） | `fp8e4m3 / fp8e5m2 / fp8e4m3fn / fp8e4m3b15` | `tl.float8e4m3 / tl.float8e5m2 / tl.float8e4m3fn / tl.float8e4m3b15` |

FP8 格式的目标架构可用性由 Triton 编译期校验（B01），Tila 透传。指针类型无映射——表面语言不含指针（`type-system.md` §1.1）。

## 7. Launcher 生成

模板（`<kernel>_launch`，跟在 kernel 定义之后）。**grid 与全部断言来自 launch analysis phase 存入 `TKernel.launch_plan` 的推导结果（`type-checker.md` §5——typing 之后的独立 pass，不在表达式推导里），lowering 只发射不推导**——可失败的部分全部留在 checker 侧：

```python
def <name>_launch(<buffer 参数>, <constexpr 参数: int = 默认值>):
    assert <t1>.dim() == r and <t2>.dim() == r and …        # 每张量的注解 rank
    [assert <t>.shape[i] == k …]                            # 仅注解含静态维时发射
    <sym> = <首个绑定该符号的 buffer>.shape[<i>]              # 符号维按维取值
    assert <其余绑定同一符号的 buffer>.shape[<i>] == <sym>    # 符号维运行时契约
    grid = (<cdiv(该轴的维, 该轴的 tiling constexpr)> × 轴数)
    <name>[grid](<buffers…>, <syms…>, <constexpr 名=值…>)
```

- **shape 契约三件套**（补齐"静态维无运行时校验"与"rank 不校验"两个缺口）：rank 断言每张量必发；静态维断言（`Tensor[f32, 1024]` → `assert a.shape[0] == 1024`）；符号维相等断言（`language-spec.md` §7 运行时契约的执行点）。符号维一律按 `shape[i]` 取值/比较，不再用 `numel()`（可推广到多维）。
- **grid 推导规则（launch_plan，launch analysis phase，失败 → E17）**：轴数 = kernel 内实际使用的 `program_id` 最大轴 + 1；每轴表达式 = `cdiv(维, tiling constexpr)`，其中"维"是该轴 mask 谓词覆盖的符号维（静态维直接代入字面量），"tiling constexpr" 是在该轴 `pid * CE` 中被引用的那个——**每轴必须唯一确定**（两个 constexpr 同时参与同一轴的 tiling 即 E17）。v0.1 实际只有轴 0。这是**语义模式识别**（识别 tiling 惯用法）而非类型检查，故独立成 phase——未来 autotune、多种 grid 策略、persistent kernel 都挂这一层。
- **内存布局契约（v0.3 起；v0.4 增空真守卫）**：Strided buffer 逐轴
  `assert b.stride(i) == <声明值>` / 绑定符号；RowMajor（默认）buffer 断言
  实际连续（`stride(r−1) == 1` 且 `stride(i) == shape[i+1]`，期望步长永远由
  shape 推导——declarative, not inferred）。**size-0 数组的契约空真**
  （v0.4-kloop §12：torch 对空维报告的 stride 无意义，如 `(100,0).stride()
  == (1,1)`；无元素可错读）——每个 RowMajor 断言带
  `<b>.numel() == 0 or (…)` 守卫（复合式外层括号必须带：顶层以 `and`
  连接而 `or` 结合度更低）。零迭代恒等式（K=0 的 matmul）由此在 GPU 侧成立。

## 8. 编译驱动、CLI 与后端校验层

```python
def compile_kernel(source: str,
                   overrides: dict[str, int],
                   *,
                   target: str = "triton",
                   validate_backend: bool = True) -> str:
    pyast = ast.parse(source)                    # SyntaxError 包装成 TilaError(E11)
    tila_ast = frontend.convert(pyast)           # 子集校验 + intrinsic resolution（E11–E15）
    tir = checker.check_kernel(tila_ast, overrides)
    return lower_triton.emit(tir)                # 模块源码字符串
```

内部流水线（评审 §34 的原样结构）：

```
parse → subset validation → tila_ast → type check → TIR → launch analysis
      → backend validation → Triton lowering → Triton compile
```

CLI（v0.1 两条命令；`--constexpr NAME=VALUE` 可重复给出，作为 checker 的**检查特化值**——每个 override 组合对应一次完整编译（TIR 是特化产物，跨值不复用，§3）；body 按模型 B 发射名字；`--explain` 打印判定日志；`--verify` 触发后端校验层）：

```
python -m tila build examples/add.tila -o build/ --constexpr BLOCK=64 --explain
                                                      # 写出 build/add.triton.py（+ add.tir.txt dump）
python -m tila run  examples/add.tila --constexpr BLOCK=64
                                                      # 内存编译 + exec + 调 launcher 跑 smoke（需 GPU）
```

`build` 产物即黄金比对物；`run` 仅用于本地验收，不属于规范。

**后端校验层（与 lowering 分离）**：`lower_triton.emit` 必须 total；生成源码在 Triton 侧的合法性由独立的 `validate_generated_triton(source)` 承担——在 `run` 或 `build --verify` 时以 `exec` + 触发一次 Triton 编译实现（无 GPU 环境则到编译器语义检查为止）。**compiler correctness（Tila 自己的检查）与 backend compatibility（Triton 接受产物）由此分开**。后端兼容性错误用独立诊断码：

| 码 | 场景 |
|---|---|
| B01 | 目标架构不支持该 dtype（如 fp8e4m3 需要 SM89+ 等，Triton 编译期报告） |
| B02 | tile 元素数超过 TRITON_MAX_TENSOR_NUMEL（2^20） |
| B03 | requested tile shape cannot be lowered to the selected Triton configuration（num_warps/num_stages 等与 tile 不匹配） |

B 码由 `backend/triton/validator.py` 报告，报 "backend validation"，不走 Tila 诊断渲染（E 码）；B04+ 预留。这样 `lowering total` 不被破坏：**"check backend → fail" 而不是 "lower → 半路发现不支持 → fail"**。

## 9. 验收与测试

**v0.1 目标**（评审 §42 原句）：*Demonstrate that a strongly typed, layout-aware subset of Triton can be compiled to Triton with a total and semantics-preserving lowering.* 程序集：**add、saxpy、masked_add、fp8_add**。

**里程碑定义**（README）：`add.tila` → 生成的 Triton 源码经现有 Triton 编译器在 GPU 上运行，与 `torch.add`（f32、随机 N 含非 2 次幂余数情形）逐元素一致。

| 测试层 | 内容 |
|---|---|
| Phase 0 oracle | 实现前先做：手写 add 的 Triton kernel，对 BLOCK ∈ {32, 64, 128, 256} 跑通并检视 TTIR/TTGIR，验证 Tila 的类型假设（arange/broadcast/load/add/mask 的实际 encoding 行为——"identity 抽象是否够用"，`development-plan.md` §2） |
| 黄金源码 | 生成模块与 `tests/golden/add.triton.py` **逐字节一致**（含空行/缩进/换行符；实现注意：产物为 LF + 单尾换行，Windows 下写文件须显式 `open(..., "w", newline="\n")` 或二进制写，否则 CRLF 会击穿逐字节比对） |
| 黄金 TIR | `build` 同时写出 dump，与 `tests/golden/add.tir.txt` 一致（checker 侧已覆盖，双保险） |
| 单元 | 签名顺序（buffer→sym→constexpr）、最小括号、匿名内联、dtype 表 |
| 差分（Stage 6） | 与 reference interpreter 对拍：同一 TIR 走 interpreter 与 Triton，结果一致（`development-plan.md` §8） |
| 集成（可选 marker `gpu`） | torch 参考对比；N ∈ {1, 127, 128, 129, 1000}；BLOCK ∈ {32, 128} |
| 反向 | 对 checker 的全部非法程序（E01–E17 矩阵）断言不产出任何源码 |

## 10. v0.2 展望（不实现，仅占位）

- `import tila as tl`——API-level source compatibility（评审 §20）：允许别名后，大量 Triton 风格代码可直接迁移；
- 直接构造 Triton AST / TritonIR（`emit` 换后端，映射表不变）；
- 2D：`tila.expand_dim`/size-1 广播/`&`（已定稿为预览片段 `docs/v0.2-preview-2d.md`，lowering 侧仅新增 `TExpandDim` 一条平移指令）、`transpose/reshape` 非擦除 layout term、`convert` 显式布局转换；
- `dot` 与 MMA layout 约束（Tila 侧等价类 + Triton 侧 encoding 的对接——桥接问题的第一个实例）；
- 循环与累加（打破单赋值，引入 φ 或显式可变绑定）；
- launcher 的 `--safe` 连续性/维度断言、多维 grid；
- 内建扩充：`num_programs`、`zeros`/`full`/`where`、`abs`/`exp`/`log`/`sqrt`、`max`/`sum`、`dot`（`language-spec.md` §2 的 API 规划）。