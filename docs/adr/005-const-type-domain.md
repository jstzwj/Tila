# ADR-005：Const 类型域与 staged bool 边界

- 状态：**Accepted**
- 日期：2026-09-18
- 实施状态：`Const[int]` 主路径已存在；本文冻结其公共边界，严格门禁与延迟断言按迁移清单验收
- 关联：[ADR-004](004-refinement-construction-syntax.md)

## 背景

现有文档同时使用了 `Const[int]`、`Const[bool]`、模块级 Python 常量、运行时
`ti.bool` 和 Triton `tl.constexpr`。实现却只有 `Const[int, Refinement, ...]` 参数构造，
static-if 的条件是对 Const int 的比较结果，runtime-if 则消费普通 scalar bool。如果把这些
概念都称为 `Const[bool]`，会产生三个错误承诺：用户可以声明 bool Const 参数、bool 会形成独立
特化键，以及任意编译期 Python 值都属于 Tila Const 类型域。

本 ADR 只决定 0.2.x 的公共类型域与 staging 边界；整数负数除法、取模、溢出和移位的精确语义
由 ADR-007 决定。

## 决策

### 公共参数域只有 `Const[int]`

0.2.x 唯一可声明的 Const 参数形式是：

```text
Const[int, Refinement, ...]
```

其内部规范表示为：

```text
ConstParamType {
  value_kind: Int
  name: Symbol
  refinements: tuple[ValueRefinement, ...]
  default: ExactInt | None
}
```

`Const[bool]`、`Const[float]`、`Const[dtype]` 以及泛化的 `Const[T]` 均不属于 0.2.x
公共语言。它们在状态表中保持 Designed，不得出现在 current 示例、当前 intrinsic 签名或
兼容性承诺中。

`ExactInt` 指 `type(value) is int`：Python 的 `bool` 虽然是 `int` 子类，也必须拒绝；暂不
隐式接受 NumPy integer、字符串或可通过 `int(value)` 转换的对象。默认值、launch override、
直接 materialize 参数和缓存键必须采用同一判定。

<!-- tila-example: current; mode=exec -->
```python
import tila as ti


@ti.jit
def specialized(tile: ti.Const[int, ti.PowerOfTwo] = 64):
    if tile >= 128:
        pass
    else:
        pass
```

Const 参数不消耗普通位置实参。其具体整数在 specialization 时绑定，值与影响语义的
refinement 结果进入 specialization/cache key；同一符号在 checker、interpreter 和 Triton
lowering 中保持同一身份。

### stagedness 不是第二套表面类型

内部把“值的 Tila 类型”和“何时已知”分成正交维度：

```text
ValueType = Scalar[i32] | Scalar[bool] | Block[...] | ...
Stage     = Stage1Known | SpecializationKnown[ConstSymbols] | Runtime
```

`Const[int]` 参数产生 specialization-known int。对它做比较，或用 `not/and/or` 组合比较，
产生 specialization-known bool predicate；这个中间结果可驱动 `TStaticIf`，但不能写成参数
注解 `Const[bool]`，也不会创建新的 bool 特化参数。

三类 bool 必须明确区分：

| 来源 | 类型/阶段 | 允许用途 |
|---|---|---|
| 模块级 Python bool 或可立即折叠的比较 | Stage 1 已知谓词 | 装饰期直接选择分支 |
| `Const[int]` 比较及其布尔组合 | specialization-known bool predicate | `TStaticIf`、可延迟的 `static_assert` |
| Tila `Scalar[bool]` | runtime | runtime-if；不得用于 static assertion 或 Const 语境 |

模块级 Python int/bool/float 是 frontend 捕获并字面化的宿主常量，不因此成为可声明的
`Const[T]` 参数。只有模块级 int 能进入要求 `Const[int]` 的整数语境；bool 只进入静态条件，
float 只作为普通数值字面量。任意 Python 对象、可变容器和函数均不属于可捕获常量集合。

### Const int 表达式子集

0.2.x 的 Const int 表达式由整数字面量、`Const[int]` 名字、模块级 int 字面量化结果、
一元负号以及 `+ - * // %` 组成。比较产生 staged bool，`not/and/or` 只组合 staged bool。
运行时 scalar、Dim、program id、loop induction variable 或 Block 一旦参与，结果就是 runtime，
不能用于 shape、arange bound、range step 或其他 Const int 语境。

除零必须给出稳定的 `TILA-CONST` 诊断，不能退化为 Python traceback。`//`、`%` 对负数的
结果和有限 dtype 溢出由已接受的 [ADR-007](007-integer-semantics.md) 定义：
staged 整数保持任意精度及 floor 除法，runtime 整数按位宽回绕；新增数学
索引推理必须有无溢出依据。

### static-if 与 `static_assert`

- Stage 1 已知条件立即折叠，未选分支不进入 TIR。
- 只依赖 `Const[int]` 的 staged bool 形成 `TStaticIf`；两分支在 Stage 1 独立检查，
  specialization 时选择唯一分支。
- 混入任意 runtime 值的 scalar bool 形成普通 `TIf`。
- `static_assert(pred)` 接受 Stage 1 已知或只依赖 Const int 的 staged bool。前者立即检查，
  后者记录 deferred assertion 并在 specialization 检查。
- 0.2.x 只有单参数 `static_assert(pred)`；`static_assert(pred, msg)` 与字符串常量域一起延后。

因此，`static_assert` 的规范签名使用 `StagedBool`，而不是虚构的 `Const[bool]`。

## Soundness 与缓存

- 只有显式 Const 参数可形成 specialization 维度；模块级常量已固化在捕获源码/前端输入中，
  不与 launch override 混用。
- 缓存键必须包含按参数声明顺序规范化后的 Const int 值以及影响代码生成的语义版本；不得把
  `True` 与 `1`、`False` 与 `0` 合并。
- staged bool 记录它依赖的 Const symbol 集合，不以运行时 scalar bool 冒充静态条件。
- refinement 必须在 specialization 校验后才能进入 proof/optimization facts，规则沿用 ADR-004。

## 诊断要求

- 非 `int` 的 Const 参数注解使用注解/语法诊断，并明确显示只支持 `Const[int, ...]`。
- bool、float 或可转换对象作为默认值/override 时，统一报 `TILA-CONST`，同时显示参数名、
  实际 Python 类型和值。
- runtime 值进入 Const int 语境时，显示 found stage 与 required `Const[int]`。
- runtime bool 进入 `static_assert` 时，明确建议改为运行时控制流或只依赖 Const 参数。

## 被拒绝的方案

1. **现在开放泛化 `Const[T]`**：需要同时定义 dtype 值、字符串、序列、序列化、缓存相等性和
   后端 ABI，远超当前语言闭环。
2. **开放 `Const[bool]` 只为 static-if**：static-if 已能由 `Const[int]` 比较表达；新增参数域
   只会扩大 ABI 和缓存状态，且无法解决 dtype 泛型问题。
3. **把模块级 Python 常量都当 Const 参数**：会把捕获环境、launch override 和缓存失效混成
   一个协议，并允许不可序列化对象渗入 IR。
4. **用 Python `isinstance(v, int)` 定义整数域**：会静默接受 bool，造成 `True` 与 `1` 的
   特化键和诊断歧义。
5. **把 stagedness 编进 `ConstBoolT/ConstIntT` 类型层级**：比较结果和 runtime bool 会迫使
   所有算子复制类型规则；独立 Stage 维度更直接。

## 迁移与验收

版本边界：本 ADR 保留 0.2.x 的历史契约。自 **0.3.0.dev0** 起，
[ADR-013](013-const-bool-domain.md) 扩展公共参数域，允许 exact Python bool 的
`Const[bool]`；本 ADR 的 int-only 限制仅适用于 0.2.x，ExactInt 与 staging
正交原则继续适用。没有 0.2.x 回移或隐式 0/1 转 bool 的兼容路径。
[ADR-014](014-host-integer-normalization.md) 的显式宿主整数转换仍为 Proposed。

- 将 current 文档中的 `Const[bool]` 改为 `StagedBool` 或“只依赖 Const int 的静态谓词”。
- future API 若需要 bool/dtype 参数，必须另写 ADR，给出 ABI、缓存和序列化规则。
- frontend、launch、直接 materialize 和 interpreter 都必须使用 ExactInt 门禁。
- 为模块级 bool、Const 比较 static-if、runtime bool if 建立三路正反测试矩阵。
- 为无默认值 Const 上的 deferred `static_assert`、失败诊断和缓存键区分建立回归。
