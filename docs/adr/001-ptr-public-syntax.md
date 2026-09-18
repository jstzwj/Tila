# ADR-001：Ptr 公共参数语法

- 状态：**Accepted**
- 日期：2026-09-18
- 实施状态：M1-01 已实现（2026-09-18）
- 依赖：[ADR-002](002-region-id-and-extent.md)

## 背景

当前实现接受 `Ptr[T, Access, Region?, Alignment?]`，但第三项实际同时被 bounds
逻辑当作长度使用。旧设计文档又展示了五参数 Ptr，并把 Region 当成用户可声明的
身份。这造成两个问题：公共语法与实现不一致；内存身份与可访问长度被混为一谈。

公共类型参数必须能够表达调用者可以验证的契约，但不能允许调用者伪造编译器用于
effect、alias 和 race 分析的内存身份。

## 决策

Ptr 的规范完整形式固定为：

```text
Ptr[Element, AddressSpace, Access, Extent, Alignment]
```

| 参数 | 含义 | v0 约束 |
|---|---|---|
| `Element` | 元素 dtype | 使用 Tila 标量 dtype |
| `AddressSpace` | 地址空间 | 只实现 `Global`；`Shared/Local` 暂时拒绝 |
| `Access` | 权限 | `ReadOnly`、`WriteOnly` 或 `ReadWrite` |
| `Extent` | 从基址开始可访问的元素数量 | 非负整数、维度表达式或 unknown |
| `Alignment` | 基址字节对齐 | 正的 2 次幂或 unknown |

`RegionId` **不是公共 Ptr 参数**。它由编译器在参数绑定、Buffer 构造和内部分配时
生成，详见 ADR-002。

完整写法：

<!-- tila-example: current; mode=exec -->
```python
from tila import Ptr, f32, Global, ReadWrite

values: Ptr[f32, Global, ReadWrite, 1024, 16]
```

### 紧凑形式

为常见 Global 指针保留紧凑形式，并按参数个数无歧义地展开：

```text
Ptr[T]                         -> Ptr[T, Global, ReadWrite, ?, ?]
Ptr[T, Access]                 -> Ptr[T, Global, Access, ?, ?]
Ptr[T, Access, Extent]         -> Ptr[T, Global, Access, Extent, ?]
Ptr[T, Access, Extent, Align]  -> Ptr[T, Global, Access, Extent, Align]
```

短别名固定访问权限，支持 `Alias[T]`、`Alias[T, Extent]` 和
`Alias[T, Extent, Alignment]`：

<!-- tila-example: current; mode=exec -->
```python
from tila import ReadPtr, WritePtr, f32

src: ReadPtr[f32, 1024, 16]
dst: WritePtr[f32, 1024, 16]
```

### Extent 和 Alignment 的单位

- `Extent` 以元素为单位，不以字节为单位；指针加法也使用元素 offset。
- `Alignment` 以字节为单位，必须为正的 2 次幂。
- 派生指针继承 extent，但 bounds 检查以“原始 extent 与当前 offset”共同判断。
- 未知 extent 不自动获得安全结论；严格模式下需要 Buffer 契约、证明或显式 unsafe。

### Launch 绑定

原始 Ptr 参数只能绑定到 Tila 已支持的连续 tensor-like 存储。已声明的静态 extent
不得大于实际可用元素数；未绑定的维度 extent 可由实际元素数特化；unknown 保持
unknown。地址空间和 dtype 不匹配时，在执行 kernel 前报错。

## 兼容与迁移

现有四项形式 `Ptr[T, Access, X, Alignment]` 在 M1 迁移期保留，但 `X` 被正式解释为
`Extent`，这与当前 bounds 代码的实际用途一致。类型打印、文档和 golden 统一改用
Extent 术语。旧文档中把该位置称为 Region 的内容不再有效。

引入完整五项形式后，四项形式仍是稳定简写，不依赖启发式区分 Region 与 Extent。

## 被拒绝的方案

1. **公开 `RegionId` 参数**：调用者可以给无关指针填写相同身份，或给可能别名的
   指针填写不同身份，从而破坏 effect/race 分析的 soundness。
2. **让同一参数兼作 Region 和 Extent**：相等长度不表示相同内存身份，且身份不能
   参与数值 bounds 比较。
3. **只保留 `Ptr[T]` 并依赖运行时信息**：无法在签名中表达只读、长度和对齐契约，
   也不能为静态检查提供稳定输入。
4. **v0 同时开放 Shared/Local**：现有 launch 协议与后端尚不能兑现其生命周期、
   分配和同步语义。

## 影响与实施要求

- 公共 API、annotation parser、类型打印与诊断必须接受同一组规范形式。
- bounds 代码只读取 Extent；effect/race 代码不能从 Extent 推断身份。
- launch 层必须验证 dtype、连续性、可用元素数和对齐契约。
- 缓存键必须使用规范化后的完整 Ptr 类型，而不是用户采用的简写拼法。
- M1-01 已覆盖公共解析、规范化打印与 launch 契约；RegionId/alias 的完整内部
  模型仍由 M1-02 继续实现。
