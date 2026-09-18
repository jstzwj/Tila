# ADR-002：RegionId 与 Extent 的拆分模型

- 状态：**Accepted**
- 日期：2026-09-18
- 实施状态：M1-02 已实现（2026-09-18）
- 关联：[ADR-001](001-ptr-public-syntax.md)

## 背景与不变量

旧模型中的 `region` 既像内存身份，又被当作长度参与 bounds 比较。这会让等长的两块
存储看似同一区域，也可能让相同存储的不同长度视图看似不同身份。

本决策固定以下不变量：

1. `RegionId` 只回答“这个值可能来自哪块存储”，不表示大小。
2. `Extent` 只回答“从基址开始有多少元素可访问”，不表示身份。
3. Bounds only consume Extent；Effects only consume RegionId。
4. 不同 `RegionId` 本身不构成 `noalias` 证明。

## 内部模型

规范化的指针类型包含：

```text
PtrType {
  element: ElementType,
  address_space: AddressSpace,
  access: Access,
  region_id: RegionId,
  extent: Extent,
  alignment: Alignment
}

RegionId = ParamRegion | BufferRegion | InternalRegion | UnknownRegion
Extent   = LinearExtent(DimExpr) | UnknownExtent
```

指针 SSA 值另外携带从该存储基址计算的 `offset: IndexExpr`。`offset` 不属于 Ptr 的
公共类型参数，因为同一静态类型的值可以在控制流中具有不同 offset。

派生指针继承 `region_id`、`extent`、`address_space` 和 `access`；offset 随指针运算
改变，对齐信息可以被削弱但不能无证明增强。

## Bounds、Effect 与 Alias 的职责边界

Bounds obligation 使用 `extent` 和值的 `offset`：例如访问宽度为 `w` 时证明
`0 <= offset` 且 `offset + w <= extent`。RegionId 不得出现在数值不等式中。

Effect 记录以 `RegionId` 为对象，例如对某个 region 的 read/write。Extent 不得用作
effect key。这样同一区域的不同视图仍可汇总为同一内存 effect。

Alias 是第三个独立维度：

```text
AliasRelation = MustAlias | MayAlias | NoAlias
```

来自不同外部参数的裸 Ptr 默认是 `MayAlias`。不同 RegionIds do not imply noalias；
只有 launch/runtime 契约、未来显式 noalias 契约或可证明的内部独立分配才能得到
`NoAlias`。由同一指针派生的值通常是 `MustAlias` 于同一底层 region，但访问范围仍
需结合 offset/extent 判断是否重叠。

## RegionId 的来源

- 每个裸 Ptr 形参获得 `ParamRegion(parameter_index)`；调用绑定不会接受用户提供
  RegionId。
- 每个 Buffer 形参获得 `BufferRegion(parameter_index)`，`buf.ptr` 继承该身份。
- 编译器控制的分配获得稳定的 `InternalRegion(allocation_id)`。
- 无法确定来源时使用 `UnknownRegion`，分析必须保守处理。

M1 首轮 `buf.ptr` 只为 rank-1、stride-1 Buffer 建立与裸 Ptr 等价的线性 extent；
多维/任意 stride 的线性化在 Buffer stride ADR 中另行决定。

## Launch 绑定与视图

launch 层验证实际参数是受支持的连续存储，并将动态元素数绑定到 Extent。两个形参
即使绑定到同一底层对象，仍保留各自的静态 ParamRegion；运行时绑定信息将它们的
alias 关系提升为 `MustAlias`，不能通过篡改 RegionId 表达。

切片或偏移视图保留底层 RegionId，独立携带 offset 和可用 extent。未来支持非连续
视图时，必须扩展 layout/stride 模型，不能把 stride 塞回 RegionId 或 Extent。

## 被拒绝的方案

1. **`RegionId == Extent`**：身份和数值范围不是同一种信息，等长不等于同一存储。
2. **不同 RegionId 自动 NoAlias**：两个外部参数仍可能绑定同一对象，会导致错误的
   重排和漏报 race。
3. **把 offset 编进 RegionId**：同一存储的派生指针会失去 effect 汇总能力。
4. **完全依赖 Python 对象 identity**：它只在一次 launch 中存在，不能作为稳定类型、
   缓存或 ahead-of-time 检查的语义基础。

## IR、诊断与迁移要求

- TIR/explain 分别打印 `region`、`extent`、`offset` 和 alias 结论，不能再输出含糊的
  单一 `region=N`。
- 旧 `PtrType.region` 迁移成 `extent`；RegionId 由参数/Buffer 来源重新生成。
- bounds golden 只比较 extent obligation；effect/race golden 使用稳定 RegionId。
- 序列化与缓存版本必须升级，避免读取旧的混合字段。
- 未知信息保守化：UnknownExtent 不能证明 bounds，UnknownRegion/MayAlias 不能证明
  独立访问。

## 实施顺序与验收

1. 引入两个不可互换的 IR 类型，并迁移类型打印与缓存键。
2. 让参数和 Buffer 构造分配 RegionId，迁移派生指针传播。
3. 将 bounds 消费方全部切换到 Extent。
4. 将 effect/race 消费方全部切换到 RegionId 与独立 alias relation。
5. 更新 launch 绑定、explain、golden 和差分测试。

验收时必须有反例证明：两个相同 Extent 的参数不会合并 RegionId；两个不同
ParamRegion 默认仍是 MayAlias；同一 Buffer 的 `buf.ptr + offset` 保留 RegionId；
RegionId 无法进入 bounds 算术，Extent 无法作为 effect key。
