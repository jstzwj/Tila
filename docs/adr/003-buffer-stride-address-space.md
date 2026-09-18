# ADR-003：Buffer stride 与 address-space 公共语法

- 状态：**Accepted**
- 日期：2026-09-18
- 实施状态：M1-03 已实现（2026-09-18）
- 关联：[ADR-001](001-ptr-public-syntax.md)、[ADR-002](002-region-id-and-extent.md)

## 背景

旧类型总览展示过包含 `Strides` 和 `AddressSpace` 的 Buffer 完整形式，但当前公共
构造器实际接受 `Buffer[T, Shape, Access, Alignment?]`。与此同时，launcher 已从
NumPy/torch 参数读取实际 stride，Buffer 坐标访问也按实际 stride 工作。若把任意
stride 表达式直接放进公共类型参数，就会同时引入布局相等、动态绑定、负 stride、
storage offset 和缓存规范化问题，而 Shared/Local 又尚无分配与生命周期语义。

## 决策

### 公共语法

Buffer v0 的规范形式固定为：

```text
Buffer[Element, Shape, Access, Alignment]
```

用户可写以下 2–4 项形式：

```text
Buffer[T, Shape]
Buffer[T, Shape, Access]
Buffer[T, Shape, Access, Alignment]
```

省略 `Access` 时为 `ReadWrite`，省略 `Alignment` 时为 `UnknownAlignment`。
规范打印始终显示四项。Alignment 以字节为单位，规则与 ADR-004 的 `Aligned[k]`
一致。

<!-- tila-example: current; mode=exec -->
```python
import tila as ti

M = ti.Dim("M")
N = ti.Dim("N")


@ti.jit
def copy_shape_only(
    src: ti.Buffer[ti.f32, (M, N), ti.ReadOnly],
    dst: ti.Buffer[ti.f32, (M, N), ti.WriteOnly, ti.Aligned[16]],
):
    pass
```

`Strides` 和 `AddressSpace` 不属于 v0 公共参数。`Global` 是 Buffer v0 的隐式地址
空间；五项或六项 Buffer 注解必须拒绝，不能靠参数类型猜测不同布局。

### 内部规范表示

```text
BufferType {
  element: ElementType,
  shape: tuple[DimExpr, ...],
  access: Access,
  alignment: Alignment,
  address_space: Global,
  strides: BoundStrides | UnboundStrides,
  region_id: BufferRegion
}

BoundStrides = tuple[StrideExpr, ...]  # 单位：元素
```

公共注解求值只建立 `UnboundStrides`。checker 为每个轴建立稳定的内部 stride
符号；launch 从实际参数绑定这些符号。stride 不是内存身份，也不进入 RegionId。

### Buffer 坐标访问

`load(buf, coords)` / `store(buf, coords, value)` 是逻辑坐标访问。只要 rank、shape、
dtype 和后端表示能力满足，正 stride、转置视图和带 storage offset 的视图都可以
绑定；lowering 使用实际元素 stride 计算地址。Bounds 仍逐逻辑轴对 shape 证明，
不把 stride 当作 extent。

负 stride、重叠/零 stride、设备不支持的 view 必须由 launch/target capability
显式拒绝或保守处理，不能静默当作 contiguous。它们的精确支持矩阵属于后续
Target ADR，而不是公共 Buffer 类型语法。

### `buf.ptr` 线性化边界

v0 只有 rank-1 且运行时 `stride[0] == 1` 的 Buffer 可以取 `buf.ptr`。结果：

- 继承 Buffer 的 element、access、Global、alignment 和 BufferRegion；
- Extent 等于唯一 shape 维；
- offset 从零开始，指针加法以元素为单位。

rank 大于 1 或 stride 不为 1 时，`buf.ptr` 必须拒绝并建议使用 Buffer 坐标访问。
即使一个多维参数运行时连续，v0 也不隐式展平；显式 flatten/view 操作需单独设计，
避免把行主序假设写死在类型系统里。

### Stride refinement

stride 性质属于由 launch 事实产生、供优化消费的 refinement，而不是 Buffer 的位置
参数。M1 只标准化内部事实词汇：

```text
StrideEq[axis, value]
Contiguous
```

这些名字暂不作为用户可写公共注解。未来开放时必须证明它们能由 launch 校验、能
进入缓存键，并与 target/layout 模型一致；任意用户书写的 stride DimExpr 本 ADR
明确不开放。

## Launch 与缓存契约

launch 必须绑定并校验 dtype、rank、shape、每轴元素 stride、storage offset、
alignment 和 device。缓存键不按每次具体 stride 数值无限特化；只有 lowering
实际依赖的、已声明或已选择特化的 stride refinement 才进入缓存键。

## 被拒绝的方案

1. **`Buffer[T, Shape, Strides, AddressSpace, Access]` 立即公开**：动态 stride 的
   等价、合法性和缓存语义尚未闭环，会把内部 ABI 固化成语言语法。
2. **所有 Buffer 强制 contiguous**：会丢失现有转置/切片的坐标访问能力，也不符合
   Buffer 作为逻辑多维视图的定位。
3. **任意 Buffer 都可 `buf.ptr`**：线性 Ptr 没有 stride/layout 字段，会错误解释
   多维或非连续地址。
4. **现在开放 Shared/Local**：尚无分配、作用域、同步和 backend capability 契约。
5. **用 RegionId 编码 stride/layout**：内存身份与地址映射是正交信息。

## 迁移与验收

- 现有 2–4 项 Buffer 注解保持兼容；旧文档中的显式 Strides/AddressSpace 形式失效。
- annotation parser、内部类型、规范打印和诊断必须使用同一四项顺序。
- 测试必须覆盖转置 Buffer 坐标访问、非连续 `buf.ptr` 拒绝、rank-2 `buf.ptr`
  拒绝、stride 元素单位和 BufferRegion 继承。
- M1-03 已实现规范四项打印、独立内部 stride 绑定、rank-2 `.ptr` 的 Stage 1
  拒绝、rank-1 非单位 stride 的 launch 拒绝，以及能力/BufferRegion 继承。
