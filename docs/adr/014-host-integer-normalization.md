# ADR-014：宿主整数的显式白名单规范化

- 状态：**Proposed**（低优先级推荐方案，未实现）
- 日期：2026-09-19
- 阶段：M2-07c；不作为 Mask/Const bool 实施前置条件
- 关联：ADR-005、ADR-007

## 推荐决策与语法

维持 Const[int] 的 ExactInt 门禁，不让 NumPy scalar 自动通过。建议增加仅限
宿主调用的 `ti.host_int(value)`，显式返回精确 Python int，再交给原有 API。

<!-- tila-example: future; milestone=M2 -->
```python
block = ti.host_int(np.int64(128))
kernel[(1,)](data, BLOCK=block)
```

允许 exact Python int，以及 NumPy 的 signed/unsigned 8/16/32/64 位整数 scalar
具体类型。使用具体类型集合而非 `isinstance(np.integer)`；别名去重为同一类型。
拒绝 Python/NumPy bool、float、字符串、零维 ndarray、用户子类和任意实现
`__int__`/`__index__` 的对象。先检查白名单再转换，不执行不可信对象的转换方法。
不接受 torch tensor，也不在 kernel frontend 中允许调用此宿主 helper。

## 范围、表示与缓存

规范化保留数学整数值，不截断、不回绕、不先转换为平台 intp。
因此 np.uint64 的最大值可变成任意精度 Python int；这不意味着能通过 shape/grid
的 i32 门禁。使用点仍独立检查 Const refinement、range/shape/grid/stride 范围和
runtime scalar dtype。数值域检查不能由 helper 成功替代。

不增加 TIR 类型或指令。helper 输出进入已有 ExactInt 绑定流程，与用户直接传
同值 Python int 得到相同规范键；原 NumPy dtype 不成为额外特化维度。
未显式调用 helper 的 np.int64 override 仍拒绝，以保持当前入口契约。

## 诊断、迁移与验收

拒绝时给出稳定 TYPE 族诊断，包含实际宿主类型和接受列表；精确错误码在实现时
登记到 diagnostic registry，不把现有错误码复用于不同含义。host helper 独立登记
公共导出但不是设备 intrinsic；文档将它与 kernel 内整数 cast 明确分开。

验收：8 类 NumPy 整数的两端值、别名、uint64 最大值、零维数组/bool/浮点/子类
拒绝；恶意转换方法未被执行；转换后仍触发使用点的范围/refinement 检查；不同
来源同数学值的缓存一致。无需增加运行时 NumPy 协议的泛化接纳。

## 未选择的方案

隐式接受所有 np.integer 会修改 ADR-005 的统一入口语义；接受 numbers.Integral
或任意转换协议会扩大用户代码执行与缓存规范化边界。要求用户随意写 int(value)
虽然可用，却会无意接受 bool/浮点截断；显式 helper 提供更窄的可检查契约。
