# ADR-004：Refinement 构造语法与契约

- 状态：**Accepted**
- 日期：2026-09-18
- 实施状态：M1-04 已实现（2026-09-18）

## 背景

设计文档使用 `Range[32, 1024]`、`MultipleOf[16]` 和 `Aligned[16]`，当前 Python
API 却只有 `Range(32, 1024)`、`MultipleOf(16)` 可构造；`Aligned[k]` 已使用下标
语法但只检查正数。语法、区间端点、参数合法性、适用位置和事实传播因此不一致。

## 决策

### 统一构造语法

所有带参数的 refinement 使用下标语法：

```text
Range[lo, hi]
MultipleOf[k]
Aligned[k]
```

无参数 refinement 使用名字本身，不调用、不加空下标：

```text
Positive
NonNegative
PowerOfTwo
```

规范打印严格使用上述形式。`Range(...)`、`MultipleOf(...)` 和 `Aligned(...)` 不是
语言语法；M1-04 实现后应拒绝，避免两套可拼写形式长期并存。

规范示例：

<!-- tila-example: current; mode=exec -->
```python
import tila as ti


@ti.jit
def configured(
    n: ti.i32 | ti.Positive,
    tile: ti.Const[int, ti.Range[32, 1024], ti.PowerOfTwo,
                   ti.MultipleOf[16]],
):
    pass
```

### 参数域

- `Range[lo, hi]`：`lo`、`hi` 为编译期整数且 `lo <= hi`；语义为闭区间
  `lo <= value <= hi`。
- `MultipleOf[k]`：`k` 为正的编译期整数；负数、零和 bool 拒绝。
- `Aligned[k]`：`k` 为正的 2 次幂整数，单位为字节；bool 拒绝。
- `Positive`：`value > 0`。
- `NonNegative`：`value >= 0`。
- `PowerOfTwo`：值是正的 2 次幂；因此隐含 Positive 和 MultipleOf 的部分事实，
  但规范打印不擅自删除用户写出的其他 refinement。

v0 的带参数 refinement 只接受整数字面量，不接受 Dim、运行期标量或任意表达式。
这保证注解求值、缓存键和错误位置稳定。未来若需要符号端点，另写 ADR。

### 适用位置

| Refinement | 合法位置 | 检查时机 |
|---|---|---|
| `Positive/NonNegative/Range` | 有序数值 scalar、`Const[int, ...]` | scalar 在 launch；Const 在特化 |
| `MultipleOf/PowerOfTwo` | 整数 scalar、`Const[int, ...]` | scalar 在 launch；Const 在特化 |
| `Aligned[k]` | Buffer/Ptr 的 Alignment 参数 | launch 实测 data pointer |

Scalar 组合继续使用 `dtype | refinement`；多个 scalar refinement 使用 tuple：

```text
i32 | (Positive, MultipleOf[16])
```

Const 组合继续按参数顺序书写：

```text
Const[int, Range[32, 1024], PowerOfTwo, MultipleOf[16]]
```

`Aligned[k]` 是 memory alignment specification，不得放入 scalar 或 Const refinement
列表；数值 refinement 也不得占用 Buffer/Ptr Alignment 参数。

### 组合与规范化

- refinement 组合是逻辑合取；所有约束都必须满足。
- 完全重复的 refinement 可在内部去重，规范打印保持首次出现顺序。
- 明显矛盾的字面约束应在注解/Stage 1 拒绝，例如
  `Range[1, 3] + MultipleOf[4]` 没有可行整数。
- 未实现通用可满足性求解前，不能证明矛盾的组合可以留到 launch/特化逐项检查，
  但不得因此生成优化事实。

### Proof facts 与优化 facts

- launch/特化成功后，`Positive`、`NonNegative`、`Range` 进入数值区间事实。
- `MultipleOf[k]`、`PowerOfTwo` 进入整除/位性质事实；lowering 只有在具体消费方
  需要且 target 支持时才发射 hint。
- `Aligned[k]` 只在实际地址校验通过后成为地址对齐事实。
- refinement 不等于 assume：声明必须由 launch/特化验证，不能仅相信用户。

## 诊断要求

非法构造在注解求值时给出 `TILA-SYN-010` 包装的明确原因；值违反已合法声明时使用
launch/Const 契约错误，并同时显示参数名、实际值和规范 refinement 文本。不得把
非法 refinement 延迟成普通 Python `TypeError` 泄漏到用户界面。

## 被拒绝的方案

1. **保留调用与下标两套语法**：破坏规范打印和文档 smoke test 的唯一性。
2. **Range 使用半开区间**：现有实现与文档均更接近闭区间，半开形式容易与 shape
   上界语义混淆；需要半开约束时可组合比较事实。
3. **允许任意 Python 表达式作为参数**：注解将依赖运行期对象身份，缓存和序列化
   不稳定。
4. **Aligned 允许任意正数**：非 2 次幂地址对齐无法形成通用、可组合的对齐格。
5. **把 Aligned 当普通整数 refinement**：值整除与地址字节对齐不是同一事实域。

## 迁移与验收

- `Range(lo, hi)` 与 `MultipleOf(k)` 迁移为对应下标形式。
- 构造器、规范打印、文档示例和错误消息必须只有一种拼写。
- 测试覆盖边界值、bool、零、负数、倒置 Range、非法适用位置、多个 refinement
  合取，以及 launch/Const 违反契约。
- M1-04 已实现唯一构造拼写、稳定去重、明显矛盾检查、规范打印、launch/Const
  契约验证，以及数值区间、整除和实测地址对齐事实传播。
