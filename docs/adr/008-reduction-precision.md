# ADR-008：归约输入、累加与输出精度

状态：**Accepted**

日期：2026-09-19；适用 `0.3.0.dev0`，M2-08 实现。

## 决策

`sum(x, axis)` / `max(x, axis)` 接受非空 arithmetic Block；axis 是范围内的
exact Python int 字面量，不接受 bool、负轴或动态轴。删除所归约的轴，最后一个
轴归约后返回标量；输出 dtype 始终与输入相同。`min` 尚未实现。

| 输入 dtype | sum 累加 dtype | max 比较值域 | 输出 dtype |
|---|---|---|---|
| i8/i16/i32/i64、u8/u16/u32/u64 | 输入整数模 2^w 值域 | 输入整数值域 | 输入 dtype |
| f16、bf16 | f32 | 输入浮点值域，可精确加宽比较 | 输入 dtype |
| f32 | f32 | f32 | f32 |
| f64 | f64 | f64 | f64 |

整数 sum 与精确整数求和后按 ADR-007 回绕等价，允许中间回绕；不能依赖
NumPy/Triton 的默认整数提升。CPU 用任意精度整数求和再 wrap，GPU 显式指定
累加 dtype。窄整数 max 的 Triton 结果显式恢复 dtype，避免后续运算偷偷变宽。
浮点 sum 先将输入精确转换至累加 dtype，累加后舍回输出 dtype。
`TReduce.input_dtype / accumulation_dtype / output_dtype` 集中表达上述策略，
CPU 和 lowering 共同消费；目前输入与输出强制相同，尚无用户可选 accumulator 参数。

bool 用 `.any()` / `.all()`，numeric sum/max 报 `TILA-TYPE-028`。
FP8 为 storage-only，必须先显式 cast 到 arithmetic dtype，否则报 `TILA-TYPE-036`。
这些是补齐原有能力域门禁，不开放新的隐式提升。

## 数值与特殊值边界

浮点加法顺序不固定，不承诺 CPU/GPU 位级相等，也不承诺跨 GPU 架构的归约树一致。
sum 遵循所选累加精度的浮点运算：NaN 传播；异号无穷相加为 NaN；允许溢出成无穷。
max **传播任何输入 NaN**，NaN payload 不保证；GPU 使用显式
`tl.maximum(..., propagate_nan=tl.PropagateNan.ALL)` combine，不能使用忽略 NaN 的
默认 tl.max。非 NaN max 返回最大值，包括无穷。sum/max 的零结果不保证符号位。

masked load 的 `other` 是真实参与归约的数据，不自动排除 lane。
sum 通常填零；max 应选择不大于有效数据的下界，默认零不适合全负输入。
空轴归约不在当前契约内；不开放空轴 identity 或 masked reduction 新接口。

## 对照准则

整数 sum 用 Python 任意精度参考并回绕，整数/有限 max 精确相等。
浮点有限 sum 对 CPU 和 GPU 分别检查高精度 `math.fsum` 参考，避免两端共同出错：
对本次无上溢/下溢的样本，以 `gamma(n-1) * sum(abs(x))` 作为累加误差上界，
再加输出舍入误差；`gamma(k)=k*eps_acc/(1-k*eps_acc)`，eps 取 machine epsilon，
是保守界而非统一固定 atol。特殊值单独比较分类，不能用宽容差吞掉 NaN。
一般次正规数/极端溢出顺序的完整跨设备验证继续归 M3，不能从有限样本推导全域支持。

证据：`tests/test_reduction_contract.py`、`tests/gpu_semantics.py`；环境及执行方式见
[ADR-009](009-gpu-validation-baseline.md) 和 [GPU 审计](../m2-gpu-audit.md)。
