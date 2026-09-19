# ADR-015：显式目标 dtype 的浮点常量构造

- 状态：**Proposed**（推荐语法与舍入规则，未实现）
- 日期：2026-09-19
- 阶段：M2-07d；独立于 Const[bool]，不开放 Const[float]
- 关联：ADR-005、ADR-007、ADR-006

## 问题与推荐语法

严格隐式字面量规则拒绝在目标 dtype 中不能精确表示的数值；不能为了方便使用
0.1 而把隐式转换统一改为静默舍入。建议提供显式构造：

<!-- tila-example: future; milestone=M2 -->
```python
scale = ti.constant[ti.f16](0.1)
value = value * scale
```

初版目标仅 f16/bf16/f32/f64。输入限 exact Python int/float 字面量、捕获的模块
数值常量及其一元负号；拒绝 bool、字符串、NumPy scalar、运行时值、Const 参数
以及任意调用/算术表达式。结果是 Stage1Known 的 Scalar[目标 dtype]，不是新的
Const 参数。该 narrow surface 后续可扩展，但不能默默调用 Python 求值器。

## 舍入与边界

唯一模式是 round-to-nearest、ties-to-even；构造动作本身就是显式舍入请求，
不增加字符串 round 参数。int 源按任意精度精确整数处理；float 源按 Python 已
解析的 binary64 位值处理，**不承诺原十进制源码的精确实数**。

直接从源精确值舍入到目标，不经过额外 f32 中转，避免双重舍入。保留负零，支持
次正规数及舍入到带符号零；非有限输入拒绝，目标舍入为无穷也拒绝，不饱和、
不静默产出 inf。NaN payload、FP8、整数目标以及可选舍入模式留给其他设计。

## 内部表示与执行路径

增加 typed constant payload：`dtype + canonical IEEE/bf16 bits + Stage1Known`。
checker 在 Stage 1 计算唯一位模式，CPU 按位恢复；Triton lowering 从整数位模式
显式 bitcast 到目标 dtype。禁止依赖后端再次解析十进制字符串得到相同数值。
若特定目标组合无法表达位模式，必须给出 target 诊断，不退回可能二次舍入的字面量。

规范打印使用 dtype、位模式和便于阅读的数值；语义摘要/编译缓存包含 dtype 与
位模式（+0/-0 不合并），不使用 float 相等性作为唯一键。原始源码仍属于编译键，
不要求不同源码形式的同值常量共享编译产物。该值不提供整数索引或 bounds 事实。

## 实施、诊断与验收

按 ADR-006 登记 intrinsic、公共导出、frontend 下标调用、checker、TIR payload、
CPU/Triton 支持矩阵及语义 revision；常量解析/舍入失败使用 CONST/NUM 族定向
诊断，在开放前登记具体代码。所有普通字面量、cast 以及 Const[int] 规则保持不变。

验收包括：0.1、正负零、ties-even 两侧及相邻值、最小次正规数、下溢、最大有限
值和溢出、巨大整数、非有限拒绝、f64→f16/bf16 的双重舍入反例、位模式缓存隔离。
使用整数/有理数参考产生期望值，对 CPU/GPU 按位比较；bf16 的 NumPy 显示值不
作为唯一舍入 oracle。未通过目标验证前不得宣称对应 GPU 组合支持。

## 未选择的方案

放宽所有字面量破坏可审计的隐式转换契约；把常量先转成 f32 再 cast 可能双重
舍入；使用十进制字符串需要新增语言常量域；开放 Const[float] 会引入参数序列化
及 NaN/零值缓存问题，均不作为这次显式构造的前置扩张。
