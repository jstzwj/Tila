# ADR-007：整数运算、转换与索引语义

- 状态：**Accepted**
- 日期：2026-09-19
- 实施状态：M2-01 已实现基础执行语义与保守数值门禁；完整 BitVec/SMT 属 M2-03
- 关联：ADR-005（Const 域）、ADR-011（SMT 与信任来源）

## 三条路径的原有行为与差异

| 项目 | checker / proof 原行为 | CPU 原行为 | Triton 原行为 | 本次处理 |
|---|---|---|---|---|
| 加减乘、取负 | 索引使用数学整数等价与非负事实，没有中间溢出前提 | Python/NumPy 的标量和数组提升不统一 | 有限位宽 tensor 运算，标量 dtype 由实参推断 | 普通值按位宽回绕；索引数学推理必须通过无溢出门禁 |
| 负数 `//`、`%` | DimExpr 使用 Python floor | Python/NumPy floor | runtime tensor 向零取整，与纯 constexpr 不同 | Tila 统一 floor，后端显式修正 |
| 除零、非法移位 | 无完整运算定义域检查 | 异常、warning 或平台相关结果 | 可能出现未定义行为 | 无法验证定义域则拒绝，不能用 bounds warn 绕过 |
| int cast | 只检查类型/shape | 标量和数组窄化路径不同 | 后端转换 | 统一低位保留/符号解释 |
| float→int | 无范围/NaN 门禁 | NumPy 平台转换 | 非有限或不可表示结果不可移植 | 先验证有限且截断后可表示；否则拒绝 |
| scalar ABI | 有声明 dtype | Python 值可能保持无界/float64 | 未声明参数类型，Python float 可按 f32 传递 | CPU 按声明 dtype 绑定；生成显式 Triton 参数类型 |
| shape/grid/stride | 隐式 i32；stride 被当成非负 | grid 可被 int() 截断，shape 无位宽门禁 | 后端推断位宽 | 拒绝非法窄化，允许合法负 stride，地址线性化使用 i64 |
| range/arange | 非零 arange 的坐标事实从零开始 | range 每步转换 i32 | range 最后一次增量可能越过 i32 | 修正 arange 事实，range 内部 i64 归纳、暴露 i32 值 |

GPU 核对还发现动态生成源码无法被 Triton inspect 找回、rank-1 zeros 丢失 tuple
逗号；本次分别注册源码到 linecache、复用 shape tuple 渲染，以执行数值对照。

## 语义决定

### 数学整数与有限位宽值

Const[int] 和纯 staged 的 `+ - * // %` 使用任意精度整数；除零是诊断。
进入 runtime 数值语境必须满足目标 dtype 的表示范围，或使用显式 cast。
纯 staged 条件保留短路求值，不因未执行的除零分支失败。

对宽度 w 的 runtime signed/unsigned 整数，加、减、乘、取负、按位运算及
合法左移保留低 w 位；signed 结果按二进制补码解释。安全 widening 在运算前
执行，结果保持 checker 推导的 dtype；CPU 不依赖 NumPy 的隐式提升。
整数 sum 按返回 dtype 模 2**w 累加，不绕经 float32；浮点归约精度仍归 ADR-008。

### 除法和取模

非零除数下，数学商 `q = floor(a / b)`，余数 `r = a - b*q`；非零余数与 b 同号。
runtime 商转回目标位宽。唯一 signed 商溢出 `MIN // -1` 明确定义为 MIN，
`MIN % -1` 为 0；lowering 在实际除法之前替换这一特殊除数，避免后端溢出行为。

除零没有可执行结果：编译/特化能判定时立即诊断，否则在每次 launch 前验证。
不能证明除数非零时拒绝。逐元素运算是 eager 的，内存 mask 不豁免运算定义域。
数据依赖除数可通过可验证的构造约束范围；本阶段不读取 tensor 内容作宿主检查，
也不承诺理解任意 mask/assume/分支对除数的约束，完整谓词支持留给 M2-02/03。

### 移位

运算前按现有安全 widening 规则确定共同整数 dtype。移位量须在 `[0, w)`；
不隐式对 w 取模。signed 右移补符号位，unsigned 右移补零；左移按位宽回绕。
负移位量、过宽移位量或无法验证的范围都诊断。

### 转换

- int→int：窄化保留低位，widening 按源 signedness 扩展；跨 signedness 的
  显式转换以目标位宽的模运算和符号解释定义。
- staged int→int 显式 cast：先在编译期对目标位宽取模，允许大于 u64 的 Const。
- float→int：向零截断；必须有限且截断后的数学值可表示，不饱和、不回绕，
  NaN/Inf 与超范围都拒绝。验证必须计入源浮点 dtype 的舍入。
- int→float：按目标格式舍入；后续 float→int 必须再次验证，不能因原整数
  可表示而忽略浮点舍入后的越界（如 i64 MAX→f64→i64）。完整低精度/FP8 路线不变。
- bool 转换保持现有显式转换语义，不开放 Mask cast 或隐式算术。

### 索引与 launch 边界

- 当前 shape 各维、grid 各轴、program_id/num_programs、arange/range 暴露的
  整数使用 i32；各维/grid 非负且不超过 2**31-1。stride 可为负，但必须可表示为 i32。
- grid 有 1–3 个轴，直接值/回调返回 exact Python int；cdiv 要求 exact int
  `a >= 0, b > 0`，不接受 bool/float 截断。auto grid 使用相同范围门禁。
- 显式整数标量按声明 dtype 校验，不允许宿主值在 ABI 边界静默溢出。
- Buffer 坐标的 stride 乘加用 i64；绑定时验证线性元素范围不超过 signed i64。
- range 的 start/end/positive step 须可表示为 i32；后端使用 i64 归纳变量，
  每个实际迭代值转换为 i32，避免最后一次步进溢出。arange 事实为 `[start,end)`。

## 保守 proof 迁移

2026-09-20 补充：[ADR-020](020-correctness-closure.md)固定 Ptr 每步/累计字节
位移的 checked i64 地址域，以及实际浮点 ABI 转换后再检查入口精化的规则。

`TBin/TUna` 标记 staged 与 checked_index。对元数据、pid、lane、循环索引、Const
派生的符号索引，保留原始运算节点并逐个验证无溢出；不允许规范化消掉危险中间值。
launch 门禁覆盖所有执行 lane，而不是仅检查最终地址或最终 mask 为真的 lane。

数据依赖/普通显式标量算术不自动继承数学等价或 contiguous 事实；赋值获得
新的符号身份，可用实际结果上的上下界 mask 证明访问。安全整数 widening
cast 保留符号身份，narrowing 丢弃原有数学关系。

本阶段用保守区间门禁，不实现 SMT。不能证明的索引无溢出/运算定义域报
`TILA-NUM-001`；与 bounds 的 strict/warn 独立，unsafe 内存访问也不豁免它。
materialize/explain 显示尚依赖实参/grid 的整数 launch contracts，实际调用每次
复查，包括缓存命中。生成代码必须经 Tila launcher（或履行等价契约的宿主）启动。
因此无 launch 的 bounds 摘要不能被理解为无条件的 GPU 安全证明。

## 兼容性与被拒绝方案

- 保留 Python 风格 floor 除法，不因 Triton 默认行为改变 Tila/DimExpr 语义。
- 不把所有运算都提升到 i64；位宽是语言语义，且 i64 也可能溢出。
- 不把索引算术的回绕结果继续当数学整数用于证明；通过显式 widening 或
  实际结果 mask 表达安全意图。旧程序可能因缺少无溢出依据而被更早拒绝。
- 不采用“所有负数除法都拒绝”或“非法 cast 使用平台结果”的替代规则。
- 不承诺本阶段区间分析完备；更复杂的合法程序可能被拒绝，后续 SMT 增强证明能力。

## 验证与交接

- `tests/test_integer_semantics.py`：全部整数位宽标量/数组 oracle、负数商余数、
  MIN/-1、标量 ABI、转换、移位、grid/shape、循环边界、中间溢出抵消反例、
  widening 正例及不向数据依赖计算泄漏数学事实。
- `tests/gpu_integer_smoke.py`：显式执行的 GPU differential，CPU 默认 pytest
  不收集且不产生 skipped；缺 CUDA 时明确失败。覆盖 i32/i64 商余数、u64 ABI/左移、
  i8 回绕/窄化、signed 右移、f64 标量精度、Const 窄化及 i64 索引/循环边界。
- 一次验证环境：Python 3.11、PyTorch 2.10.0+cu128、Triton 3.6.0、CUDA runtime
  12.8、RTX 3090；这不是 GPU CI 或完整 M3 支持矩阵，其他版本/架构仍待验证。
- M2-02 接手 predicate DAG 与信任来源；M2-03 将保守门禁语义编码到 SMT；
  M2-04 补完整数据流，M2-08/M3 扩充 GPU 用例和持续验证。

执行验证命令：

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python tests/gpu_integer_smoke.py
```

后端适配依据：[Triton 整数除法与转换语义](https://github.com/triton-lang/triton/blob/main/docs/python-api/triton-semantics.rst)。
该文档的 runtime 向零取整与不可移植 float→int 边界均由 Tila 的显式规则处理；
本 ADR 的兼容性承诺以本地回归和上述有限环境证据为准。
