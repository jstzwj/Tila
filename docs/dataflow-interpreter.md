# M2-04：数据流与 CPU 解释器

## 分支和提前返回

runtime-if 与 static-if 只合并能到达后续语句的前驱。单侧 `return` 时保留另一侧的
定义、类型和路径条件；双侧返回后停止检查后续语句。嵌套的可能未定义状态按 OR
合并，分支相关字面量不再误作为常量折叠。

两侧整数表达式不同时创建新的 phi 身份，并在路径 DAG 中记录条件等式
`(condition ∧ phi == then_value) ∨ (¬condition ∧ phi == else_value)`。
phi 本身不运算，使用数学 Int 选择已经满足 dtype 的值；后续算术仍遵守 ADR-007
的回绕规则。布尔值同样按条件选择，其他事实只保留交集。终止路径不参与类型合并。

## 循环的保守不动点

归纳变量是循环局部、使用新名字，不允许遮蔽已有变量；范围为 `start <= i < end`，
step 必须为正编译期整数。start/end 接受整数标量或字面量。循环局部不能逃逸。
循环携带变量维持 dtype/shape，`+=` 明确拒绝并提示改写为 `x = x + value`。

当前值域采用一个小型保守不动点：未赋值、所有更新均为自赋值、或所有更新均写回
初始同值字面量时保留不变量；其余携带变量在入口和出口扩大为新的未知身份。
扫描包含嵌套分支和嵌套循环，归纳变量绑定也算赋值。初始值或一次迭代的
`assume` 不能充当所有迭代的不变量。尚不推导递增计数器、仿射递推或一般区间不变量。

字面量空循环保留全部初始值；符号循环出口记录零次执行时携带值不变的条件关系，
可在 Const 特化为零时消除不必要的 Unknown。循环体事实不直接泄漏到出口。
循环体必然返回时，后续语句只由零次执行边到达；确定非空则后续代码不可达。
`return` 退出整个 program instance，不是 continue。数值定义域校验也遵守该终止行为，
不执行实际空循环的体内运算检查。

## SAT 候选的确认范围

首个内存访问若只依赖显式整数标量、已绑定常量以及精确有限位宽定义，且路径和 mask
不存在未知布尔或 UserAssumption，可将 SAT 确认为 ProvenUnsafe。隐式 shape/stride、
lane、加载值、phi 代理、循环中及循环后的抽象值仍保守返回 Unknown；前序内存访问
之后也不宣称已证明完整执行可达。缓存键包含输入来源和执行上下文精确性。

这不是通用执行器：测试从 SMT 模型取得负索引，在 CPU debug 路径实际重放越界；
另外执行循环第二次迭代越界反例，确认该案例虽可重现，证明器仍如实报告 Unknown。

## CPU 内存与数值行为

- load/store 先广播坐标与 mask，再仅选择 active lanes 索引；不再裁剪坐标。
  空数组的全 false mask、负数或极大 uint64 的 inactive 索引均不访问内存。
- active 越界在 debug 抛 AssertionError，非 debug 抛 IndexError；`unsafe` 和
  `--safety=warn` 只影响静态证明，不能将 CPU 实际越界变成合法读写。
- 坐标索引保留 NumPy 切片、转置、负 stride、只读零 stride 视图。torch CPU
  非连续视图共享原存储；bf16 经同宽 uint16/NumPy bf16 视图桥接，写入不是副本。
- masked other 和 where 分支遵守已检查 dtype；标量 where 返回标量。浮点二元
  运算按 checker 的共同 dtype 执行；f64 sum 使用 f64，f16/bf16 sum 用 f32
  累加再恢复元素 dtype；整数运算及归约沿用 ADR-007。
- debug 下 `assume` 逐 lane 检查，发生在后续访问之前。关闭 debug 不运行该断言，
  但解释器的 active bounds 检查仍保留。

CPU 参考实现不承诺浮点归约与 GPU 树形顺序逐位相同，也不新增 FP8 运算、任意
共享存储重叠写入的确定性或通用循环求解。系统性质测试归 M2-05，更广 GPU 对照归 M2-08。

验证入口：`tests/test_dataflow_interp.py`、`tests/test_proof_model.py`、
`tests/test_smt_solver.py`；GPU 整数回归为 `tests/gpu_integer_smoke.py`。
