# Tila 实现状态

状态：当前工作区的唯一能力状态清单

基线日期：2026-09-19

对应版本：`0.3.0.dev0` 开发基线（未发布正式 0.3.0；不回移 Const bool 至 0.2.x）

验证基线：`PYTHONPATH=src python -m pytest -q` = 877 passed、零 skipped

M3-01 固定环境已从 `ci/gpu/uv.lock` 重建；专用 GPU runner 和 workflow 已配置，
当前覆盖 112 个 GPU 测试节点/256 个语义案例，初始支持仅 RTX 3090/SM86。
默认分支定时验收须合并验证分支后启用；操作与边界见 [GPU 支持](gpu-support.md)。

M3-02 已完成 grid/零启动门禁、集中 target policy、lowering 前结构 verifier、
源码/ABI/布局缓存指纹和 hint/alignment 负测试，见 [Launch 与 target](launch-target.md)。
本阶段已通过本地严格验收；不将此前远端 CI 结果视为当前修改的 CI 证据。

M2-08 已完成 [固定环境 GPU 审计](m2-gpu-audit.md)：70 个 pytest 节点、214 个语义
案例通过，零 skipped。统一命令 `PYTHONPATH=src python tools/gpu_audit.py`，
记录环境、源码、失败数据与重放命令；版本不符/无 GPU 不可验收。
[ADR-008](adr/008-reduction-precision.md) 固定归约输入/累加/输出精度及 NaN 传播，
[ADR-009](adr/009-gpu-validation-baseline.md) 固定本地组合。已修复宽无符号
PyTorch dtype 绑定、窄整数 max 恢复、浮点 max NaN 一致性与 bool/FP8 归约门禁。
跨架构/版本支持仍未验证，归 M3 后续。

2026-09-19 设计更新：[ADR-011](adr/011-smt-proof-and-trust.md) 接受 Z3 默认
通用证明引擎、布尔 DAG、整数编码与信任来源分离。M2-01 已实现 ADR-007 的
基础整数语义与 launch 门禁；M2-02 已实现 DAG、ProofResult、信任来源及作用域
隔离。M2-03 已接入默认 Z3、Int/BitVec、预算与有界进程内 proof 缓存，
详见 [证明器实现边界](smt-prover.md)。M2-04 已收口活跃分支合并、简单循环不变量、
零次循环出口和 CPU 内存访问语义，边界见 [数据流与解释器](dataflow-interpreter.md)。
M2-05 已补小位宽穷举、性质测试、差异审计和失败重放，覆盖范围见
[证明审计](m2-proof-audit.md)。M2-06 已固定 [audit explain v1](explain-audit.md)、
CLI/关键错误 golden、反例与重放附件边界。ExactInt 边界不变。

M2-07 已形成四份独立提案：[布尔 tile/Mask](adr/012-boolean-tile-mask.md)、
[Const bool](adr/013-const-bool-domain.md)、[宿主整数转换](adr/014-host-integer-normalization.md)、
[显式舍入常量](adr/015-rounded-typed-constants.md)。M2-07a/b/c/d 均已实现。
`constant[dtype](value)` 是显式 RNE 浮点常量 intrinsic；`host_int` 为宿主端显式整数转换，
39 项专项覆盖具体类型白名单、无损值、拒绝路径、使用点范围检查及缓存等价。
Const[bool] 自 0.3.0.dev0 生效；ExactInt 不放宽，布尔绑定与整数事实/缓存域分离。
47 项 Const bool 专项覆盖 exact 绑定、staging/短路、分支/循环、CLI、缓存与审计快照。
11 组 Const bool CPU/GPU 对照通过同下述 GPU 环境（含短路跳过除零），命令为
`PYTHONPATH=src python tests/gpu_const_bool_smoke.py`。

显式浮点常量的 39 项专项与两份 golden 覆盖直接舍入、位模式、类型/表达式门禁及缓存。
46 组 CPU/GPU 按位对照通过同下述 GPU 环境（f16/bf16/f32/f64），命令为
`PYTHONPATH=src python tests/gpu_constant_smoke.py`；普通字面量与 Const 参数域不放宽。

布尔 tile 的 39 项专项包含身份/广播、分支/循环、Ptr、debug assume 与 explain golden。
68 组 CPU/GPU 对照已通过 RTX 3090 + PyTorch 2.10.0+cu128 / Triton 3.6.0 / CUDA 12.8，
命令为 `PYTHONPATH=src python tests/gpu_boolean_smoke.py`；不替代正式 GPU 支持矩阵。

本文回答一个问题：**当前代码究竟支持什么？** 设计目标和未来排期分别见
`design-principles.md` 与 `../plan.md`；M1 冻结项的逐项证据见
[`m1-exit-audit.md`](m1-exit-audit.md)。当其他文档的阶段描述与本文冲突时，
实现状态以本文为准；语言语义仍以各规范文档为准。

<!-- public-api: jit,assume_launch,cdiv,host_int,Dim,bool,i8,i16,i32,i64,u8,u16,u32,u64,f8e4m3fn,f8e5m2,f16,bf16,f32,f64,Buffer,Ptr,ReadPtr,WritePtr,RWPtr,Const,ReadOnly,WriteOnly,ReadWrite,Global,Shared,Local,PowerOfTwo,Positive,NonNegative,Range,MultipleOf,Aligned,program_id,num_programs,arange,range,load,store,unsafe_load,unsafe_store,cast,constant,where,dot,zeros,sum,max,exp,exp2,assume,static_assert,byte_offset,reshape -->
<!-- frontend-intrinsics: program_id,num_programs,arange,range,load,store,unsafe_load,unsafe_store,cast,constant,where,dot,zeros,sum,max,exp,exp2,assume,static_assert,byte_offset,reshape -->

---

## 1. 状态定义

每项能力只能有一个当前状态：

| 状态 | 含义 |
|---|---|
| `Implemented` | 公共语法、checker、相关执行/生成路径和测试已经闭环 |
| `Partial` | 已有可用子集，但公共 API、语义、后端或测试至少有一项明显缺失 |
| `Designed` | 规范或路线已经定义，当前代码尚未提供可用实现 |
| `Deferred` | 已明确不在近期支持范围；可能有定向拒绝诊断 |
| `Removed` | 旧版本曾支持或设计，当前基线已移除且不再有效 |

后端列使用：

- `Check`：可被 frontend/checker 静态处理；
- `Host`：仅 Python 宿主端执行，不属于 kernel intrinsic；
- `Specialize`：Const 代入后、生成/执行前复查；
- `Launch`：按每次实际参数和 grid 检查；
- `CPU`：reference interpreter 可执行；
- `Triton`：可生成 Triton 源码；
- `GPU verified`：已在固定 CUDA/Triton 环境持续执行验证。

当前 GPU CI 在验证分支建立，默认分支持续验收尚待合并。暂不批量升级为
`GPU verified`；已验证的操作、dtype、形状范围以 gpu-support.md 为准。

M2-01 新增可显式运行的 `tests/gpu_integer_smoke.py`，已在 RTX 3090 /
PyTorch 2.10.0+cu128 / Triton 3.6.0 上通过 23 组整数相关 CPU/GPU 对照。
这是一套有限的本地证据，不是完整支持矩阵或持续 GPU CI。

---

## 2. 编译与执行管线

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| `@tila.jit` 源码解析 | `Implemented` | Check | `inspect.getsource + ast.parse`；不执行 kernel 函数体 |
| HIR 与 typed TIR | `Implemented` | Check | frontend、checker、canonical TIR dump 已有测试 |
| Stage 1 定义期检查 | `Implemented` | Check | 装饰函数时完成子集、类型、shape、capability 和 obligation 生成 |
| Stage 2 Const 特化 | `Implemented` | Check | 默认值/CLI const/launch const、延迟 shape 约束和 bounds 求值 |
| NumPy reference interpreter | `Implemented` | CPU | add、matmul、attention、控制流、Ptr 1D 等路径有测试 |
| Triton 源码 lowering | `Partial` | Triton | 已有结构/静态 target verifier，未知节点明确失败；编译后资源检查和完整 source map 待续 |
| CUDA 自动后端选择 | `Partial` | Triton | 集中检查同设备、RTX 3090/SM86、Triton 3.6.0、grid/tile/dot 限制；完整组合由严格审计认证 |
| 特化缓存 | `Implemented` | Triton | 进程内 JIT callable 缓存含源码、Const/位模式、debug、target UUID/软件版本、num_warps、dtype/布局/alignment；不承诺持久化或容量上限 |
| Source map/后端错误回映射 | `Designed` | — | 尚无 Tila 源位置到 Triton 编译错误的完整映射 |
| 多后端/直接 PTX | `Deferred` | — | Triton 是当前唯一计划后端 |

---

## 3. 公共宿主 API 与 Launch

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| `tila.jit` | `Implemented` | Check | 仅普通顶层函数；依赖可取回源码 |
| `kernel[grid](...)` | `Implemented` | CPU / Triton | 支持至多三个 grid 轴 |
| `kernel.launch_auto(...)` | `Implemented` | CPU / Triton | 从 bounds obligation 的 `pid*STEP+lane`/裸 pid 模式推导 |
| `tila.cdiv(a, b)` grid 标记 | `Implemented` | Launch | 同时提供数值 ceil-div 与 grid contract 识别 |
| `tila.host_int(value)` | `Implemented` | Host | exact Python int 与八种 NumPy 整数具体类型；拒绝 bool/数组/用户子类，保留数学值，使用点独立检查范围；不允许在 kernel 内调用 |
| `@tila.assume_launch(...)` | `Partial` | Launch | 支持受限表达式和运行时检查；装饰器接口当前接收字符串，规范示例仍需统一 |
| dtype/shape/stride/alignment 绑定 | `Implemented` | Launch | NumPy、torch CPU/CUDA 基本绑定；stride 单位为元素 |
| 多 tensor device 一致性 | `Designed` | — | 当前只用第一个 tensor 决定后端，尚未统一验证所有设备 |
| `num_warps` launch option | `Partial` | Triton | 当前固定为 4，尚无类型化用户参数；不能宣称支持 `{1,2,4,8}` 全集合 |
| 固定 CUDA/Triton 支持矩阵 | `Designed` | — | 依赖未锁定，尚无 compute capability 表 |
| CLI `check/build/run/dump/explain` | `Implemented` | Check / CPU | Windows UTF-8/cp1252 回归已覆盖；`run` 执行文件自身的 `main()` |

---

## 4. 类型和值类别

### 4.1 DType

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| `bool` | `Implemented` | Check / CPU / Triton | scalar bool 与 Mask 分离 |
| `i8/i16/i32/i64` | `Implemented` | Check / CPU / Triton | widening、字面量范围和整数运算有矩阵测试 |
| `u8/u16/u32/u64` | `Implemented` | Check / CPU / Triton | signed/unsigned 隐式混算拒绝 |
| `f16/f32/f64` | `Implemented` | Check / CPU / Triton | 基础算术、cast、load/store；真实 GPU 组合未验证 |
| `bf16` 基础值语义 | `Implemented` | Check / CPU / Triton | `ml_dtypes` 为 dev/interp 依赖；masked load 和加法有端到端测试 |
| `f8e4m3fn/f8e5m2` storage gate | `Partial` | Check / Triton | 可声明、load/store/cast，直接算术会拒绝；CPU storage 和真实 GPU/FP8 dot 未闭环 |
| FP8 完整 load→compute→store | `Designed` | — | 目标能力、舍入和 GPU 验证待 M5 |
| dtype capability 公共对象（`Float` 等） | `Designed` | — | 当前只有内部 tuple，没有公共 `tila.Float`/`DotInput` 类型 |

### 4.2 类型构造

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| Scalar 参数 | `Implemented` | Check / CPU / Triton | dtype 注解形成运行期标量 |
| `Block[T, Shape]` 内部类型 | `Implemented` | Check / CPU / Triton | 用户不直接声明 Block 参数，由内建推导 |
| `Mask[Shape]` | `Implemented` | Check / CPU / Triton | 独立于 Block bool，携带共享 DAG 谓词用于 bounds |
| `Unit` | `Implemented` | Check | store/assume 等返回；赋值和参与运算会拒绝 |
| `Const[int]` | `Implemented` | Check / CPU / Triton | 支持默认值、特化和 refinement；ADR-005 将其固定为 0.2.x 唯一公共 Const 参数域 |
| `Const[int]` ExactInt 全入口门禁 | `Implemented` | Frontend / Specialize / Launch | 默认值、materialize/explain、CLI 和 launch override 均只接受 exact Python int；bool、float、字符串及 NumPy integer 拒绝 |
| `Const[bool]` | `Implemented` | Frontend / Specialize / Launch / CPU / Triton | ADR-013；exact Python bool，CLI true/false；不接受 0/1、NumPy bool 或 refinement；支持 staged 布尔表达式与类型标签缓存 |
| 其他数值 Const 参数 | `Designed` | — | Const[float/dtype/str] 不开放 |
| `Buffer[T, Shape, Access, Alignment?]` | `Implemented` | Check / CPU / Triton | 2–4 项公共形式规范化为四项打印；隐式 Global；内部显式区分 `BoundStrides/UnboundStrides`，不能由用户伪造 |
| `Ptr[T, AddressSpace, Access, Extent, Alignment]` 与紧凑形式 | `Implemented` | Check / CPU / Triton | ADR-001 五参数完整形式和 1–4 项紧凑形式统一规范化；Access/AddressSpace 为受控 enum，Extent/Alignment 为结构化类型；v0 只接受 Global |
| `ReadPtr/WritePtr/RWPtr[T, Extent?, Alignment?]` | `Implemented` | Check / CPU / Triton | 可携带 Extent/Alignment；省略 Extent 时为 UnknownExtent，strict bounds 下需要额外证明或 unsafe |
| 编译器管理的 `RegionId` 与 alias relation | `Implemented` | Check / Launch | `ParamRegion/BufferRegion/InternalRegion/UnknownRegion` 结构化；派生指针继承；alias 独立为 `MustAlias/MayAlias/NoAlias`，外部参数静态保守、launch 按实际存储区间细化 |
| `Global` address space | `Implemented` | Check / CPU / Triton | 当前 Ptr/Buffer 实际恒为 Global |
| `Shared/Local` address space | `Designed` | — | 名字已导出，但公共构造器和后端没有真正支持，不能视为可用能力 |
| `TypeVar` 泛型 | `Designed` | — | `tila.TypeVar`/能力 bound 均不存在；现有规范示例是未来设计 |
| Layout 用户类型 | `Deferred` | — | 当前不进入 `Block` 公共语法 |

---

## 5. Shape、DimExpr 与转换

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| `tila.Dim(name)` | `Implemented` | Check / Launch | shape 绑定与同名显式标量一致性有检查 |
| 仿射 `DimExpr` | `Implemented` | Check / Launch | `+ - * // % ceildiv` 的受限规范化/求值 |
| `min/max` DimExpr | `Partial` | Check | 内部节点与 Z3 编码已存在；公共构造及完整规范化尚未闭环 |
| shape 等价与延迟 Const 约束 | `Implemented` | Check | dot/broadcast/store/reshape 的 Const-only 差异可延迟到 Stage 2 |
| trailing-dimension broadcast | `Implemented` | Check / CPU / Triton | 相等或 size-1；不合法时报 shape 错误 |
| scalar→Block broadcast | `Implemented` | Check / CPU / Triton | 按内建/运算语境广播 |
| 隐式安全 widening | `Implemented` | Check | 有符号、无符号、浮点链分离；int/float 混算拒绝 |
| 语境化字面量 | `Implemented` | Check | 整数范围、f16/bf16 精确可表示性有测试 |
| 显式 `cast[U](x)` | `Implemented` | Check / CPU / Triton | 保持 shape；不支持 Mask cast |
| 显式 `constant[U](value)` | `Implemented` | Frontend / Check / CPU / Triton | 仅 f16/bf16/f32/f64；原生 int/float 字面量或模块常量及其一元负号，直接 RNE，保留零符号/次正规数，拒绝非有限与溢出；按位 payload |
| `cast(x, U)` 函数形式 | `Designed` | — | 文档曾称为同义语法，frontend 实际不支持 |

---

## 6. Refinement 与事实

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| `PowerOfTwo` | `Implemented` | Check / Launch | Const 特化期校验 |
| `Positive/NonNegative` | `Implemented` | Check / Launch | scalar launch 校验并进入部分 bounds 事实 |
| `Range[lo, hi]` | `Implemented` | Check / Launch | 闭区间；整数 literal 端点且 lo <= hi；旧调用语法拒绝；成功契约注入区间事实 |
| `MultipleOf[k]` refinement | `Implemented` | Check / Launch | k 为正整数；仅整数值；成功契约注入 `value % k == 0` 事实 |
| `Aligned[k]` | `Implemented` | Check / Launch | k 为正的 2 次幂字节数；仅用于 Buffer/Ptr Alignment；实测通过后记录 RegionId 对齐事实 |
| contiguous fact | `Implemented` | Check / Triton | 基于 arange 的事实生成 `tl.max_contiguous` |
| multiple-of lowering fact | `Implemented` | Check / Triton | `pid*STEP + arange` 拆基并发射 `tl.multiple_of`，有 golden/负例测试 |
| 内部 `StrideEq` / `Contiguous` facts | `Designed` | — | ADR-003 明确不提供公共构造；launch 推导与性能分析消费路径尚未闭环 |
| fact provenance | `Partial` | Check / Triton | ProofResult 记录信任来源；两类已发射 hint 均展示结构性依据/位置；不承诺最小证明或全局事实推导图 |

---

## 7. 控制流与 Python 子集

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| 普通赋值 | `Implemented` | Check / CPU / Triton | 不支持解构赋值和任意 Python 对象 |
| runtime `if/elif/else` | `Implemented` | Check / CPU / Triton | scalar bool；只合并存活前驱，确定赋值与整数 phi 条件等式 |
| 模块常量 constexpr-if | `Implemented` | Check | Stage 1 直接折叠 |
| Const 参数 static-if | `Implemented` | Check / CPU / Triton | TStaticIf 延迟到特化；variant 使用点诊断已覆盖 |
| `for i in tila.range(...)` | `Implemented` | Check / CPU / Triton | step 为正 Const，start/end 支持整数标量/字面量；归纳变量新名字，简单不变量与零次出口关系 |
| loop-carried 类型稳定性 | `Implemented` | Check | dtype/shape 漂移拒绝 |
| 裸 `return` | `Implemented` | Check / CPU / Triton | 支持提前结束当前 program instance |
| 值 `return` | `Deferred` | Check | 定向拒绝；kernel 结果写入输出 Buffer |
| `while`、`break/continue` | `Deferred` | Check | 不属于当前 Python 子集 |
| `+=` 显式累加器 | `Deferred` | Check | 定向拒绝，提示改写为普通赋值 |
| 闭包、嵌套函数、任意 Python 调用 | `Deferred` | Check | 静态子集定向拒绝 |

---

## 8. Intrinsic 状态

`src/tila/intrinsics.py` 是 ADR-006 定义的机器可读 catalog；公共导出、表面形式、
checker handler、effect/bounds、可达 TIR、backend expectation、target 与下表状态键均由
完整性测试关联。以下能力细分表仍是用户边界的事实来源，不能只从名字已导出推断实现完成。

### 8.0 Registry 状态索引

此表为每个 registry spec 提供唯一状态键；状态必须与 `IntrinsicSpec.availability` 一致。

| Registry 状态键 | 状态 | 表面边界 |
|---|---|---|
| `intrinsic:program_id` | `Implemented` | public call |
| `intrinsic:num_programs` | `Implemented` | public call |
| `intrinsic:arange` | `Implemented` | public call |
| `intrinsic:range` | `Implemented` | loop form；普通 call 定向拒绝 |
| `intrinsic:load` | `Partial` | Buffer 闭环；Ptr 能力仍有边界 |
| `intrinsic:store` | `Partial` | Buffer 闭环；Ptr 能力仍有边界 |
| `intrinsic:unsafe_load` | `Implemented` | public call |
| `intrinsic:unsafe_store` | `Implemented` | public call |
| `intrinsic:cast` | `Implemented` | subscript call |
| `intrinsic:constant` | `Implemented` | subscript call；exact int/float 源，显式 RNE 舍入 |
| `intrinsic:where` | `Implemented` | public call |
| `intrinsic:dot` | `Implemented` | v0 f16 signature |
| `intrinsic:zeros` | `Implemented` | public call |
| `intrinsic:sum` | `Implemented` | public call |
| `intrinsic:max` | `Implemented` | public call |
| `intrinsic:exp` | `Implemented` | public call |
| `intrinsic:exp2` | `Implemented` | public call |
| `intrinsic:assume` | `Implemented` | public call |
| `intrinsic:static_assert` | `Implemented` | Stage 1 或 Const specialization staged bool |
| `intrinsic:byte_offset` | `Deferred` | public reserved name；定向拒绝 |
| `intrinsic:reshape` | `Implemented` | public call |
| `intrinsic:any` | `Implemented` | method only |
| `intrinsic:all` | `Implemented` | method only |

### 8.1 索引、构造与形状

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| `program_id(axis)` | `Implemented` | Check / CPU / Triton | axis 为字面量 0..2 |
| `num_programs(axis)` | `Implemented` | Check / CPU / Triton | axis 为字面量 0..2 |
| `arange(start, end)` | `Implemented` | Check / CPU / Triton | start 当前必须是字面量；end 是 Const 语境 |
| `range(start, end, step)` | `Implemented` | Check / CPU / Triton | 仅用于 for；运行期 int start 已支持 |
| `zeros(shape, dtype)` | `Implemented` | Check / CPU / Triton | shape 每维为 Const 语境 |
| `full(shape, value, dtype)` | `Deferred` | — | 尚未实现 |
| `reshape(x, shape)` | `Implemented` | Check / CPU / Triton | 目标维为 Const；numel 可即时或特化期证明 |
| `x[:, None]`/`x[None, :]` expand | `Implemented` | Check / CPU / Triton | 仅这两种下标语法糖 |
| `trans` | `Deferred` | — | 当前示例用坐标顺序表达转置访问 |
| `cat` | `Deferred` | — | 尚未实现 |

### 8.2 内存访问

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| Buffer `load` | `Implemented` | Check / CPU / Triton | 逐轴坐标、mask、other 与 bounds obligation |
| Buffer `store` | `Implemented` | Check / CPU / Triton | value dtype/shape 精确匹配；capability 检查 |
| Ptr `load/store` | `Partial` | Check / CPU / Triton | 裸 Ptr 与合法 `buf.ptr` 采用 flat 元素语义；减法、byte offset 与更完整 pointer arithmetic 尚未实现 |
| `unsafe_load/unsafe_store` | `Implemented` | Check / CPU / Triton | 只豁免当前 obligation，不产生事实；进入审计输出 |
| `buf.ptr` | `Implemented` | Check / Launch / CPU / Triton | v0 仅允许 rank-1、stride-1；多维在 Stage 1、非单位 stride 在 launch 明确拒绝；继承 BufferRegion、Extent、Access、Alignment |
| `p + element_offset` | `Implemented` | Check / CPU / Triton | offset 必须为整数，继承 pointer capability |
| `p - element_offset` | `Deferred` | — | 已从 v0 规范语法移除；未来需先定义负 offset 与 bounds 规范化 |
| `byte_offset` | `Deferred` | Check | 名字存在但调用固定报 `TILA-SYN-050` |

### 8.3 数值、选择与归约

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| 算术 `+ - * / // %` | `Implemented` | Check / CPU / Triton | `/` 仅 Float，`// %` 仅 Int；dtype 转换严格 |
| ADR-007 基础整数语义与数值门禁 | `Implemented` | Check / Specialize / Launch / CPU / Triton | runtime 回绕、floor 商余数、MIN/-1、移位/转换定义域；每次验证中间索引溢出及 i32 ABI，未知数据域保守拒绝 |
| 整数位运算/移位 | `Implemented` | Check / CPU / Triton | 操作数须为兼容 Int dtype |
| 比较与 Mask 谓词 | `Implemented` | Check / CPU / Triton | Block 比较生成 Mask 与可提取谓词 |
| Mask `& | ~` | `Implemented` | Check / CPU / Triton | 可混合 Block bool 与 scalar bool；tile 返回 Mask，DAG 保留未知身份与相对 lane 轴 |
| bool tile 消费 | `Implemented` | Check / CPU / Triton | load/store mask、where 支持 Block bool；不自动取得 bounds 事实，不合并 Mask/Block 类型 |
| `mask.any()/mask.all()` | `Implemented` | Check / CPU / Triton | Mask/Block bool，仅方法形式，归约为 scalar bool |
| `where` | `Implemented` | Check / CPU / Triton | eager 两侧；dtype 必须一致，shape 可广播 |
| `dot` f16 输入 | `Implemented` | Check / CPU / Triton | rank-2，acc 支持 f16/f32；固定 RTX 3090 示例 GPU 对照覆盖 f32 acc，完整矩阵待续 |
| `dot` bf16/FP8 输入 | `Designed` | — | 当前 `DOT_INPUT` 仅 f16 |
| `sum/max` | `Implemented` | Check / CPU / Triton | ADR-008：exact int axis；显式累加与输出 dtype、NaN 传播；12 dtype 双轴 GPU 对照；bool/FP8 先拒绝或 cast |
| `min` reduction | `Deferred` | — | 尚未实现 |
| `exp` | `Implemented` | Check / CPU / Triton | Float 域逐元素 |
| `exp2` | `Implemented` | Check / CPU / Triton | 公共占位符、frontend、checker、interpreter 与 lowering 已对齐 |
| `log/sqrt/rsqrt/abs/floor/ceil` | `Deferred` | — | 尚未实现 |

### 8.4 断言与提示

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| `assume(pred)` | `Implemented` | Check / CPU / Triton | 只接受可符号化合取；debug interpreter/device assert 路径存在 |
| `static_assert(pred)` | `Implemented` | Check / Specialize / Launch | Stage 1 谓词立即检查；只依赖 Const 参数的 staged bool 每次 specialization 复查，保留 `and/or` 短路语义 |
| `static_assert(pred, msg)` | `Designed` | — | Python 子集不接受字符串，消息形式尚未实现 |
| 手工 multiple-of/max-contiguous hint | `Deferred` | — | 不暴露隐形扁平入口；未来若加入，仅采用经过设计的 `tila.hint.*` 表面 API |
| 自动 optimization hint | `Partial` | Triton | contiguous/multiple-of 已有；alignment 与完整 provenance 尚缺 |

---

## 9. Bounds、Effect 与并发

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| load/store proof obligation | `Implemented` | Check / Launch | Buffer 逐轴、Ptr 单 extent；访问点保存谓词快照 |
| interval/nonnegative fast path | `Implemented` | Check | 默认 Z3 前的小型充分条件捷径；不绕过缺失依赖诊断 |
| predicate DAG | `Implemented` | Check | 无 DNF 展开；共享子式、未知布尔身份、路径、源位置及 lane/broadcast 映射；支持 Z3 否定/析取蕴含 |
| grid cdiv/exact-dim contract | `Implemented` | Launch | 显式 grid 和 launch_auto 均可登记事实 |
| 四态结论 | `Implemented` | Check / Launch | ProvenSafe、ProvenUnsafe、Unknown、Exempted；SafeUnderContract 仅为已检查契约的显示摘要 |
| `--safety strict/warn` | `Implemented` | CLI / Launch | Unknown 可降 warning；ProvenUnsafe 永远 error |
| 默认 Z3 通用证明引擎 | `Implemented` | Check / Specialize / Launch | 锁定 z3-solver 4.16.0.0；统一结果/协议、可重放查询；缺失时明确配置错误 |
| 否定谓词 SMT 编码 | `Implemented` | Check / Specialize / Launch | And/Or/Not 共享编码；复杂查询可因预算返回 Unknown |
| 有限位宽 proof 与执行语义对齐 | `Partial` | Specialize / Launch | Int/BitVec 编码覆盖回绕、floor 商余、整数 cast、位运算/移位；加载内容/复杂数据流仍近似，浮点不进入 SMT |
| ProofResult 信任来源与 Exempted | `Implemented` | Check / Launch | 不可变结果、来源集合/位置/trace；assume 不泄漏作用域，unsafe 局部豁免；符号 grid 仅 pending，launch 每次重验 |
| SMT 预算与证明缓存 | `Implemented` | Check / Specialize / Launch | timeout/rlimit、kernel 时间/查询数、构建大小；来源/版本/绑定隔离的有界 LRU，Unknown 不缓存；launch 契约仍逐次重验 |
| SAT 反例可达性 | `Partial` | Check / Launch | 首次访问的精确整数标量/常量/条件路径可确认；加载、phi、lane、循环及前序内存效果仍为 Unknown 候选 |
| 布尔 tile 作为执行 mask | `Designed` | — | 与是否携带边界谓词分开；待独立接口设计，不自动开放当前 API |
| kernel effect 汇总 | `Implemented` | Check | `Read/Write[region]` 出现在 report/explain；当前为聚合列表 |
| per-instruction effect IR | `Designed` | — | TLoad/TStore 尚无统一原生 effect 字段 |
| `where` eager memory warning | `Implemented` | Check | `TILA-EFFECT-007` 已生成；完整 effect/并发系统归 M4 |
| alias 声明/运行时 alias 检查 | `Designed` | — | `tila.alias`、`--check-alias` 尚不存在 |
| atomic | `Designed` | — | 无公共名字、checker、TIR 或后端实现 |
| inter-program race analysis | `Designed` | — | 尚未实现 |
| uniformity/barrier | `Designed` | — | 尚未实现 |
| shared-memory typestate | `Deferred` | — | 长期研究项 |

---

## 10. 诊断、审计与测试覆盖

| 能力 | 状态 | 覆盖 | 当前边界/证据 |
|---|---|---|---|
| 机器可读诊断 registry | `Implemented` | Host / Frontend / Check / Specialize / Launch | 所有当前发射 code 有唯一 family、phase、severity、summary 和默认修复建议；源码/registry 完整性测试双向对齐 |
| 统一诊断渲染 | `Implemented` | CLI / Runtime | error/launch error/warning 均输出 code、location、phase、details 和 fix；未知内部异常仅在 `TILA_DEBUG=1` 暴露 traceback |
| `TILA-SYN/TYPE/SHAPE/CONST/MEM/BOUNDS/EFFECT` | `Implemented` | Check / Launch | 当前错误码目录见 `docs/diagnostics.md`；兼容性变更必须同步 registry 与测试 |
| `TILA-TARGET` 完整诊断族 | `Partial` | Launch | 缺 triton/CUDA tensor 有诊断；完整硬件 capability 尚无实现 |
| `TILA-RACE/UNIFORM` | `Designed` | — | 错误码只存在于设计文档 |
| `explain` 类型/事实/effect/obligation | `Implemented` | CLI | audit explain v1 固定章节/状态字段；CLI/诊断 golden、只读与跨 hash seed 回归 |
| proof trace | `Partial` | CLI | fast/SMT 路线与信任来源统一展示，原始查询可选重放；非完整 Z3 proof/minimal core |
| add TIR/Triton golden | `Implemented` | Test | 逐字节比较 |
| matmul/attention/fused-attention golden | `Designed` | — | 示例有 CPU smoke，但尚无 TIR/Triton/explain golden |
| 官方示例 CPU smoke | `Implemented` | CPU | 五个示例以 subprocess 运行，Windows cp1252 场景有回归 |
| GPU differential | `Partial` | CPU / Triton | 固定环境 112 节点/256 案例，五个官方示例多配置；默认分支调度待合并、完整 dtype/target 矩阵仍待 M3 |
| property/fuzz tests | `Implemented` | Test | M2-05 小位宽有界穷举、固定种子变形/执行对照和缓存/预算隔离；不是全输入空间证明 |

---

## 11. 已移除的旧基线

以下属于 2026-08 旧实现，已移动到 `trash/`，不能作为当前能力引用：

| 能力 | 状态 | 替代/说明 |
|---|---|---|
| `.tila` 独立源文件语法 | `Removed` | 当前使用 Python 子集与 `@tila.jit` |
| DistExpr layout 代数 | `Removed` | 当前 layout 不进入公共类型；未来重新设计 |
| 坐标寻址一级原语旧 API | `Removed` | 当前使用 Buffer 坐标或 Ptr 元素 offset |
| E01–E21/B 旧诊断码 | `Removed` | 当前使用 `TILA-<FAMILY>-NNN` |
| 旧 `src/tila2` 迁移设想 | `Removed` | 新实现已经直接位于 `src/tila` |

---

## 12. 如何维护本文

任何公共能力变更必须在同一变更集中更新本文：

1. 新增名字：加入对应表格，并给出唯一状态；
2. `Designed → Partial`：列出已经闭环和仍缺失的路径；
3. `Partial → Implemented`：必须满足 `plan.md` 的完成定义；
4. 删除能力：标为 `Removed` 并给迁移路径；
5. 改动 `tila.__all__` 或 frontend intrinsic allowlist：同步更新本文顶部的机器可检验清单；
6. 测试基线变化：更新页首数字，但不以测试数量代替语义覆盖说明。

状态更新不得只凭“存在类/函数”判断。例如 `Shared` 已公开导出，但无法进入
当前 Ptr/Buffer 类型和后端，因此仍是 `Designed`。表面 intrinsic 还必须同时
具备公共占位符、frontend 入口和 checker handler；相关集合由
`tests/test_intrinsic_registry.py` 校验。

### 12.1 Markdown 示例标签

所有 `README.md` 与 `docs/*.md` 中的 Python fence 必须紧邻下列标签之一：

```text
<!-- tila-example: current; mode=exec -->
<!-- tila-example: current; mode=syntax -->
<!-- tila-example: diagnostic -->
<!-- tila-example: future; milestone=M4 -->
<!-- tila-example: generated -->
```

- `current/exec`：真实执行，含 `@tila.jit` 时必须通过 frontend/checker；
- `current/syntax`：当前语法片段，至少通过 Python 编译 smoke；
- `diagnostic`：故意展示当前拒绝路径；
- `future`：不可作为当前 API 使用，必须指明 M1–M6 归属；
- `generated`：后端生成物，不是 Tila 表面源码。

未标记的 Python fence、缺少里程碑的 future 块或无法执行的 current/exec 块
都会由 `tests/test_docs_examples.py` 拒绝。
