# ADR-017：最小 atomic_add 契约

- 状态：**Accepted**（2026-09-20，M4-03a 设计冻结；M4-03b/c CPU 与固定 RTX 3090 GPU 已实现）
- 前置：ADR-002、006、007、009、010、015、016。
- 范围：确定 atomic_add 的语言语义、Effect IR 扩展、CPU 参考与 GPU 验收。
- 非目标：本阶段不新增公共 API、执行节点或支持声明；不实现 CAS、其他原子、
  fence、barrier、race、uniformity、系统级通信或性能优化。

以上“本阶段”指 M4-03a 设计阶段。后续 M4-03b 实施边界见
[CPU 实现记录](../atomic-cpu.md)；M4-03c 的限定 GPU 支持及证据见
[GPU 实现记录](../atomic-gpu.md)。

## 1. 初始范围与公共接口

首个实现批次只接受 Global、ReadWrite、自然对齐的 i32/u32/f32 元素。
CPU 参考模型使用相同类型域；GPU 验收目标仅为 ADR-009 的 RTX 3090/SM86、
PyTorch 2.10.0+cu128、Triton 3.6.0、CUDA runtime 12.8 固定组合。
这是已通过本地严格验收的最小矩阵，不代表其他组合或持续 GPU CI。

以下为未来接口，遵循现有 load/store 的 Buffer 与 Ptr 两种寻址形式：

```text
atomic_add(buf, coord0, ..., coordR_minus_1, value,
           *, mask=None, order=ti.Relaxed, scope=ti.GPU) -> old
atomic_add(ptr, value,
           *, mask=None, order=ti.Relaxed, scope=ti.GPU) -> old
```

- Buffer rank 决定坐标数，至少一维；Ptr 为 RWPtr 或 Buffer 派生的可读写指针。
  返回标量或 Block，元素 dtype 与内存元素相同。
- 首版接受标量或一维访问 tile。Buffer 多个坐标可为标量或相同的一维 shape，
  标量坐标可扩展到该 shape；Ptr 支持标量或一维 pointer tile。高阶 tile 暂拒绝。
- value 为同 dtype 标量（允许扩展到访问 shape）或严格同形 Block；不作一般
  NumPy 广播，不让 value 扩大指针 shape。mask 可省略、标量 bool（扩展）或
  严格同形布尔 tile/Mask，布尔互操作沿用 ADR-012。
- 已类型化 value 必须与元素 dtype 精确一致；literal 仅按现有期望类型规则
  实例化，整数必须可表示，bool 不当作整数。需舍入的浮点使用 ADR-015 常量
  接口，其他类型用显式 cast；不得继承后端隐式 narrowing/promotion。
- 无 `other`、`out`、`unsafe_atomic_add` 或字符串 order/scope；无隐式地址转换。
  ReadOnly、WriteOnly、Shared、Local、bool、窄整数、i64/u64、f16/bf16/f64、
  FP8 全部在 Tila 层拒绝，不尝试后端碰运气。

自然对齐为 4 字节。launch 必须检查真实基地址（包含 view offset）以及 stride
为元素字节数的整数倍；typed pointer 的整数元素偏移保留自然对齐。不能以
Aligned 声明、缓存命中或空 grid 替代本次检查。不支持裸宿主整数地址。
正 stride/offset view 与重复坐标纳入首版验收；可写零 stride/负 stride 的
Buffer 绑定暂拒绝，CPU/GPU 同域，不放宽现有可写布局门禁。

## 2. 求值、返回值与重复地址

参数按源码顺序各求值一次，之后才执行该 atomic；mask 不短路参数求值。
嵌在 pointer/coords/value/mask 中的 load 或 atomic 拥有独立访问 site。

每个 active 逻辑元素执行一个不可分割的读—加—写，返回该次操作看到的旧值。
每个 inactive 元素不访问目标内存，返回同 dtype 的零（f32 为 +0）。省略 mask
等于全 true。零返回是 Tila 的显式规定：lowering 先求 atomic 到唯一临时值，
再选择 `where(mask, temporary, typed_zero)`；不能重复内联 atomic，也不能
把零写入 inactive 地址。masked-off 地址仍需满足地址计算的整数语义，active
地址进入现有 bounds 义务；debug 对 active 地址检查，bounds 策略不改原子性。

同 tile、同 program 或不同 program 可以访问同一地址。每个 active 元素都
贡献一次更新；不允许通过向量高级索引一次赋值丢失重复项。每个地址分别有
符合依赖和线程语义的原子更新顺序；不规定 lane/program 间谁先获得哪个旧值，
也不承诺整个 Block 是一个事务或存在跨地址的统一顺序。

例如初值 0，四个 active 元素均加 1：最终值为 4，返回旧值的多重集为
{0,1,2,3}；不要求第 k 个 lane 得到 k。`old = atomic_add(...)` 的多次值引用
不再更新内存；两次独立调用即使参数相同也必须产生两次更新。返回值无人使用
仍保留副作用，不得作为纯表达式删除。

同址重叠的普通 load/store 与 atomic 混用不能靠 atomic 标签推出安全；首版
没有 race 分析，调用者须避免无同步的并发非原子冲突。初始化在启动前完成，
验收在同步后读回；本接口不提供跨 kernel/stream 的自动排序。

## 3. 数值与内存顺序

| 项目 | 固定规则 |
|---|---|
| i32/u32 | 加法逐次按 32 位回绕；i32 按二补码解释，沿用 ADR-007 |
| f32 更新 | 每次以 f32 RNE 相加；本 Global target 将次正规输入和结果归为保留符号的零 |
| f32 旧值 | 返回实际读到的旧 f32 值；更新算术的归零规则不先改写旧值返回 |
| NaN/Inf | 保留浮点分类语义；不承诺 NaN payload、符号或跨设备逐位一致 |
| order | 类型 `MemoryOrder`；首版唯一可用值 `ti.Relaxed` |
| scope | 类型 `MemoryScope`；首版唯一可用值 `ti.GPU` |

两个枚举属于编译期专用值，不是字符串、整数、dtype、运行期 scalar 或新的
Const 类型域。可按其他类型化限定符方式解析别名，但不得任意执行 Python。
Acquire/Release/AcqRel、CTA/Sys 保留给后续设计；首版不导出对应可用成员，
伪造/变换 TIR 携带未支持值必须被 verifier 拒绝。没有“静默降级为 relaxed”。

Relaxed 只提供作用域内的原子更新，不发布/获取其他地址的数据，也不作为
barrier 或全局进度保证。lowering 必须显式写 `sem="relaxed", scope="gpu"`。
Tila 默认值不依赖后端默认：Triton 文档的默认 sem 为 acq_rel，scope 为 gpu。
参见 [Triton atomic_add](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)。

f32 Global atomic 的 RNE/次正规归零及自然对齐依据固定 CUDA 12.8 的
[PTX 8.7 atom 规范](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-atom)。
这条规则仅限本 atomic，不改变普通 f32 加法或 ADR-008 的归约契约。浮点
更新顺序不同会导致不同结果，不能要求并发执行与 CPU 固定顺序逐位一致。

## 4. Effect IR、verifier 与现有消费者

新增唯一执行节点 `TAtomicAdd`，返回类型为内存元素的标量/Block；其操作数字段
遵循 TLoad/TStore 的寻址结构：buffer 或 ptr、coords、value、mask，以及 line。
order/scope 为类型化静态属性。允许表达式和丢弃返回值的调用语句；后者必须
进入有副作用的 TIR 语句载体，不得被前端当作无用纯表达式忽略。

扩展不可变 `MemoryEffect`：

```text
kind: Read | Write | Atomic
atomic: None | AtomicInfo(op=Add, order=Relaxed, scope=GPU)
# site_id / region_id / address_space / element_dtype / location 保留
```

atomic 节点与 effect 的属性必须一致，verifier 独立重算确认；Read/Write 不得
携带 AtomicInfo，Atomic 不得缺少它。一个原子事件不是两个独立 Read+Write
事件，避免汇总、未来 alias/race 消费者丢失其不可分割性。判定访问权限时
Atomic 同时要求可读和可写。RegionId 不与 Extent、实际地址或 order/scope 混合。

实施必须覆盖所有消费者，而非只添加 lowering：

| 消费者 | 必须完成的扩展 |
|---|---|
| intrinsic registry / frontend / checker | 注册签名与副作用、enum 参数和 literal 规则；产生同源 bounds 义务；废弃返回仍执行 |
| effect_ir / verifier | OPERANDS 穷尽遍历；definition 绑定；唯一 site、元数据、权限、dtype、shape、enum、target 与来源校验；未知节点拒绝 |
| effect_summary | 参数访问先于 atomic；独立 path/mask/loop，false mask 只排除本访问；Const/return/零循环和 assume 规则沿用 ADR-016 |
| kernel effects | 增加 `Atomic[region]` 投影，保留 op/order/scope 在局部明细；不拆成普通写也不忽略读性质 |
| where 策略 | 将读取检测扩为 load/atomic 事件，沿用最近值分支归属与定义引用截断；已存 old 的复用不告警 |
| CPU / lowering | 逐逻辑元素参考更新；每次 atomic 求值恰好一次，不能复制、CSE 或删除；映射源 site |
| cache / target | atomic dtype/order/scope 进入 capability 校验；语义 revision 升级，属性计入生成源码/结构指纹；逐次 launch 校验不由缓存代替 |

TILA-EFFECT-007 扩为“eager memory effect”，诊断区分 Read 与 Atomic(Add)。
where 条件本身的 atomic 不额外触发该值分支告警，但仍保留 effect；`effects=off`
不关闭权限、target、bounds、enum 或完整性校验。atomic 不因 unsafe 上下文豁免。

explain 保持现有章节位置，访问记录显示 Atomic/Add、order、scope、site、源行、
RegionId、dtype、path/mask/loop。详细 JSON 使用 `tila.effect-details.v2` 表达
AtomicInfo；v1 不静默改变解释。实施时显式迁移相关 golden 和消费方，Read/Write
的 atomic 字段为 null，粗粒度 Read/Write 行格式保留。M4-03b 已完成该 v2 迁移。

错误分类沿用现有族：dtype 不一致 TILA-TYPE-030；权限 MEM；bounds BOUNDS；
不支持的 target/enum 组合 TARGET；缺失/冲突 effect 或未知节点走 verifier。
具体新增错误码在 M4-03b 集中注册并加 golden，本设计不先注册不可执行错误码。

## 5. CPU 确定性参考模型

保留现有 program 顺序：pid2 外层、pid1 中层、pid0 内层递增。每个 program
按 TIR 语句/操作数求值顺序运行；atomic 的 active 元素按逻辑 C-order 展平
索引递增处理，依据真实 stride/view 寻址。先求各参数，再逐元素取旧值、更新、
填返回数组；禁止 `array[indices] += values` 式批量更新替代重复地址序列。

整数显式模 2**32，避免依赖 NumPy 溢出异常/版本行为；f32 用显式归零与逐步
f32 舍入实现 target 模型，保存更新前旧值。NaN payload 不进入等价要求。
对同一输入 CPU 顺序可重放，但只是一个参考调度，不是 GPU lane 顺序规范，
不能用单个参考轨迹证明 GPU 无竞争或实现 acquire/release。

## 6. GPU 验收：观察不变量而非逐 lane 顺序

所有案例先初始化、后同步读回；固定输入种子并记录环境、源码、Const、grid、
num_warps、dtype、mask 和初值。重复运行用于暴露问题，不声称穷尽线程调度。

1. **无碰撞**：逐元素核对 old 和最终整数值；f32 用已定义数值规则核对。
   全 false、尾 mask、标量/一维、Buffer/Ptr/派生 Ptr、正 stride/offset view 必测。
2. **整数碰撞**：tile 内及跨 program 的同址加一，验证最终 active 计数和返回
   old 的完整多重集（小计数不溢出）；任意整数贡献另验证模和，加入回绕边界。
   两个别名参数访问同一存储也应保留所有更新，不能按 RegionId 分开累加。
3. **浮点碰撞**：先以可精确表示的小整数 f32 加法核对无丢失更新；随机有限
   normal 值使用预先计算的误差界，不使用任意固定 allclose 容忍度。若 m 次
   加法无溢出/下溢且 m*u<1，u=2**-24，可用 gamma_m=m*u/(1-m*u) 乘以
   `abs(initial)+sum(abs(values))` 与高精度总和比较；这是验收包络，不是唯一结果。
   小规模用合法更新排列枚举联合检查旧值及终值；不把两组互不相容结果各自通过
   容差就当作一条合法轨迹。次正规、±0、NaN、Inf 用无碰撞/受控顺序专项测试，
   不套 normal 随机误差公式。
4. **副作用完整性**：返回未用仍更新；old 复用不重复更新；独立相同调用更新
   两次；masked-off 返回零且目标未写；嵌套调用在参数中各执行一次。用源码/IR
   与执行双证据检查，不要求后端固定使用 atom 还是等价 red 指令。
   特别检查返回值使用/丢弃两条路径：后端若先聚合浮点贡献，只有保持本契约
   允许的逐元素结果才可接受。用消去/舍入反例及小规模排列枚举核对；若无法
   保证，须禁用该聚合或暂不启用对应 f32 路径，不能以误差容忍度掩盖语义变化。
5. **门禁**：不支持 dtype/shape/access/space/order/scope、错型 value、自然
   对齐失败、元数据冲突，必须在启动前拒绝。热缓存、空 grid、effects=off 也测。
6. **审计**：补 TIR/source/explain/golden 与错误快照，加入 tools/gpu_audit.py
   和重放清单，再更新预期节点数/支持矩阵。现有 300 节点不算 atomic 证据。

## 7. 实施拆分与退出条件

- **M4-03a（本阶段 DONE）**：设计 ADR 和计划同步，atomic 仍标 Designed。
- **M4-03b（DONE）**：最小 enum/签名、TAtomicAdd/AtomicInfo、verifier、
  bounds、summary、where 与 explain v2；CPU 参考实现及负测试。后端路径未完成
  时显式拒绝 GPU atomic，registry 仅在可用后端标 Partial，不能宣称完整支持。
- **M4-03c（DONE）**：RTX 3090 lowering、target 门禁、source map/缓存、严格 GPU
  验收与重放；88 项 atomic 专项通过，最小矩阵升级为 Implemented。

后续扩大 dtype、tile rank、order/scope 或架构必须追加设计与证据；不能因为
底层有 API 就默认接受。M3 的隔离 GPU 持续验收仍缺失，本 ADR 不解除该阻塞。

## 8. 设计依据与排除方案

本地锁定 Triton 3.6.0 的 `language/semantic.py::atom_red_typechecking_impl`
包含 value 隐式 cast 和广播；Tila 前置规则故意更窄，不能靠后端替代校验。
上游对应源码：[Triton v3.6.0 semantic.py](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/semantic.py)。

拒绝直接暴露全部 Triton dtype/order/scope：证据范围过大；拒绝 masked-off
返回未定义值：会把后端差异传给用户；拒绝把 Atomic 降为 Write：丢失 RMW 信息；
拒绝以 CPU 唯一顺序逐位比较浮点并发：会把合法调度差异判成错误。mask 零返回、
窄支持矩阵、参考调度均为本项目设计选择，不是对后端原生行为的描述。
