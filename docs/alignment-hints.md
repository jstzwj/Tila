# M3-05：alignment 契约到 hint 的发射

版本：0.3.0.dev0；验证范围仍为 ADR-009 的 RTX 3090/SM86 固定组合。

## 推导规则与单位

`Aligned[A]` 表示传入 view 的基地址 P 是 A **字节**的倍数。检查使用实际
data_ptr，已经包含 storage offset；不是仅检查底层 allocation 地址。
元素大小为 E 字节、元素 stride 为 S 时，第 i 个地址为 `P + E*S*i`。
若另有证明 `i` 为 M 的倍数，可推导该地址的字节对齐为 `gcd(A, E*S*M)`。
不能由基地址对齐反推 `i` 的整除性，也不能把 A 直接当成元素个数。

例如 f32 基地址 16B 对齐、stride=1 时，任意相邻元素只保证 4B 对齐；
只有另证 i 可被 4 整除，才恢复 16B。f16 对应 2B 和 8 个元素。
负 stride 的乘积取绝对值；零 stride 地址不变，但这些布局本阶段不新增提示。

本阶段只发射**基地址**提示，不对未证明的派生地址向量发射 alignment assertion。
现有 pid*step 的 multiple_of 和连续索引的 max_contiguous 仍使用独立静态证据。

## 证据与发射条件

新增 `alignment.collect` 只接受参数上的显式 Aligned 声明，且绑定必须满足：
dtype 匹配、一维、stride=1、非空指针值、实际地址通过声明的字节对齐校验。
适用于 Buffer 和 Ptr。Ptr 本身仍执行既有 contiguous/Extent 检查。
非零 storage offset 的连续 view 可以发射，但前提是该 view 自己的地址对齐。

无声明时，即使 allocator 恰好返回 256B 对齐地址，也不增加此提示。非连续、
负/零 stride、多维绑定暂不发射；不会由 shape 或 view 的相同底层 storage 猜测。
用户 assume、残留的 `TKernel.runtime_alignments` 都不能授权发射。

证据是每次调用重新收集的不可变 AlignmentFact，包含参数、字节对齐、元素字节数、
stride，来源标记为 CheckedLaunchContract。静态 hint 继续标记 StaticFact；
没有把用户信任来源提升为静态证明。

lowering 把参数指针转换为 uint64 字节地址，标注整数 multiple_of，再转回原 pointer
dtype。这样发射单位明确，既不引入偏移变换，也不依赖把 pointer metadata 当作
元素单位解释。转换使用原参数名，不引入可能与用户变量冲突的临时名。

## Launch、explain 与缓存

所有参数/shape/stride/alignment、Const、launch、安全和 target 门禁通过后才收集
证据并执行。每次调用先清空上一轮用于发射的证据；契约失败不会进入 kernel，
缓存命中也要重验。零 grid 仍检查宿主契约，但不收集发射证据、不编译或启动。

离线 materialize/build 和默认 Lowering 没有实际绑定，因此不发射新 alignment hint。
成功调用后，`last_alignment_facts` 保留该次验证的快照；explain 的 hint dependency
明确写 `last validated binding`，显示参数、单位和 stride，不是对未来调用的承诺。
CPU 成功调用也可记录已检查的绑定，explain 展示该证据可授权的 lowering 提示；
这不表示 CPU 执行过 Triton。失败调用后 explain 不复用旧 alignment 发射证据。
契约的源行记为 0，表示 launch 来源，不伪造某条函数体语句的位置。

缓存键显式包含 AlignmentFact 元组，同时保留生成源码 hash 和实际布局/offset/
alignment 指纹。相同对象从连续 view 切换到 strided view，不会复用带连续绑定证据
的 callable；有提示/无提示源码也分离。后端失败附件记录 alignment_facts，生成源码
本身包含对应断言，仍可用 M3-03 工具仅编译重放。

## 验收与限制

CPU 新增 16 项：字节/元素推导、两种参数形式、无声明/非连续/负 stride/广播不
发射、失败和空启动清理、缓存证据隔离、静态/launch 来源并列的 explain golden。
GPU 新增 9 项：i8/f16/f32/f64 × Buffer/Ptr，非零对齐 offset、13 元素尾块，
同一 JIT 的提示开/关和重复命中；连续到非连续绑定切换；未对齐调用在执行前拒绝。
测试关闭新 hint 使用局部测试替换，不增加公共 launch 选项。

完整基线为 CPU 926 passed、严格 GPU 160 节点/304 案例，零 skipped。
这些是语义/发射/缓存证据，不是性能提升或向量化承诺；也未扩展对其他 GPU 架构的支持。
更一般 stride/多维/控制流下的派生地址提示仍保守不发射。
[M3-06 退出审计](m3-exit-audit.md)已完成，结论为 NOT READY；自动 GPU CI 已撤下，
仅保留本地验收，不据本阶段通过宣告整个 M3 完成。
