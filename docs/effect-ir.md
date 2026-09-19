# M4-01b：局部 Effect IR 与定义引用

2026-09-19，[ADR-016](adr/016-instruction-effect-ir.md) 已评审冻结为 Accepted。
本文记录 M4-01b 的指令元数据与完整性检查；后续 [M4-01c](effect-summary.md)已实现
path/mask/loop 上下文与汇总迁移，并发分析仍未实现。

`TLoad.effect` / `TStore.effect` 保存不可变 MemoryEffect：结构路径 site_id、
Read/Write、RegionId、address_space、element_dtype、EffectLocation(line)。
load 和 unsafe_load 都是 Read，store 和 unsafe_store 都是 Write；同一行的
多次访问按结构路径区分。保留节点原有 mask/coords/ptr/unsafe 字段，不复制其语义。
源行是 Tila 函数片段内的行号，沿用 source map 约定；不承诺列精度或绝对文件位置。

checker 完成 typed TIR 后统一绑定；`effect_ir.OPERANDS` 穷尽列出后端可接受节点
的执行操作数，控制流 body 使用独立词法环境。TName 持有不可变 ValueRef，引用
参数或定义点；重赋值产生不同身份。分支合并与循环携带值使用显式 merge/loop
身份，不把它们当作精确 phi 或循环不变量。提前 return 的分支不作为后续定义的
活跃前驱；但本阶段不计算访问的路径条件或删除不可达 effect。

遍历嵌套 load、条件、mask、other、store value 和循环界限，遇到 TName 不回访
其生产者，因此不会把 `a + a` 变成两次读取。纯表达式可共享；同一个内存节点
对象不能代表多个求值出现点，跨定义作用域共享同一个 TName 也被拒绝。

verifier 在原结构/target 校验后独立重算元数据并比较，验证过程不补齐或修复
缺失信息。缺失/错误 effect、伪造 site/region/dtype/位置、陈旧定义引用、共享内存
出现点和未知节点都报 `TILA-TARGET-009`。访问类型从当前定义点取得，不依赖
`tk.types` 的最终名字快照。内部构造/改写 TIR 的工具需显式重新绑定后再验证，
不能把验证器当作自动接受任意篡改的修复入口。

M4-01b 当时保留原 `TKernel.effects` 列表与输出；M4-01c 已将其替换为只读派生
投影，输出差异见后续文档。`verify_effects()` 返回的是结构访问清单，包含所有
分支和循环体，不是运行时事件序列、执行次数或 kernel may-effect summary。
UserAssumption、mask=false 和 unsafe 不删除此清单中的节点。

registry semantic revision 提升为 9，隔离旧元数据语义缓存。没有新增 launch
overlay、solver 查询或公共 API，没有改变 CPU 计算或 Triton 源码发射。

验收：新增 16 项专项，覆盖嵌套/unsafe/重复值使用、字段篡改、缺失及冲突、定义
重绑定、早退、循环携带指针、RegionId/Extent 隔离、where/other 急切读取、完整
节点覆盖及确定性。一个手工构造 TIR 的旧 hint 测试增加显式绑定，保留原断言。
全量 CPU 953 项、RTX 3090 严格 GPU 300 节点（444 案例）通过，均零跳过；golden
未改写。本地 GPU 报告：`artifacts/ci-gpu/20260919T144732Z-fylqrfl8/report.json`。
这些是当前工作区的本地证据，旧提交的托管 CPU run 不替代本次远端验收。

M4-01c 已派生 path/mask/loop 上下文与 kernel summary，移除 checker 平行列表。
[M4-01d](effect-audit.md)已实现可选详细输出与缓存/绑定隔离；atomic/race/uniformity 暂缓。
M3 仍因缺少隔离 GPU CI 保持未完成。
