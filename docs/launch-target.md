# M3-02：Launch、target verifier 与缓存

适用版本：0.3.0.dev0。支持环境仍限于 [GPU 支持矩阵](gpu-support.md)。

C1 新增边界：verifier 独立检查普通/unsafe 访问的读写权限和 cast 来源/shape。
GPU 的 runtime if/循环内 return 提前报 TARGET-009，顶层与 Const 分支 return
保持支持；CPU 完整提前退出语义不变。实测原因与反例见 [C1 复核](c1-correctness-review.md)。

## Grid 与空启动

CPU/CUDA 共用 launch 域：grid 为 1–3 轴 tuple，各轴必须是 exact Python int，
上限依次为 `(2147483647, 65535, 65535)`，下限为 0。显式、callable、cdiv 和
自动 grid 都经过同一检查；违反时在调用 Triton 前报 `TILA-TYPE-104`。
即使某轴为 0，也检查其他轴；空 tuple 不是空启动，而是非法 grid。

上限对应 NVIDIA 的 MAX_GRID_DIM_X/Y/Z，已用固定设备上的 cuDeviceGetAttribute
核对；属性定义见 [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)。

任意轴为 0 时，先检查参数、dtype、alignment、Const/refinement、assume_launch、
device/target 与 TIR，再返回 no-op，不生成或执行内核。没有执行程序，因此不执行
逐程序数值/边界检查；last_proof_results 为空，last_report 明确标记 no-op，
不将其包装为 ProvenSafe。零长度 tensor 配正 grid 则正常执行带 mask 的程序。

## Target 与 TIR 门禁

`src/tila/target.py` 集中定义不可变 capability：设备/架构、Triton 版本、grid、
num_warps、tile 大小及 dot dtype/K 限制。混合 device、非 CPU/CUDA device、
不支持的 CUDA target 在执行前拒绝。完整软件/driver 版本认证仍由严格审计入口负责。

`src/tila/verifier.py` 在 specialization 和 lowering 前检查准确节点类型、嵌套
结构、操作符、dtype、已知 rank、标识符及 operand/statement 角色。未知节点
（包括未知子类）、循环对象图、非法子节点报 `TILA-TARGET-009`；lowering 不再输出
占位注释后继续。可用时诊断保留源行号。

CUDA tile 维度必须是正的二次幂，总元素不超过 2^20；dot 当前仅 f16 输入、
f16/f32 输出、二维输入，已知 K 至少 16。离线 symbolic shape 延迟到 Const
specialization 检查，执行前不能留下未解析的 tile 维度。CPU 仍允许非二次幂 tile。

这是结构与已实现 target 规则的门禁，不替代类型检查、SSA/dominance 或安全证明。
M3-03 已补 [编译后硬资源门禁与语句 source map](backend-diagnostics.md)；更广操作
矩阵与性能分析仍待后续阶段，不能推导“所有通过 verifier 的程序都必然通过后端编译”。

## 缓存指纹

这里缓存生成的 Triton JIT callable；实际二进制特化仍由 Triton 管理。键包括：

- registry revision、函数身份、带类型 Const、显式常量位模式、debug、num_warps；
- 定义源码 SHA256 与当前生成源码 SHA256；
- target 的 device index、UUID、软件版本与 capability；
- Buffer/Ptr 声明类型及实际 dtype、shape、stride、torch storage offset、有效地址
  alignment（最多区分至 256 字节），以及 scalar ABI dtype。

不使用完整数据地址；运行时 scalar 值交给 Triton 分派。所有 launch 契约每次重验，
缓存命中不豁免检查。生成源码的 linecache 名称也使用源码 hash，防止旧 callable
读到被覆盖的源码。缓存为进程内、JIT 实例内缓存，当前没有持久化或 LRU 容量承诺。

## Hint 与 alignment

max_contiguous 必须匹配当前定义行且来源为 STATIC；无证据、用户假设或冲突定义
不发射。分支 PID 事实取交集，循环边界清理事实，每次 lowering 重置状态。
multiple_of 的步长要求二次幂字面量、PowerOfTwo Const，或与 arange 上限同一
Const（执行前有 tile 二次幂门禁）。普通 Const 的整数折叠独立处理，避免影响大整数 cast。

Buffer/Ptr 的 Aligned 契约检查实际 view 地址，失败时即使 grid 为零也拒绝，
不进入 kernel 执行。M3-05 已在 [独立的已检查绑定证据](alignment-hints.md) 下
增加一维连续 Buffer/Ptr 基地址提示，不将此处的静态索引提示等同于地址对齐。

## 验收

新增 `tests/test_m3_launch.py` 35 项与 `tests/gpu_launch.py` 8 项，覆盖上限/零轴、
失败契约、非法 TIR、hint 来源/控制流隔离，以及同一 JIT 的源码、stride、warp
变化与真实 GPU 缓存隔离。全量 CPU 877 passed；严格 GPU 112 节点/256 案例通过，
均零 skipped。本阶段证据为本地固定环境验收；历史远端 CI 结果见支持矩阵，不能
将其视为当前未提交修改已通过远端 CI。
