# M3-03：编译后资源诊断与 source map

适用 0.3.0.dev0、ADR-009 的 RTX 3090/SM86 与 Triton 3.6.0 固定组合。

## 编译、检查与启动

CUDA launch 先通过 M3-02 的契约与 TIR 门禁，然后执行：生成源码、构建 Triton
JIT callable、warmup 编译、资源检查、加载二进制，最后才启动 kernel。
warmup 不执行 kernel。缓存命中仍重验资源门禁；空 grid 不走编译/加载路径。
实际执行期间的 CUDA/device assert 错误不归类为编译错误，不重试执行。

`TILA-TARGET-010` 表示生成模块装载、后端编译或二进制加载失败；
`TILA-TARGET-011` 表示硬资源限制失败。稳定诊断包含阶段、可确认的 Tila 位置，
资源错误另含资源名、需求和上限。原始后端文本/traceback 不混入稳定 golden，
保留为异常 cause 和附件，避免依赖编译器的文本格式。

## 源位置

Lowering 同时产生源码与 `source_map`：生成模块的 1-based 行号映射到原始 Tila
函数片段的 1-based 行号。递归发射保留内层语句位置；单条语句展开的 hint 等
附加行指向该语句。源码文本未加入标记，因此既有源码 golden 不变。

Triton CompilationError 的行号相对其 `src` 函数片段，适配层先在生成模块内
唯一匹配片段，再换算位置。诊断显示 Tila 行及源语句；附件还含文件名、函数
片段起始行，可按 `source_start_line + tila_line - 1` 定位到原文件。
列精度不作承诺。辅助函数、自动 scalar cast、无法唯一匹配的片段、ptxas 或
设备加载的全局错误保持 unknown，不猜测某条用户语句。硬资源错误通常属于整个
特化内核，也不伪造语句级位置；附件保留完整映射供检查。

## 资源规则

编译 metadata 的 shared bytes 与活动设备的 max_shared_mem 比较，超限拒绝。
加载后比较 `num_warps * 32` 与二进制的 `n_max_threads`，超限拒绝；Triton
加载器自身的 OutOfResources 也转换成同一诊断。重复命中已加载对象仍重新检查
线程上限，避免此前失败留下的句柄状态绕过检查。

成功后 `kernel.last_backend_resources` 记录 shared bytes/上限、线程数/上限、
registers_per_thread 和 spill_count。后两项仅为后端性能信息，不设置任意阈值，
不把高寄存器压力或 spill 当成语义错误或性能结论。每次调用先清空报告，失败、
CPU 和空 grid 不留下旧的成功数据。本实现使用锁定版本的私有加载接口，升级
Triton 时必须重新审计该适配层；尚不覆盖其他架构资源和完整 occupancy 分析。

## 附件与重放

捕获 `TilaError` 后可从 `error.backend_artifact` 取得目录。默认在系统临时目录
创建，`TILA_BACKEND_ARTIFACTS` 可指定保留根目录；GPU audit 已将其纳入产物目录。
附件包含 `kernel.py`、`failure.json`，记录 source map、原 Tila 源码/文件位置、
Const、scalar、dtype/布局/alignment、grid/warp、target、原始原因和 traceback。
资源失败还保留结构化资源需求与上限。文件系统写入失败不掩盖原错误，路径为 None，
写入原因可读 `artifact_error`。临时目录可能被系统清理，需要长期保留时指定根目录。

在固定环境中重放可信附件：

```sh
PYTHONPATH=src ci/gpu/.venv/bin/python tools/replay_backend.py /path/to/tila-backend-xxx
```

重放读取生成代码，以 mock tensor 恢复 dtype/alignment 等编译绑定，仅编译/加载，
不分配原数据、不启动 kernel。它验证当前支持 target 与记录的 torch/CUDA runtime
版本；相同 SM86 设备可重放，不要求原 UUID。工具不会自动安装环境，完整源码和
依赖锁由 GPU audit 附件保存。运行时数据相关错误不属于该工具的复现承诺。

## 验收边界

CPU 新增 7 项：源码不变、嵌套位置、相对行换算、三份诊断 golden、原始附件与
附件写入失败。GPU 新增 3 项：受控发射错误经过真实 Triton 编译并重放；真实
matmul 编译产物共享内存超限且重复调用均不启动；成功/缓存命中的资源报告。
负测试拦截非 warmup 调用，同时确认输出未被修改。

当前本地全量基线为 CPU 884 项、严格 GPU 115 节点/259 案例，零 skipped。
新增编译失败案例为受控故障注入，不代表发现了合法用户程序的新语言限制。
本阶段不增加架构支持、不作性能承诺，也不将历史远端 CI 结果视为当前修改的证据。
