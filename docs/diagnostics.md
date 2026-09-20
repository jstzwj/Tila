# Tila 诊断契约与错误码目录

状态：M1-06 实现基线，M2-01 于 2026-09-19 新增数值契约诊断。

本文定义用户可见诊断的兼容性契约。机器事实来源是
`src/tila/errors.py::DIAGNOSTIC_REGISTRY`；测试会扫描 `src/tila/*.py` 中出现的所有具体
`TILA-<FAMILY>-<NNN>`，要求它们与 registry 双向一致。新增、删除或复用错误码必须在同一
变更中更新 registry、本文和正反测试。

## 统一结构

每个 registry 表项固定包含：

```text
code
phases              # host/frontend/check/specialize/launch/target/internal
severity            # error/warning/error-or-warning
summary
default_fix
```

每个实际诊断至少渲染：

```text
error[TILA-<FAMILY>-<NNN>]: <具体原因>
    at: <源码位置、launch boundary 或 unknown>
    phase: <阶段>
    <found/required、类型、shape、事实或实际值等上下文>
    fix:
      <可执行修复建议>
```

`TilaError`、`TilaLaunchContractError` 和 `Warning_` 共用 code registry 与字段约束。
调用点提供更具体的 details/fix；缺省时使用 registry 的 summary/default fix，不能产生没有
修复建议的用户诊断。

## 阶段与边界

| 阶段 | 负责范围 | 主要入口 |
|---|---|---|
| `host` | 显式宿主整数转换的类型白名单 | `ti.host_int(value)` |
| `frontend` | Python 子集、注解、表面形式 | `@ti.jit`、CLI 模块加载 |
| `check` | dtype、shape、capability、控制流、静态 proof | `@ti.jit` Stage 1 |
| `specialize` | Const、deferred shape/assert、生成前契约 | `materialize`、launch Stage 2 |
| `launch` | 实参、张量、grid、alignment、launch contract | `kernel[grid](...)` |
| `target` | 后端包、设备与目标能力 | Triton/GPU 路径 |
| `internal` | 非预期实现故障 | CLI 安全边界 |

CLI 默认只渲染结构化诊断，不输出 Python traceback。设置 `TILA_DEBUG=1` 后，未知内部异常
会原样抛出，供开发者取得 traceback；已知 `TilaError`/launch contract 仍保持稳定渲染。

## 当前错误码

以下是 M1-06 冻结的活动目录。每个具体编号只能表示一个原因族；同一原因跨路径不得随机换码。

| Family | 活动编号 | 所有者 |
|---|---|---|
| `TILA-SYN` | 000–004、010–012、014–015、020–024、026–028、030–041、050、060–063 | frontend/表面语法 |
| `TILA-TYPE` | 012–037、101–105 | checker；037 为 host_int 宿主转换；101+ 为 launch 参数契约 |
| `TILA-SHAPE` | 003–006、008–012 | checker/specialization shape |
| `TILA-CONST` | 001–011 | Const/StagedBool/specialization；011 为显式浮点常量源/目标限制 |
| `TILA-NUM` | 001–002 | 001 为整数定义域/索引/ABI；002 为显式浮点常量非有限/溢出 |
| `TILA-MEM` | 001–006 | memory capability/offset/alignment |
| `TILA-BOUNDS` | 001–003、010 | proof 与 launch contract |
| `TILA-EFFECT` | 007、008 | where 急切读取（独立 off/warn/error）；非法 effects 策略 |
| `TILA-TARGET` | 004–011 | backend/target、launch options、device/TIR 门禁、编译/加载失败与硬资源超限 |
| `TILA-INTERNAL` | 001 | CLI 内部故障封装 |

尚未实现的 `TILA-RACE`、`TILA-UNIFORM` 只存在于未来设计，不进入活动 registry；实际实现前
不得以占位代码伪装成当前能力。

## Const 诊断约定

M2-03 的 `TILA-PROOF-001` 表示默认 Z3 依赖缺失或预算配置非法。
求解 timeout/rlimit/累计预算耗尽仍是 Unknown，按既有 bounds strict/warn 分派，
诊断附带 ProofResult 原因与候选反例；不是配置错误。

M2-01 的 `TILA-NUM-001` 覆盖除数非零、合法移位、有限且可表示的 float→int、
索引中间值溢出和 shape/stride/标量 ABI 范围。已知非法值在 materialize 时拒绝，
需要实参的契约每次 launch 检查；与 bounds strict/warn 独立，不能通过 unsafe
内存操作豁免。grid 类型/轴数/range 继续使用 `TILA-TYPE-104`，Const 断言中的
算术求值失败继续使用 `TILA-CONST-009`。

- `TILA-CONST-008`：所有入口上的 ExactInt 失败。details 必须包含参数名、实际类型和值；
  Python bool、float、字符串、NumPy integer 和任意可转换对象均拒绝。
- `TILA-CONST-010`：specialization 收到未声明的 Const override，不能降级成普通 unknown kwarg。
- `TILA-CONST-005`：`static_assert` 在 Stage 1 或 specialization 求值为 false。
- `TILA-CONST-006`：谓词不是 Stage 1/staged bool，例如依赖 runtime scalar。
- `TILA-CONST-009`：合法 staged 表达式在 specialization 求值失败，例如实际执行到除零。

`and/or` 保持 Python 短路语义：未执行分支中的除零不产生 `TILA-CONST-009`。

## 维护门禁

1. 所有源码中出现的具体错误码必须存在于 registry，registry 不保留无人使用的活动码。
2. registry/spec 不可变；family、phase、severity 和默认修复建议不能为空。
3. 新错误路径至少有 code、location、phase、关键上下文和可执行 fix 的断言。
4. CLI 测试必须确认普通模式不包含 `Traceback`；内部故障统一为 `TILA-INTERNAL-001`。
5. 错误码语义若不兼容，不得静默复用旧编号；先更新本目录并说明迁移。
