# M1 Exit Audit

状态：**PASS**

审计日期：2026-09-19

范围：M1-01 至 M1-06、ADR-001 至 ADR-006，以及 M1 的四项退出标准。

本报告判定 M1 可以退出并进入 M2。它证明核心语言与内存模型已经冻结，不表示
GPU target、完整 proof solver、并发 effect 或 TypeVar 已经实现；这些能力仍按
`status.md` 和 `roadmap.md` 的状态与阶段执行。

## 1. 退出条件结论

| 退出条件 | 结论 | 主要证据 |
|---|---|---|
| Region/Extent 不再混用 | PASS | `RegionId`、`Extent`、`AliasRelation` 为独立类型；obligation 只保存解包后的 extent 表达式，effect 只保存 RegionId；派生指针继承 RegionId |
| Ptr/Buffer/refinement 语法和内部类型一致 | PASS | Access/AddressSpace 为受控 enum；Extent、Alignment、Bound/UnboundStrides 无字符串或 `None` 哨兵；规范打印与正反构造测试一致 |
| intrinsic registry 成为单一事实来源 | PASS | 不可变 catalog 关联公共导出、frontend 名字、checker handler、effect/bounds、可达 TIR、backend、target 与状态键，并有双向完整性门禁 |
| 所有公开类型构造有正反测试与稳定诊断 | PASS | Ptr、Buffer、Const、Range、MultipleOf、Aligned 均覆盖合法构造、非法参数及稳定错误；诊断 registry 与非 debug traceback 门禁通过 |

## 2. ADR 与实现追踪

| ADR | 决定 | 实现/验证落点 |
|---|---|---|
| ADR-001 | Ptr 公共参数顺序与无歧义简写 | `types.py` 的统一规范化、五项打印；`test_ptr_syntax.py` |
| ADR-002 | RegionId、Extent、alias relation 分离 | checker/runtime/TIR 的独立字段与事实；`test_region_alias_model.py` |
| ADR-003 | Buffer 隐式 Global、launch-bound stride、受限 `buf.ptr` | `BoundStrides/UnboundStrides`；rank-1 check 与 stride-1 launch 门禁；`test_buffer_syntax.py` |
| ADR-004 | refinement 统一使用下标构造 | Range/MultipleOf/Aligned 校验、打印、事实传播；`test_refinement_syntax.py` |
| ADR-005 | 0.2.x Const 域固定为 exact Python int | 所有入口拒绝 bool/float/string/NumPy integer；staged bool 与 runtime bool 分离；`test_diagnostic_contract.py` |
| ADR-006 | 完整 intrinsic registry 元数据 | registry 完整性、handler/TIR/backend/status 关联；`test_intrinsic_registry.py` |

六份 ADR 均保持 `Accepted`；M1-01 至 M1-06 均为 `DONE`。

## 3. 审计中发现并修正的问题

1. `Access`/`AddressSpace` 虽已限制公开构造，内部仍使用字符串协议。现已迁移为
   受控 enum，`PtrT`/`BufferT` 在边界立即拒绝字符串。
2. `Extent`、`Alignment` 和 Buffer strides 仍用 `None` 或裸值表达 unknown/unbound。
   现已分别使用 `LinearExtent | UnknownExtent`、`Alignment | UnknownAlignment`、
   `BoundStrides | UnboundStrides`，证明与 launch 消费点显式解包。
3. 文档曾把未实现的 `p - offset` 与已实现的指针加法并列。v0 现只承诺
   `p + offset`；减法明确为 Deferred。
4. 类型总览曾把 `Const[bool]`/dtype 写成当前参数域。现与 ADR-005 对齐为
   0.2.x 仅 `Const[int]`。

## 4. 自动化证据

最终基线：

```text
python -m pytest -q
427 passed in <environment-dependent time>
```

- skipped/xfail 标记：0；
- Markdown current 示例、README/status/roadmap 漂移门禁：通过；
- Ptr/Buffer/Region/refinement/Const/intrinsic/diagnostic 专项回归：通过；
- `git diff --check`：通过。

`tests/test_m1_exit_audit.py` 固化本报告、ADR 状态、结构化类型表示、零 skipped/xfail
以及 roadmap/plan 里程碑状态，防止后续回退。

## 5. M2 交接边界

M1 无遗留退出阻塞项。以下仍是明确的后续范围，不影响本次 PASS：

- proof 分层、四态结论、统一 proof trace 与可选 slow solver：M2；
- 固定 GPU/Triton 支持矩阵和真实 GPU differential：M3；
- per-instruction effect、atomic、race、uniformity：M4；
- TypeVar、target capability 与完整 FP8：M5；
- `p - offset`、`byte_offset`、Shared/Local allocation：保持 Deferred/Designed，
  开放前必须有独立语义和后端闭环。
