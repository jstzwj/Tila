# Tila 架构决策记录

本目录记录公共语法与核心 IR 的提案和已冻结决策。ADR 回答“设计应当是什么”，
[状态表](../status.md)回答“当前实现到了哪里”；ADR 被接受不等于功能已经实现。

| ADR | 状态 | 主题 |
|---|---|---|
| [ADR-001](001-ptr-public-syntax.md) | Accepted | Ptr 公共参数语法 |
| [ADR-002](002-region-id-and-extent.md) | Accepted | RegionId 与 Extent 的拆分模型 |
| [ADR-003](003-buffer-stride-address-space.md) | Accepted | Buffer stride 与 address-space 公共语法 |
| [ADR-004](004-refinement-construction-syntax.md) | Accepted | Refinement 构造语法与契约 |
| [ADR-005](005-const-type-domain.md) | Accepted | Const 类型域与 staged bool 边界 |
| [ADR-006](006-intrinsic-registry.md) | Accepted | Intrinsic registry 架构与完整性门禁 |
| [ADR-007](007-integer-semantics.md) | Accepted | 整数回绕、floor 除法、转换、索引门禁与 ABI；M2-01 已实现 |
| [ADR-008](008-reduction-precision.md) | Accepted | sum/max 输入、累加、输出 dtype 与 NaN 策略；M2-08 已实现 |
| [ADR-009](009-gpu-validation-baseline.md) | Accepted | 固定本地 GPU 验收组合与重放；正式 CI/支持仍归 M3 |
| [ADR-010](010-effect-diagnostic-policy.md) | Accepted | M4-02 where 急切读取诊断、独立 effects 策略；race 另行设计 |
| [ADR-011](011-smt-proof-and-trust.md) | Accepted | M2-02/03 已实现 DAG/ProofResult、默认 Z3、预算及缓存；系统可达性/审计继续 |

ADR-011 不替代 ADR-007 对具体整数运算语义的决策。

M2-07 独立决策（ADR-012 至 ADR-015 均已接受并实现）：

| ADR | 状态 | 主题 |
|---|---|---|
| [ADR-012](012-boolean-tile-mask.md) | Accepted | 布尔 tile 与 Mask 的受限互操作；M2-07a 已实现 |
| [ADR-013](013-const-bool-domain.md) | Accepted | 自 0.3.0.dev0 实现 Const[bool] 独立类型域与带类型标签的缓存键 |
| [ADR-014](014-host-integer-normalization.md) | Accepted | 显式 host_int 白名单转换已实现，保留 ExactInt |
| [ADR-015](015-rounded-typed-constants.md) | Accepted | 显式目标 dtype、RNE 舍入与按位常量表示；M2-07d 已实现 |

M4-01 分阶段实施：

| ADR | 状态 | 主题 |
|---|---|---|
| [ADR-016](016-instruction-effect-ir.md) | Accepted | M4-01b/c/d 已实现局部 effect、定义引用/verifier、path/mask/loop、派生汇总及可选详细输出与隔离验收 |

ADR-016 不替代 ADR-010 的诊断严格度决策，不引入 atomic/race/uniformity 功能。

状态含义：

- `Proposed`：讨论中，不能作为兼容性承诺；
- `Accepted`：设计已冻结，后续实现必须遵守；
- `Superseded`：已由另一份 ADR 替代；
- `Rejected`：保留被拒绝方案及理由。
