# Tila 架构决策记录

本目录记录已经冻结的公共语法与核心 IR 决策。ADR 回答“设计应当是什么”，
[状态表](../status.md)回答“当前实现到了哪里”；ADR 被接受不等于功能已经实现。

| ADR | 状态 | 主题 |
|---|---|---|
| [ADR-001](001-ptr-public-syntax.md) | Accepted | Ptr 公共参数语法 |
| [ADR-002](002-region-id-and-extent.md) | Accepted | RegionId 与 Extent 的拆分模型 |
| [ADR-003](003-buffer-stride-address-space.md) | Accepted | Buffer stride 与 address-space 公共语法 |
| [ADR-004](004-refinement-construction-syntax.md) | Accepted | Refinement 构造语法与契约 |
| [ADR-005](005-const-type-domain.md) | Accepted | Const 类型域与 staged bool 边界 |
| [ADR-006](006-intrinsic-registry.md) | Accepted | Intrinsic registry 架构与完整性门禁 |
| [ADR-011](011-smt-proof-and-trust.md) | Accepted | SMT 默认证明引擎、整数编码边界与信任来源；待 M2 实现 |

ADR-007 至 ADR-010 的议题仍在 `plan.md` 中处于 Proposed，尚未形成独立决策文件；
ADR-011 不替代 ADR-007 对具体整数运算语义的决策。

状态含义：

- `Proposed`：讨论中，不能作为兼容性承诺；
- `Accepted`：设计已冻结，后续实现必须遵守；
- `Superseded`：已由另一份 ADR 替代；
- `Rejected`：保留被拒绝方案及理由。
