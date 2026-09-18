# Tila 边界安全：静态证明模型

状态：设计基线；2026-09-19 按 ADR-011 更新 M2 目标，SMT 尚未实现。
前置阅读：`refinements.md`（事实来源与传播）、`type-system.md` §9
（Buffer/Ptr 访问形式）。

边界检查是 Tila 的杀手级特性：**每个内存访问都必须被证明在界内**，
证明的输入是类型系统的 shape 声明与 refinement 事实。M2 采用
[ADR-011](adr/011-smt-proof-and-trust.md) 的 SMT 默认引擎：数学整数与
有限位宽整数分别编码；当前实现仍是区间/DNF/grid 手写证明器。

---

## 1. Proof obligation

对每一次 `load / store / atomic`，checker 生成一条义务：

```text
Buffer 形式:
    访问 buf: Buffer[T, (D0, …, Dk−1)] 于坐标 (c0, …, ck−1)
    义务（逐轴）:  0 <= ci < Di     i = 0..k−1
    掩码形式:      mask ⇒ 逐轴义务

Ptr 形式:
    访问 p: Ptr/Block[Ptr[T,…, Extent=E, …]]，值偏移为 idx
    义务:        0 <= idx < E      （E 为元素数量；
                                     UnknownExtent 时义务为 Unknown，见 §6）
```

`RegionId` 不参与该数值义务；它仅标识 effect/race 分析的内存来源。
职责拆分由 [ADR-002](adr/002-region-id-and-extent.md) 固定。

义务在**事实环境**（refinements.md §3）中求解，产出四态结论之一
（§6）。义务不由任何隐式机制放宽——放宽只有三条合法路径：
mask 证明（§3）、launch 契约（§5）、`assume/unsafe`（refinements.md §5）。

---

## 2. 索引表达式与事实环境

索引保持**符号结构**，不做纯数值近似：

```text
IndexExpr ::=
    Const
  | Symbol                  维符号 N、M、循环上界 K…
  | ProgramId(axis)
  | Arange(start, end)
  | LoopIndex(var)
  | IndexExpr + IndexExpr | − IndexExpr | * Const
  | IndexExpr // Const | % Const
  | min(·,·) | max(·,·)
```

环境中的典型事实（全部由推导产生，无需用户声明）：

```text
0 <= pid < grid[ax]            ← grid 契约（§5）
0 <= lane < B                  ← arange(0, B)
0 <= i < K                     ← range 归纳变量
0 <= N                         ← launch 契约注解
N % BLOCK == 0                 ← 显式 launch 契约（§5.2）
```

例：

<!-- tila-example: current; mode=syntax -->
```python
offs = pid * BLOCK + tila.arange(0, BLOCK)
```

在数学整数语义或已证明所有相关中间运算不溢出的前提下，得到
`pid*BLOCK <= offs < pid*BLOCK + BLOCK` 与 `0 <= offs`
（由 `pid >= 0, BLOCK > 0, lane >= 0`）。仅凭这些**没有有限上界**
相对 N——这正是需要 mask 或契约的地方。

---

## 3. Mask 证明

`mask = offs < N` 的比较被 checker 记录在 Mask 类型的谓词槽里
（type-system.md §3.3）。masked 访问的义务变为：

```text
mask ⇒ 0 <= offs < N
```

判定方式：mask 谓词合取进入事实环境后，义务的可满足性检查。
对 `offs < N` 与已有的 `offs >= 0`：成立 ✓。

### 3.1 逐轴证明与"错误维度的 mask"

二维访问逐轴独立证明：

```text
mask ⇒ 0 <= rows < M
mask ⇒ 0 <= cols < N
```

mask 的 shape 正确但比较了错误的维度（如 `cols < M`）**不构成**
该轴的证明——每条义务必须匹配到正确的维符号。报错信息直接指出
哪一轴缺失：

```text
error[TILA-BOUNDS-002]:
    tila.load(x, (rows, cols), mask=rm & cm)
cannot prove axis 1 in bounds
    required: 0 <= cols < N
    mask provides: cols < M
    known: cols ∈ [0, N) unconstrained by mask
consider: cm = cols < N
```

### 3.2 Mask 组合

当前实现的谓词按 **DNF 子句**组织：`&` 对子句做笛卡尔积、`|` 拼接子句、`~`
保守地产生空子句（不携带可证信息）。证明器对每个子句独立做合取推理：
义务的证明要求目标谓词在**每一个**子句下成立
（`(C1 ∨ C2) ⇒ G ⟺ (C1 ⇒ G) ∧ (C2 ⇒ G)`）。`~` 引入的空子句使证明退化为
Unknown（sound 保守），不产生错误 hint。比较操作数为字面量时同样
产生谓词（`offs < 1000` 是合法谓词）。

M2 目标：用共享子表达式 DAG 保留 `And/Or/Not`，将 lane/broadcast 关系和
路径条件一并编码到 SMT，不再强制展开 DNF；`~` 在支持的编码内直接表示否定。
这项迁移尚未完成，当前 `~` 的保守边界仍以状态表为准。

---

## 4. 求解器架构

M2 的已接受设计是 **Z3 默认通用引擎 + 小型快速路径**，不是可选 slow path。
soundness 是底线：Unknown 永不升级为 Safe，Exempted 不等于证明。

```text
统一 obligation + facts + path + mask + 信任来源
    → 小型常量折叠/直接事实匹配（可选捷径）
    → Z3 查询 facts ∧ path ∧ mask ∧ ¬in_bounds
    → ProofResult + 源位置/前提/反例或 Unknown 原因
```

查询结果：unsat 表示在记录前提下安全；sat 在编码精确或反例确认可达时才是
ProvenUnsafe，保守循环/数据流近似下的未确认反例为 Unknown；求解 unknown
或超时仍为 Unknown。不可达路径与矛盾用户假设要在报告中显式区分。

工程要求（待实现）：

- Const/shape 使用 Int；kernel 整数按位宽及 signedness 编码。ADR-007 先
  定义除法、溢出、移位等语义，未覆盖时不得猜测；仅在有无溢出依据时使用
  数学整数推理，shape/grid 窄化也必须检查。
- 保留小型快速路径，不继续扩建第二套复杂求解器；不承诺全流程线性复杂度。
- 单查询 timeout/rlimit、kernel 累计预算及公式构建大小均有限制；具体默认值
  由基准测试确定。预算耗尽返回 Unknown，不截掉析取分支后宣称安全。
- 默认引擎切换时纳入标准依赖/锁文件，缺失时明确报配置错误。
- 缓存包含义务、facts/来源、路径、mask、特化/launch 绑定、编码版本及求解器
  版本/配置；命中后仍重验 launch 契约，Unknown 不跨更高预算永久复用。
- explain 显示证明依据、用户假设、候选反例、超时或不支持原因；unsat core
  不承诺是最小证明。详细迁移和验收见 ADR-011。

---

## 5. Grid 契约与 launch 契约

### 5.1 标准 grid 的 pid 事实

launcher 以 `cdiv(N, BLOCK)` 启动一维 kernel 时，自动注入：

```text
0 <= pid < ceildiv(N, BLOCK)
```

事实登记的三条途径（实现语义）：
1. `ti.cdiv(N, BLOCK)` grid 标记：launcher 把 (a, b) 数值匹配回
   （维符号, Const 符号）登记 grid 契约；
2. **精确维 grid 启发式**：普通 int grid 的数值恰好等于某个维的
   绑定值时，登记 `grid_ax == 该维`（`pid_ax < dim`，softmax 的
   行-per-program 模式即依赖它）——若不匹配任何维则无事实；
3. **`launch_auto`**（§5.3）从义务模式直接推导 (bound, step) 并登记，
   不经过数值匹配。

配合 `0 <= lane < BLOCK` 可推出弱上界
`offs < ceildiv(N,BLOCK)*BLOCK <= N + BLOCK − 1`——不充分，
因此**尾块 mask 仍然必要**（这是正确的设计，不是缺陷）。

### 5.2 显式 launch 契约

当用户声明可整除时，完整 tile 访问可以被证明：

<!-- tila-example: future; milestone=M1 -->
```python
@tila.assume_launch(N % BLOCK == 0)
@tila.jit
def kernel(x: Buffer[f32, (N,)], N: i32, BLOCK: Const[int, PowerOfTwo]):
    ...  # 无 mask 的完整 tile 访问 → SafeUnderContract
```

契约由 launcher 在每次启动前检查（失败抛 `TilaLaunchContractError`，
不受 `python -O` 影响）。契约在报告中标明该 kernel 的哪些访问
依赖它。

### 5.3 launch_auto：grid 自动推导

`kernel.launch_auto(args...)` 从**义务模式**推导 grid：对每个被
使用的 pid 轴，在义务中找 `pid_ax * STEP + lane`（或裸 `pid_ax` 标量
坐标，STEP=1）模式，取 `bound`（须为运行期维符号）与 `STEP`（须为
Const/字面量），`grid_ax = ceildiv(bound, STEP)`；未使用的轴为 1。
推导出的 (bound, step) **直接登记为 grid 契约**（比数值匹配更精确，
可喂给 cdiv 战术）。找不到模式或多个义务不一致时报
`TILA-TYPE-105`，提示改用显式 grid。

---

## 6. 四态结论与诊断策略

当前实现及兼容显示使用以下四态：

| 状态 | 含义 | 默认处理 |
|---|---|---|
| `ProvenSafe` | 事实足以证明逐轴在界内 | 通过 |
| `SafeUnderContract` | 证明依赖 launch 契约 | 通过（报告标注依赖） |
| `Unknown` | 证不出，也未证伪 | strict：error；warn：warning |
| `ProvenUnsafe` | 事实可证越界 | **无条件 error**（任何模式） |

M2 将结论与信任来源拆开：内部 verdict 为 ProvenSafe / ProvenUnsafe /
Unknown / Exempted，dependencies 独立记录 StaticFact、CheckedLaunchContract
和 UserAssumption。`SafeUnderContract` 保留为显示摘要；有用户假设时必须
同时标注。unsafe 仅为 Exempted，不能作为后续 proof 或 hint 的安全依据。
这项迁移尚未实现；当前聚合报告不能视为完整的信任来源审计。

```text
error[TILA-BOUNDS-001]:
    x = tila.load(xbuf, offs)
possible out-of-bounds access
    index:  offs = pid * BLOCK + arange(0, BLOCK)
    known:  0 <= offs,  0 <= pid < ceildiv(N, BLOCK)
    cannot prove:  offs < N
fix:
    mask = offs < N
    x = tila.load(xbuf, offs, mask=mask)
    （或 @tila.assume_launch(N % BLOCK == 0)，或 tila.unsafe_load）
```

诊断要求：index 的符号式、已知的全部相关事实、缺的那一步、
至少一个修复建议——四要素缺一不可。

---

## 7. 走查示例

### 7.1 masked add（一维尾块）

<!-- tila-example: current; mode=syntax -->
```python
offs = pid * BLOCK + tila.arange(0, BLOCK)
mask = offs < N
a = tila.load(x, offs, mask=mask)
```

义务 `mask ⇒ 0 <= offs < N`：谓词给上界，事实 `0 <= offs` 给下界 →
ProvenSafe ✓。

### 7.2 matmul 的 K 尾块

<!-- tila-example: current; mode=syntax -->
```python
for k0 in tila.range(0, K, BK):
    a = tila.load(am, (rows, k0 + ks), mask=rm & (k0 + ks < K))
```

归纳事实 `k0 <= K`（`k0 = BK*i < K`）+ 谓词 → 逐轴可证。
遗漏 `(k0+ks < K)` 时轴 1 义务 Unknown → 报错并建议补谓词。

### 7.3 2D tile 的行列 mask

<!-- tila-example: current; mode=syntax -->
```python
rows = pid_m * BM + tila.arange(0, BM)
cols = pid_n * BN + tila.arange(0, BN)
rm = rows < M
cm = cols < N
a = tila.load(am, (rows[:, None], cols[None, :]), mask=rm[:, None] & cm[None, :])
```

外积 mask 的逐轴归因：`rm[:,None]` 只作用于轴 0、`cm[None,:]`
只作用于轴 1——证明器利用 broadcast 结构把义务拆回单轴。
（广播本身的合法性由 TILA-SHAPE 规则先行检查。）

### 7.4 gather（数据依赖索引）

<!-- tila-example: diagnostic -->
```python
idx = tila.load(idx_buf, offs, mask=m)
x = tila.load(data, idx)          # Unknown → strict error
```

合法路径（按推荐顺序）：

<!-- tila-example: current; mode=syntax -->
```python
tila.assume((idx >= 0) & (idx < N))       # 事实 + debug 断言
x = tila.load(data, idx)                   # ProvenSafe
# 或：x = tila.unsafe_load(data, idx)       # 显式豁免
```

---

## 8. 与其他文档的关系

- 事实的产生与传播规则：`refinements.md` §3；
- 访问形式（Buffer/Ptr）与 Region 语义：`type-system.md` §8–§9；
- 契约的 launcher 实现：`surface-language.md` §6；
- 硬件层（对齐/向量化宽度）消费对齐事实：`intrinsics.md` §4。
