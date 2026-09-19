# ADR-013：Const[bool] 的独立参数域

- 状态：**Accepted**（M2-07b；自 0.3.0.dev0 生效）
- 日期：2026-09-19
- 阶段：M2-07b；目标为 0.3.x，发布前版本评审
- 关联：ADR-005；保留其对 0.2.x 的历史承诺

## 推荐决策

从 0.3.0.dev0 开发基线新增 `ti.Const[bool]`，默认值、materialize/explain override 和 launch
override 统一要求 `type(value) is bool`。0/1、NumPy bool、字符串和可转换对象
均拒绝。Const[int] 继续只接受 ExactInt；两者不能通过 Python 子类关系混用。

该变更不回移到 0.2.x；版本号同步 pyproject.toml 与 tila.__version__，不代表
已发布 0.3.0 正式版。Const 类型域、编译与 proof 缓存 revision 一并升级。

<!-- tila-example: current; mode=syntax -->
```python
@ti.jit
def optional_path(ENABLED: ti.Const[bool] = True):
    if ENABLED:
        pass
```

Const bool 可用于 staged `not/and/or`、bool 等值比较、static-if 和 static_assert。
不能用于 shape、arange/range 的整数参数、数值算术或整数 refinement；初版不允许
bool 参数附加 refinement。与 runtime bool 组合后为 Runtime；常量 False/True
触发短路时不得求值被跳过的无效表达式。两分支装饰期检查与 TStaticIf 规则保持。

## 表示、ABI 与缓存

内部 Const 参数增加 `value_kind: Int | Bool`，与 Stage 正交；bool 不进入 DimExpr、
Facts.num 或隐式 i32 索引 ABI。staged 布尔表达式直接保存 bool 参数依赖，在特化
时绑定；尚未绑定不能解释为 False。CPU 使用真实 bool，Triton 参数仍是 constexpr。

所有编译/特化/proof 缓存及序列化使用带类型标签的规范键，例如
`(parameter_name, "Int", 1)` 与 `(parameter_name, "Bool", true)`，不能只存裸值。
模型版本必须进入键并升级；不能因 Python 的 True==1 而跨域复用。
绑定、cache lookup、契约检查的顺序沿用逐次检查要求。

CLI 先解析参数声明，再解释 `--const ENABLED=true/false`（小写）；int 参数继续
接受十进制整数。CLI 不靠字符串猜测类型，不接受 bool 参数的 `1`、int 参数的
`true`。多 kernel 文件逐 kernel 按声明解析并报告冲突，不能共用无类型转换结果。

## 迁移与验收

实施前修订 ADR-005，保留 0.2.x 历史边界并明确新版本生效点；更新 ConstT、TIR、
frontend、staged 求值、runtime 绑定、缓存键、CLI 和规范打印。现有 int 默认值、
动态 scalar bool 和模块 bool 捕获规则不变。沿用 CONST 诊断族区分类型、缺值和
无效语境；不开放 Const[float/dtype/str] 或任意 Const[T]。

测试矩阵必须覆盖 True/False 与 1/0 的跨域拒绝、所有绑定入口、默认值覆盖、
缺参、嵌套 static-if、短路、延迟 static_assert、冷/热缓存隔离、CPU/Triton 路径
一致，以及 CLI 的合法/非法文本。达标前状态表保持 Designed。

## 实现证据与保守边界

M2-07b 已接通 ConstT/TIR 的 Bool 域、exact bool 全入口绑定、staged 布尔表达式
与别名、static-if、延迟 static_assert、声明驱动 CLI 和类型标签缓存键。
嵌套 static-if 中的断言只在所选特化路径求值；已知短路条件不构造未执行右侧
的 SMT 运算定义域，布尔常量传播仅局部折叠，保持共享 DAG 的有界迭代遍历。
Facts.bools 与 Facts.num 分离；SMT 使用独立 Bool 节点，未绑定时不会确认可达
反例，绑定后重新判定；bool 不进入 DimExpr、shape/grid 或整数 launch 契约。
registry semantic revision 为 4，proof encoding revision 为 3。

分支合并与循环仍沿用 M2-04 的保守 staging；失去稳定值身份的局部变量降为
Runtime，不把控制流相关别名错误地内联为某一分支的 Const 值。
`tests/test_const_bool.py` 覆盖各入口、类型拒绝、短路、嵌套分支/循环、缓存隔离、
CLI 冲突和 explain golden。11 组 CPU/GPU 对照通过 RTX 3090、PyTorch 2.10.0+cu128、
Triton 3.6.0、CUDA 12.8；命令：`PYTHONPATH=src python tests/gpu_const_bool_smoke.py`。

## 未选择的方案

继续用 Const[int] 约定 0/1 虽兼容，却不能表达精确 bool 参数域；隐式把 int 当 bool
会扩大歧义。将所有 Const 泛型一次开放，或把 ConstBoolT 做成第二套算子类型体系，
都会超出本议题。此次仅规划独立参数域，不改变 staging 与值类型正交的原则。
