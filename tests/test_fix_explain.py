"""--explain 审计输出（surface-language.md §8）：section 头、义务证明链
（哪条谓词/哪条路线证出的）、Unknown 只展示不 raise、CLI explain /
check --explain、以及 explain() 的只读性。

已知局限（实现记录，见 runtime.JITFunction.explain 的 docstring）：
- grid 事实是 launch 期契约，explain 的 Facts 里 grid_facts 为空——依赖
  cdiv 战术（SafeUnderContract）的义务在 explain 里显示 Unknown；
- checker 的 assume 全局谓词不持久化到 TKernel；但每条义务自带
  preds_snapshot（访问点快照），assume 路线的证明经由它照常渲染
  （gather 用例验证这一点）。
"""

import importlib.util
import os

import tila as ti
from tila.cli import main

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")

N = ti.Dim("N")

SECTIONS = ("types:", "facts:", "hints:", "effects:", "aliases:", "obligations:",
            "warnings:", "notes:")


def _load_example(name: str):
    path = os.path.join(EXAMPLES, name)
    spec = importlib.util.spec_from_file_location(f"_explain_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# (a) add_kernel：六段 section 头 + 类型 + 每条义务 ProvenSafe + mask 路线
# ---------------------------------------------------------------------------

def test_add_kernel_explain_sections_and_routes():
    k = _load_example("add_kernel.py").add_kernel
    out = k.explain()

    for header in SECTIONS:
        assert header in out, f"missing section header: {header}"

    assert "offs : Block[i32" in out      # Block[i32, (BLOCK,)]
    assert "mask : Mask[" in out          # Mask[(BLOCK,)]

    # 三条义务（load x / load y / store out）全部 ProvenSafe
    assert len(k.tk.obligations) == 3
    assert out.count("state: ProvenSafe") == 3
    assert "state: Unknown" not in out

    # 证明路线：mask 子句里的谓词直证。谓词以 canonical 符号式渲染
    # （Pred 只存 DimExpr，无源码名回映射——`(BLOCK * pid0) + __lane1 < N`
    # 即源码 `offs < N` 的规范形态）。
    assert "direct predicate `" in out
    assert "< N`" in out
    assert "mask clause" in out


# ---------------------------------------------------------------------------
# (b) 无 mask 尾块：Unknown + "无上界谓词"路线；绝不 raise（不 launch）
# ---------------------------------------------------------------------------

def test_unmasked_tail_explain_unknown_without_raising():
    @ti.jit
    def tail(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
             BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
        pid = ti.program_id(0)
        offs = pid * BLOCK + ti.arange(0, BLOCK)
        v = ti.load(x, offs)              # 无 mask 尾块：Unknown
        ti.store(x, offs, v)

    out = tail.explain()                  # 副作用自由：不 raise TilaError
    assert "state: Unknown" in out
    assert "no upper-bound predicate" in out
    assert "< N" in out                   # 目标谓词渲染（coord < N）
    assert "state: ProvenSafe" not in out


# ---------------------------------------------------------------------------
# (c) gather + ti.assume：assume 谓词经由义务的 preds_snapshot 可审计
# ---------------------------------------------------------------------------

def test_gather_assume_route_visible_via_snapshot():
    # 局限记录：assume 的全局谓词不在 TKernel 上持久化（不改 checker），
    # 但 _add_obligation 把访问点快照放进 ob.preds_snapshot——explain 的
    # 证明池包含它，assume 路线完整渲染。
    @ti.jit
    def gather(idx_buf: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
               data: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
               out: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
               BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
        pid = ti.program_id(0)
        offs = pid * BLOCK + ti.arange(0, BLOCK)
        m = offs < N
        idx = ti.load(idx_buf, offs, mask=m, other=0)
        ti.assume((idx >= 0) & (idx < N))
        v = ti.load(data, idx)            # 数据依赖坐标：靠 assume 证明
        ti.store(out, offs, v, mask=m)

    out = gather.explain()
    # 三条义务（idx_buf / data / out）全部 ProvenSafe
    assert len(gather.tk.obligations) == 3
    assert out.count("state: ProvenSafe") == 3
    # data-load 的证明走访问点快照里的 assume 谓词（上界 idx < N +
    # 下界 idx >= 0）
    assert "global predicate snapshot at access point (assume/contract)" in out
    assert ">= 0" in out
    assert "< N" in out


# ---------------------------------------------------------------------------
# (d) CLI：explain 命令 + check --explain
# ---------------------------------------------------------------------------

def test_cli_explain_command(tmp_path, capsys):
    src = open(os.path.join(EXAMPLES, "add_kernel.py"),
               encoding="utf-8").read()
    f = tmp_path / "add_kernel.py"
    f.write_text(src, encoding="utf-8")
    rc = main(["explain", str(f)])
    out = capsys.readouterr().out
    assert rc == 0
    for header in SECTIONS:
        assert header in out
    assert "state: ProvenSafe" in out


def test_cli_check_explain_flag(capsys):
    rc = main(["check", os.path.join(EXAMPLES, "add_kernel.py"), "--explain"])
    out = capsys.readouterr().out
    assert rc == 0
    for header in SECTIONS:
        assert header in out
    assert "state: ProvenSafe" in out
    assert "< N`" in out                  # canonical 谓词渲染（offs < N）


# ---------------------------------------------------------------------------
# (e) 只读性：两次调用输出一致，tk 快照不变
# ---------------------------------------------------------------------------

def test_explain_is_read_only_and_idempotent():
    k = _load_example("add_kernel.py").add_kernel
    dump_before = k.tk.dump()
    report_before = k.last_report
    o1 = k.explain()
    o2 = k.explain()
    assert o1 == o2
    assert k.tk.dump() == dump_before
    assert k.last_report == report_before
