"""M1 退出审计的可执行门禁。"""

from pathlib import Path
import re

import pytest

import tila as ti
from tila import types as TY


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (ROOT / "docs/m1-exit-audit.md").read_text(encoding="utf-8")
PLAN = (ROOT / "plan.md").read_text(encoding="utf-8")
ROADMAP = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")


def test_audit_passes_all_exit_conditions_and_tracks_every_m1_adr():
    assert "状态：**PASS**" in AUDIT
    assert AUDIT.count("| PASS |") == 4
    for number in range(1, 7):
        assert f"ADR-{number:03d}" in AUDIT


def test_m1_task_ledger_and_roadmap_are_closed():
    for number in range(1, 8):
        assert re.search(rf"\| M1-{number:02d} \| DONE \|", PLAN)
    assert "| M1 | 已完成 |" in ROADMAP
    assert "[退出审计通过](m1-exit-audit.md)" in ROADMAP


def test_memory_type_axes_use_controlled_structured_types():
    ptr = ti.ReadPtr[ti.f32]
    buf = ti.Buffer[ti.f32, (16,)]
    assert isinstance(ptr.access, TY.Access)
    assert isinstance(ptr.address_space, TY.AddressSpace)
    assert isinstance(ptr.extent, TY.UnknownExtent)
    assert isinstance(ptr.alignment, TY.UnknownAlignment)
    assert isinstance(buf.strides, TY.UnboundStrides)
    assert not isinstance(ti.ReadOnly, str)


def test_structured_variants_reject_old_sentinel_protocols():
    with pytest.raises(TypeError, match="extent must be Extent"):
        TY.PtrT(ti.f32, extent=None)
    with pytest.raises(TypeError, match="alignment must be Alignment"):
        TY.BufferT(ti.f32, (), alignment=None)
    with pytest.raises(TypeError, match="strides must be"):
        TY.BufferT(ti.f32, (), strides=None)


def test_no_tests_are_hidden_by_skip_or_xfail_markers():
    marker = re.compile(r"pytest\.(?:skip|xfail)|pytest\.mark\.(?:skip|xfail)")
    offenders = []
    for path in (ROOT / "tests").glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        if marker.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == []


def test_deferred_pointer_subtraction_is_not_a_v0_promise():
    status = (ROOT / "docs/status.md").read_text(encoding="utf-8")
    type_system = (ROOT / "docs/type-system.md").read_text(encoding="utf-8")
    assert "`p - element_offset` | `Deferred`" in status
    assert "`p - offset`、指针比较和指针差" in type_system
