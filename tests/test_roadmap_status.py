"""roadmap 使用当前 M0–M6 体系，并锁定易混淆能力的唯一归属。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROADMAP = (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")


def test_roadmap_uses_current_milestones():
    for milestone in range(7):
        assert f"M{milestone}" in ROADMAP
    assert "Phase 0：" not in ROADMAP
    assert "Phase 1：" not in ROADMAP
    assert "Phase 2：" not in ROADMAP
    assert "Phase 3：" not in ROADMAP


def test_roadmap_has_no_parallel_tila2_migration():
    forbidden = "src/" + "tila2"
    assert forbidden not in ROADMAP
    assert "新语义实现已经直接位于 `src/tila`" in ROADMAP


def test_cross_stage_capabilities_have_explicit_ownership():
    required = (
        "M5 完整归属",
        "完整 effect/并发语义统一归 M4",
        "完整 `TILA-TARGET` 能力归 M5",
        "TypeVar 不能",
    )
    for statement in required:
        assert statement in ROADMAP


def test_roadmap_links_status_and_plan_as_authorities():
    assert "[status.md](status.md)" in ROADMAP
    assert "[../plan.md](../plan.md)" in ROADMAP
