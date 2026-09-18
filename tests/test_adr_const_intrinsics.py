"""Accepted ADR-005/006 decisions for Const and the intrinsic registry."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_INDEX = (ROOT / "docs/adr/README.md").read_text(encoding="utf-8")
ADR_005 = (ROOT / "docs/adr/005-const-type-domain.md").read_text(
    encoding="utf-8"
)
ADR_006 = (ROOT / "docs/adr/006-intrinsic-registry.md").read_text(
    encoding="utf-8"
)


def test_adrs_are_accepted_and_indexed():
    assert "[ADR-005](005-const-type-domain.md) | Accepted" in ADR_INDEX
    assert "[ADR-006](006-intrinsic-registry.md) | Accepted" in ADR_INDEX
    assert "状态：**Accepted**" in ADR_005
    assert "状态：**Accepted**" in ADR_006


def test_const_domain_and_staging_are_separate():
    assert "0.2.x 唯一可声明的 Const 参数形式" in ADR_005
    assert "Const[int, Refinement, ...]" in ADR_005
    assert "`Const[bool]`、`Const[float]`、`Const[dtype]`" in ADR_005
    assert "`ExactInt` 指 `type(value) is int`" in ADR_005
    assert "stagedness 不是第二套表面类型" in ADR_005
    assert "specialization-known bool predicate" in ADR_005
    assert "`static_assert(pred)`" in ADR_005


def test_registry_owns_metadata_not_complex_typing_rules():
    for field in (
        "surface_forms",
        "availability",
        "status_key",
        "checker_handler",
        "effect_rule",
        "tir_ops",
        "backend_expectation",
        "target_requirement",
    ):
        assert field in ADR_006
    assert "专用 checker handler 仍管理复杂" in ADR_006
    assert "跨模块 Python callable" in ADR_006
    assert "避免 import cycle 和导入时副作用" in ADR_006
    assert "typed TIR 是 backend 边界" in ADR_006


def test_registry_has_machine_checkable_completeness_gates():
    assert "由 catalog 派生" in ADR_006
    assert "availability 与 `docs/status.md` 对应行一致" in ADR_006
    assert "非 Pure 项有 effect rule" in ADR_006
    assert "schema/semantic revision 进入编译缓存版本" in ADR_006
    assert "M1-05 已完成不可变 catalog" in ADR_006
