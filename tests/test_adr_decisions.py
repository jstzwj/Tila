"""Accepted ADRs that constrain the M1 memory model."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_INDEX = (ROOT / "docs/adr/README.md").read_text(encoding="utf-8")
ADR_001 = (ROOT / "docs/adr/001-ptr-public-syntax.md").read_text(encoding="utf-8")
ADR_002 = (ROOT / "docs/adr/002-region-id-and-extent.md").read_text(encoding="utf-8")


def test_adrs_are_accepted_and_indexed():
    assert "[ADR-001](001-ptr-public-syntax.md) | Accepted" in ADR_INDEX
    assert "[ADR-002](002-region-id-and-extent.md) | Accepted" in ADR_INDEX
    assert "状态：**Accepted**" in ADR_001
    assert "状态：**Accepted**" in ADR_002


def test_ptr_public_syntax_carries_extent_not_region_id():
    assert "Ptr[Element, AddressSpace, Access, Extent, Alignment]" in ADR_001
    assert "`RegionId` **不是公共 Ptr 参数**" in ADR_001
    assert "`Alias[T, Extent]`" in ADR_001
    assert "`Extent`，这与当前 bounds" in ADR_001


def test_region_extent_and_alias_are_independent_dimensions():
    for invariant in (
        "RegionId = ParamRegion | BufferRegion | InternalRegion | UnknownRegion",
        "Extent   = LinearExtent(DimExpr) | UnknownExtent",
        "Bounds only consume Extent；Effects only consume RegionId",
        "AliasRelation = MustAlias | MayAlias | NoAlias",
        "不同 RegionIds do not imply noalias",
    ):
        assert invariant in ADR_002


def test_adr_defines_conservative_unknown_and_external_alias_rules():
    assert "来自不同外部参数的裸 Ptr 默认是 `MayAlias`" in ADR_002
    assert "UnknownExtent 不能证明 bounds" in ADR_002
    assert "UnknownRegion/MayAlias 不能证明" in ADR_002
    assert "RegionId 无法进入 bounds 算术" in ADR_002
    assert "Extent 无法作为 effect key" in ADR_002
