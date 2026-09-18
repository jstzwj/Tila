"""Accepted ADR-003/004 decisions that constrain the rest of Batch B."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADR_INDEX = (ROOT / "docs/adr/README.md").read_text(encoding="utf-8")
ADR_003 = (ROOT / "docs/adr/003-buffer-stride-address-space.md").read_text(
    encoding="utf-8"
)
ADR_004 = (ROOT / "docs/adr/004-refinement-construction-syntax.md").read_text(
    encoding="utf-8"
)


def test_adrs_are_accepted_and_indexed():
    assert "[ADR-003](003-buffer-stride-address-space.md) | Accepted" in ADR_INDEX
    assert "[ADR-004](004-refinement-construction-syntax.md) | Accepted" in ADR_INDEX
    assert "状态：**Accepted**" in ADR_003
    assert "状态：**Accepted**" in ADR_004


def test_buffer_public_syntax_excludes_strides_and_address_space():
    assert "Buffer[Element, Shape, Access, Alignment]" in ADR_003
    assert "`Strides` 和 `AddressSpace` 不属于 v0 公共参数" in ADR_003
    assert "`Global` 是 Buffer v0 的隐式地址" in ADR_003
    assert "strides: BoundStrides | UnboundStrides" in ADR_003
    assert "单位：元素" in ADR_003


def test_buffer_ptr_linearization_is_explicitly_restricted():
    assert "rank-1 且运行时 `stride[0] == 1`" in ADR_003
    assert "rank 大于 1 或 stride 不为 1 时，`buf.ptr` 必须拒绝" in ADR_003
    assert "v0 也不隐式展平" in ADR_003
    assert "StrideEq[axis, value]" in ADR_003
    assert "这些名字暂不作为用户可写公共注解" in ADR_003


def test_refinement_syntax_and_domains_are_frozen():
    for spelling in ("Range[lo, hi]", "MultipleOf[k]", "Aligned[k]"):
        assert spelling in ADR_004
    assert "语义为闭区间" in ADR_004
    assert "`k` 为正的编译期整数" in ADR_004
    assert "`k` 为正的 2 次幂整数" in ADR_004
    assert "只接受整数字面量" in ADR_004


def test_refinement_fact_domains_and_positions_are_separate():
    assert "Scalar 组合继续使用 `dtype | refinement`" in ADR_004
    assert "`Aligned[k]` 是 memory alignment specification" in ADR_004
    assert "refinement 组合是逻辑合取" in ADR_004
    assert "refinement 不等于 assume" in ADR_004
    assert "`Range(...)`、`MultipleOf(...)` 和 `Aligned(...)` 不是" in ADR_004
