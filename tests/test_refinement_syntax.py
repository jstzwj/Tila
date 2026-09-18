"""ADR-004 / M1-04：统一 refinement 构造、校验、打印与事实传播。"""

import numpy as np
import pytest

import tila as ti
from tila import types as TY
from tila.dims import Cst
from tila.errors import TilaError, TilaLaunchContractError


def test_parameterized_refinements_use_canonical_subscript_syntax():
    range_ref = ti.Range[-2, 7]
    multiple_ref = ti.MultipleOf[16]
    alignment = ti.Aligned[32]

    assert isinstance(range_ref, TY.RangeRefinement)
    assert isinstance(multiple_ref, TY.MultipleOfRefinement)
    assert isinstance(alignment, TY.Alignment)
    assert range_ref.text == "Range[-2, 7]"
    assert multiple_ref.text == "MultipleOf[16]"
    assert alignment.text == "Aligned[32]"


@pytest.mark.parametrize(
    "make",
    [
        lambda: ti.Range(1, 2),
        lambda: ti.MultipleOf(4),
        lambda: ti.Aligned(16),
        lambda: ti.Range[1],
        lambda: ti.Range[1, 2, 3],
        lambda: ti.Range[True, 2],
        lambda: ti.Range[3, 2],
        lambda: ti.MultipleOf[True],
        lambda: ti.MultipleOf[0],
        lambda: ti.MultipleOf[-2],
        lambda: ti.Aligned[True],
        lambda: ti.Aligned[0],
        lambda: ti.Aligned[3],
    ],
)
def test_invalid_refinement_construction_is_rejected(make):
    with pytest.raises(TypeError):
        make()


def test_refinement_positions_and_contradictions_are_checked():
    with pytest.raises(TypeError):
        _ = ti.f32 | ti.MultipleOf[2]
    with pytest.raises(TypeError):
        _ = ti.Const[int, ti.Aligned[16]]
    with pytest.raises(TypeError):
        _ = ti.Const[int, ti.Range[1, 3], ti.MultipleOf[4]]
    with pytest.raises(TypeError):
        _ = ti.Const[int, ti.PowerOfTwo, ti.MultipleOf[6]]


def test_refinement_normalization_deduplicates_stably():
    @ti.jit
    def configured(tile: ti.Const[
            int, ti.Range[32, 1024], ti.PowerOfTwo,
            ti.MultipleOf[16], ti.MultipleOf[16]] = 64):
        pass

    refinements = configured.tk.consts[0].refinements
    assert [r.text for r in refinements] == [
        "Range[32, 1024]", "PowerOfTwo", "MultipleOf[16]",
    ]


def test_scalar_refinement_printing_and_launch_validation():
    spec = ti.i32 | (ti.Range[0, 15], ti.MultipleOf[4])
    assert spec.describe() == "i32 | Range[0, 15] | MultipleOf[4]"

    @ti.jit
    def scalar(n: ti.i32 | (ti.Range[0, 15], ti.MultipleOf[4])):
        pass

    scalar[(1,)](12)
    with pytest.raises(TilaLaunchContractError) as exc:
        scalar[(1,)](14)
    assert exc.value.code == "TILA-TYPE-103"
    assert "MultipleOf[4]" in exc.value.render()


def test_float_positive_remains_supported_but_integer_only_predicates_do_not():
    @ti.jit
    def scale(value: ti.f32 | ti.Positive):
        pass

    scale[(1,)](0.25)
    with pytest.raises(TilaLaunchContractError):
        scale[(1,)](0.0)


def test_const_refinement_violation_uses_canonical_text():
    @ti.jit
    def configured(tile: ti.Const[int, ti.Range[8, 16],
                                  ti.MultipleOf[4]] = 8):
        pass

    configured.materialize({"tile": 12})
    with pytest.raises(TilaError) as exc:
        configured.materialize({"tile": 10})
    assert exc.value.code == "TILA-CONST-003"
    assert "MultipleOf[4]" in exc.value.render()


def test_range_and_multiple_of_enter_checker_facts():
    @ti.jit
    def indexed(x: ti.Buffer[ti.f32, (8,), ti.ReadOnly],
                index: ti.i32 | (ti.Range[0, 7], ti.MultipleOf[2])):
        value = ti.load(x, index)

    assert indexed.tk.sym_lo["index"] == Cst(0)
    assert indexed.tk.sym_hi["index"] == Cst(8)
    obligation = indexed.tk.obligations[0]
    predicates = {predicate.key() for predicate in obligation.preds_snapshot}
    assert "==:(index % 2):0" in predicates
    indexed[(1,)](np.arange(8, dtype=np.float32), 6)


def test_combined_ranges_propagate_their_intersection_independent_of_order():
    @ti.jit
    def bounded(value: ti.i32 | (ti.Range[2, 8], ti.Range[0, 10],
                                 ti.Positive)):
        pass

    assert bounded.tk.sym_lo["value"] == Cst(2)
    assert bounded.tk.sym_hi["value"] == Cst(9)


def test_verified_alignment_becomes_a_launch_fact():
    @ti.jit
    def aligned(x: ti.Buffer[ti.f32, (8,), ti.ReadOnly, ti.Aligned[16]]):
        pass

    array = np.zeros(8, dtype=np.float32)
    aligned[(1,)](array)
    region = aligned.tk.buffers[0].region_id
    assert aligned.tk.runtime_alignments == {region: 16}
    assert "aligned BufferRegion[0:x]: 16 bytes" in aligned.explain()
