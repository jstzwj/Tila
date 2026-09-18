"""ADR-006 intrinsic catalog and cross-component completeness gates."""

from dataclasses import replace
from pathlib import Path
import re

import pytest

import tila
from tila import _Intrinsic
from tila import tir as T
from tila.checker import CHECKER_INTRINSIC_HANDLERS, Checker
from tila.errors import TilaError
from tila.frontend import INTRINSIC_NAMES
from tila.interp import INTERPRETER_TIR_HANDLERS, INTERPRETER_TIR_OPS
from tila.intrinsics import (
    Backend,
    BoundsRule,
    CHECKER_INTRINSIC_NAMES,
    EffectRule,
    INTRINSICS,
    INTRINSIC_REGISTRY,
    INTRINSIC_REGISTRY_SCHEMA_VERSION,
    INTRINSIC_REGISTRY_SEMANTIC_REVISION,
    METHOD_INTRINSIC_NAMES,
    PUBLIC_INTRINSIC_NAMES,
    SurfaceForm,
    call_shape_problem,
    intrinsic_names_for_surface,
    validate_registry,
)
from tila.lowering import TRITON_TIR_HANDLERS, TRITON_TIR_OPS
from tila.runtime import _triton_cache_key


ROOT = Path(__file__).resolve().parents[1]
STATUS = (ROOT / "docs/status.md").read_text(encoding="utf-8")
INTRINSIC_DOCS = (ROOT / "docs/intrinsics.md").read_text(encoding="utf-8")


def _checker_handlers():
    return {
        name.removeprefix("_in_")
        for name in dir(Checker)
        if name.startswith("_in_") and callable(getattr(Checker, name))
    }


def test_public_intrinsic_registry_has_no_duplicates():
    assert len(PUBLIC_INTRINSIC_NAMES) == len(set(PUBLIC_INTRINSIC_NAMES))
    assert len(CHECKER_INTRINSIC_NAMES) == len(set(CHECKER_INTRINSIC_NAMES))
    assert tuple(INTRINSIC_REGISTRY) == CHECKER_INTRINSIC_NAMES
    assert INTRINSIC_REGISTRY_SCHEMA_VERSION >= 1
    assert INTRINSIC_REGISTRY_SEMANTIC_REVISION >= 1


def test_public_placeholders_and_frontend_are_exactly_aligned():
    expected = set(PUBLIC_INTRINSIC_NAMES)
    assert set(INTRINSIC_NAMES) == expected
    assert expected <= set(tila.__all__)
    for name in expected:
        value = getattr(tila, name)
        assert isinstance(value, _Intrinsic), name
        assert value.name == name


def test_checker_handlers_are_exhaustive_without_hidden_flat_intrinsics():
    assert _checker_handlers() == set(CHECKER_INTRINSIC_NAMES)
    expected_ids = {spec.checker_handler for spec in INTRINSICS}
    assert set(CHECKER_INTRINSIC_HANDLERS) == expected_ids
    assert all(callable(handler)
               for handler in CHECKER_INTRINSIC_HANDLERS.values())


def test_method_intrinsics_are_methods_only():
    for name in METHOD_INTRINSIC_NAMES:
        assert name not in INTRINSIC_NAMES
        assert name not in tila.__all__
        assert not hasattr(tila, name)


def test_exp2_is_a_real_public_placeholder():
    assert "exp2" in tila.__all__
    assert tila.exp2.name == "exp2"


def test_surface_forms_are_catalog_data_not_frontend_side_lists():
    public_calls = set(intrinsic_names_for_surface(SurfaceForm.PUBLIC_CALL))
    assert public_calls == set(PUBLIC_INTRINSIC_NAMES) - {"cast", "range"}
    assert set(intrinsic_names_for_surface(SurfaceForm.SUBSCRIPT_CALL)) == {"cast"}
    assert set(intrinsic_names_for_surface(SurfaceForm.LOOP_FORM)) == {"range"}
    assert set(intrinsic_names_for_surface(SurfaceForm.METHOD_CALL)) == {"any", "all"}


def test_every_spec_has_complete_machine_metadata():
    assert validate_registry(INTRINSICS) == INTRINSICS
    for spec in INTRINSICS:
        assert spec.surface_forms
        assert spec.status_key == f"intrinsic:{spec.name}"
        assert spec.docs_anchor
        assert spec.checker_handler
        assert spec.target_requirement.value
        assert Backend.CHECKER in spec.backend_expectation
        assert spec.arity.min_positional <= spec.arity.max_positional


def test_status_keys_and_availability_are_exactly_aligned():
    pairs = re.findall(
        r"^\| `(intrinsic:[a-z0-9_]+)` \| "
        r"`(Implemented|Partial|Designed|Deferred)` \|",
        STATUS,
        flags=re.MULTILINE,
    )
    status_index = dict(pairs)
    assert len(status_index) == len(pairs), "duplicate intrinsic status key"
    assert status_index == {
        spec.status_key: spec.availability.value for spec in INTRINSICS
    }


def test_docs_anchors_resolve_to_real_signature_sections():
    headings = set(re.findall(r"^### ([0-9]+\.[0-9]+) ", INTRINSIC_DOCS,
                              flags=re.MULTILINE))
    assert {spec.docs_anchor for spec in INTRINSICS} <= headings


def test_declared_tir_ops_resolve_and_have_required_backend_handlers():
    for handlers in (INTERPRETER_TIR_HANDLERS, TRITON_TIR_HANDLERS):
        assert len(handlers) == len(set(handlers.values()))
    for spec in INTRINSICS:
        for op in spec.tir_ops:
            cls = getattr(T, op, None)
            assert isinstance(cls, type), (spec.name, op)
            assert issubclass(cls, (T.TOperand, T.TStmt)), (spec.name, op)
        if Backend.INTERPRETER in spec.backend_expectation:
            assert set(spec.tir_ops) <= INTERPRETER_TIR_OPS, spec.name
        if Backend.TRITON in spec.backend_expectation:
            assert set(spec.tir_ops) <= TRITON_TIR_OPS, spec.name


def test_effect_and_bounds_are_independent_and_explicit():
    memory = {EffectRule.READ_MEMORY, EffectRule.WRITE_MEMORY}
    for spec in INTRINSICS:
        if spec.effect_rule in memory:
            assert spec.bounds_rule is BoundsRule.BUFFER_OR_PTR_ACCESS
        else:
            assert spec.bounds_rule is None


def test_registry_and_backend_views_are_immutable():
    with pytest.raises(TypeError):
        INTRINSIC_REGISTRY["new"] = INTRINSICS[0]
    with pytest.raises(TypeError):
        TRITON_TIR_HANDLERS["Fake"] = "fake"
    with pytest.raises(TypeError):
        INTERPRETER_TIR_HANDLERS["Fake"] = "fake"


def test_registry_semantic_revision_is_part_of_compilation_cache_key():
    marker = object()
    key = _triton_cache_key(marker, {"BLOCK": 128})
    assert key[0] == INTRINSIC_REGISTRY_SEMANTIC_REVISION
    assert key[1] == id(marker)
    assert key[2] == (("BLOCK", 128),)


def test_validator_rejects_missing_metadata_and_duplicate_identity():
    first = INTRINSICS[0]
    with pytest.raises(ValueError, match="duplicate intrinsic name"):
        validate_registry((first, first))
    with pytest.raises(ValueError, match="requires declared TIR ops"):
        validate_registry((replace(first, tir_ops=()),))
    load = INTRINSIC_REGISTRY["load"]
    with pytest.raises(ValueError, match="memory effect and bounds rule"):
        validate_registry((replace(load, bounds_rule=None),))


def test_call_shape_gate_owns_arity_keywords_and_duplicates():
    dot = INTRINSIC_REGISTRY["dot"]
    assert call_shape_problem(dot, 2, ("acc",)) is None
    assert "exactly 2" in call_shape_problem(dot, 1, ())
    assert "unknown keyword" in call_shape_problem(dot, 2, ("mask",))
    assert "duplicate keyword" in call_shape_problem(dot, 2, ("acc", "acc"))


def test_frontend_enforces_registry_surface_forms_and_keywords():
    with pytest.raises(TilaError) as cast_error:
        @tila.jit
        def bad_cast():
            value = tila.cast(1)
    assert cast_error.value.code == "TILA-SYN-035"

    with pytest.raises(TilaError) as range_error:
        @tila.jit
        def bad_range():
            value = tila.range(0, 4)
    assert range_error.value.code == "TILA-SYN-028"

    with pytest.raises(TilaError) as keyword_error:
        @tila.jit
        def bad_keyword():
            value = tila.exp(1.0, typo=True)
    assert keyword_error.value.code == "TILA-SYN-036"
    assert "unknown keyword" in keyword_error.value.title
