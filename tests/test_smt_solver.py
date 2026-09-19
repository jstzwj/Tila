"""M2-03 solver semantics, bounded failure and cache/trust isolation."""
from dataclasses import replace
import builtins

import numpy as np
import pytest
import tila as ti
from tila import predicates as P
from tila.dims import Cst, Sym, Add, FloorDiv, Mod
from tila.facts import (Facts, Pred, Obligation, evaluate_obligation,
                        PROVEN_SAFE, PROVEN_UNSAFE, UNKNOWN)
from tila.solver import ProofConfig, ProofSession, Encoder, _CACHE
from tila.errors import TilaError


def negated_obligation():
    mask = P.negate(P.atom(Pred(">=", Sym("i"), Cst(8))))
    return Obligation("load", "x", 0, Sym("i"), Cst(8), mask)


def test_default_solver_proves_negation_and_replayable_query():
    session = ProofSession()
    result = session.prove(negated_obligation(), Facts(), {"i"})
    assert result.verdict == PROVEN_SAFE
    assert "Z3" in result.render()
    solver = session.z.Solver()
    solver.from_string(result.query)
    assert solver.check() == session.z.unsat


def test_nontrivial_disjunction_not_reduced_to_common_atoms():
    a = P.atom(Pred("<", Sym("i"), Cst(4)))
    b = P.atom(Pred("==", Sym("i"), Cst(7)))
    ob = replace(negated_obligation(), mask=P.disjunction(a, b))
    assert evaluate_obligation(ob, Facts(), {"i"}).verdict == PROVEN_SAFE


def test_general_contradiction_preserves_user_assumption():
    origin = frozenset({P.Origin(P.USER, 3, "assume")})
    preds = (Pred("<", Sym("i"), Cst(1), origin),
             Pred(">", Sym("i"), Cst(2), origin))
    result = evaluate_obligation(replace(negated_obligation(), preds_snapshot=preds), Facts(), set())
    assert result.verdict == UNKNOWN
    assert "inconsistent premises" in result.reason
    assert origin <= result.dependencies
    exhausted = ProofSession(ProofConfig(max_queries=0)).prove(
        replace(negated_obligation(), preds_snapshot=preds), Facts(), set())
    assert exhausted.verdict == UNKNOWN and origin <= exhausted.dependencies


def test_sat_constant_vs_abstract_candidate():
    ob = Obligation("load", "x", 0, Cst(9), Cst(8), P.TRUE)
    result = evaluate_obligation(ob, Facts(), set())
    assert result.verdict == PROVEN_UNSAFE
    assert ("coordinate", "9") in result.candidate_counterexample
    result = evaluate_obligation(replace(ob, coord=Sym("loaded")), Facts(), set())
    assert result.verdict == UNKNOWN and result.candidate_counterexample
    assert "reachability" in result.reason
    abstract = Facts(sym_lo={"loaded": Cst(9)}, sym_hi={"loaded": Cst(10)})
    result = evaluate_obligation(replace(ob, coord=Sym("loaded")), abstract, set())
    assert result.verdict == UNKNOWN and result.candidate_counterexample


@pytest.mark.parametrize("config,reason", [
    (ProofConfig(timeout_ms=0), "per-query"),
    (ProofConfig(rlimit=0), "per-query"),
    (ProofConfig(total_ms=0), "cumulative"),
    (ProofConfig(max_queries=0), "cumulative"),
    (ProofConfig(max_nodes=1), "construction"),
])
def test_budget_exhaustion_is_explicit_unknown(config, reason):
    result = ProofSession(config).prove(negated_obligation(), Facts(), {"i"})
    assert result.verdict == UNKNOWN
    assert reason in result.reason


def test_actual_z3_resource_unknown_is_not_safety():
    result = ProofSession(ProofConfig(rlimit=1)).prove(negated_obligation(), Facts(), {"i"})
    assert result.verdict == UNKNOWN
    assert "Z3 unknown" in result.reason


def test_solver_timeout_reason_is_preserved(monkeypatch):
    session = ProofSession(ProofConfig(cache_entries=0))
    solver_type = type(session.z.Solver())
    monkeypatch.setattr(solver_type, "check", lambda *args: session.z.unknown)
    monkeypatch.setattr(solver_type, "reason_unknown", lambda self: "timeout")
    result = session.prove(negated_obligation(), Facts(), {"i"})
    assert result.verdict == UNKNOWN and "timeout" in result.reason


def test_large_integer_construction_budget():
    ob = replace(negated_obligation(), extent=Cst(1 << 5000))
    result = evaluate_obligation(ob, Facts(), {"i"})
    assert result.verdict == UNKNOWN and "construction budget" in result.reason


def test_invalid_environment_budget_is_configuration_error(monkeypatch):
    monkeypatch.setenv("TILA_PROOF_TIMEOUT_MS", "unlimited")
    with pytest.raises(TilaError) as exc:
        ProofSession()
    assert exc.value.code == "TILA-PROOF-001"


def test_cumulative_queries_are_shared_across_accesses():
    session = ProofSession(ProofConfig(max_queries=2, cache_entries=0))
    assert session.prove(negated_obligation(), Facts(), {"i"}).verdict == PROVEN_SAFE
    result = session.prove(negated_obligation(), Facts(), {"i"})
    assert result.verdict == UNKNOWN and "cumulative" in result.reason


def test_cache_reuses_success_but_isolates_origin_binding_and_budget():
    _CACHE.clear()
    session = ProofSession()
    ob = negated_obligation()
    first = session.prove(ob, Facts(), {"i"})
    assert first.verdict == PROVEN_SAFE
    assert session.prove(ob, Facts(), {"i"}) == first
    assert session.cache_hits == 1
    assumed = replace(ob, preds_snapshot=(Pred(">=", Sym("i"), Cst(0),
                         frozenset({P.Origin(P.USER, 5)})),))
    proof = session.prove(assumed, Facts(), {"i"})
    assert P.USER in {o.kind for o in proof.dependencies}
    assert session.cache_hits == 1
    result = session.prove(ob, Facts(num={"i": -1}), set())
    assert result.verdict == PROVEN_UNSAFE
    assert ProofSession(ProofConfig(rlimit=1)).prove(ob, Facts(), {"i"}).verdict == UNKNOWN
    assert ProofSession().prove(ob, Facts(), {"i"}).verdict == PROVEN_SAFE


def test_cache_is_bounded_and_unknown_not_retained():
    _CACHE.clear()
    session = ProofSession(ProofConfig(cache_entries=2))
    for line in range(4):
        session.prove(replace(negated_obligation(), loc_line=line), Facts(), {"i"})
    assert len(_CACHE) == 2
    before = tuple(_CACHE)
    session.prove(negated_obligation(), Facts(), set())
    assert tuple(_CACHE) == before


def test_missing_solver_is_configuration_error(monkeypatch):
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "z3":
            raise ImportError("test missing solver")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(TilaError) as exc:
        ProofSession()
    assert exc.value.code == "TILA-PROOF-001"


@pytest.mark.parametrize("a,b", [(-7, 3), (7, -3), (-7, -3), (7, 3)])
def test_math_floor_division_and_remainder_match_adr007(a, b):
    session = ProofSession()
    encoder = Encoder(negated_obligation(), Facts(), set(), session.config)
    for expr, expected in ((FloorDiv(Cst(a), Cst(b)), a // b),
                           (Mod(Cst(a), Cst(b)), a % b)):
        assert session.z.simplify(encoder.expr(expr)).as_long() == expected


@pytest.mark.parametrize("dtype,value,expected", [
    ("i8", 128, -128), ("u8", -1, 255), ("i32", 2**31, -2**31),
    ("u64", -1, 2**64 - 1),
])
def test_bitvector_narrowing_and_overflow(dtype, value, expected):
    ob = replace(negated_obligation(), value_defs=(("v", ("wrap", dtype, Cst(value), Cst(0))),))
    session = ProofSession()
    encoder = Encoder(ob, Facts(), set(), session.config)
    assert session.z.simplify(encoder.expr(Sym("v"))).as_long() == expected


def test_undefined_division_is_not_vacuously_safe():
    ob = Obligation("load", "x", 0, FloorDiv(Cst(1), Cst(0)), Cst(8), P.TRUE)
    result = evaluate_obligation(ob, Facts(), set())
    assert result.verdict == UNKNOWN and "domain" in result.reason
    masked = replace(ob, mask=P.conjunction(
        P.atom(Pred(">=", ob.coord, Cst(0))), P.atom(Pred("<", ob.coord, Cst(8)))))
    result = evaluate_obligation(masked, Facts(), set())
    assert result.verdict == UNKNOWN and "domain" in result.reason


@pytest.mark.parametrize("dtype,value,count,expected", [
    ("i8", -8, 2, -2), ("u8", 248, 2, 62),
])
def test_right_shift_signedness(dtype, value, count, expected):
    ob = replace(negated_obligation(), value_defs=(("v", (">>", dtype, Cst(value), Cst(count))),))
    session = ProofSession()
    encoder = Encoder(ob, Facts(), set(), session.config)
    assert session.z.simplify(encoder.expr(Sym("v"))).as_long() == expected


def test_launch_binding_changes_do_not_hit_proof_cache():
    _CACHE.clear()
    session = ProofSession()
    session.prove(negated_obligation(), Facts(launch_bindings=(("N", 8),)), {"i"})
    session.prove(negated_obligation(), Facts(launch_bindings=(("N", 9),)), {"i"})
    assert session.cache_hits == 0


def test_bitwise_index_proved_and_executed_without_mask():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (8,), ti.WriteOnly], index: ti.i32):
        bounded = index & 7
        ti.store(out, bounded, 42)
    for index in (-1, 0, 8, 2**31 - 1):
        out = np.zeros(8, dtype=np.int32)
        k[(1,)](out, index)
        assert out[index & 7] == 42 and out.sum() == 42
    assert "Z3" in k.last_proof_results[0][1].render()


def test_unsigned_cast_range_is_proved_from_bitwidth():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (256,), ti.WriteOnly], index: ti.i32):
        bounded = ti.cast[ti.u8](index)
        ti.store(out, bounded, 7)
    out = np.zeros(256, dtype=np.int32)
    k[(1,)](out, -1)
    assert out[255] == 7


def test_wraparound_prevents_false_mathematical_proof():
    index, bounded = Sym("index"), Sym("bounded")
    ob = Obligation("store", "out", 0, bounded, Cst(8),
        P.atom(Pred("<", bounded, Cst(8))),
        value_types=(("index", "i32"), ("bounded", "i32")),
        value_defs=(("bounded", ("wrap", "i32", Add(index, Cst(1)), Cst(0))),))
    result = evaluate_obligation(ob, Facts(num={"index": 2**31 - 1}), {"index"})
    assert result.verdict == PROVEN_UNSAFE
    assert ("coordinate", str(-2**31)) in result.candidate_counterexample


def test_unknown_boolean_row_and_column_are_distinct_lanes():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (8,), ti.ReadOnly],
          out: ti.Buffer[ti.f32, (8, 8), ti.WriteOnly]):
        lane = ti.arange(0, 8)
        value = ti.load(x, lane)
        positive = value > 0.0
        rows = lane[:, None] + 8
        cols = lane[None, :]
        ti.store(out, (rows, cols), ti.zeros((8, 8), ti.f32),
                 mask=positive[:, None] & ~positive[None, :])
    # Different vector elements can have different signs, so the mask is not
    # contradictory. Treating row/column booleans as equal would prove this OOB safe.
    with pytest.raises(TilaError) as exc:
        k.materialize()
    assert exc.value.code == "TILA-BOUNDS-001"


def test_repeated_unknown_lane_mapping_preserves_identity():
    unknown = P.unknown(shape=(Cst(8),))
    shape = (Cst(8), Cst(1))
    mapping = ("expand", 1, (Cst(8),))
    a = P.map_atoms(unknown, lambda e: e, shape, mapping)
    b = P.map_atoms(unknown, lambda e: e, shape, mapping)
    ob = Obligation("load", "x", 0, Cst(9), Cst(8), P.conjunction(a, P.negate(b)))
    result = evaluate_obligation(ob, Facts(), set())
    assert result.verdict == PROVEN_SAFE and "unreachable" in result.reason
