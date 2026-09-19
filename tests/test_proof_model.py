"""M2-02: formula size, lane/scope isolation and auditable trust boundaries."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import tila as ti
from tila import predicates as P
from tila.dims import Cst, Sym
from tila.errors import TilaError, TilaLaunchContractError
from tila.facts import (Facts, Obligation, Pred, ProofResult, evaluate_obligation,
                        explain_obligation, PROVEN_SAFE, UNKNOWN, EXEMPTED)

N = ti.Dim("N")


def obligation(mask=P.TRUE, coord=Sym("i"), extent=Sym("N"), **kwargs):
    return Obligation("load", "x", 0, coord, extent, mask, **kwargs)


def test_upper_bound_does_not_prove_nonnegative_index():
    upper = P.atom(Pred("<", Sym("i"), Sym("N")))
    result = evaluate_obligation(obligation(upper), Facts(), set())
    assert result.verdict == UNKNOWN
    assert "lower bound is unproven" in result.render()
    lower = P.atom(Pred(">=", Sym("i"), Cst(0)))
    assert evaluate_obligation(obligation(P.conjunction(lower, upper)),
                               Facts(), set()).verdict == PROVEN_SAFE


def test_boolean_product_does_not_expand_to_exponential_dnf():
    guard = P.atom(Pred("<", Sym("i"), Sym("N")))
    formula = guard
    # DNF would need 2**60 clauses. Shared formulas stay small and prove the
    # same guard without dropping any disjunct.
    for _ in range(60):
        formula = P.conjunction(formula, P.disjunction(P.unknown(), P.unknown()))
    assert len(tuple(P.nodes(formula))) == 241
    assert evaluate_obligation(obligation(formula), Facts(), {"i"}).verdict == PROVEN_SAFE
    assert evaluate_obligation(obligation(P.disjunction(formula, P.unknown())),
                               Facts(), {"i"}).verdict == UNKNOWN


def test_deep_shared_formula_walk_is_iterative():
    formula = P.atom(Pred("<", Sym("i"), Sym("N")))
    for _ in range(1500):
        formula = P.conjunction(formula, P.disjunction(formula, P.unknown()))
    assert len(tuple(P.nodes(formula))) == 4501
    assert evaluate_obligation(obligation(formula), Facts(), {"i"}).verdict == PROVEN_SAFE


def test_result_preserves_multiple_trust_sources_and_locations():
    user = P.Origin(P.USER, 17, "assume lower bound")
    checked = P.Origin(P.CHECKED, 4, "host checked upper bound")
    low = Pred(">=", Sym("i"), Cst(0), frozenset({user}))
    facts = Facts()
    facts.add_pred(Pred("<", Sym("i"), Sym("N"), frozenset({checked})))
    ob = obligation(preds_snapshot=(low,), loc_line=21)
    result = evaluate_obligation(ob, facts, set())
    assert isinstance(result, ProofResult)
    assert result.verdict == PROVEN_SAFE
    assert result.dependencies == frozenset({user, checked})
    assert result.summary == "SafeUnderContract [UserAssumption]"
    assert result.source_locations == (4, 17, 21)
    assert explain_obligation(ob, facts, set()) == result.render()
    assert evaluate_obligation(ob, facts, set()) == result
    with pytest.raises(FrozenInstanceError):
        result.verdict = UNKNOWN
    with pytest.raises(FrozenInstanceError):
        ob.path = P.FALSE


def test_unsafe_is_local_exemption_and_never_a_fact():
    facts = Facts()
    ob = obligation()
    waived = evaluate_obligation(replace(ob, kind="unsafe_load"), facts, set())
    assert waived.verdict == EXEMPTED
    assert not waived.dependencies
    assert evaluate_obligation(ob, facts, set()).verdict == UNKNOWN
    assert not facts.preds


def test_symbolic_grid_is_pending_until_launch_checked():
    ob = obligation(coord=Sym("pid0"))
    facts = Facts(grid_facts={"pid0": (Sym("N"), Cst(1))})
    result = evaluate_obligation(ob, facts, {"pid0"})
    assert result.verdict == UNKNOWN and result.pending_contracts
    assert P.CHECKED not in {d.kind for d in result.dependencies}
    facts.grid_checked = True
    result = evaluate_obligation(ob, facts, {"pid0"})
    assert result.verdict == PROVEN_SAFE
    assert result.summary == "SafeUnderContract"


def test_contradictory_assumptions_are_not_unconditional_safety():
    origin = frozenset({P.Origin(P.USER, 12, "assume")})
    preds = (Pred(">=", Sym("i"), Cst(0), origin),
             Pred("<", Sym("i"), Cst(0), origin))
    result = evaluate_obligation(obligation(preds_snapshot=preds), Facts(), set())
    assert result.verdict == UNKNOWN
    assert "inconsistent premises" in result.reason
    assert result.dependencies == origin


def test_masked_out_access_is_reported_as_unreachable():
    mask = P.atom(Pred("<", Cst(12), Cst(8)))
    result = evaluate_obligation(obligation(mask, Cst(12), Cst(8)), Facts(), set())
    assert result.verdict == PROVEN_SAFE
    assert "unreachable" in result.reason


def test_masked_interval_candidate_does_not_claim_reachability():
    result = evaluate_obligation(obligation(P.unknown(), Cst(12), Cst(8)), Facts(), set())
    assert result.verdict == UNKNOWN
    assert result.candidate_counterexample
    assert "reachability" in result.render()


def test_branch_join_unions_origins_and_drops_branch_only_bounds():
    a, b = Facts(), Facts()
    pred = Pred(">=", Sym("i"), Cst(0))
    a.add_pred(replace(pred, origins=frozenset({P.Origin(P.USER, 1)})))
    b.add_pred(replace(pred, origins=frozenset({P.Origin(P.USER, 2)})))
    a.sym_lo["i"] = Cst(0)
    joined = a.intersect(b)
    assert {o.line for o in joined.preds[pred.key()].origins} == {1, 2}
    assert not joined.sym_lo


def test_scalar_path_conditions_prove_only_inside_guard():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly], i: ti.i32):
        if i >= 0 and i < N:
            v = ti.load(x, i)
        w = ti.load(x, i)

    inside, outside = k.tk.obligations
    assert inside.path is not P.TRUE and outside.path is P.TRUE
    assert evaluate_obligation(inside, Facts(), set()).verdict == PROVEN_SAFE
    assert evaluate_obligation(outside, Facts(), set()).verdict == UNKNOWN


def test_assume_cannot_prove_an_earlier_access():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly], i: ti.i32):
        v = ti.load(x, i)
        ti.assume(i >= 0 and i < N)
        w = ti.load(x, i)

    earlier, later = k.tk.obligations
    assert evaluate_obligation(earlier, Facts(), set()).verdict == UNKNOWN
    proof = evaluate_obligation(later, Facts(), set())
    assert proof.verdict == PROVEN_SAFE
    assert {o.kind for o in proof.dependencies} == {P.USER}
    assert all(o.line for o in proof.dependencies)


def test_assume_from_branch_and_zero_trip_loop_does_not_escape():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly], i: ti.i32, flag: ti.i32,
          END: ti.Const[int] = 0):
        if flag > 0:
            ti.assume(i >= 0 and i < N)
        for j in ti.range(0, END):
            ti.assume(i >= 0 and i < N)
        v = ti.load(x, i)

    ob = k.tk.obligations[0]
    assert not ob.preds_snapshot
    assert evaluate_obligation(ob, Facts(), set()).verdict == UNKNOWN


def test_reused_loop_variable_has_fresh_symbol_and_local_interval():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (8,), ti.ReadOnly],
          FIRST: ti.Const[int] = 8, SECOND: ti.Const[int] = 16):
        for j in ti.range(0, FIRST):
            v = ti.load(x, j)
        for j in ti.range(8, SECOND):
            v = ti.unsafe_load(x, j)

    a, b = k.tk.obligations
    assert a.coord != b.coord
    assert dict(a.sym_hi)[a.coord.name] == Sym("FIRST")
    assert dict(b.sym_hi)[b.coord.name] == Sym("SECOND")
    assert evaluate_obligation(a, Facts(num={"FIRST": 8}), set()).verdict == PROVEN_SAFE


def test_entry_assumption_is_not_a_loop_carried_invariant():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly], i: ti.i32,
          END: ti.Const[int] = 2):
        ti.assume(i >= 0 and i < N)
        for j in ti.range(0, END):
            v = ti.load(x, i)
            i = i - 1

    # The first iteration might be safe, but the next can use -1.
    assert evaluate_obligation(k.tk.obligations[0], Facts(), set()).verdict == UNKNOWN


def test_broadcast_row_mask_cannot_prove_column_coordinates():
    @ti.jit
    def wrong(x: ti.Buffer[ti.f32, (N, N), ti.ReadOnly]):
        lane = ti.arange(0, 8)
        rows = lane[:, None]
        cols = lane[None, :]
        m = rows < N
        v = ti.load(x, (rows, cols), mask=m)

    a, b = wrong.tk.obligations
    assert a.coord != b.coord
    assert any(n.op == "map" for n in P.nodes(a.mask))
    with pytest.raises(TilaError) as exc:
        wrong.materialize()
    assert exc.value.code == "TILA-BOUNDS-001"


def test_expanded_mask_and_expanded_coordinate_share_lane_mapping():
    @ti.jit
    def right(x: ti.Buffer[ti.f32, (N, N), ti.ReadOnly]):
        lane = ti.arange(0, 8)
        valid = lane < N
        v = ti.load(x, (lane[:, None], lane[None, :]),
                    mask=valid[:, None] & valid[None, :])

    right.materialize()
    assert all(r.verdict == PROVEN_SAFE for _, r in right.last_proof_results)


def test_unknown_boolean_identity_and_negation_survive_assignment():
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly]):
        lane = ti.arange(0, 8)
        val = ti.load(x, lane, mask=lane < N)
        a = val > 0.0
        b = a
        v = ti.load(x, lane, mask=(a | ~b) & (lane < N))

    # The formula retains unknown identity; a | ~a is a true fast simplification.
    assert all(r.verdict == PROVEN_SAFE for _, r in k._evaluate_obligations({}, {}, []))


def test_launch_contract_is_rechecked_with_same_const_specialization():
    @ti.assume_launch("N % BLOCK == 0")
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite],
          BLOCK: ti.Const[int, ti.PowerOfTwo] = 8):
        offsets = ti.program_id(0) * BLOCK + ti.arange(0, BLOCK)
        value = ti.load(x, offsets)
        ti.store(x, offsets, value)

    k[(ti.cdiv(8, 8),)](np.ones(8, dtype=np.float32))
    assert all(P.CHECKED in {o.kind for o in r.dependencies}
               for _, r in k.last_proof_results)
    with pytest.raises(TilaLaunchContractError):
        k[(ti.cdiv(9, 8),)](np.ones(9, dtype=np.float32))
