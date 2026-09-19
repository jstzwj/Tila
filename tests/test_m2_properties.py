"""M2-05 bounded exhaustive/differential audit; fixed seed, no new dependency."""
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

import tila as ti
from tila import dtypes as D, predicates as P
from tila.dims import Cst, Sym, Add, Sub, Mul, FloorDiv, Mod
from tila.facts import (Facts, Obligation, Pred, PROVEN_SAFE, PROVEN_UNSAFE,
                        UNKNOWN, fast_obligation)
from tila.solver import Encoder, ProofConfig, ProofSession, _CACHE
from tila.interp import Interp, run_kernel
from m2_audit_support import SEED, OPS, reference, cpu_operation, compare, fail

# Tests use a declared larger audit budget. Production defaults are unchanged.
AUDIT = ProofConfig(timeout_ms=2000, total_ms=10_000, rlimit=2_000_000, cache_entries=0)


def domain(dtype):
    return range(-128, 128) if dtype == "i8" else range(256)


def encoded_operation(op, dtype, right, input_dtype=None):
    x = Sym("x")
    math = {"+": Add, "-": Sub, "*": Mul, "//": FloorDiv, "%": Mod}
    left = math[op](x, Cst(right)) if op in math else x
    definition = ("wrap" if op in math or op == "cast" else op,
                  dtype, left, Cst(right))
    ob = Obligation("load", "audit", 0, Sym("value"), Cst(8), P.TRUE,
                    value_types=(("x", input_dtype or dtype),),
                    value_defs=(("value", definition),), reachability_inputs=("x",))
    encoder = Encoder(ob, Facts(), set(), AUDIT)
    expression = encoder.expr(ob.coord)
    return ob, encoder, expression


def ground(enc, expression, value):
    # Bind the *raw bitvector*, not the BV2Int expression: signedness/width of
    # inputs is part of the audit rather than an assumption of the test oracle.
    raw = getattr(enc, "_audit_raw_input", None)
    if raw is None:
        raw, = enc.z.z3util.get_vars(enc.symbols["x"])
        enc._audit_raw_input = raw
    return enc.z.simplify(enc.z.substitute(expression,
                         (raw, enc.z.BitVecVal(value, raw.size()))))


def witness_query(enc, expression, values, expected):
    def serialize(index):
        solver = enc.z.Solver()
        solver.add(enc.symbols["x"] == values[index], expression != expected[index])
        return solver.to_smt2()
    return serialize


@pytest.mark.parametrize("dtype", ["i8", "u8"])
@pytest.mark.parametrize("op", list(OPS))
def test_exhaustive_cpu_pairs_and_smt_boundary_slices(dtype, op):
    values = list(domain(dtype))
    rights = list(range(8)) if op in ("<<", ">>") else values
    if op in ("//", "%"):
        rights = [v for v in rights if v != 0]
    # All legal operand pairs on the CPU (including MIN/-1), not random samples.
    pairs = [(a, b) for a in values for b in rights]
    left = np.array([a for a, _ in pairs], dtype=D.ALL[dtype].np_dtype)
    right = np.array([b for _, b in pairs], dtype=D.ALL[dtype].np_dtype)
    expected = [reference(op, dtype, a, b) for a, b in pairs]
    compare({"op": op, "dtype": dtype, "stage": "CPU pairs"}, pairs, expected,
            cpu_operation(op, dtype, left, right))
    # SMT: every left operand, each edge/sign boundary right operand; every
    # legal shift count. Ground substitution uses the encoded input identity.
    slices = rights if op in ("<<", ">>") else sorted(set(rights) &
               {values[0], values[-1], -3, -1, 0, 1, 3, 7})
    for b in slices:
        ob, enc, expression = encoded_operation(op, dtype, b)
        actual = [ground(enc, expression, a).as_long() for a in values]
        expected = [reference(op, dtype, a, b) for a in values]
        compare({"op": op, "dtype": dtype, "stage": "SMT enumeration"},
                [(a, b) for a in values], expected, actual,
                witness_query(enc, expression, values, expected))


@pytest.mark.parametrize("source", ["i8", "u8"])
@pytest.mark.parametrize("target", ["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"])
def test_exhaustive_integer_cast_matrix(source, target):
    values = list(domain(source))
    ob, enc, expression = encoded_operation("cast", target, 0, input_dtype=source)
    expected = [reference("cast", target, a) for a in values]
    actual = [ground(enc, expression, a).as_long() for a in values]
    case = {"op": "cast", "dtype": target, "source": source}
    compare(case, [(a, 0) for a in values], expected, actual,
            witness_query(enc, expression, values, expected))
    compare(case, [(a, 0) for a in values], expected,
            cpu_operation("cast", target, np.asarray(values, dtype=D.ALL[source].np_dtype)))


@pytest.mark.parametrize("dtype", ["i8", "u8"])
def test_unary_not_exhaustive(dtype):
    values = list(domain(dtype))
    _, enc, expr = encoded_operation("~", dtype, 0)
    expected = [reference("~", dtype, a) for a in values]
    actual = [ground(enc, expr, a).as_long() for a in values]
    compare({"op": "~", "dtype": dtype}, [(a, 0) for a in values], expected, actual,
            witness_query(enc, expr, values, expected))
    compare({"op": "~", "dtype": dtype}, [(a, 0) for a in values], expected,
            cpu_operation("~", dtype, np.array(values, dtype=D.ALL[dtype].np_dtype)))


@pytest.mark.parametrize("op,dtype,right", [("&", "i8", 7), ("%", "i8", 8),
                                            ("+", "i8", 1), ("*", "u8", 17)])
@pytest.mark.parametrize("guarded", [False, True])
def test_safety_verdict_matches_exhaustive_execution(op, dtype, right, guarded):
    ob, _, _ = encoded_operation(op, dtype, right)
    if guarded:
        ob = replace(ob, mask=P.conjunction(P.atom(Pred(">=", ob.coord, Cst(0))),
                                            P.atom(Pred("<", ob.coord, Cst(8)))))
    values = [reference(op, dtype, x, right) for x in domain(dtype)]
    unsafe = [x for x, v in zip(domain(dtype), values) if not 0 <= v < 8 and not guarded]
    result = ProofSession(AUDIT).prove(ob, Facts(), set())
    expected = PROVEN_UNSAFE if unsafe else PROVEN_SAFE
    if result.verdict != expected:
        fail({"op": op, "dtype": dtype, "guarded": guarded, "right": right},
             f"expected {expected}, got {result.render()}", result.query)
    cpu = Interp.__new__(Interp)
    cpu.debug = True
    if unsafe:
        witness = int(dict(result.candidate_counterexample)["x"])
        index = int(cpu_operation(op, dtype, witness, right))
        assert witness in unsafe
        with pytest.raises(AssertionError, match="bounds check"):
            cpu._load(np.arange(8), (), [index], None, None)
    else:
        indices = cpu_operation(op, dtype, np.array(list(domain(dtype))), right)
        mask = (indices >= 0) & (indices < 8) if guarded else None
        cpu._load(np.arange(8), (), [indices], mask, 0)


def atom(op, value):
    return P.atom(Pred(op, Sym("x"), Cst(value)))


def boolean_eval(node, x):
    if node.op == "atom":
        p = node.atom
        return {"<": x < p.right.value, ">=": x >= p.right.value,
                "==": x == p.right.value}[p.op]
    if node.op == "not": return not boolean_eval(node.args[0], x)
    if node.op == "and": return all(boolean_eval(c, x) for c in node.args)
    if node.op == "or": return any(boolean_eval(c, x) for c in node.args)
    return node.op == "true"


def test_seeded_boolean_metamorphisms_and_truth_tables():
    rng = random.Random(SEED)
    for trial in range(24):
        a, b, c = [atom(rng.choice(["<", ">=", "=="]), rng.randrange(-128, 128))
                   for _ in range(3)]
        pairs = [(P.negate(P.conjunction(a, b)), P.disjunction(P.negate(a), P.negate(b))),
                 (P.conjunction(a, P.disjunction(b, c)),
                  P.disjunction(P.conjunction(a, b), P.conjunction(a, c))),
                 (a, P.negate(P.negate(a))),
                 (P.disjunction(a, b), P.disjunction(b, a))]
        for index, (left, right) in enumerate(pairs):
            expected = [boolean_eval(left, x) for x in domain("i8")]
            assert expected == [boolean_eval(right, x) for x in domain("i8")]
            mismatch = P.disjunction(P.conjunction(left, P.negate(right)),
                                     P.conjunction(P.negate(left), right))
            ob = Obligation("load", "boolean-audit", 0, Cst(9), Cst(8), mismatch,
                            value_types=(("x", "i8"),), reachability_inputs=("x",))
            result = ProofSession(AUDIT).prove(ob, Facts(), set())
            if result.verdict != PROVEN_SAFE:
                fail({"trial": trial, "law": index, "formula": repr(mismatch)},
                     result.render(), result.query)
            enc = Encoder(ob, Facts(), set(), AUDIT)
            formula = enc.boolean(left)
            actual = [enc.z.is_true(ground(enc, formula, x)) for x in domain("i8")]
            compare({"trial": trial, "law": index, "formula": repr(left)},
                    list(domain("i8")), expected, actual,
                    witness_query(enc, formula, list(domain("i8")), expected))


def test_shared_boolean_pressure_preserves_meaning_under_budget_changes():
    formula = P.conjunction(atom(">=", 0), P.negate(atom(">=", 8)))
    initial = len(tuple(P.nodes(formula)))
    for _ in range(160):
        # Absorption: root AND (root OR arbitrary) == root. The shared DAG is
        # linear even though recursively expanding its subexpressions is not.
        formula = P.conjunction(formula, P.disjunction(formula, P.unknown()))
    assert len(tuple(P.nodes(formula))) == initial + 160 * 3
    ob = Obligation("load", "dag-pressure", 0, Sym("x"), Cst(8), formula,
                    value_types=(("x", "i8"),))
    good = ProofSession(AUDIT).prove(ob, Facts(), set())
    if good.verdict != PROVEN_SAFE:
        fail({"case": "shared-dag", "layers": 160}, good.render(), good.query)
    small = ProofSession(replace(AUDIT, max_nodes=16)).prove(ob, Facts(), set())
    assert small.verdict == UNKNOWN and "construction" in small.reason


def test_failure_artifact_contains_only_first_witness(monkeypatch, tmp_path):
    monkeypatch.setenv("TILA_AUDIT_FAILURE_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="minimal enumerated witness"):
        compare({"op": "+", "dtype": "i8"}, [[126, 1], [127, 1]],
                [127, -128], [127, 128], "(check-sat)\n")
    path, = tmp_path.glob("failure-*/case.json")
    record = json.loads(path.read_text())
    assert record["seed"] == SEED and record["input"] == [127, 1]
    assert record["expected"] == -128 and record["actual"] == 128
    assert path.with_name("query.smt2").read_text() == "(check-sat)\n"
    replay = subprocess.run([sys.executable, str(Path(__file__).with_name("m2_audit_support.py")),
                             str(path)], capture_output=True, text=True, check=True)
    assert "Python oracle: -128" in replay.stdout and "CPU interpreter: -128" in replay.stdout
    assert "SMT replay: sat" in replay.stdout


@pytest.mark.parametrize("dtype", ["i8", "u8"])
def test_intermediate_wrap_is_not_cancelled_before_division(dtype):
    x, middle, result = Sym("x"), Sym("middle"), Sym("result")
    ob = Obligation("load", "intermediate", 0, result, Cst(128), P.TRUE,
                    value_types=(("x", dtype),),
                    value_defs=(("middle", ("wrap", dtype, Add(x, Cst(1)), Cst(0))),
                                ("result", ("wrap", dtype, FloorDiv(middle, Cst(2)), Cst(0)))))
    enc = Encoder(ob, Facts(), set(), AUDIT)
    expr = enc.expr(result)
    inputs = list(domain(dtype))
    expected = [reference("//", dtype, reference("+", dtype, a, 1), 2) for a in inputs]
    cpu = cpu_operation("//", dtype, cpu_operation("+", dtype, np.array(inputs), 1), 2)
    actual = [ground(enc, expr, a).as_long() for a in inputs]
    case = {"dtype": dtype, "pipeline": "wrap(x + 1) // 2"}
    compare(case, inputs, expected, cpu)
    compare(case, inputs, expected, actual, witness_query(enc, expr, inputs, expected))


def test_fast_vs_smt_conclusion_audit_with_concrete_memory():
    # These are the retained fast tactic's conclusions, not an invented replay
    # of an old compiler. Every changed conclusion has a concrete truth table.
    x = Sym("x")
    base = Obligation("load", "audit", 0, x, Cst(8), P.TRUE,
                      value_types=(("x", "i8"),), reachability_inputs=("x",))
    cases = [
        ("negation", replace(base, mask=P.negate(atom(">=", 8))), {"x"}, PROVEN_SAFE),
        ("disjunction", replace(base, mask=P.disjunction(atom("<", 4), atom("==", 7))),
         {"x"}, PROVEN_SAFE),
        ("reachable-negative", replace(base, path=atom("<", 0)), set(), PROVEN_UNSAFE),
        ("opaque-negative", replace(base, path=atom("<", 0), execution_context_exact=False),
         set(), UNKNOWN),
    ]
    cpu = Interp.__new__(Interp)
    cpu.debug = True
    for label, ob, nonneg, expected in cases:
        fast = fast_obligation(ob, Facts(), nonneg)
        assert fast.verdict == UNKNOWN, label
        result = ProofSession(AUDIT).prove(ob, Facts(), nonneg)
        if result.verdict != expected:
            fail({"case": label}, result.render(), result.query)
        reached_bad = []
        for value in domain("i8"):
            active = (not nonneg or value >= 0) and boolean_eval(ob.path, value) and boolean_eval(ob.mask, value)
            if active and not 0 <= value < 8:
                reached_bad.append(value)
                with pytest.raises(AssertionError):
                    cpu._load(np.arange(8), (), [value], True, 0)
            else:
                cpu._load(np.arange(8), (), [value], active, 0)
        assert bool(reached_bad) == (expected != PROVEN_SAFE)


def test_unbounded_integer_abstraction_would_hide_a_real_overflow():
    x = Sym("x")
    mathematical = Add(x, Cst(1))
    legacy = Obligation("load", "unsound-math-abstraction", 0, mathematical, Cst(8),
                        P.atom(Pred("<", mathematical, Cst(8))))
    assert fast_obligation(legacy, Facts(), {"x"}).verdict == PROVEN_SAFE
    actual, _, _ = encoded_operation("+", "i8", 1)
    actual = replace(actual, mask=P.atom(Pred("<", actual.coord, Cst(8))),
                     preds_snapshot=(Pred(">=", x, Cst(0)),))
    result = ProofSession(AUDIT).prove(actual, Facts(), set())
    if result.verdict != PROVEN_UNSAFE:
        fail({"case": "mathematical-versus-wrapped"}, result.render(), result.query)
    witness = int(dict(result.candidate_counterexample)["x"])
    assert witness == 127
    assert reference("+", "i8", witness, 1) == int(cpu_operation("+", "i8", witness, 1)) == -128


def test_cache_metamorphism_order_binding_origin_and_reachability():
    x, n = Sym("x"), Sym("N")
    mask = P.negate(P.atom(Pred(">=", x, n)))
    base = Obligation("load", "cache-audit", 0, x, n, mask,
                      value_types=(("x", "i8"),), reachability_inputs=("x",))
    cases = []
    for kind in (P.STATIC, P.USER, P.CHECKED):
        low = Pred(">=", x, Cst(0), frozenset({P.Origin(kind, 17, "audit lower bound")}))
        for extent in (1, 8, 127):
            cases.append((replace(base, preds_snapshot=(low,)), Facts(num={"N": extent})))
    cases.extend([
        (base, Facts(num={"N": 8})),
        (replace(base, execution_context_exact=False), Facts(num={"N": 8})),
        (replace(base, reachability_inputs=()), Facts(num={"N": 8})),
        (replace(base, value_types=(("x", "u8"),)), Facts(num={"N": 8})),
        (base, Facts(num={"N": 8, "x": -1})),
    ])
    rng = random.Random(SEED)
    rng.shuffle(cases)
    _CACHE.clear()
    config = replace(AUDIT, cache_entries=128)
    signatures = []
    for ob, facts in cases:
        expected = ProofSession(AUDIT).prove(ob, facts, set())
        actual = ProofSession(config).prove(ob, facts, set())
        signature = (expected.verdict, expected.dependencies, expected.pending_contracts)
        assert (actual.verdict, actual.dependencies, actual.pending_contracts) == signature
        signatures.append(signature)
    for (ob, facts), signature in reversed(list(zip(cases, signatures))):
        session = ProofSession(config)
        actual = session.prove(ob, facts, set())
        assert (actual.verdict, actual.dependencies, actual.pending_contracts) == signature
        assert session.cache_hits == (actual.verdict != UNKNOWN)
    # Warm cache cannot satisfy a different, exhausted query budget.
    ob, facts = cases[0]
    limited = ProofSession(replace(config, max_queries=0))
    assert limited.prove(ob, facts, set()).verdict == UNKNOWN
    assert limited.cache_hits == 0


@pytest.mark.parametrize("limit", ["timeout_ms", "rlimit", "total_ms", "max_queries", "max_nodes", "max_depth", "max_query_bytes"])
def test_budget_failure_never_becomes_a_cached_safety_fact(limit):
    mask = P.negate(atom(">=", 8))
    ob = Obligation("load", "budget-audit", 0, Sym("x"), Cst(8), mask)
    _CACHE.clear()
    good = replace(AUDIT, cache_entries=16)
    assert ProofSession(good).prove(ob, Facts(), {"x"}).verdict == PROVEN_SAFE
    before = tuple(_CACHE)
    failed = ProofSession(replace(good, **{limit: 0})).prove(ob, Facts(), {"x"})
    assert failed.verdict == UNKNOWN and failed.reason
    assert tuple(_CACHE) == before


@ti.jit
def carried_return_audit(out: ti.Buffer[ti.i8, (1,), ti.WriteOnly], seed: ti.i8,
                         END: ti.Const[int]):
    value = seed
    for step in ti.range(0, END):
        if value < 0:
            return
        value = value + 1
    ti.store(out, 0, value)


def test_control_flow_exhaustive_i8_and_zero_trip_boundaries():
    # Run the checked TIR oracle for all inputs; sample the same edge cases
    # through public launch so metadata/domain validation is exercised as well.
    for end in range(6):
        for seed in domain("i8"):
            value, returned = seed, False
            for _ in range(end):
                if value < 0:
                    returned = True
                    break
                value = reference("+", "i8", value, 1)
            expected = 42 if returned else value
            out = np.array([42], dtype=np.int8)
            run_kernel(carried_return_audit.tk, {"out": out},
                       {"seed": seed, "out_stride0": 1}, {"END": end}, (1,), debug=True)
            compare({"kernel": "carried_return_audit", "end": end}, [seed], [expected], out)
            if seed in (-128, -1, 0, 126, 127):
                out[:] = 42
                carried_return_audit[(1,)](out, seed, END=end)
                compare({"kernel": "carried_return_audit", "end": end, "public_launch": True},
                        [seed], [expected], out)


def test_seeded_broadcast_view_and_mask_permutation():
    rng = np.random.default_rng(SEED)
    cpu = Interp.__new__(Interp)
    cpu.debug = True
    for trial in range(40):
        rows, cols = (int(v) for v in rng.integers(1, 8, size=2))
        backing = rng.integers(-128, 128, (rows * 2, cols * 2), dtype=np.int16)
        view = backing[::2, ::-2]
        r, c = np.arange(rows + 1)[:, None], np.arange(cols + 1)[None, :]
        mask = (r < rows) & (c < cols) & (rng.random((rows + 1, cols + 1)) > .4)
        expected = np.full(mask.shape, -9, np.int16)
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]: expected[i, j] = view[i, j]
        result = cpu._load(view, (), [r, c], mask, -9)
        compare({"trial": trial, "shape": [rows, cols]},
                list(np.ndindex(mask.shape)), expected, result)
        transposed = cpu._load(view.T, (), [c.T, r.T], mask.T, -9)
        compare({"trial": trial, "transpose": True},
                list(np.ndindex(mask.shape)), expected, transposed.T)
        original = backing.copy()
        cpu._store(view, (), [r, c], np.full(mask.shape, 7, np.int16), mask)
        reference_view = original[::2, ::-2]
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]: reference_view[i, j] = 7
        compare({"trial": trial, "scatter": True}, list(np.ndindex(backing.shape)), original, backing)


@ti.jit
def assumption_before(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], index: ti.i32):
    ti.assume(index >= 0 and index < 8)
    value = ti.unsafe_load(x, index)


@ti.jit
def assumption_after(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], index: ti.i32):
    value = ti.unsafe_load(x, index)
    ti.assume(index >= 0 and index < 8)


@ti.jit
def assumption_branch(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], index: ti.i32, flag: ti.i32):
    if flag > 0:
        ti.assume(index >= 0 and index < 8)
    value = ti.unsafe_load(x, index)


@ti.jit
def assumption_loop(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], index: ti.i32, END: ti.Const[int]):
    for j in ti.range(0, END):
        ti.assume(index >= 0 and index < 8)
    value = ti.unsafe_load(x, index)


@pytest.mark.parametrize("kernel,verdict", [(assumption_before, PROVEN_SAFE),
    (assumption_after, PROVEN_UNSAFE), (assumption_branch, PROVEN_UNSAFE),
    (assumption_loop, UNKNOWN)])
def test_assumption_placement_never_leaks_into_other_paths(kernel, verdict):
    # Real checker snapshots; unsafe lets the audit execute the deliberately
    # invalid example, but is removed only from the separately tested obligation.
    ob = replace(kernel.tk.obligations[0], kind="load")
    result = ProofSession(AUDIT).prove(ob, Facts(), set())
    if result.verdict != verdict:
        fail({"kernel": kernel.__name__}, result.render(), result.query)
    assert (P.USER in {o.kind for o in result.dependencies}) == (kernel is assumption_before)
    for index in (-1, 3, 8):
        scalars = {"index": index, "x_stride0": 1, "flag": 0}
        args = (kernel.tk, {"x": np.zeros(8, np.int32)}, scalars, {"END": 0}, (1,))
        if index == 3:
            run_kernel(*args, debug=True)
        else:
            message = "device_assert" if kernel is assumption_before else "bounds check"
            with pytest.raises(AssertionError, match=message):
                run_kernel(*args, debug=True)
