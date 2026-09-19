"""M2-06 audit ABI: stable status vocabulary, order and opt-in replay."""
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys

import pytest
import tila as ti
from tila import predicates as P
from tila.cli import main
from tila.dims import Cst, Sym
from tila.facts import (Facts, Pred, Obligation, ProofResult, audit_result_lines, predicate_audit_text,
                        PROVEN_SAFE, PROVEN_UNSAFE, UNKNOWN, EXEMPTED)
from tila.solver import ProofSession, ProofConfig, _CACHE

GOLDEN = Path(__file__).parent / "golden" / "m2_explain"

CLI_SOURCES = {
    "unknown": """import tila as ti
@ti.jit
def gather(data: ti.Buffer[ti.i32, (8,), ti.ReadOnly], indices: ti.Buffer[ti.i32, (1,), ti.ReadOnly]):
    index = ti.load(indices, 0)
    value = ti.load(data, index)
""",
    "unsafe": """import tila as ti
@ti.jit
def bad(data: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):
    value = ti.load(data, 9)
""",
    "budget": """import tila as ti
@ti.jit
def budget(data: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):
    value = ti.load(data, 0)
""",
}


def golden(name, text):
    assert text.rstrip() + "\n" == (GOLDEN / name).read_text(encoding="utf-8")


def cases():
    x = Sym("x")
    base = Obligation("load", "data", 0, x, Cst(8), P.TRUE, loc_line=10)
    return {
        "safe": (replace(base, mask=P.atom(Pred("<", x, Cst(8)))),
                 ProofResult(PROVEN_SAFE, frozenset({P.Origin(P.STATIC, 9, "comparison")}),
                             ("direct predicate `x < 8`; lower bound is nonnegative",), (9, 10))),
        "unsafe": (base, ProofResult(PROVEN_UNSAFE, source_locations=(10,),
                      reason="reachable scalar out-of-bounds access",
                      candidate_counterexample=(("x", "-1"), ("coordinate", "-1"), ("extent", "8")),
                      query="(check-sat)\n", cache_status="miss")),
        "candidate": (base, ProofResult(UNKNOWN, source_locations=(10,),
                      reason="candidate counterexample; concrete reachability unverified",
                      candidate_counterexample=(("x", "-1"), ("coordinate", "-1"), ("extent", "8")),
                      query="(check-sat)\n")),
        "exempted": (replace(base, kind="unsafe_load"), ProofResult(EXEMPTED,
                      source_locations=(10,), reason="explicitly waived (unsafe access); no safety fact produced")),
        "unreachable": (replace(base, path=P.FALSE), ProofResult(PROVEN_SAFE,
                      source_locations=(10,), reason="unreachable access: Z3 proved premises unsatisfiable")),
        "assumption": (base, ProofResult(PROVEN_SAFE,
                      frozenset({P.Origin(P.USER, 7, "assume bound")}),
                      ("Z3: facts ∧ path ∧ mask ∧ ¬in_bounds is unsat",), (7, 10),
                      query="(check-sat)\n", cache_status="hit")),
        "contract": (base, ProofResult(PROVEN_SAFE,
                      frozenset({P.Origin(P.CHECKED, detail="validated launch grid")}),
                      source_locations=(10,))),
        "pending": (base, ProofResult(UNKNOWN, source_locations=(10,),
                      reason="pending launch contract", pending_contracts=("grid must satisfy recorded relation",))),
        "budget": (base, ProofResult(UNKNOWN, source_locations=(10,),
                      reason="kernel cumulative proof budget exhausted")),
    }


@pytest.mark.parametrize("name", tuple(cases()))
def test_audit_result_vocabulary_golden(name):
    ob, result = cases()[name]
    golden(name + ".txt", "\n".join(audit_result_lines(ob, result, show_witness=True, show_cache=True)))


def test_ordering_and_model_input_order_are_stable():
    ob, result = cases()["unsafe"]
    assert audit_result_lines(ob, result) == audit_result_lines(
        ob, replace(result, candidate_counterexample=tuple(reversed(result.candidate_counterexample))))
    unknown = replace(result, verdict=UNKNOWN, reason="Z3 unknown: canceled")
    timeout = replace(unknown, reason="Z3 unknown: timeout")
    assert audit_result_lines(ob, unknown) == audit_result_lines(ob, timeout)


def test_cache_telemetry_does_not_change_proof_equality():
    _CACHE.clear()
    x = Sym("x")
    ob = Obligation("load", "data", 0, x, Cst(8), P.negate(P.atom(Pred(">=", x, Cst(8)))))
    session = ProofSession()
    fresh = session.prove(ob, Facts(), {"x"})
    cached = session.prove(ob, Facts(), {"x"})
    assert fresh == cached
    assert (fresh.cache_status, cached.cache_status) == ("miss", "hit")
    assert "proof cache: hit" in audit_result_lines(ob, cached, show_cache=True)
    assert audit_result_lines(ob, fresh) == audit_result_lines(ob, cached)


def test_real_budget_failure_has_no_candidate_or_query():
    ob = cases()["safe"][0]
    result = ProofSession(ProofConfig(total_ms=0)).prove(ob, Facts(), set())
    assert result.verdict == UNKNOWN
    assert not result.candidate_counterexample and not result.query
    assert "counterexample:\n  (none)" in "\n".join(audit_result_lines(ob, result))


@ti.jit
def safe_kernel(data: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):
    value = ti.load(data, 0)


def test_explain_cli_golden(tmp_path, capsys):
    source = "import tila as ti\n\n@ti.jit\ndef safe_kernel(data: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):\n    value = ti.load(data, 0)\n"
    path = tmp_path / "safe.py"
    path.write_text(source)
    assert main(["explain", str(path)]) == 0
    golden("cli-safe.txt", capsys.readouterr().out)


def test_cli_unknown_explain_golden(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TILA_SAFETY", "strict")
    path = tmp_path / "unknown.py"
    path.write_text(CLI_SOURCES["unknown"])
    assert main(["explain", str(path)]) == 0
    golden("cli-unknown.txt", capsys.readouterr().out)


def test_default_cli_is_stable_across_process_hash_seeds(tmp_path):
    path = tmp_path / "unknown.py"
    path.write_text(CLI_SOURCES["unknown"])
    outputs = []
    for seed in (1, 73):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        proc = subprocess.run([sys.executable, "-m", "tila", "explain", str(path)],
                              env=env, capture_output=True, text=True, check=True)
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1]


def test_query_is_opt_in_and_replayable():
    @ti.jit
    def unknown(data: ti.Buffer[ti.i32, (8,), ti.ReadOnly], index: ti.i32):
        # Unsafe keeps construction legal; use its obligation for the query test.
        value = ti.unsafe_load(data, index)

    ob = replace(unknown.tk.obligations[0], kind="load", execution_context_exact=False)
    unknown.tk.obligations = [ob]
    plain = unknown.explain()
    assert "(check-sat)" not in plain
    detailed = unknown.explain(show_query=True)
    assert "SMT-LIB replay queries:" in detailed and "(check-sat)" in detailed
    query = detailed.split("  - obligation 1:\n", 1)[1].split("warnings:", 1)[0]
    query = "\n".join(line[4:] for line in query.splitlines())
    session = ProofSession()
    solver = session.z.Solver()
    solver.from_string(query)
    assert solver.check() == session.z.sat


@pytest.mark.parametrize("name", tuple(CLI_SOURCES))
def test_cli_diagnostic_golden(name, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TILA_SAFETY", "strict")
    if name == "budget":
        monkeypatch.setenv("TILA_PROOF_TOTAL_MS", "0")
    source = tmp_path / "kernel.py"
    source.write_text(CLI_SOURCES[name])
    assert main(["check", str(source)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    golden("error-" + name + ".txt", captured.err)


def test_cli_error_query_replays_and_exception_retains_result(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TILA_SAFETY", "strict")
    source = tmp_path / "kernel.py"
    source.write_text(CLI_SOURCES["unsafe"])
    assert main(["check", str(source), "--show-query", "--show-witness", "--show-cache"]) == 2
    error = capsys.readouterr().err
    assert "coordinate: 9" in error and "proof cache telemetry:" in error
    query = error.split("SMT-LIB replay query:\n", 1)[1]
    session = ProofSession()
    solver = session.z.Solver()
    solver.from_string(query)
    assert solver.check() == session.z.sat
    from tila.cli import _load
    from tila.errors import TilaError
    with pytest.raises(TilaError) as exc:
        _load(str(source))
    assert exc.value.proof_result.verdict == PROVEN_UNSAFE
    assert exc.value.proof_result.query


def test_predicate_rendering_preserves_unknown_identity_and_is_bounded():
    a, b = P.unknown(), P.unknown()
    text = predicate_audit_text(P.conjunction(a, P.negate(b)))
    assert "unknown#0" in text and "unknown#1" in text
    root = a
    for _ in range(60):
        root = P.conjunction(root, P.disjunction(root, P.unknown()))
    assert len(predicate_audit_text(root)) < 50_000
    for _ in range(100):
        root = P.conjunction(root, P.unknown())
    assert "display budget exceeded" in predicate_audit_text(root)


def test_default_explain_is_identical_across_cache_and_solver_model_changes():
    ob, result = cases()["candidate"]
    other = replace(result, candidate_counterexample=(("x", "-42"), ("coordinate", "-42"), ("extent", "8")),
                    cache_status="hit")
    assert audit_result_lines(ob, result) == audit_result_lines(ob, other)
    assert audit_result_lines(ob, result, show_witness=True) != audit_result_lines(ob, other, show_witness=True)


def test_generated_hint_provenance_is_auditable():
    @ti.jit
    def hinted(data: ti.Buffer[ti.i32, (8,), ti.ReadOnly], BLOCK: ti.Const[int] = 8):
        pid = ti.program_id(0)
        index = pid * BLOCK + ti.arange(0, BLOCK)
        value = ti.load(data, index, mask=index < 8)

    before = hinted.tk.dump()
    report = hinted.explain()
    assert "multiple_of(index_base, BLOCK)" in report
    assert "max_contiguous(index, BLOCK)" in report
    assert "structural pid * step; integer launch guards required" in report
    assert "structural contiguous span; integer launch guards required" in report
    assert hinted.tk.dump() == before
    golden("hints.txt", report.split("hints:\n", 1)[1].split("effects:\n", 1)[0])


def test_parameter_refinements_and_section_order_are_preserved():
    @ti.jit
    def refined(index: ti.i32 | ti.NonNegative,
                BLOCK: ti.Const[int, ti.PowerOfTwo] = 8):
        pass

    report = refined.explain()
    assert "index : i32 | NonNegative" in report
    assert "BLOCK : Const[int, PowerOfTwo]" in report
    headers = ["parameters:", "types:", "facts:", "obligations:",
               "hints:", "effects:", "aliases:", "warnings:", "notes:"]
    offsets = [report.index(header) for header in headers]
    assert offsets == sorted(offsets)
