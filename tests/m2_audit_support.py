"""Deterministic audit oracle and persistent, single-input failure replay.

Replay: PYTHONPATH=src:tests python tests/m2_audit_support.py CASE.json
"""
import json
import operator
import os
from pathlib import Path
import tempfile
import shlex

import numpy as np

from tila import dtypes as D, tir as T, types as TY
from tila.interp import Interp

SEED = 205_2026
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul,
       "//": operator.floordiv, "%": operator.mod, "&": operator.and_,
       "|": operator.or_, "^": operator.xor, "<<": operator.lshift,
       ">>": operator.rshift}


def reference_cast(value, dtype):
    """Python arbitrary precision oracle; no production numeric helpers."""
    dt = D.ALL[dtype]
    value = int(value) % (2 ** dt.bits)
    return value - 2 ** dt.bits if dt.kind == "int" and value >= 2 ** (dt.bits - 1) else value


def reference(op, dtype, left, right=0):
    if op == "cast":
        return reference_cast(left, dtype)
    if op == "~":
        return reference_cast(~int(left), dtype)
    return reference_cast(OPS[op](int(left), int(right)), dtype)


def cpu_operation(op, dtype, left, right=0):
    dt = D.ALL[dtype]
    vt = TY.ScalarT(dt)
    if op == "cast":
        expr = T.TCast(vt, dt, T.TName("left"))
    elif op == "~":
        expr = T.TUna(vt, "~", T.TName("left"))
    else:
        expr = T.TBin(vt, op, T.TName("left"), T.TName("right"))
    return Interp.__new__(Interp).e(expr, {"left": left, "right": right}, (0, 0, 0))


def fail(case, message, query=""):
    root = Path(os.environ.get("TILA_AUDIT_FAILURE_DIR", "artifacts/m2-proof-audit"))
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="failure-", dir=root))
    node = os.environ.get("PYTEST_CURRENT_TEST", "tests/test_m2_properties.py").rsplit(" (", 1)[0]
    record = {"seed": SEED, **case, "failure": message,
              "reproduce": "PYTHONPATH=src python -m pytest -q " + shlex.quote(node)}
    (directory / "case.json").write_text(json.dumps(record, indent=2) + "\n")
    if query:
        (directory / "query.smt2").write_text(query)
    raise AssertionError(f"{message}; minimal enumerated witness: {directory / 'case.json'}; "
                         "replay with PYTHONPATH=src:tests python tests/m2_audit_support.py CASE.json")


def compare(case, inputs, expected, actual, query=""):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape:
        fail(case, f"shape mismatch {expected.shape} != {actual.shape}", query)
    bad = np.flatnonzero(expected.ravel() != actual.ravel())
    if len(bad):
        # Enumeration order is stable. Retain one failing input instead of the
        # entire batch: minimal witness size, not a claim of global AST minimality.
        index = int(bad[0])
        fail({**case, "input": inputs[index], "expected": expected.ravel()[index].item(),
              "actual": actual.ravel()[index].item()}, "oracle mismatch",
             query(index) if callable(query) else query)


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1])
    record = json.loads(path.read_text())
    print(json.dumps(record, indent=2))
    print("Deterministic test replay:", record.get("reproduce", "see test_m2_properties.py"))
    if "op" in record and "input" in record:
        left, right = record["input"]
        print("Python oracle:", reference(record["op"], record["dtype"], left, right))
        print("CPU interpreter:", cpu_operation(record["op"], record["dtype"], left, right))
    query = path.with_name("query.smt2")
    if query.exists():
        import z3
        solver = z3.Solver()
        solver.set(timeout=10_000, rlimit=2_000_000)
        solver.from_string(query.read_text())
        status = solver.check()
        print("SMT replay:", status)
        if status == z3.sat:
            print(solver.model())
