"""Default Z3 proof adapter, bounded construction/queries and provenance cache.

Mathematical expressions are justified by the mandatory numeric launch gate.
Runtime definitions wrap through BitVec; opaque loads/loop values overapproximate
execution, so SAT is only a candidate outside exact scalar dataflow.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
from hashlib import sha256
from time import monotonic
from threading import RLock
from typing import Protocol
import os

from . import dims as D, predicates as P
from .dtypes import ALL
from .errors import TilaError
from .facts import (ProofResult, fast_obligation, PROVEN_SAFE, PROVEN_UNSAFE,
                    UNKNOWN, EXEMPTED)

ENCODING_VERSION = 2
_CACHE = OrderedDict()
_LOCK = RLock()  # Z3's default context and the process cache are shared.


@dataclass(frozen=True)
class ProofConfig:
    timeout_ms: int = 100
    rlimit: int = 200_000
    total_ms: int = 2000
    max_queries: int = 128
    max_nodes: int = 8000
    max_depth: int = 128
    max_integer_bits: int = 4096
    max_query_bytes: int = 2_000_000
    cache_entries: int = 256

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in self.__dict__.values()):
            raise ValueError("proof budgets must be nonnegative integers")

    @classmethod
    def environment(cls):
        try:
            return cls(**{name: int(os.environ["TILA_PROOF_" + name.upper()])
                          for name in cls.__dataclass_fields__
                          if "TILA_PROOF_" + name.upper() in os.environ})
        except ValueError as exc:
            raise TilaError("TILA-PROOF-001", str(exc)) from exc


class SolverProtocol(Protocol):
    def prove(self, ob, facts, nonneg) -> ProofResult: ...


class Limit(Exception):
    pass


class Unsupported(Exception):
    pass


def z3_module():
    try:
        import z3
    except ImportError as exc:
        raise TilaError("TILA-PROOF-001",
                        "default prover requires z3-solver==4.16.0.0; install project dependencies") from exc
    return z3


class Encoder:
    def __init__(self, ob, facts, nonneg, config):
        self.z = z3_module()
        self.ob, self.facts, self.nonneg, self.config = ob, facts, nonneg, config
        self.types, self.defs = dict(ob.value_types), dict(ob.value_defs)
        self.memo, self.symbols, self.booleans = {}, {}, {}
        self.unknown_views = {}
        self.domain, self.origins = [], set()
        self.count = 0
        self.deadline = monotonic() + config.total_ms / 1000

    def tick(self, depth=0):
        if monotonic() >= self.deadline:
            raise Limit("kernel cumulative proof budget exhausted during construction")
        self.count += 1
        if self.count > self.config.max_nodes or depth > self.config.max_depth:
            raise Limit("formula construction budget exhausted")

    def expr(self, e, depth=0):
        # Cache by identity; DimExpr structural hashing recursively revisits DAGs.
        if id(e) in self.memo:
            return self.memo[id(e)]
        self.tick(depth)
        z = self.z
        if isinstance(e, D.Cst):
            if e.value.bit_length() > self.config.max_integer_bits:
                raise Limit("integer constant construction budget exhausted")
            value = z.IntVal(e.value)
        elif isinstance(e, D.Sym):
            value = self.symbol(e.name, depth + 1)
        else:
            a, b = self.expr(e.left, depth + 1), self.expr(e.right, depth + 1)
            if isinstance(e, D.Add): value = a + b
            elif isinstance(e, D.Sub): value = a - b
            elif isinstance(e, D.Mul): value = a * b
            elif isinstance(e, (D.FloorDiv, D.Mod, D.CeilDiv)):
                self.domain.append(b != 0)
                quotient = z.If(b > 0, a / b, (-a) / (-b))
                value = (quotient if isinstance(e, D.FloorDiv) else
                         a - b * quotient if isinstance(e, D.Mod) else
                         -z.If(b > 0, (-a) / b, a / (-b)))
            elif isinstance(e, D.Min): value = z.If(a < b, a, b)
            elif isinstance(e, D.Max): value = z.If(a > b, a, b)
            else: raise Unsupported(f"unsupported mathematical expression {type(e).__name__}")
        self.memo[id(e)] = value
        return value

    def symbol(self, name, depth):
        if name in self.symbols:
            return self.symbols[name]
        self.tick(depth)
        z = self.z
        if name in self.defs:
            op, dtype, left, right = self.defs[name]
            dt = ALL[dtype]
            a, b = self.expr(left, depth + 1), self.expr(right, depth + 1)
            av, bv = z.Int2BV(a, dt.bits), z.Int2BV(b, dt.bits)
            if op == "wrap": value = av
            elif op == "&": value = av & bv
            elif op == "|": value = av | bv
            elif op == "^": value = av ^ bv
            elif op == "~": value = ~av
            elif op in ("<<", ">>"):
                self.domain.append(z.And(b >= 0, b < dt.bits))
                value = av << bv if op == "<<" else av >> bv if dt.kind == "int" else z.LShR(av, bv)
            else: raise Unsupported(f"unsupported runtime integer operation {op}")
            value = z.BV2Int(value, is_signed=dt.kind == "int")
        elif name in self.types:
            dt = ALL[self.types[name]]
            value = z.BV2Int(z.BitVec("value:" + name, dt.bits), is_signed=dt.kind == "int")
        else:
            value = z.Int("index:" + name)
        self.symbols[name] = value
        return value

    def pred(self, p):
        self.origins.update(p.origins)
        a, b = self.expr(p.left), self.expr(p.right)
        if p.op == "<": return a < b
        if p.op == "<=": return a <= b
        if p.op == ">": return a > b
        if p.op == ">=": return a >= b
        if p.op == "==": return a == b
        if p.op == "!=": return a != b
        raise Unsupported(f"unsupported comparison {p.op}")

    def boolean(self, root):
        z = self.z
        todo = [(root, False)]
        while todo:
            node, ready = todo.pop()
            if node in self.booleans:
                continue
            if not ready:
                self.tick()
                todo.append((node, True))
                todo.extend((c, False) for c in node.args if c not in self.booleans)
                continue
            args = [self.booleans[c] for c in node.args]
            if node.op == "true": value = z.BoolVal(True)
            elif node.op == "false": value = z.BoolVal(False)
            elif node.op == "unknown": value = z.Bool(f"unknown:{len(self.booleans)}")
            elif node.op == "unknown_view":
                key = (node.args[0], node.mapping)
                if key not in self.unknown_views:
                    self.unknown_views[key] = z.Bool(f"unknown-view:{len(self.unknown_views)}")
                value = self.unknown_views[key]
            elif node.op == "atom": value = self.pred(node.atom)
            elif node.op == "and": value = z.And(*args)
            elif node.op == "or": value = z.Or(*args)
            elif node.op == "not": value = z.Not(args[0])
            elif node.op == "map": value = args[0]  # symbols already use right-relative lane axes
            else: raise Unsupported(f"unsupported Boolean node {node.op}")
            self.booleans[node] = value
        return self.booleans[root]

    def build(self):
        z, ob, facts = self.z, self.ob, self.facts
        premises = [self.boolean(ob.path), self.boolean(ob.mask)]
        for pred in tuple(facts.preds.values()) + ob.preds_snapshot:
            premises.append(self.pred(pred))
        for name, value in sorted(facts.num.items()):
            self.tick()
            if value.bit_length() > self.config.max_integer_bits:
                raise Limit("integer binding construction budget exhausted")
            premises.append(self.symbol(name, 0) == value)
        lo, hi = dict(facts.sym_lo), dict(facts.sym_hi)
        lo.update(ob.sym_lo)
        hi.update(ob.sym_hi)
        for name, value in sorted(lo.items()):
            premises.append(self.symbol(name, 0) >= self.expr(value))
        for name, value in sorted(hi.items()):
            premises.append(self.symbol(name, 0) < self.expr(value))
        for name in sorted(self.nonneg):
            premises.append(self.symbol(name, 0) >= 0)
        if facts.grid_checked:
            for name, (bound, step) in sorted(facts.grid_facts.items()):
                b, s = self.expr(bound), self.expr(step)
                self.domain.append(s > 0)
                pid = self.symbol(name, 0)
                premises.extend((pid >= 0, pid < (b + s - 1) / s))
            if facts.grid_facts:
                self.origins.add(P.Origin(P.CHECKED, detail="validated launch grid"))
        self.origins.add(P.Origin(P.STATIC, detail="integer encoding and access-point intervals"))
        coord, extent = self.expr(ob.coord), self.expr(ob.extent)
        goal = z.And(coord >= 0, coord < extent)
        return premises, goal


class ProofSession:
    """One kernel check/specialization/launch; budgets never reset per access."""
    def __init__(self, config=None):
        self.z = z3_module()  # Missing solver is a configuration error, even on fast paths.
        self.config = config or ProofConfig.environment()
        self.elapsed_ms = 0.0
        self.queries = 0
        self.cache_hits = 0

    def check(self, solver, *extra):
        remaining = self.config.total_ms - self.elapsed_ms
        if remaining <= 0 or self.queries >= self.config.max_queries:
            raise Limit("kernel cumulative proof budget exhausted")
        if not self.config.timeout_ms or not self.config.rlimit:
            raise Limit("per-query timeout/resource budget exhausted")
        solver.set(timeout=max(1, min(self.config.timeout_ms, int(remaining))),
                   rlimit=self.config.rlimit)
        self.queries += 1
        start = monotonic()
        try:
            return solver.check(*extra)
        finally:
            self.elapsed_ms += (monotonic() - start) * 1000

    def prove(self, ob, facts, nonneg):
        with _LOCK:
            return self._prove(ob, facts, nonneg)

    def _prove(self, ob, facts, nonneg):
        if ob.kind.startswith("unsafe"):
            return fast_obligation(ob, facts, nonneg)
        if ob.coord is None or ob.extent is None:
            return fast_obligation(ob, facts, nonneg)
        start = monotonic()
        elapsed_before = self.elapsed_ms
        query = ""
        encoder = None
        try:
            encoder = Encoder(ob, facts, nonneg, self.config)
            encoder.deadline = start + max(0, self.config.total_ms - elapsed_before) / 1000
            premises, goal = encoder.build()
            self.elapsed_ms += (monotonic() - start) * 1000
            if self.elapsed_ms >= self.config.total_ms:
                raise Limit("kernel cumulative proof budget exhausted during construction")
            fast = fast_obligation(ob, facts, nonneg)
            if fast.verdict == PROVEN_SAFE and not encoder.domain and not any(
                    o.kind == P.USER for o in encoder.origins):
                return fast  # Small, sufficient direct/interval/grid proof.
            solver = self.z.Solver()
            solver.add(*premises)
            # Z3's serializer preserves shared expressions with let bindings.
            # The normalized SMT query, all origins, source and configuration
            # form a process-local bounded cache key. Unknown is never cached.
            solver.push()
            solver.add(self.z.Not(goal))
            query = solver.to_smt2()
            solver.pop()
            normalized = self.z.And(*premises, self.z.Not(goal)).sexpr()
            if len(query.encode()) + len(normalized.encode()) > self.config.max_query_bytes:
                raise Limit("serialized query construction budget exhausted")
            key = sha256(repr((ENCODING_VERSION, self.z.get_version_string(),
                self.config, normalized, tuple(sorted(encoder.origins)), ob.loc_line,
                ob.source, ob.kind, facts.grid_checked, facts.launch_bindings,
                ob.path is P.TRUE, ob.mask is P.TRUE,
                ob.reachability_inputs, ob.execution_context_exact,
                (D.free_syms(ob.coord) | D.free_syms(ob.extent)) <= facts.num.keys(),
                tuple(d.sexpr() for d in encoder.domain))).encode()).hexdigest()
            self.elapsed_ms = elapsed_before + (monotonic() - start) * 1000
            if self.elapsed_ms >= self.config.total_ms:
                raise Limit("kernel cumulative proof budget exhausted during serialization")
            if self.config.cache_entries and key in _CACHE:
                self.cache_hits += 1
                _CACHE.move_to_end(key)
                return _CACHE[key]
            result = self.solve(ob, facts, nonneg, encoder, solver, goal, query)
            if self.config.cache_entries and result.verdict != UNKNOWN:
                _CACHE[key] = result
                while len(_CACHE) > self.config.cache_entries:
                    _CACHE.popitem(last=False)
            return result
        except (Limit, Unsupported, RecursionError) as exc:
            origins = frozenset(encoder.origins if encoder is not None else ())
            locations = tuple(sorted({ob.loc_line} | {o.line for o in origins if o.line}))
            return ProofResult(UNKNOWN, origins, source_locations=locations,
                               reason=str(exc) or "formula construction depth exceeded", query=query)
        finally:
            self.elapsed_ms = elapsed_before + (monotonic() - start) * 1000

    def solve(self, ob, facts, nonneg, encoder, solver, goal, query):
        z = self.z
        deps = frozenset(encoder.origins)
        locs = tuple(sorted({ob.loc_line} | {o.line for o in deps if o.line}))

        def result(verdict, reason="", candidate=(), trace=()):
            return ProofResult(verdict, deps, trace, locs, reason, candidate, query=query)

        # Domain checks must not turn an undefined operation into a vacuous proof.
        if encoder.domain:
            status = self.check(solver, z.Not(z.And(*encoder.domain)))
            if status != z.unsat:
                return result(UNKNOWN, "integer operation domain unproven" if status == z.sat
                              else "Z3 unknown: " + solver.reason_unknown())
        reachable = self.check(solver)
        if reachable == z.unknown:
            return result(UNKNOWN, "Z3 unknown: " + solver.reason_unknown())
        if reachable == z.unsat:
            user = any(o.kind == P.USER for o in deps)
            return result(UNKNOWN if user else PROVEN_SAFE,
                          "inconsistent premises involving UserAssumption; reachability unverified"
                          if user else "unreachable access: Z3 proved premises unsatisfiable")
        status = self.check(solver, z.Not(goal))
        if status == z.unsat:
            # Keep familiar short proof routes where available; Z3 is the
            # authority, including consistency checking of user assumptions.
            fast = fast_obligation(ob, facts, nonneg)
            trace = fast.trace if fast.verdict == PROVEN_SAFE else ()
            return result(PROVEN_SAFE, trace=trace + ("Z3: facts ∧ path ∧ mask ∧ ¬in_bounds is unsat",))
        if status == z.unknown:
            return result(UNKNOWN, "Z3 unknown: " + solver.reason_unknown())
        model = solver.model()
        candidate = tuple((name, str(model.eval(value, model_completion=True)))
                          for name, value in sorted(encoder.symbols.items()))
        candidate += (("coordinate", str(model.eval(encoder.expr(ob.coord), model_completion=True))),
                      ("extent", str(model.eval(encoder.expr(ob.extent), model_completion=True))))
        # Exact scalar dataflow (including guarded finite-width arithmetic) can
        # witness an execution. Loads, lanes and loop abstractions cannot.
        fast = fast_obligation(ob, facts, nonneg)
        if self.exact_reachability(ob, facts, deps):
            return result(PROVEN_UNSAFE, "reachable scalar out-of-bounds access", candidate,
                          ("exact scalar inputs and path; no load/loop abstraction",))
        if fast.pending_contracts:
            return replace(fast, candidate_counterexample=candidate, query=query)
        return result(UNKNOWN, "candidate counterexample; concrete reachability unverified",
                      candidate, fast.trace)

    @staticmethod
    def exact_reachability(ob, facts, dependencies):
        if not ob.execution_context_exact or any(o.kind == P.USER for o in dependencies):
            return False
        allowed = set(ob.reachability_inputs) | set(facts.num)
        definitions = dict(ob.value_defs)
        needed = set(D.free_syms(ob.coord) | D.free_syms(ob.extent))
        for root in (ob.path, ob.mask):
            for node in P.nodes(root):
                if node.op in ("unknown", "unknown_view"):
                    return False
                if node.op == "atom":
                    needed.update(D.free_syms(node.atom.left) | D.free_syms(node.atom.right))
        for pred in tuple(facts.preds.values()) + ob.preds_snapshot:
            needed.update(D.free_syms(pred.left) | D.free_syms(pred.right))
        # Interval-only identities (lanes and induction variables) are not
        # executable scalar inputs, even when their intervals have SAT models.
        for name, value in tuple(facts.sym_lo.items()) + tuple(facts.sym_hi.items()) + ob.sym_lo + ob.sym_hi:
            needed.add(name)
            needed.update(D.free_syms(value))
        seen = set()
        while needed:
            name = needed.pop()
            if name in allowed or name in seen:
                continue
            if name not in definitions:
                return False
            seen.add(name)
            _, _, left, right = definitions[name]
            needed.update(D.free_syms(left) | D.free_syms(right))
        return True
