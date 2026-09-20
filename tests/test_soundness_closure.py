"""Concrete review regressions and independent finite-domain proof oracles."""
from itertools import product
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import tila as ti
from tila import dims as D, numeric, predicates as P
from tila.errors import TilaError, TilaLaunchContractError
from tila.facts import Facts, Obligation, Pred, PROVEN_SAFE, fast_obligation, evaluate_obligation
from tila.interp import Interp
from tila.solver import Encoder, ProofConfig, ProofSession


@ti.jit
def signed_mask(x: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    i = ti.arange(0, 4)
    ti.store(x, 2 * i, i, mask=-2 * i < 4)


@ti.jit
def positive_f32(x: ti.Buffer[ti.f32, (1,), ti.WriteOnly], n: ti.f32 | ti.Positive):
    ti.store(x, 0, n)


@ti.jit
def pointer_chain(x: ti.ReadPtr[ti.i32, 4]):
    i = ti.arange(0, 4)
    big = ti.zeros((4,), ti.i32) + 2147483647
    two = ti.zeros((4,), ti.i32) + 2
    p = x + i
    q = p + big
    r = q + big
    end = r + two


def test_signed_mask_never_proves_safe_and_launch_does_not_write():
    report = signed_mask.explain()
    assert 'conclusion: ProvenSafe' not in report
    golden = Path(__file__).with_name('golden') / 'soundness-explain.txt'
    assert report + '\n' == golden.read_text()
    out = np.full(4, -17, np.int32)
    with pytest.raises(TilaError):
        signed_mask[(1,)](out)
    np.testing.assert_array_equal(out, -17)
    with pytest.raises(TilaError):
        signed_mask.materialize()


def test_refinement_checks_rounded_value_before_execution():
    out = np.full(1, -17, np.float32)
    for tiny in (1e-50, 2**-150, np.float64(1e-50)):
        with pytest.raises(TilaLaunchContractError, match='violates Positive'):
            positive_f32[(1,)](out, tiny)
        assert out[0] == -17
    for value in (0.125, np.float64(0.125)):
        positive_f32[(1,)](out, value)
        assert out[0] == 0.125


@pytest.mark.parametrize('dt,tiny', [(ti.f16, 2**-26), (ti.bf16, 2**-135),
                                    (ti.f32, 2**-151)])
def test_float_abi_underflow_signed_zero_and_finite_overflow(dt, tiny):
    assert numeric.scalar_float(tiny, dt) == 0
    assert np.signbit(numeric.scalar_float(-tiny, dt))
    with pytest.raises(TilaError, match='overflow|infinity'):
        numeric.scalar_float(10**400, dt)


def test_float_abi_exact_integer_no_double_rounding():
    # binary64 would round this to the f32 midpoint before the final cast.
    source = 2**80 + 2**56 + 1
    assert numeric.scalar_float(source, ti.f32) == float(2**80 + 2**57)


def test_static_different_shapes_rejected_at_use():
    with pytest.raises(TilaError, match='TILA-TYPE-020'):
        @ti.jit
        def kernel(x: ti.Buffer[ti.i32, (4,), ti.WriteOnly], FLAG: ti.Const[bool] = True):
            if FLAG:
                i = ti.arange(0, 4)
            else:
                i = ti.arange(0, 8)
            ti.store(x, i, ti.zeros((4,), ti.i32), mask=i < 4)


def test_static_variant_cannot_escape_through_another_merge():
    with pytest.raises(TilaError, match='TILA-TYPE-020'):
        @ti.jit
        def kernel(FLAG: ti.Const[bool], OTHER: ti.Const[bool]):
            if FLAG:
                if OTHER:
                    value = ti.arange(0, 4)
                else:
                    value = ti.arange(0, 8)
            else:
                value = ti.arange(0, 4)
            result = value + 1


def test_static_variant_in_else_cannot_escape_outer_merge():
    with pytest.raises(TilaError, match='TILA-TYPE-020'):
        @ti.jit
        def kernel(FLAG: ti.Const[bool], OTHER: ti.Const[bool]):
            if FLAG:
                value = ti.arange(0, 4)
            else:
                if OTHER:
                    value = ti.arange(0, 4)
                else:
                    value = ti.arange(0, 8)
            result = value + 1


def test_pointer_accumulation_uses_address_domain():
    array = np.zeros(4, np.int32)
    env = {'x': ('ptr', 'x', 0)}
    interp = Interp(pointer_chain.tk, {'x': array}, {}, {}, (1,))
    interp.stmts(pointer_chain.tk.body, env, (0, 0, 0))
    np.testing.assert_array_equal(env['end'][2], np.arange(4, dtype=np.int64) + 2**32)
    assert numeric.validate(pointer_chain.tk, {}, {}, (1, 1, 1)) == []
    source, _ = pointer_chain.materialize()
    assert 'tl.cast(big, tl.int64)' in source


@pytest.mark.parametrize('delta,size', [(2**63, 1), (2**61, 4), (-2**61 - 1, 4)])
def test_pointer_address_overflow_fails_before_memory(delta, size):
    with pytest.raises(TilaError, match='address domain'):
        numeric.pointer_add(0, delta, size)


def test_pointer_intermediate_overflow_cannot_cancel():
    @ti.jit
    def kernel(x: ti.ReadPtr[ti.i32, 4], delta: ti.i64):
        p = x + delta
        end = p + -delta

    with pytest.raises(TilaError, match='pointer byte displacement'):
        kernel[(1,)](np.zeros(4, np.int32), 2**61)


@ti.jit
def pointer_roundtrip(x: ti.ReadPtr[ti.i32, 4], out: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    i = ti.arange(0, 4)
    big = ti.cast[ti.i32](2147483647)
    two = ti.cast[ti.i32](2)
    back = ti.cast[ti.i64](-4294967296)
    p = x + i
    q = p + big
    r = q + big
    far = r + two
    near = far + back
    ti.store(out, i, ti.load(near))


def test_pointer_wide_roundtrip_reads_original_elements():
    x = np.arange(4, dtype=np.int32) + 31
    out = np.zeros_like(x)
    pointer_roundtrip[(1,)](x, out)
    np.testing.assert_array_equal(out, x)


def test_canonical_equality_preserves_small_domain_semantics():
    x, y = D.Sym('x'), D.Sym('y')
    expressions = [D.Cst(v) for v in range(-3, 4)] + [x, y]
    for a, b, c in product(range(-3, 4), repeat=3):
        expressions.append(a * x + b * y + c)
        expressions.append(c + b * y + a * x)
    expressions += [D.Mul(D.Cst(a), op(x + 1, D.Cst(2)))
                    for a in (-2, -1, 1, 2)
                    for op in (D.FloorDiv, D.CeilDiv, D.Mod, D.Min, D.Max)]
    classes = {}
    bindings = [dict(x=a, y=b) for a, b in product(range(-4, 5), repeat=2)]
    for expr in expressions:
        values = tuple(D.eval_num(expr, env) for env in bindings)
        key = D.canon(expr)
        assert key not in classes or classes[key] == values, (key, expr)
        classes[key] = values
    assert not D.equal(2*x, -2*x)


def test_direct_and_interval_safe_routes_against_enumeration():
    i = D.Sym('i')
    facts = Facts(sym_lo={'i': D.Cst(0)}, sym_hi={'i': D.Cst(4)})
    # Enumerate mismatched signs, constants and both complementary relations.
    for a, b, c, bound, op in product((-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2),
                                     (-1, 0, 1), (1, 4, 8), ('<', '>=')):
        coord = a * i + c
        pred = Pred(op, b * i, D.Cst(bound))
        ob = Obligation('store', 'x', 0, coord, D.Cst(bound), mask=P.atom(pred))
        result = fast_obligation(ob, facts, {'i'})
        if result.verdict == PROVEN_SAFE:
            for lane in range(4):
                active = b*lane < bound if op == '<' else b*lane >= bound
                assert not active or 0 <= a*lane+c < bound, (a,b,c,bound,op,result)


@pytest.mark.parametrize('route', ['direct', 'interval', 'complement', 'constant-false',
                                  'exact-grid', 'cdiv', 'smt-unreachable', 'smt-boolean'])
def test_safe_rule_entailments_checked_without_fast_return(route):
    i, n, pid, step = map(D.Sym, ('i', 'N', 'pid0', 'STEP'))
    facts, nonneg = Facts(), {'i', 'pid0'}
    ob = Obligation('load', 'x', 0, i, n, P.TRUE)
    expected = ''
    if route == 'direct':
        ob = replace(ob, mask=P.atom(Pred('<', i, n)))
        expected = 'direct predicate'
    elif route == 'interval':
        facts = Facts(sym_lo={'i': D.Cst(0)}, sym_hi={'i': D.Cst(4)}, num={'N': 8})
        expected = 'numeric interval'
    elif route == 'complement':
        ob = replace(ob, mask=P.conjunction(P.atom(Pred('<', i, D.Cst(0))),
                                           P.atom(Pred('>=', i, D.Cst(0)))))
        expected = 'contradictory'
    elif route == 'constant-false':
        ob = replace(ob, mask=P.atom(Pred('<', D.Cst(8), D.Cst(4))))
        expected = 'contradictory'
    elif route == 'exact-grid':
        facts = Facts(grid_facts={'pid0': (n, D.Cst(1))}, grid_checked=True)
        ob = replace(ob, coord=pid)
        expected = 'exact-dim grid'
    elif route == 'cdiv':
        nonneg.add('STEP')
        facts = Facts(sym_lo={'i': D.Cst(0)}, sym_hi={'i': step},
                      num={'N': 8, 'STEP': 4}, grid_facts={'pid0': (n, step)}, grid_checked=True)
        facts.add_pred(Pred('==', D.Mod(n, step), D.Cst(0)))
        ob = replace(ob, coord=pid*step+i)
        expected = 'cdiv tactic'
    elif route == 'smt-unreachable':
        ob = replace(ob, mask=P.FALSE)
    else:
        ob = replace(ob, mask=P.negate(P.atom(Pred('>=', i, n))))
    if expected:
        fast = fast_obligation(ob, facts, nonneg)
        assert fast.verdict == PROVEN_SAFE
        assert expected in fast.render()
    result = evaluate_obligation(ob, facts, nonneg)
    assert result.verdict == PROVEN_SAFE
    # Rebuild the implication directly; do not call any Safe-producing route.
    encoder = Encoder(ob, facts, nonneg, ProofConfig())
    premises, goal = encoder.build()
    z = encoder.z
    solver = z.Solver()
    solver.add(*premises)
    if encoder.domain:
        assert solver.check(z.Not(z.And(*encoder.domain))) == z.unsat
    assert solver.check(z.Not(goal)) == z.unsat


def test_signed_predicates_do_not_poison_cache_or_unreachable_path():
    i = D.Sym('i')
    facts = Facts(sym_lo={'i': D.Cst(0)}, sym_hi={'i': D.Cst(4)})
    session = ProofSession()
    good = Obligation('store', 'x', 0, 2*i, D.Cst(4),
                      P.negate(P.atom(Pred('>=', 2*i, D.Cst(4)))))
    bad = replace(good, mask=P.negate(P.atom(Pred('>=', -2*i, D.Cst(4)))))
    assert session.prove(good, facts, {'i'}).verdict == PROVEN_SAFE
    assert session.prove(good, facts, {'i'}).cache_status == 'hit'
    assert session.prove(bad, facts, {'i'}).verdict != PROVEN_SAFE
    # Both predicates hold at i=1: a lost minus used to manufacture a contradiction.
    reachable = replace(good, coord=D.Cst(8), mask=P.conjunction(
        P.atom(Pred('<', -2*i, D.Cst(0))), P.atom(Pred('>=', 2*i, D.Cst(0)))))
    assert session.prove(reachable, facts, {'i'}).verdict != PROVEN_SAFE
