"""DimExpr：规范化与等价判定（type-system.md §4）。"""

from tila.dims import (Add, Cst, Dim, Mul, Sym, canon, equal, eval_num,
                       free_syms, int_value)

N = Dim("N")
B = Dim("B")


def test_canon_folds_constants():
    assert canon(Cst(2) + Cst(3)) == "5"
    assert canon(Add(Sym("N"), Cst(0))) == "N"


def test_canon_sorts_terms():
    a = Sym("N") + Sym("B")
    b = Sym("B") + Sym("N")
    assert canon(a) == canon(b)
    assert equal(a, b)


def test_linear_scaling():
    assert equal(N * Cst(2) + B, B + N + N)


def test_opaque_atoms_keyed_structurally():
    x = Mul(Sym("pid0"), Sym("B"))
    y = Mul(Sym("pid0"), Sym("B"))
    z = Mul(Sym("pid1"), Sym("B"))
    assert equal(x, y)
    assert not equal(x, z)


def test_atom_plus_lane_combines():
    x = Mul(Sym("pid0"), Sym("B")) + Sym("__lane1")
    y = Sym("__lane1") + Mul(Sym("B"), Sym("pid0"))
    assert equal(x, y)


def test_int_value():
    assert int_value(Cst(4) * Cst(8) - Cst(2)) == 30
    assert int_value(N) is None


def test_eval_num():
    e = N * Cst(2) + Cst(1)
    assert eval_num(e, {"N": 5}) == 11


def test_free_syms():
    assert free_syms(N + B * Cst(2)) == {"N", "B"}


def test_sym_str_is_name():
    assert str(N) == "N"
    assert str(N + Cst(1)) == "N + 1"
