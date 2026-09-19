"""ADR-013 exact bool domain, staged execution and binding isolation."""
import numpy as np
import pytest

import tila as ti
from tila import tir as T
from tila.errors import TilaError, TilaLaunchContractError
from tila.runtime import _triton_cache_key

FLAG_DIM = ti.Dim("FLAG")


@ti.jit
def choose(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly],
           ENABLED: ti.Const[bool] = True, OTHER: ti.Const[bool] = False):
    if ENABLED and not OTHER:
        ti.store(out, 0, 7)
    else:
        ti.store(out, 0, 3)


@ti.jit
def required(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], FLAG: ti.Const[bool]):
    if FLAG == True:
        ti.store(out, 0, 1)
    else:
        ti.store(out, 0, 0)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("other", [True, False])
def test_static_execution_and_materialization(enabled, other):
    assert isinstance(choose.tk.body[0], T.TStaticIf)
    out = np.zeros(1, np.int32)
    choose[(1,)](out, ENABLED=enabled, OTHER=other)
    assert out[0] == (7 if enabled and not other else 3)
    source, typed = choose.materialize({"ENABLED": enabled, "OTHER": other})
    assert "ENABLED: tl.constexpr" in source
    assert "Const[bool]" in typed
    assert "Const[bool]" in choose.explain({"ENABLED": enabled, "OTHER": other})


@pytest.mark.parametrize("value", [0, 1, np.bool_(True), np.int64(1), "true", None, 1.0])
@pytest.mark.parametrize("entry", ["materialize", "explain", "launch"])
def test_exact_bool_all_entries(value, entry):
    with pytest.raises((TilaError, TilaLaunchContractError), match="TILA-CONST-008"):
        if entry == "launch":
            choose[(1,)](np.zeros(1, np.int32), ENABLED=value)
        else:
            getattr(choose, entry)({"ENABLED": value})


def test_defaults_and_missing_value():
    out = np.zeros(1, np.int32)
    choose[(1,)](out)
    assert out[0] == 7
    for entry in (required.materialize, required.explain):
        with pytest.raises(TilaError, match="TILA-CONST-007"):
            entry()
    with pytest.raises(TilaLaunchContractError, match="TILA-CONST-007"):
        required[(1,)](out)
    for flag in (True, False, True):
        required[(1,)](out, FLAG=flag)
        assert out[0] == int(flag)


def test_invalid_bool_default_and_refinement():
    with pytest.raises(TilaError, match="TILA-CONST-008"):
        @ti.jit
        def k(FLAG: ti.Const[bool] = 1):
            pass
    with pytest.raises(TypeError, match="refinements"):
        ti.Const[bool, ti.Positive]


def test_static_assert_and_short_circuit():
    @ti.jit
    def k(FLAG: ti.Const[bool], DIV: ti.Const[int] = 0):
        ti.static_assert(FLAG or (1 // DIV > 0))
    k.materialize({"FLAG": True})
    with pytest.raises(TilaError, match="TILA-CONST-009"):
        k.materialize({"FLAG": False})
    k.materialize({"FLAG": False, "DIV": 1})


@ti.jit
def short_circuit(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], FLAG: ti.Const[bool],
                  DIV: ti.Const[int]):
    if FLAG or (1 // DIV > 0):
        ti.store(out, 0, 1)
    else:
        ti.store(out, 0, 2)


def test_static_if_short_circuit_skips_invalid_division():
    out = np.zeros(1, np.int32)
    short_circuit[(1,)](out, FLAG=True, DIV=0)
    assert out[0] == 1
    with pytest.raises(TilaError, match="TILA-NUM-001"):
        short_circuit[(1,)](out, FLAG=False, DIV=0)


def test_binding_changes_recheck_unreachable_path():
    @ti.jit
    def k(data: ti.Buffer[ti.i32, (1,), ti.ReadOnly], FLAG: ti.Const[bool]):
        if FLAG:
            value = ti.load(data, 9)
    data = np.zeros(1, np.int32)
    k[(1,)](data, FLAG=False)
    with pytest.raises(TilaError, match="TILA-BOUNDS"):
        k[(1,)](data, FLAG=True)
    k[(1,)](data, FLAG=False)
    assert "Const[bool](FLAG)" in k.explain({"FLAG": False})


def test_typed_cache_keys():
    assert _triton_cache_key(choose, {"ENABLED": True}) != _triton_cache_key(choose, {"ENABLED": 1})
    assert _triton_cache_key(choose, {"ENABLED": False}) != _triton_cache_key(choose, {"ENABLED": True})


def test_bool_not_integer_shape_or_arithmetic():
    with pytest.raises(TilaError, match="TILA-CONST-001"):
        @ti.jit
        def k(FLAG: ti.Const[bool]):
            i = ti.arange(0, FLAG)
    with pytest.raises(TilaError, match="TILA-TYPE"):
        @ti.jit
        def k(FLAG: ti.Const[bool]):
            value = FLAG + 1


def test_runtime_mix_stays_runtime():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], runtime: ti.bool,
          FLAG: ti.Const[bool] = True):
        if FLAG and runtime:
            ti.store(out, 0, 1)
    assert isinstance(k.tk.body[0], T.TIf)
    with pytest.raises(TilaError, match="TILA-CONST-006"):
        @ti.jit
        def invalid(runtime: ti.bool, FLAG: ti.Const[bool]):
            ti.static_assert(FLAG and runtime)


def test_staged_alias_and_nested_branch():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], FLAG: ti.Const[bool],
          OTHER: ti.Const[bool]):
        alias = FLAG
        both = alias and not OTHER
        ti.static_assert(both == (FLAG and not OTHER))
        if alias:
            if both:
                ti.store(out, 0, 1)
            else:
                ti.store(out, 0, 2)
        else:
            ti.store(out, 0, 3)
    out = np.zeros(1, np.int32)
    for flag, other in ((True, True), (False, False), (True, False)):
        k[(1,)](out, FLAG=flag, OTHER=other)
        assert out[0] == (3 if not flag else 2 if other else 1)


def test_assertions_only_evaluate_selected_static_branches():
    @ti.jit
    def k(FLAG: ti.Const[bool], OTHER: ti.Const[bool]):
        if FLAG:
            ti.static_assert(FLAG)
            if OTHER:
                ti.static_assert(False)
        else:
            ti.static_assert(not FLAG)
    for flag, other in ((False, False), (False, True), (True, False)):
        k.materialize({"FLAG": flag, "OTHER": other})
    with pytest.raises(TilaError, match="TILA-CONST-005"):
        k.materialize({"FLAG": True, "OTHER": True})


def test_bool_dimension_name_rejected():
    with pytest.raises(TilaError, match="TILA-CONST-001"):
        @ti.jit
        def k(out: ti.Buffer[ti.i32, (FLAG_DIM,), ti.WriteOnly], FLAG: ti.Const[bool]):
            pass


def test_none_default_rejected():
    with pytest.raises(TilaError, match="TILA-CONST-008"):
        @ti.jit
        def k(FLAG: ti.Const[bool] = None):
            pass


def test_module_boolean_assertions():
    @ti.jit
    def k():
        ti.static_assert(True != False)
        ti.static_assert(not False)
        alias = not False
        ti.static_assert(alias)
    k.materialize()


@ti.jit
def declaration(FLAG: ti.Const[bool] = False):
    ti.static_assert(not FLAG)


def test_bool_declaration_golden():
    from pathlib import Path
    expected = (Path(__file__).parent / "golden" / "m2_explain" / "const-bool.txt").read_text()
    declaration.materialize()
    assert declaration.explain().rstrip() + "\n" == expected
    with pytest.raises(TilaError, match="TILA-CONST-005"):
        declaration.materialize({"FLAG": True})


def test_launch_integer_contract_rejects_boolean():
    with pytest.raises(TilaError, match="TILA-SYN-063"):
        @ti.assume_launch("FLAG == 1")
        @ti.jit
        def k(FLAG: ti.Const[bool]):
            pass


def test_loop_does_not_reuse_staged_alias():
    @ti.jit
    def k(out: ti.Buffer[ti.bool, (1,), ti.WriteOnly], FLAG: ti.Const[bool],
          COUNT: ti.Const[int]):
        active = FLAG
        for i in ti.range(0, COUNT):
            active = not active
        ti.store(out, 0, active)
    out = np.zeros(1, bool)
    for count in (0, 1, 2, 3):
        k[(1,)](out, FLAG=True, COUNT=count)
        assert out[0] == (count % 2 == 0)


def test_multi_kernel_cli_type_conflict(tmp_path, capsys):
    from tila.cli import main
    path = tmp_path / "conflict.py"
    path.write_text("import tila as ti\n@ti.jit\ndef a(X: ti.Const[bool]):\n    pass\n"
                    "@ti.jit\ndef b(X: ti.Const[int]):\n    pass\n")
    assert main(["check", str(path), "--const", "X=true"]) == 2
    output = capsys.readouterr()
    assert "base-10 integer" in output.err
    assert "通过" not in output.out


@pytest.mark.parametrize("text,valid", [("true", True), ("false", True), ("1", False),
                                        ("True", False), ("0", False)])
def test_cli_typed_bool(tmp_path, capsys, text, valid):
    from tila.cli import main
    file = tmp_path / "bool_kernel.py"
    file.write_text("import tila as ti\n@ti.jit\ndef k(FLAG: ti.Const[bool]):\n    pass\n")
    assert main(["check", str(file), "--const", f"FLAG={text}"]) == (0 if valid else 2)
    capsys.readouterr()
