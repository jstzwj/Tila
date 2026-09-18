"""M1-06: diagnostic registry, ExactInt, and deferred static assertions."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import re

import numpy as np
import pytest

import tila as ti
from tila import cli
from tila.errors import (DIAGNOSTIC_REGISTRY, DiagnosticPhase, TilaError,
                         TilaLaunchContractError)


ROOT = Path(__file__).resolve().parents[1]


class IntSubclass(int):
    pass


def _raised_codes_in_sources():
    found = set()
    for path in (ROOT / "src/tila").glob("*.py"):
        if path.name == "errors.py":
            continue
        found.update(re.findall(r"TILA-[A-Z]+-[0-9]{3}",
                                path.read_text(encoding="utf-8")))
    return found


def test_every_emitted_diagnostic_code_is_registered():
    emitted = _raised_codes_in_sources()
    assert emitted
    assert emitted <= set(DIAGNOSTIC_REGISTRY)
    assert set(DIAGNOSTIC_REGISTRY) - emitted == set()


def test_diagnostic_specs_are_immutable_and_well_formed():
    for code, spec in DIAGNOSTIC_REGISTRY.items():
        assert code == spec.code
        assert spec.phases
        assert spec.summary
        assert spec.default_fix
    with pytest.raises(TypeError):
        DIAGNOSTIC_REGISTRY["TILA-FAKE-001"] = next(
            iter(DIAGNOSTIC_REGISTRY.values())
        )
    with pytest.raises(FrozenInstanceError):
        next(iter(DIAGNOSTIC_REGISTRY.values())).summary = "changed"


def test_all_diagnostic_kinds_render_phase_location_and_fix():
    error = TilaError("TILA-CONST-008", "bad Const")
    launch = TilaLaunchContractError("TILA-CONST-008", "bad launch Const")
    for rendered in (error.render(), launch.render()):
        assert "at:" in rendered
        assert "phase:" in rendered
        assert "fix:" in rendered
    assert DiagnosticPhase.SPECIALIZE in error.spec.phases
    assert DiagnosticPhase.LAUNCH in launch.spec.phases


def test_const_default_requires_exact_python_int():
    with pytest.raises(TilaError) as exc:
        @ti.jit
        def bad(BLOCK: ti.Const[int] = True):
            pass
    assert exc.value.code == "TILA-CONST-008"
    assert "bool" in exc.value.render()

    with pytest.raises(TilaError) as exc:
        @ti.jit
        def bad_float(BLOCK: ti.Const[int] = 1.0):
            pass
    assert exc.value.code == "TILA-CONST-008"
    assert "float" in exc.value.render()


@pytest.mark.parametrize("value", [True, 64.0, "64", np.int64(64)])
def test_materialize_and_explain_reject_const_coercion(value):
    @ti.jit
    def kernel(BLOCK: ti.Const[int]):
        pass

    for entry in (kernel.materialize, kernel.explain):
        with pytest.raises(TilaError) as exc:
            entry({"BLOCK": value})
        assert exc.value.code == "TILA-CONST-008"
        assert type(value).__name__ in exc.value.render()


def test_unknown_const_override_has_specific_diagnostic():
    @ti.jit
    def kernel(BLOCK: ti.Const[int] = 64):
        pass

    with pytest.raises(TilaError) as exc:
        kernel.materialize({"BLOCK": 64, "TYPO": 1})
    assert exc.value.code == "TILA-CONST-010"
    assert "TYPO" in exc.value.render()


@pytest.mark.parametrize("value", [True, 64.0, np.int64(64), IntSubclass(64)])
def test_launch_rejects_non_exact_const_int(value):
    @ti.jit
    def kernel(BLOCK: ti.Const[int]):
        pass

    with pytest.raises(TilaLaunchContractError) as exc:
        kernel[(1,)](BLOCK=value)
    assert exc.value.code == "TILA-CONST-008"
    assert type(value).__name__ in exc.value.render()


def test_static_assert_is_rechecked_for_every_specialization():
    @ti.jit
    def kernel(BLOCK: ti.Const[int] = 32):
        ti.static_assert(BLOCK % 16 == 0)

    kernel.materialize({"BLOCK": 64})
    with pytest.raises(TilaError) as exc:
        kernel.materialize({"BLOCK": 30})
    assert exc.value.code == "TILA-CONST-005"
    assert "BLOCK" in exc.value.render()
    assert "30" in exc.value.render()


def test_static_assert_without_default_is_deferred_and_supports_bool_ops():
    @ti.jit
    def kernel(BLOCK: ti.Const[int]):
        ti.static_assert(BLOCK >= 16 and not (BLOCK % 16 != 0))

    assert any(item["kind"] == "static_assert" for item in kernel.tk.deferred)
    kernel.materialize({"BLOCK": 32})
    with pytest.raises(TilaError) as exc:
        kernel.materialize({"BLOCK": 8})
    assert exc.value.code == "TILA-CONST-005"


def test_static_assert_rejects_runtime_bool():
    with pytest.raises(TilaError) as exc:
        @ti.jit
        def kernel(flag: ti.bool):
            ti.static_assert(flag)
    assert exc.value.code == "TILA-CONST-006"
    rendered = exc.value.render()
    assert "found:" in rendered and "required:" in rendered


def test_static_assert_arithmetic_failure_is_structured():
    @ti.jit
    def kernel(BLOCK: ti.Const[int]):
        ti.static_assert(BLOCK // 0 > 1)

    with pytest.raises(TilaError) as exc:
        kernel.materialize({"BLOCK": 32})
    assert exc.value.code == "TILA-CONST-009"
    assert "ZeroDivisionError" in exc.value.render()


def test_static_assert_bool_ops_preserve_short_circuiting():
    @ti.jit
    def kernel(BLOCK: ti.Const[int]):
        ti.static_assert(BLOCK == 0 or 64 // BLOCK > 1)

    kernel.materialize({"BLOCK": 0})
    kernel.materialize({"BLOCK": 16})
    with pytest.raises(TilaError) as exc:
        kernel.materialize({"BLOCK": 64})
    assert exc.value.code == "TILA-CONST-005"


def test_cli_bad_const_and_internal_failure_do_not_leak_traceback(
        tmp_path, capsys, monkeypatch):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import tila as ti\n\n"
        "@ti.jit\n"
        "def kernel(BLOCK: ti.Const[int] = 64):\n"
        "    pass\n",
        encoding="utf-8",
    )
    assert cli.main(["check", str(source), "--const", "BLOCK=nope"]) == 2
    err = capsys.readouterr().err
    assert "TILA-CONST-008" in err and "Traceback" not in err

    monkeypatch.setattr(cli, "_load", lambda path: (_ for _ in ()).throw(
        RuntimeError("boom")))
    assert cli.main(["check", str(source)]) == 4
    err = capsys.readouterr().err
    assert "TILA-INTERNAL-001" in err and "RuntimeError: boom" in err
    assert "Traceback" not in err

    monkeypatch.setenv("TILA_DEBUG", "1")
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["check", str(source)])
