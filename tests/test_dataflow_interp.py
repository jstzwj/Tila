"""M2-04: control-flow counterexamples and executable CPU memory oracle."""
from dataclasses import replace

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError
from tila.facts import Facts, evaluate_obligation, PROVEN_UNSAFE, UNKNOWN
from tila.interp import Interp, run_kernel

N = ti.Dim("N")


def test_return_guard_dominates_and_defines_continuation():
    @ti.jit
    def k(x: ti.Buffer[ti.i32, (N,), ti.ReadWrite], i: ti.i32):
        if i < 0 or i >= N:
            unused = 1.0
            return
        else:
            unused = i
        ti.store(x, unused, 7)

    for i in (-1, 0, 3, 4):
        x = np.zeros(4, dtype=np.int32)
        k[(1,)](x, i)
        expected = np.zeros_like(x)
        if 0 <= i < 4:
            expected[i] = 7
        np.testing.assert_array_equal(x, expected)


def test_branch_phi_and_literals_match_executed_branch():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (3,), ti.WriteOnly], flag: ti.i32):
        if flag > 0:
            index = 1
        else:
            index = 2
        if index == 1:
            value = 11
        else:
            value = 22
        ti.store(out, index, value)

    for flag in (0, 1):
        out = np.zeros(3, dtype=np.int32)
        k[(1,)](out, flag)
        np.testing.assert_array_equal(out, [0, 11, 0] if flag else [0, 0, 22])


def test_nested_definition_is_not_definite_on_all_paths():
    with pytest.raises(TilaError, match="undefined|not defined|every path"):
        @ti.jit
        def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], a: ti.i32, b: ti.i32):
            if a > 0:
                if b > 0:
                    value = 1
            else:
                value = 2
            ti.store(out, 0, value)


def test_static_return_keeps_only_live_definitions():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], MODE: ti.Const[int]):
        if MODE > 0:
            value = 1.0
            return
        else:
            value = 4
        ti.store(out, 0, value)

    for mode in (0, 1):
        out = np.zeros(1, dtype=np.int32)
        k[(1,)](out, MODE=mode)
        assert out[0] == (4 if mode == 0 else 0)


def test_identity_and_literal_loop_invariants():
    @ti.jit
    def k(x: ti.Buffer[ti.i32, (N,), ti.ReadWrite], i: ti.i32):
        if i < 0 or i >= N:
            return
        index = i
        value = 5
        for j in ti.range(0, 3):
            index = index
            quotient = 10 // value
            value = 5
            ti.store(x, index, quotient)
        ti.store(x, index, value)

    x = np.zeros(4, dtype=np.int32)
    k[(1,)](x, 2)
    np.testing.assert_array_equal(x, [0, 0, 5, 0])


def test_zero_trip_preserves_initial_values():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
        index = 0
        for j in ti.range(2, 1):
            index = 10
        ti.store(out, index, 9)

    out = np.zeros(1, dtype=np.int32)
    k[(1,)](out)
    assert out[0] == 9


def test_symbolic_zero_trip_edge_preserves_initial_index():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], END: ti.Const[int]):
        index = 0
        for j in ti.range(0, END):
            index = 10
        ti.store(out, index, 9)

    out = np.zeros(1, dtype=np.int32)
    k[(1,)](out, END=0)
    assert out[0] == 9
    with pytest.raises(TilaError, match="out-of-bounds"):
        k[(1,)](out, END=1)


def test_returning_loop_continuation_only_on_empty_edge():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (8,), ti.WriteOnly], end: ti.i32):
        for j in ti.range(0, end):
            return
        if end >= 0:
            ti.store(out, end, 6)

    for end in (-1, 0, 1, 8):
        out = np.zeros(8, np.int32)
        k[(1,)](out, end)
        assert out[0] == (6 if end == 0 else 0)


def test_nonempty_loop_return_exits_program_and_skips_continuation():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
        for j in ti.range(0, 2):
            ti.store(out, 0, 3)
            return
        ti.store(out, 999, 4)

    out = np.zeros(1, dtype=np.int32)
    k[(1,)](out)
    assert out[0] == 3
    assert len(k.tk.obligations) == 1


def test_augassign_has_explicit_rewrite_diagnostic():
    with pytest.raises(TilaError, match="write x = x"):
        @ti.jit
        def k(i: ti.i32):
            i += 1


def test_induction_variable_cannot_shadow_outer_value():
    with pytest.raises(TilaError, match="fresh name"):
        @ti.jit
        def k(i: ti.i32):
            for i in ti.range(0, 2):
                pass


def test_scalar_candidate_replayed_in_interpreter():
    @ti.jit
    def k(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], i: ti.i32):
        if i < 0:
            v = ti.unsafe_load(x, i)

    ob = replace(k.tk.obligations[0], kind="load")
    proof = evaluate_obligation(ob, Facts(), set())
    assert proof.verdict == PROVEN_UNSAFE
    i = int(dict(proof.candidate_counterexample)["i"])
    assert i < 0
    with pytest.raises(AssertionError, match="bounds check failed"):
        run_kernel(k.tk, {"x": np.zeros(8, np.int32)}, {"i": i, "x_stride0": 1}, {}, (1,), debug=True)
    assert evaluate_obligation(replace(ob, execution_context_exact=False),
                               Facts(), set()).verdict == UNKNOWN


def test_loop_counterexample_remains_candidate_but_cpu_reproduces():
    @ti.jit
    def k(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):
        index = 0
        for j in ti.range(0, 2):
            v = ti.unsafe_load(x, index)
            index = index - 1

    ob = replace(k.tk.obligations[0], kind="load")
    assert evaluate_obligation(ob, Facts(), set()).verdict == UNKNOWN
    with pytest.raises(AssertionError, match="bounds check failed"):
        run_kernel(k.tk, {"x": np.zeros(8, np.int32)}, {"x_stride0": 1}, {}, (1,), debug=True)


def memory(debug=True):
    interpreter = Interp.__new__(Interp)
    interpreter.debug = debug
    return interpreter


@pytest.mark.parametrize("dtype", [np.int8, np.uint64, np.float16, np.float32])
def test_empty_and_scalar_masked_load_never_dereferences(dtype):
    cpu = memory()
    arr = np.empty(0, dtype=dtype)
    result = cpu._load(arr, (1,), [np.array([2**64 - 1, 0], np.uint64)],
                       np.array([False, False]), 3)
    np.testing.assert_array_equal(result, [3, 3])
    assert result.dtype == dtype
    assert cpu._load(arr, (1,), [-1], False, 2) == 2
    cpu._store(arr, (1,), [-1], 1, False)


@pytest.mark.parametrize("coord", [-1, 4, np.uint64(2**64 - 1)])
@pytest.mark.parametrize("debug", [False, True])
def test_active_bounds_checked_for_load_and_store(coord, debug):
    cpu = memory(debug)
    arr = np.arange(4, dtype=np.int32)
    error = AssertionError if debug else IndexError
    for op in (lambda: cpu._load(arr, (1,), [coord], None, None),
               lambda: cpu._store(arr, (1,), [coord], 9, None)):
        with pytest.raises(error, match="bounds check failed"):
            op()
    np.testing.assert_array_equal(arr, np.arange(4))


def test_mask_broadcast_and_noncontiguous_scatter():
    cpu = memory()
    base = np.zeros((6, 8), np.float16)
    view = base[::2, ::-2].T
    rows, cols = np.arange(4)[:, None], np.arange(3)[None, :]
    mask = np.array([[True, False, True]])
    cpu._store(view, (), [rows, cols], np.arange(4)[:, None], mask)
    result = cpu._load(view, (), [rows, cols], mask, -1)
    np.testing.assert_array_equal(result, np.where(mask, np.arange(4)[:, None], -1))
    assert result.dtype == np.float16
    assert np.count_nonzero(base) == 6


@ti.jit
def copy_view(x: ti.Buffer[ti.f32, (4, 3), ti.ReadOnly],
              out: ti.Buffer[ti.f32, (4, 3), ti.WriteOnly]):
    rows = ti.arange(0, 4)[:, None]
    cols = ti.arange(0, 4)[None, :]
    value = ti.load(x, (rows, cols), mask=cols < 3)
    ti.store(out, (rows, cols), value, mask=cols < 3)


def test_numpy_negative_stride_views_end_to_end():
    source = np.arange(48, dtype=np.float32).reshape(6, 8)[::2, ::-2].T
    backing = np.zeros((6, 8), np.float32)
    out = backing[::2, ::-2].T
    copy_view[(1,)](source, out)
    np.testing.assert_array_equal(out, source)
    assert np.count_nonzero(backing) == np.count_nonzero(source)


def test_torch_cpu_views_share_output_storage():
    import torch
    source = torch.arange(48, dtype=torch.float32).reshape(6, 8)[::2, ::2].T
    backing = torch.zeros((6, 8))
    out = backing[::2, ::2].T
    copy_view[(1,)](source, out)
    torch.testing.assert_close(out, source)
    assert backing.count_nonzero() == source.count_nonzero()


def test_torch_bfloat16_strided_output_is_not_a_copy():
    import torch

    @ti.jit
    def k(x: ti.Buffer[ti.bf16, (4,), ti.ReadOnly],
          out: ti.Buffer[ti.bf16, (4,), ti.WriteOnly]):
        i = ti.arange(0, 4)
        v = ti.load(x, i)
        ti.store(out, i, v + 1.0)

    x = torch.arange(8, dtype=torch.bfloat16)[::2]
    backing = torch.zeros(8, dtype=torch.bfloat16)
    k[(1,)](x, backing[::2])
    torch.testing.assert_close(backing[::2], x + 1)
    assert not backing[1::2].count_nonzero()


def test_debug_assume_checks_before_memory(monkeypatch):
    @ti.jit
    def k(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], i: ti.i32):
        ti.assume(i >= 0 and i < 8)
        v = ti.load(x, i)

    monkeypatch.setenv("TILA_DEBUG", "1")
    with pytest.raises(AssertionError, match="device_assert failed"):
        k[(1,)](np.zeros(8, np.int32), -1)


def test_zero_stride_readonly_view():
    source = np.broadcast_to(np.arange(3, dtype=np.float32), (4, 3))
    out = np.zeros((4, 3), np.float32)
    copy_view[(1,)](source, out)
    np.testing.assert_array_equal(out, source)


def test_float64_reduction_does_not_accumulate_in_float32():
    @ti.jit
    def k(x: ti.Buffer[ti.f64, (4,), ti.ReadOnly],
          out: ti.Buffer[ti.f64, (1,), ti.WriteOnly]):
        i = ti.arange(0, 4)
        v = ti.load(x, i)
        total = ti.sum(v, 0)
        ti.store(out, 0, total)

    x = np.array([2**40, 1, -2**40, 2], dtype=np.float64)
    out = np.zeros(1, np.float64)
    k[(1,)](x, out)
    assert out[0] == 3


def test_scalar_where_preserves_bool_control_flow():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], flag: ti.i32):
        selected = ti.where(flag > 0, True, False)
        if selected:
            ti.store(out, 0, 3)
        else:
            ti.store(out, 0, 4)

    for flag in (0, 1):
        out = np.zeros(1, np.int32)
        k[(1,)](out, flag)
        assert out[0] == (3 if flag else 4)


def test_numeric_validation_respects_runtime_return():
    @ti.jit
    def k(out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], divisor: ti.i32):
        if divisor == 0:
            return
        value = 8 // divisor
        ti.store(out, 0, value)

    out = np.zeros(1, np.int32)
    k[(1,)](out, 0)
    assert out[0] == 0
    k[(1,)](out, 2)
    assert out[0] == 4
