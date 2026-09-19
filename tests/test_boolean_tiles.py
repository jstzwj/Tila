"""ADR-012 execution, proof identity, and consumer boundaries."""
import numpy as np
import pytest
from pathlib import Path

import tila as ti
from tila.errors import TilaError

N = ti.Dim("N")


@ti.jit
def mixed_mask(flags: ti.Buffer[ti.bool, (N,), ti.ReadOnly],
               data: ti.Buffer[ti.i32, (N,), ti.ReadOnly],
               out: ti.Buffer[ti.i32, (N,), ti.WriteOnly]):
    i = ti.arange(0, 8)
    enabled = ti.load(flags, i, mask=i < N, other=False)
    active = (enabled & (i < N)) | (~enabled & (i < N))
    value = ti.load(data, i, mask=active, other=0)
    selected = ti.where(enabled, value, -1)
    ti.store(out, i, selected, mask=active)


@ti.jit
def broadcast_bool(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
                   out: ti.Buffer[ti.i32, (4, 4), ti.WriteOnly]):
    i = ti.arange(0, 4)
    enabled = ti.load(flags, i)
    active = enabled[:, None] & ~enabled[None, :]
    value = ti.where(active, 1, 0)
    ti.store(out, (i[:, None], i[None, :]), value)


@ti.jit
def direct_bool(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
                data: ti.Buffer[ti.i32, (4,), ti.ReadOnly],
                out: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    i = ti.arange(0, 4)
    enabled = ti.load(flags, i)
    value = ti.load(data, i, mask=enabled, other=-9)
    ti.store(out, i, value, mask=True)
    ti.store(out, i, i + 99, mask=False)


@ti.jit
def reductions(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
               out: ti.Buffer[ti.i32, (1,), ti.WriteOnly], toggle: ti.bool):
    i = ti.arange(0, 4)
    enabled = ti.load(flags, i)
    value = 0
    if enabled.any():
        value = 1
    if enabled.all():
        value = 2
    ti.store(out, 0, value, mask=(toggle | ~toggle) & ~False)


@pytest.mark.parametrize("n", [1, 3, 7, 8])
def test_mixed_mask_tail(n):
    flags = np.arange(n) % 2 == 0
    data = np.arange(n, dtype=np.int32)
    out = np.zeros_like(data)
    mixed_mask[(1,)](flags, data, out)
    np.testing.assert_array_equal(out, np.where(flags, data, -1))


@pytest.mark.parametrize("bits", range(16))
def test_boolean_consumers_exhaustive(bits):
    flags = np.array([bool(bits & (1 << i)) for i in range(4)])
    out = np.zeros((4, 4), dtype=np.int32)
    broadcast_bool[(1,)](flags, out)
    np.testing.assert_array_equal(out, flags[:, None] & ~flags[None, :])
    data = np.arange(4, dtype=np.int32)
    direct = np.zeros_like(data)
    direct_bool[(1,)](flags, data, direct)
    np.testing.assert_array_equal(direct, np.where(flags, data, -9))
    scalar = np.full(1, -1, dtype=np.int32)
    reductions[(1,)](flags, scalar, bool(bits % 2))
    assert scalar[0] == (2 if flags.all() else int(flags.any()))


def test_loaded_bool_does_not_prove_bounds():
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (8,), ti.ReadOnly],
          data: ti.Buffer[ti.i32, (N,), ti.ReadOnly]):
        i = ti.arange(0, 8)
        enabled = ti.load(flags, i)
        value = ti.load(data, i, mask=enabled | (i < N))

    with pytest.raises(TilaError, match="TILA-BOUNDS-001"):
        k[(1,)](np.ones(8, dtype=bool), np.zeros(3, dtype=np.int32))
    assert "Unknown" in k.explain()


def test_copy_identity_can_prove_unreachable_access():
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
          data: ti.Buffer[ti.i32, (1,), ti.ReadOnly]):
        i = ti.arange(0, 4)
        enabled = ti.load(flags, i)
        copied = enabled
        value = ti.load(data, i + 10, mask=enabled & ~copied)

    k[(1,)](np.ones(4, dtype=bool), np.zeros(1, dtype=np.int32))


def test_independent_loads_do_not_share_predicate():
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
          data: ti.Buffer[ti.i32, (1,), ti.ReadOnly]):
        i = ti.arange(0, 4)
        first = ti.load(flags, i)
        second = ti.load(flags, i)
        value = ti.load(data, i + 10, mask=first & ~second)

    with pytest.raises(TilaError, match="TILA-BOUNDS-001"):
        k[(1,)](np.ones(4, dtype=bool), np.zeros(1, dtype=np.int32))


def test_row_column_views_do_not_cancel():
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
          data: ti.Buffer[ti.i32, (1, 1), ti.ReadOnly]):
        i = ti.arange(0, 4)
        enabled = ti.load(flags, i)
        value = ti.load(data, (i[:, None], i[None, :]),
                        mask=enabled[:, None] & ~enabled[None, :])

    with pytest.raises(TilaError, match="TILA-BOUNDS-001"):
        k[(1,)](np.array([True, False, True, False]), np.zeros((1, 1), dtype=np.int32))


def test_mask_cannot_increase_access_rank():
    with pytest.raises(TilaError, match="TILA-SHAPE-010"):
        @ti.jit
        def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
              data: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
            i = ti.arange(0, 4)
            enabled = ti.load(flags, i)
            value = ti.load(data, i, mask=enabled[None, :])


def test_integer_mask_rejected():
    with pytest.raises(TilaError, match="TILA-SHAPE-010"):
        @ti.jit
        def k(data: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
            i = ti.arange(0, 4)
            value = ti.load(data, i, mask=i)


def test_tile_if_still_rejected():
    with pytest.raises(TilaError, match="TILA-TYPE"):
        @ti.jit
        def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly]):
            i = ti.arange(0, 4)
            enabled = ti.load(flags, i)
            if enabled:
                return


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("toggle", [False, True])
def test_branch_and_loop_boolean_values(count, toggle):
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
          out: ti.Buffer[ti.i32, (4,), ti.WriteOnly], choice: ti.bool,
          COUNT: ti.Const[int] = 0):
        i = ti.arange(0, 4)
        loaded = ti.load(flags, i)
        if choice:
            active = loaded & (i < 4)
        else:
            active = ~loaded & (i < 4)
        for step in ti.range(0, COUNT):
            active = ~active
        ti.store(out, i, ti.where(active, 1, 0))

    flags = np.array([True, False, False, True])
    out = np.zeros(4, dtype=np.int32)
    k[(1,)](flags, out, toggle, COUNT=count)
    expected = flags if toggle else ~flags
    np.testing.assert_array_equal(out, ~expected if count % 2 else expected)


def test_branch_cannot_leak_loaded_boolean_identity():
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
          data: ti.Buffer[ti.i32, (1,), ti.ReadOnly], choice: ti.bool):
        i = ti.arange(0, 4)
        loaded = ti.load(flags, i)
        active = loaded & (i < 4)
        if choice:
            active = ~loaded
        value = ti.load(data, i + 10, mask=active & ~loaded)

    with pytest.raises(TilaError, match="TILA-BOUNDS-001"):
        k[(1,)](np.zeros(4, dtype=bool), np.zeros(1, dtype=np.int32), True)


def test_debug_assumption_is_checked_with_boolean_mask(monkeypatch):
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
          data: ti.Buffer[ti.i32, (4,), ti.ReadOnly], index: ti.i32):
        i = ti.arange(0, 4)
        enabled = ti.load(flags, i)
        ti.assume(index >= 0 and index < 4)
        value = ti.load(data, index, mask=enabled.any())

    monkeypatch.setenv("TILA_DEBUG", "1")
    with pytest.raises(AssertionError, match="device_assert failed"):
        k[(1,)](np.zeros(4, dtype=bool), np.zeros(4, dtype=np.int32), -1)


def test_ptr_bool_mask_and_store():
    @ti.jit
    def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly],
          out: ti.Buffer[ti.bool, (4,), ti.WriteOnly]):
        i = ti.arange(0, 4)
        enabled = ti.load(flags.ptr + i)
        value = ti.load(flags.ptr + i, mask=enabled, other=False)
        ti.store(out.ptr + i, value, mask=enabled)

    flags = np.array([False, True, False, True])
    out = np.zeros(4, dtype=bool)
    k[(1,)](flags, out)
    np.testing.assert_array_equal(out, flags)


def test_incompatible_boolean_shapes_rejected():
    with pytest.raises(TilaError, match="TILA-SHAPE-003"):
        @ti.jit
        def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly]):
            i = ti.arange(0, 4)
            j = ti.arange(0, 8)
            enabled = ti.load(flags, i)
            active = enabled & (j < 4)


def test_bool_integer_combination_rejected():
    with pytest.raises(TilaError, match="TILA-TYPE-026"):
        @ti.jit
        def k(flags: ti.Buffer[ti.bool, (4,), ti.ReadOnly]):
            i = ti.arange(0, 4)
            enabled = ti.load(flags, i)
            active = enabled & i


def test_boolean_tile_explain_golden():
    expected = (Path(__file__).parent / "golden" / "m2_explain" / "boolean-tile.txt").read_text()
    fresh = ti.jit(reductions.fn)
    assert fresh.explain() + "\n" == expected
    assert fresh.explain() + "\n" == expected  # warm cache preserves audit ABI
