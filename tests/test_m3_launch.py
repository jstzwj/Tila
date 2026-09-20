"""M3-02 fail-before-execution contracts and semantic cache isolation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tila as ti
from tila import dtypes as D, tir as T, types as TY
from tila.dims import Cst
from tila.errors import TilaError, TilaLaunchContractError
from tila.lowering import Lowering, TRITON_TIR_OPS
from tila.runtime import _Launcher, _binding_signature, _triton_cache_key
from tila.target import SUPPORTED, CUDATarget, validate_grid, resolve_cuda
from tila.verifier import verify, NODE_NAMES

N = ti.Dim("N")


@ti.jit
def copy(x: ti.Buffer[ti.i32, (N,), ti.ReadWrite], B: ti.Const[int, ti.PowerOfTwo] = 8):
    i = ti.program_id(0) * B + ti.arange(0, B)
    v = ti.load(x, i, mask=i < N)
    ti.store(x, i, v + 1, mask=i < N)


@ti.jit
def aligned(x: ti.Buffer[ti.i32, (N,), ti.ReadWrite, 16]):
    i = ti.arange(0, 8)
    v = ti.load(x, i, mask=i < N)
    ti.store(x, i, v, mask=i < N)


@pytest.mark.parametrize("grid", [(), (1, 1, 1, 1), (-1,), (True,), (1.0,),
                                    (np.int64(1),), (2**31,), (0, 65536), (0, 1, 65536)])
def test_grid_domain(grid):
    with pytest.raises(TilaLaunchContractError) as exc:
        validate_grid(grid)
    assert exc.value.code == "TILA-TYPE-104"


@pytest.mark.parametrize("grid", [(0,), (0, 1), (1, 0, 1), (2**31 - 1, 65535, 65535)])
def test_grid_endpoints_without_launch(grid):
    validate_grid(grid)


def no_execution(monkeypatch):
    def execute(*args, **kwargs):
        pytest.fail("invalid/empty launch reached execution")
    monkeypatch.setattr(_Launcher, "_execute", execute)


def test_zero_grid_is_noop_after_binding(monkeypatch):
    no_execution(monkeypatch)
    x = np.arange(8, dtype=np.int32)
    copy[(0,)](x)
    np.testing.assert_array_equal(x, np.arange(8))
    copy.launch_auto(np.empty(0, np.int32))
    assert "no-op" in copy.last_report and copy.last_proof_results == ()
    with pytest.raises(TilaLaunchContractError, match="dtype mismatch"):
        copy[(0,)](np.zeros(8, np.float32))
    with pytest.raises((TilaError, TilaLaunchContractError), match="PowerOfTwo"):
        copy[(0,)](x, B=3)


def test_zero_tensor_with_nonzero_grid_executes_masked_body():
    copy[(1,)](np.empty(0, np.int32))


def test_auto_and_callable_grids_share_limits(monkeypatch):
    no_execution(monkeypatch)
    with pytest.raises(TilaLaunchContractError, match=r"grid\[1\]"):
        copy[(0, lambda meta: 65536)](np.zeros(8, np.int32))
    monkeypatch.setattr("tila.runtime._derive_grid", lambda *args: ((0, 65536), {}))
    with pytest.raises(TilaLaunchContractError, match=r"grid\[1\]"):
        copy.launch_auto(np.empty(0, np.int32))


def test_misaligned_view_never_executes_even_on_empty_grid(monkeypatch):
    no_execution(monkeypatch)
    x = np.zeros(16, np.int32)
    offset = next(i for i in range(4) if x[i:].ctypes.data % 16)
    for grid in ((0,), (1,)):
        with pytest.raises(TilaLaunchContractError) as exc:
            aligned[grid](x[offset:])
        assert exc.value.code == "TILA-MEM-003"


def test_target_policy_versions_and_arch():
    props = SimpleNamespace(major=8, minor=6, name=SUPPORTED.name, uuid="gpu")
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: props),
                            version=SimpleNamespace(cuda="12.8"), __version__="2.10.0")
    device = SimpleNamespace(index=0)
    assert resolve_cuda(torch, SimpleNamespace(__version__="3.6.0"), device).capability is SUPPORTED
    for version in ("3.5.0", "3.7.0"):
        with pytest.raises(TilaError, match="Triton version"):
            resolve_cuda(torch, SimpleNamespace(__version__=version), device)
    props.major = 9
    with pytest.raises(TilaError, match="CUDA target"):
        resolve_cuda(torch, SimpleNamespace(__version__="3.6.0"), device)


def test_verifier_coverage_matches_backend():
    from tila.lowering import TRITON_REJECTED_TIR_OPS
    assert not TRITON_TIR_OPS & TRITON_REJECTED_TIR_OPS
    assert NODE_NAMES == TRITON_TIR_OPS | TRITON_REJECTED_TIR_OPS


@pytest.mark.parametrize("nested", [False, True])
def test_unknown_node_rejected_before_source(nested):
    class Future(T.TStmt):
        pass
    body = [Future()]
    if nested:
        body = [T.TIf(T.TLit(True, D.bool_), [], body, line=23)]
    kernel = replace(copy.tk, body=body)
    with pytest.raises(TilaError) as exc:
        Lowering(kernel).kernel_source()
    assert exc.value.code == "TILA-TARGET-009"
    assert "Future" in exc.value.render()


@pytest.mark.parametrize("node", [T.TPid(TY.ScalarT(D.i32), 3),
    T.TBin(TY.ScalarT(D.i32), "**", T.TLit(1, D.i32), T.TLit(2, D.i32)),
    T.TUna(TY.ScalarT(D.i32), "sin", T.TLit(1, D.i32)),
    T.TBin(TY.ScalarT(D.i32), "+", object(), T.TLit(1, D.i32)),
    T.TArange(TY.BlockT(TY.ScalarT(D.i32), (Cst(3),)), 0, T.TLit(3, D.i32))])
def test_invalid_target_expression(node):
    with pytest.raises(TilaError) as exc:
        Lowering(replace(copy.tk, body=[T.TAssign("bad", node, 12)])).kernel_source()
    assert exc.value.code == "TILA-TARGET-009" and "line 12" in exc.value.render()


def test_cycle_rejected():
    node = T.TUna(TY.ScalarT(D.i32), "-", None)
    node.operand = node
    with pytest.raises(TilaError, match="cyclic"):
        verify(replace(copy.tk, body=[T.TAssign("bad", node)]))


def test_const_specialized_tile_gate():
    @ti.jit
    def kernel(out: ti.Buffer[ti.i32, (N,), ti.WriteOnly], B: ti.Const[int] = 8):
        i = ti.arange(0, B)
        ti.store(out, i, i, mask=i < N)
    with pytest.raises(TilaError, match="powers of two"):
        kernel.materialize({"B": 3})
    kernel[(1,)](np.zeros(3, np.int32), B=3)  # CPU still permits non-Triton tiles


def test_cache_source_abi_target_and_launch_isolation(monkeypatch):
    x = np.zeros(32, np.int32)
    base = _binding_signature(copy.tk, {"x": x[:8]})
    key = _triton_cache_key(copy, {"B": 8}, bindings=base)
    for view in (x[:16], x[::2][:8], x[1:9]):
        assert _triton_cache_key(copy, {"B": 8}, bindings=_binding_signature(copy.tk, {"x": view})) != key
    # Equivalent view metadata does not depend on its exact allocation address.
    assert _binding_signature(copy.tk, {"x": x[:8]}) == base
    assert _triton_cache_key(copy, {"B": 8}, source="different generated source", bindings=base) != key
    monkeypatch.setattr(copy, "source_fingerprint", "different original source")
    assert _triton_cache_key(copy, {"B": 8}, bindings=base) != key
    target = CUDATarget(0, "gpu0", "torch", "cuda")
    a = _triton_cache_key(copy, {"B": 8}, target=target)
    assert _triton_cache_key(copy, {"B": 8}, target=replace(target, uuid="gpu1")) != a
    assert _triton_cache_key(copy, {"B": 8}, target=target, num_warps=8) != a
    monkeypatch.setenv("TILA_DEBUG", "1")
    assert _triton_cache_key(copy, {"B": 8}, target=target) != a


def test_hint_requires_definition_specific_static_evidence():
    from tila.predicates import Origin, USER
    kernel = replace(copy.tk, hint_origins={})
    assert "tl.max_contiguous(" not in Lowering(kernel).kernel_source()
    kernel.hint_origins = {"i": {Origin(USER, 99, "assumed")}}
    assert "tl.max_contiguous(" not in Lowering(kernel).kernel_source()
    # A later assignment to the same name has no contiguous-span proof.
    kernel = replace(copy.tk, body=copy.tk.body + [T.TAssign("i", T.TLit(0, D.i32), 999)])
    source = Lowering(kernel).kernel_source()
    assert source.count("tl.max_contiguous(") == 1
    assert "i = 0\n    tl.max_contiguous" not in source


def test_ptr_alignment_contract_also_precedes_execution(monkeypatch):
    @ti.jit
    def kernel(p: ti.RWPtr[ti.i32, 8, 16]):
        v = ti.load(p)
        ti.store(p, v)
    no_execution(monkeypatch)
    data = np.zeros(16, np.int32)
    offset = next(i for i in range(4) if data[i:].ctypes.data % 16)
    with pytest.raises(TilaLaunchContractError, match="Aligned"):
        kernel[(1,)](data[offset:offset + 8])


def test_zero_grid_does_not_bypass_assume_launch(monkeypatch):
    @ti.assume_launch("N % 8 == 0")
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (N,), ti.ReadWrite]):
        pass
    no_execution(monkeypatch)
    with pytest.raises(TilaLaunchContractError) as exc:
        kernel[(0,)](np.zeros(7, np.int32))
    assert exc.value.code == "TILA-BOUNDS-010"


def test_target_resource_bounds_without_allocation():
    node = T.TZeros(TY.BlockT(TY.ScalarT(D.i32), (Cst(1 << 20), Cst(2))),
                    [T.TLit(1 << 20, D.i32), T.TLit(2, D.i32)])
    with pytest.raises(TilaError, match="element limit"):
        verify(replace(copy.tk, body=[T.TAssign("large", node)]))
    @ti.jit
    def kernel():
        a = ti.zeros((16, 8), ti.f16)
        b = ti.zeros((8, 16), ti.f16)
        c = ti.dot(a, b)
    with pytest.raises(TilaError, match="minimum 16"):
        kernel.materialize({})


def test_unconstrained_step_has_no_multiple_of_hint():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (N,), ti.WriteOnly], STEP: ti.Const[int] = 3):
        i = ti.program_id(0) * STEP + ti.arange(0, 8)
        ti.store(x, i, i, mask=(i >= 0) & (i < N))
    assert "tl.multiple_of(" not in Lowering(kernel.tk).kernel_source()


def test_pid_hint_fact_does_not_leak_out_of_branch():
    scalar = TY.ScalarT(D.i32)
    block = TY.BlockT(scalar, (Cst(8),))
    body = [T.TAssign("p", T.TPid(scalar, 0)),
            T.TIf(T.TLit(True, D.bool_), [], [T.TAssign("p", T.TLit(1, D.i32))]),
            T.TAssign("i", T.TBin(block, "+", T.TBin(scalar, "*", T.TName("p"), T.TLit(8, D.i32)),
                                   T.TArange(block, 0, T.TLit(8, D.i32))))]
    emitter = Lowering(replace(copy.tk, body=body, hints=[]))
    from tila.effect_ir import bind_effects
    bind_effects(emitter.tk)
    assert "tl.multiple_of(" not in emitter.kernel_source()
    assert emitter.kernel_source() == emitter.kernel_source()
