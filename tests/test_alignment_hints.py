import numpy as np
from pathlib import Path
import pytest

from alignment_support import kernel
from tila.alignment import collect, derived_byte_alignment
from tila.lowering import Lowering
from tila.runtime import _tensor_info, _triton_cache_key
from tila.errors import TilaLaunchContractError
from tila.predicates import CHECKED


@pytest.mark.parametrize("element,stride,multiple,expected", [(1,1,1,1),(2,1,1,2),(4,1,1,4),(8,1,1,8),(4,1,4,16),(4,2,1,8),(4,-1,1,4),(4,0,1,16)])
def test_byte_alignment_is_not_element_divisibility(element, stride, multiple, expected):
    assert derived_byte_alignment(16, element, multiple, stride) == expected


@pytest.mark.parametrize("kind", ["buffer", "ptr"])
def test_checked_binding_only_and_failed_launch_clears_evidence(tmp_path, kind):
    k = kernel(tmp_path, kind=kind)
    x = np.zeros(32, np.float32)
    assert x.ctypes.data % 16 == 0
    assert "byte_address" not in k.explain({})
    assert "tl.pointer_type" not in Lowering(k.tk).kernel_source()
    k[(2,)](x[4:17])  # aligned nonzero view offset, tail mask
    facts = k.last_alignment_facts
    assert facts and facts[0].origin.kind == CHECKED
    assert "byte_address(x_ptr), 16" in k.explain({})
    emitter = Lowering(k.tk, alignment_facts=facts)
    src = emitter.kernel_source()
    assert "tl.multiple_of(tl.cast(x_ptr, tl.uint64), 16)" in src
    assert all(origins for _, origins in emitter.hint_audit)
    assert src == emitter.kernel_source()  # no duplicate/stale audit entries
    assert _triton_cache_key(k, {}, source=src, alignment_facts=facts) != _triton_cache_key(k, {}, source=src)
    with pytest.raises(TilaLaunchContractError):
        k[(1,)](x[1:9])
    assert not k.last_alignment_facts and not k.tk.runtime_alignments
    assert "byte_address" not in k.explain({})
    with pytest.raises(TilaLaunchContractError):
        k[(1,)]()  # even an early binding failure clears previous evidence
    assert not k.last_alignment_facts


@pytest.mark.parametrize("layout", ["strided", "negative", "broadcast", "undeclared"])
def test_no_inferred_alignment_from_allocator_or_layout(tmp_path, layout):
    k = kernel(tmp_path, alignment=None if layout == "undeclared" else 16)
    x = np.zeros(32, np.float32)
    views = {"strided": x[::2], "negative": x[16::-1],
             "broadcast": np.broadcast_to(x[:1], (8,)), "undeclared": x}
    assert collect(k.tk, {"x": views[layout]}, _tensor_info) == ()
    # Stale runtime metadata (or an assumed alignment) cannot authorize emission.
    k.tk.runtime_alignments[k.tk.buffers[0].region_id] = 256
    assert "tl.pointer_type" not in Lowering(k.tk).kernel_source()


def test_zero_grid_emits_no_alignment_hint(tmp_path):
    k = kernel(tmp_path)
    x = np.zeros(8, np.float32)
    k[(1,)](x)
    assert k.last_alignment_facts
    k[(0,)](x)
    assert not k.last_alignment_facts and "byte_address" not in k.explain({})


def test_alignment_explain_provenance_golden(tmp_path):
    k = kernel(tmp_path)
    k[(1,)](np.zeros(8, np.float32))
    section = k.explain({}).split("hints:\n", 1)[1].split("effects:\n", 1)[0]
    assert section == (Path(__file__).parent / "golden/alignment_hints.txt").read_text()
