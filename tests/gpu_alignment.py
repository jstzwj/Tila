"""Aligned view/Ptr hints: real SM86 compilation, execution and cache isolation."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from alignment_support import kernel
from tila.lowering import Lowering
from tila.errors import TilaLaunchContractError
from tila.runtime import _Launcher


@pytest.mark.parametrize("kind", ["buffer", "ptr"])
@pytest.mark.parametrize("dtype,torch_dtype", [("i8",torch.int8),("f16",torch.float16),("f32",torch.float32),("f64",torch.float64)])
def test_aligned_hint_on_off(kind, dtype, torch_dtype, tmp_path, monkeypatch):
    k = kernel(Path(os.environ.get("TILA_GPU_KERNELS", tmp_path)), dtype, kind)
    size = torch.empty((), dtype=torch_dtype).element_size()
    offset = 16 // size
    base = torch.zeros(64, dtype=torch_dtype, device="cuda")
    view = base[offset:offset+13]
    k[(2,)](view)
    assert k.last_alignment_facts[0].bytes == 16
    source = Lowering(k.tk, alignment_facts=k.last_alignment_facts).kernel_source()
    assert "tl.multiple_of(tl.cast(x_ptr, tl.uint64), 16)" in source
    k[(2,)](view)
    assert len(k._kern_cache) == 1
    expected = np.zeros(64)
    expected[offset:offset+13] = 2
    np.testing.assert_array_equal(base.cpu().numpy(), expected)
    # Disable only new alignment hints, preserving the same JIT and binding.
    monkeypatch.setattr(Lowering, "alignment_prefix", lambda self: [])
    base.zero_()
    k[(2,)](view)
    k[(2,)](view)
    np.testing.assert_array_equal(base.cpu().numpy(), expected)
    assert len(k._kern_cache) == 2
    monkeypatch.setattr(_Launcher, "_execute", lambda *a: pytest.fail("invalid contract executed"))
    with pytest.raises(TilaLaunchContractError):
        k[(1,)](base[1:9])
    assert not k.last_alignment_facts


def test_strided_binding_does_not_reuse_contiguous_hint(tmp_path):
    k = kernel(Path(os.environ.get("TILA_GPU_KERNELS", tmp_path)))
    x = torch.zeros(32, device="cuda")
    k[(2,)](x[:16])
    assert k.last_alignment_facts
    x.zero_()
    k[(2,)](x[::2])
    assert not k.last_alignment_facts and len(k._kern_cache) == 2
    np.testing.assert_array_equal(x.cpu().numpy(), np.tile([1,0],16))
