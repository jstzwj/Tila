import numpy as np
import pytest
import tila as ti
from tila.errors import TilaError
from tila.runtime import _triton_cache_key


@ti.jit
def copy(x: ti.Buffer[ti.i32, (1,), ti.ReadOnly], out: ti.Buffer[ti.i32, (1,), ti.WriteOnly]):
    ti.store(out, 0, ti.load(x, 0))


@pytest.mark.parametrize("value", [True, False, 0, 1, 2, 3, 16, 4.0, np.int64(4)])
def test_launch_option_exact_domain(value):
    with pytest.raises(TilaError) as exc:
        copy[(1,)].with_options(num_warps=value)
    assert exc.value.code == "TILA-TARGET-006"


def test_options_do_not_mutate_launch_or_kernel():
    launch = copy[(1,)]
    alternative = launch.with_options(num_warps=8)
    assert launch.num_warps == 4 and alternative.num_warps == 8
    out = np.zeros(1, np.int32)
    alternative(np.array([3], np.int32), out)
    assert out[0] == 3
    assert _triton_cache_key(copy, {}, num_warps=4) != _triton_cache_key(copy, {}, num_warps=8)
    assert _triton_cache_key(copy, {}, target=("cuda", 0)) != _triton_cache_key(copy, {}, target=("cuda", 1))
