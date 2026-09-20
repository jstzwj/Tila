"""表面语言子集校验（surface-language.md §2，TILA-SYN 族）。"""

import pytest

import tila as ti
from tila.errors import TilaError

N = ti.Dim("N")


def test_while_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], BLOCK: ti.Const[int] = 64):
            i = 0
            while i < N:
                i = i + 1
    assert ei.value.code == "TILA-SYN-002"


def test_list_comprehension_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            y = [i for i in x]
    assert ei.value.code == "TILA-SYN-002"


def test_python_call_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            y = len(x)
    assert ei.value.code == "TILA-SYN-036"


def test_unknown_tila_attribute_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            y = ti.not_an_intrinsic(x)
    assert ei.value.code == "TILA-SYN-030"


def test_missing_annotation_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x):
            pass
    assert ei.value.code == "TILA-SYN-012"


def test_return_rejected():
    """带值 return 仍被拒绝（v0 kernel 不返回值）；裸 return 合法（提前退出，
    语义细节见 tests/test_fix_return.py）。"""
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            return x
    assert ei.value.code == "TILA-SYN-021"

    @ti.jit
    def k_bare(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
        return
    assert k_bare.tk.name == "k_bare"


def test_local_annotation_rejected():
    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly]):
            y: ti.f32 = ti.load(x, ti.arange(0, 8))
    assert ei.value.code == "TILA-SYN-022"


def test_runtime_loop_start_now_supported():
    """运行期起点已支持（intrinsics.md §2.1 扩展）：字面量非零起点与
    运行期 int 标量起点均合法；float 起点仍拒绝。"""
    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], K: ti.i32):
        for i in ti.range(1, K, 1):
            pass

    @ti.jit
    def k2(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly], K: ti.i32, s: ti.i32):
        for i in ti.range(s, K, 2):
            pass

    with pytest.raises(TilaError) as ei:
        @ti.jit
        def k3(x: ti.Buffer[ti.f32, (N,), ti.WriteOnly],
               s: ti.f32, K: ti.i32):
            for i in ti.range(s, K, 1):
                pass
    assert ei.value.code == "TILA-TYPE-024"


def test_import_tila_as_ti_alias_resolves():
    """`import tila as ti` 的别名命名空间可以被正确解析。"""

    @ti.jit
    def k(x: ti.Buffer[ti.f32, (N,), ti.ReadWrite], BLOCK: ti.Const[int] = 8):
        offs = ti.arange(0, BLOCK)
        m = offs < N
        v = ti.load(x, offs, mask=m)
        ti.store(x, offs, v, mask=m)
    assert k.tk.name == "k"
