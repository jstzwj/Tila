"""A4 修复验证：裸指针（Ptr）kernel 参数的注册与 launch 消费。

Ptr 参数（type-system.md §8/§9.3）此前只进 checker 的 VarInfo，从不进
TKernel——launcher 数得 0 个可消费参数，`TILA-TYPE-101 too many
arguments: expected 0`。现在：

- checker 把 Ptr 参数登记进 `tk.ptr_params`（并记录声明序
  `tk.param_order`）；extent 符号与维符号同机制（隐式标量，可由实际
  连续存储、伴随 Buffer shape 或显式标量绑定数值）；
- lowering 签名/实参序：buffer {name}_ptr → ptr {name}_ptr → 标量 → Const；
- runtime 位置实参按声明序映射 buffer|ptr|显式标量（隐式符号与 Const
  不占位置实参）；
- interp 把 Ptr 参数注册为 buffer 并播种 `env[name] = ("ptr", name, 0)`。

注：省略 Extent 的 ReadPtr/WritePtr/RWPtr 会产生 UnknownExtent——Ptr 形式访问的
bounds 义务在 strict 模式下需要 Extent（BOUNDS-001，文档行为）。因此
"可证明"的用例使用 ti.Ptr[dt, access, extent]；无 Extent 用例在
TILA_SAFETY=warn 下运行（义务降级为 warning，数值行为不变）。
"""

import numpy as np
import pytest

import tila as ti
from tila.errors import TilaError, TilaLaunchContractError

N = ti.Dim("N")
R = ti.Dim("R")


class TestPtrParamLaunch:
    # -- (a) 工作复现：伴随 Buffer 绑定 N + Ptr 参数 load/store -----------

    def test_rwptr_double_companion_buffer(self, monkeypatch):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              p: ti.RWPtr[ti.f32],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N                 # N 由伴随 Buffer 的 shape 绑定
            v = ti.load(p + offs, mask=m)
            ti.store(p + offs, v * 2.0, mask=m)

        # 简写为 UnknownExtent → strict 模式拒绝（文档 §9.3）
        x = np.arange(100, dtype=np.float32)
        with pytest.raises(TilaError) as ei:
            k[(ti.cdiv(100, 64),)](x, x, BLOCK=64)
        assert ei.value.code == "TILA-BOUNDS-001"

        # TILA_SAFETY=warn：义务降级为 warning，数值行为照常
        monkeypatch.setenv("TILA_SAFETY", "warn")
        x = np.arange(100, dtype=np.float32)
        k[(ti.cdiv(100, 64),)](x, x, BLOCK=64)
        assert np.array_equal(x, np.arange(100, dtype=np.float32) * 2.0)

    def test_region_ptr_double_strict(self):
        """紧凑 Ptr[..., ReadWrite, N]：Extent 声明使义务可证（strict）。"""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              p: ti.Ptr[ti.f32, ti.ReadWrite, N],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(p + offs, mask=m)
            ti.store(p + offs, v * 2.0, mask=m)

        n = 100
        x = np.arange(n, dtype=np.float32)
        k[(ti.cdiv(n, 64),)](x, x, BLOCK=64)
        assert np.array_equal(x, np.arange(n, dtype=np.float32) * 2.0)

    # -- (b) 同一 kernel 的读/写两个 Ptr 参数 --------------------------------

    def test_read_write_ptr_pair(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              src: ti.Ptr[ti.f32, ti.ReadOnly, N],
              dst: ti.Ptr[ti.f32, ti.WriteOnly, N],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(src + offs, mask=m)
            ti.store(dst + offs, v + 1.0, mask=m)

        n = 100
        a = np.arange(n, dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)
        k[(ti.cdiv(n, 64),)](a, a, b, BLOCK=64)
        assert np.array_equal(b, a + 1.0)
        assert np.array_equal(a, np.arange(n, dtype=np.float32))  # src 未被改

    def test_writeptr_readptr_shorthand_pair(self, monkeypatch):
        """UnknownExtent 短别名在 warn 模式下同 kernel 绑定两个。"""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              src: ti.ReadPtr[ti.f32],
              dst: ti.WritePtr[ti.f32],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(src + offs, mask=m)
            ti.store(dst + offs, v * 3.0, mask=m)

        monkeypatch.setenv("TILA_SAFETY", "warn")
        n = 100
        a = np.arange(n, dtype=np.float32)
        b = np.zeros(n, dtype=np.float32)
        k[(ti.cdiv(n, 64),)](a, a, b, BLOCK=64)
        assert np.array_equal(b, np.arange(n, dtype=np.float32) * 3.0)

    # -- 位置实参映射：声明序（buffer | ptr | 显式标量）----------------------

    def test_positional_declaration_order(self):
        @ti.jit
        def k(p: ti.Ptr[ti.f32, ti.ReadWrite, N],      # ptr 在最前
              x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              kk: ti.i32,                              # 显式标量
              BLOCK: ti.Const[int] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(p + offs, mask=m)
            ti.store(p + offs, v * 2.0 + ti.cast[ti.f32](kk), mask=m)

        n = 100
        arr = np.arange(n, dtype=np.float32)
        k[(ti.cdiv(n, 64),)](arr, arr, 1, BLOCK=64)
        assert np.array_equal(arr, np.arange(n, dtype=np.float32) * 2.0 + 1.0)

    def test_missing_ptr_tensor_named(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              p: ti.Ptr[ti.f32, ti.ReadWrite, N],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            ti.store(p + offs, 1.0, mask=offs < N)

        arr = np.zeros(64, dtype=np.float32)
        with pytest.raises(TilaLaunchContractError) as ei:
            k[(1,)](arr)
        assert ei.value.code == "TILA-TYPE-101"
        assert "pointer parameter 'p'" in ei.value.render()

    # -- (c) 契约校验 --------------------------------------------------------

    def test_ptr_dtype_mismatch(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              p: ti.Ptr[ti.f32, ti.ReadWrite, N],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            v = ti.load(p + offs, mask=offs < N)
            ti.store(p + offs, v, mask=offs < N)

        x = np.zeros(64, dtype=np.float32)
        bad = np.zeros(64, dtype=np.float64)      # 声明 f32，传入 f64
        with pytest.raises(TilaLaunchContractError) as ei:
            k[(1,)](x, bad, BLOCK=64)
        assert ei.value.code == "TILA-TYPE-101"
        assert "pointer parameter 'p'" in ei.value.render()

    def test_store_through_readptr_param_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                  p: ti.Ptr[ti.f32, ti.ReadOnly, N],
                  BLOCK: ti.Const[int] = 64):
                offs = ti.arange(0, BLOCK)
                ti.store(p + offs, 1.0, mask=offs < N)
        assert ei.value.code == "TILA-MEM-001"

    def test_load_through_writeptr_param_rejected(self):
        with pytest.raises(TilaError) as ei:
            @ti.jit
            def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
                  p: ti.Ptr[ti.f32, ti.WriteOnly, N],
                  BLOCK: ti.Const[int] = 64):
                offs = ti.arange(0, BLOCK)
                v = ti.load(p + offs, mask=offs < N)
                ti.store(p + offs, v, mask=offs < N)
        assert ei.value.code == "TILA-MEM-001"

    def test_ptr_alignment_declaration(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              p: ti.Ptr[ti.f32, ti.ReadWrite, N, 16],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            v = ti.load(p + offs, mask=offs < N)
            ti.store(p + offs, v, mask=offs < N)

        buf = np.zeros(130, dtype=np.float32)
        x = buf[:64]
        misaligned = buf[1:65]              # data_ptr 偏移 4B → 不满足 16
        with pytest.raises(TilaLaunchContractError) as ei:
            k[(1,)](x, misaligned, BLOCK=64)
        assert ei.value.code == "TILA-MEM-003"

    # -- (d) extent 符号绑定 --------------------------------------------------

    def test_unbound_extent_symbol_binds_available_storage(self):
        @ti.jit
        def k(p: ti.Ptr[ti.f32, ti.ReadWrite, R],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            v = ti.load(p + offs, mask=offs < R)
            ti.store(p + offs, v, mask=offs < R)

        arr = np.zeros(64, dtype=np.float32)
        k[(1,)](arr, BLOCK=64)

    def test_region_bound_by_companion_buffer(self):
        """extent 符号由伴随 Buffer 的 dim 提供数值 → 正常 launch。"""
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              p: ti.Ptr[ti.f32, ti.ReadWrite, N],
              BLOCK: ti.Const[int] = 64):
            offs = ti.arange(0, BLOCK)
            v = ti.load(p + offs, mask=offs < N)
            ti.store(p + offs, v + 1.0, mask=offs < N)

        n = 64
        arr = np.arange(n, dtype=np.float32)
        k[(1,)](arr, arr, BLOCK=64)
        assert np.array_equal(arr, np.arange(n, dtype=np.float32) + 1.0)


class TestPtrParamLowering:
    """(e) materialize：{name}_ptr 落在 buffer 之后、标量之前。"""

    def _kernel(self):
        @ti.jit
        def k(x: ti.Buffer[ti.f32, (N,), ti.ReadOnly],
              p: ti.Ptr[ti.f32, ti.ReadWrite, N],
              BLOCK: ti.Const[int, ti.PowerOfTwo] = 64):
            pid = ti.program_id(0)
            offs = pid * BLOCK + ti.arange(0, BLOCK)
            m = offs < N
            v = ti.load(p + offs, mask=m)
            ti.store(p + offs, v * 2.0, mask=m)
        return k

    def test_materialize_signature_order(self):
        src, dump = self._kernel().materialize({"BLOCK": 64})
        assert "def k(\n    x_ptr,\n    p_ptr,\n    N: tl.int32,\n    x_stride0: tl.int32,\n" \
            "    BLOCK: tl.constexpr\n):" in src
        # 指针算术引用 ptr 参数实参名 p_ptr
        assert "(p_ptr + " in src

    def test_launch_args_order(self):
        from tila.lowering import Lowering
        k = self._kernel()
        assert Lowering(k.tk).launch_args() == \
            ["x_ptr", "p_ptr", "N", "x_stride0", "BLOCK"]

    def test_dump_has_ptr_param_line(self):
        dump = self._kernel().tk.dump()
        assert "ptr    p : Ptr[f32, Global, ReadWrite, N, UnknownAlignment]" in dump

    def test_param_order_recorded(self):
        tk = self._kernel().tk
        assert tk.param_order == [("buffer", "x"), ("ptr", "p"),
                                  ("const", "BLOCK")]
