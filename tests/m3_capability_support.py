"""Deterministic source-backed kernels, retained by the GPU audit artifact."""
import importlib.util
import inspect
from pathlib import Path


def source_kernel(directory, name, source):
    path = directory / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def tile_kernel(directory, dtype, operation, rank):
    shape = "(16,)" if rank == 1 else "(4, 4)"
    expr = (f"ti.zeros({shape}, ti.{dtype})" if operation == "zeros" else
            f"ti.reshape(v, {shape})")
    return source_kernel(directory, f"audit_{operation}_{dtype}_{rank}", f'''import tila as ti
@ti.jit
def kernel(x: ti.Buffer[ti.{dtype}, (16,), ti.ReadOnly],
           out: ti.Buffer[ti.{dtype}, (16,), ti.WriteOnly]):
    i = ti.arange(0, 16)
    v = ti.load(x, i)
    tile = {expr}
    flat = ti.reshape(tile, (16,))
    ti.store(out, i, flat)
''')


def dot_kernel(directory, accumulator="f16", input_dtype="f16"):
    return source_kernel(directory, f"audit_dot_{input_dtype}_{accumulator}", f'''import tila as ti
@ti.jit
def kernel(a: ti.Buffer[ti.{input_dtype}, (16, 32), ti.ReadOnly],
           b: ti.Buffer[ti.{input_dtype}, (32, 16), ti.ReadOnly],
           c: ti.Buffer[ti.{accumulator}, (16, 16), ti.ReadOnly],
           out: ti.Buffer[ti.f32, (16, 16), ti.WriteOnly]):
    m = ti.arange(0, 16)[:, None]
    n = ti.arange(0, 16)[None, :]
    k0 = ti.arange(0, 32)[None, :]
    k1 = ti.arange(0, 32)[:, None]
    av = ti.load(a, (m, k0))
    bv = ti.load(b, (k1, n))
    cv = ti.load(c, (m, n))
    result = ti.dot(av, bv, acc=cv)
    widened = ti.cast[ti.f32](result)
    ti.store(out, (m, n), widened)
''')


def typed_add(directory, dtype):
    # Specialize only dtype in the actual official example; no generic API added.
    path = Path(__file__).resolve().parents[1] / "examples/add_kernel.py"
    spec = importlib.util.spec_from_file_location("audit_official_add", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = inspect.getsource(module.add_kernel.fn)
    source = source.replace("def add_kernel(", "def kernel(").replace("ti.f32", f"ti.{dtype}")
    return source_kernel(directory, f"audit_add_{dtype}", 'import tila as ti\nN = ti.Dim("N")\n' + source)


def elementwise(directory, dtype, operation, output=None, suffix=""):
    output = output or dtype
    name = f"audit_{dtype}_{operation}_{output}_{suffix}"
    expression = {"copy": "v", "cast": f"ti.cast[ti.{output}](v)",
                  "exp": "ti.exp(v)", "exp2": "ti.exp2(v)",
                  "exp_chain": "ti.cast[ti.f32](ti.exp(v))",
                  "exp2_chain": "ti.cast[ti.f32](ti.exp2(v))",
                  "fp8_e4": "ti.cast[ti.f32](ti.cast[ti.f8e4m3fn](v))",
                  "fp8_e5": "ti.cast[ti.f32](ti.cast[ti.f8e5m2](v))"}[operation]
    path = directory / f"{name}.py"
    path.write_text(f'''import tila as ti
N = ti.Dim("N")
@ti.jit
def kernel(x: ti.Buffer[ti.{dtype}, (N,), ti.ReadOnly],
           out: ti.Buffer[ti.{output}, (N,), ti.WriteOnly]):
    i = ti.program_id(0) * 8 + ti.arange(0, 8)
    v = ti.load(x, i, mask=i < N)
    result = {expression}
    ti.store(out, i, result, mask=i < N)
''')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel
