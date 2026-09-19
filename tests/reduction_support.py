"""Source-backed kernels shared by CPU contract and explicit GPU tests."""
import importlib.util
import uuid


def reduction_kernel(directory, dtype, op, axis=1):
    name = f"reduce_{dtype}_{op}_{axis}_{uuid.uuid4().hex}"
    path = directory / f"{name}.py"
    size = 4 if axis == 1 else 8
    path.write_text(f'''import tila as ti
@ti.jit
def kernel(x: ti.Buffer[ti.{dtype}, (4, 8), ti.ReadOnly],
           out: ti.Buffer[ti.{dtype}, ({size},), ti.WriteOnly]):
    rows = ti.arange(0, 4)[:, None]
    cols = ti.arange(0, 8)[None, :]
    v = ti.load(x, (rows, cols))
    result = ti.{op}(v, {axis})
    ti.store(out, ti.arange(0, {size}), result)
''')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel
