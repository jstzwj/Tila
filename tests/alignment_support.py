"""Source-backed alignment cases, shared by CPU and explicit GPU audit."""
import importlib.util


def kernel(directory, dtype="f32", kind="buffer", alignment=16):
    name = f"align_{dtype}_{kind}_{alignment}"
    suffix = f", {alignment}" if alignment else ""
    annotation = f"ti.Buffer[ti.{dtype}, (N,), ti.ReadWrite{suffix}]" if kind == "buffer" else f"ti.RWPtr[ti.{dtype}, N{suffix}]"
    address = "x, i" if kind == "buffer" else "x + i"
    path = directory / f"{name}.py"
    path.write_text(f'''import tila as ti
N = ti.Dim("N")
@ti.jit
def kernel(x: {annotation}):
    i = ti.program_id(0) * 8 + ti.arange(0, 8)
    v = ti.load({address}, mask=i < N)
    ti.store({address}, v + 1, mask=i < N)
''')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel
