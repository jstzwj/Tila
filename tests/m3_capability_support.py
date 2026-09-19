"""Deterministic source-backed kernels, retained by the GPU audit artifact."""
import importlib.util


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
