"""Source-backed atomic kernels retained by the GPU audit runner."""
import importlib.util


def kernel(directory, dtype='i32', kind='buffer'):
    name = f'atomic_{dtype}_{kind}'
    annotation = f'ti.Buffer[ti.{dtype}, (64,), ti.ReadWrite]' if kind == 'buffer' else f'ti.RWPtr[ti.{dtype}, 64]'
    target = 'x, offset' if kind == 'buffer' else 'x + offset'
    path = directory / f'{name}.py'
    path.write_text(f'''import tila as ti
@ti.jit
def kernel(x: {annotation}, values: ti.Buffer[ti.{dtype}, (64,), ti.ReadOnly],
           old: ti.Buffer[ti.{dtype}, (64,), ti.WriteOnly],
           COUNT: ti.Const[int] = 64, COLLIDE: ti.Const[bool] = True,
           KEEP: ti.Const[bool] = True):
    i = ti.program_id(0) * 32 + ti.arange(0, 32)
    if COLLIDE:
        offset = i * 0
    else:
        offset = i
    value = ti.load(values, i, mask=i < COUNT, other=0)
    if KEEP:
        before = ti.atomic_add({target}, value, mask=i < COUNT)
        ti.store(old, i, before, mask=i < 64)
    else:
        ti.atomic_add({target}, value, mask=i < COUNT)
''')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel
