import tila as ti


@ti.jit
def kernel(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly], flag: ti.bool):
    if flag:
        v = ti.load(x, 0)
    else:
        v = ti.load(x, 1)
