import tila as ti


@ti.jit
def eager(x: ti.Buffer[ti.i32, (4,), ti.ReadOnly]):
    v = ti.where(True, ti.load(x, 0), ti.load(x, 1))
