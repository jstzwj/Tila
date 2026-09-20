import numpy as np
import tila as ti


@ti.jit
def collision(x: ti.Buffer[ti.i32, (8,), ti.WriteOnly]):
    ti.store(x, 0, 1)


@ti.jit
def duplicate(x: ti.Buffer[ti.i32, (8,), ti.WriteOnly]):
    i = ti.arange(0, 4)
    ti.store(x, i * 0, i)


@ti.jit
def partition(x: ti.Buffer[ti.i32, (8,), ti.WriteOnly], STEP: ti.Const[int] = 4):
    i = ti.program_id(0) * STEP + ti.arange(0, 4)
    ti.store(x, i, i, mask=i < 8)


@ti.jit
def copy_reverse(a: ti.Buffer[ti.i32, (4,), ti.ReadOnly], x: ti.Buffer[ti.i32, (4,), ti.WriteOnly]):
    p = ti.program_id(0)
    value = ti.load(a, p, mask=p < 4, other=0)
    ti.store(x, 3 - p, value, mask=p < 4)


@ti.jit
def matrix(x: ti.Buffer[ti.i32, (2, 2), ti.WriteOnly]):
    ti.store(x, (0, 0), 4)


def main():
    collision[(2,)](np.zeros(8, np.int32))
