import tila as ti


@ti.jit
def audit(FLAG: ti.Const[bool] = False):
    if FLAG:
        return
    p = ti.program_id(0)
