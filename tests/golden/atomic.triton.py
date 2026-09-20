import triton
import triton.language as tl


@triton.jit
def _tila_atomic_add(ptr, value, mask):
    mask = tl.cast(mask, tl.int1)
    value = tl.cast(value, ptr.dtype.element_ty)
    if value.dtype == tl.float32:
        # Preserve f32 RMW semantics even for zero and unused old values.
        bits = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=r,r",
            args=[value.to(tl.uint32, bitcast=True)], dtype=tl.uint32, is_pure=False, pack=1)
        value = bits.to(tl.float32, bitcast=True)
    old = tl.atomic_add(ptr, value, mask=mask, sem="relaxed", scope="gpu")
    result = tl.where(mask, old, tl.full((), 0, ptr.dtype.element_ty))
    if value.dtype == tl.float32:
        bits = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=r,r",
            args=[result.to(tl.uint32, bitcast=True)], dtype=tl.uint32, is_pure=False, pack=1)
        result = bits.to(tl.float32, bitcast=True)
    return result


@triton.jit
def simple(
    x_ptr,
    x_stride0: tl.int32
):
    x_stride0 = tl.cast(x_stride0, tl.int32)
    _tila_effect_0 = _tila_atomic_add(x_ptr + tl.cast(x_stride0, tl.int64) * tl.cast(0, tl.int64), 1, True)
