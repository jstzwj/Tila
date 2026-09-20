"""ADR-017 atomic qualifiers and deterministic CPU arithmetic."""
from enum import Enum


class MemoryOrder(Enum):
    RELAXED = 'relaxed'


class MemoryScope(Enum):
    GPU = 'gpu'


Relaxed = MemoryOrder.RELAXED
GPU = MemoryScope.GPU


def add(old, value, dtype):
    import numpy as np
    from . import numeric
    if dtype.is_int:
        return numeric.wrap(int(old) + int(value), dtype)
    def ftz(x):
        x = np.float32(x)
        return np.copysign(np.float32(0), x) if 0 < abs(x) < np.finfo(np.float32).tiny else x
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        return ftz(np.float32(ftz(old) + ftz(value)))


def validate_bindings(kernel, tensors, tensor_info):
    from .effect_ir import verify_effects
    from . import tir as T
    from .errors import TilaLaunchContractError
    names = {node.effect.region_id.name for node in verify_effects(kernel)
             if isinstance(node, T.TAtomicAdd)}
    for name in sorted(names):
        tensor = tensors[name]
        _, _, strides, address = tensor_info(tensor)
        byte_strides = getattr(tensor, 'strides', None)
        if address % 4 or any(s <= 0 for s in strides) or (
                byte_strides is not None and any(s % 4 for s in byte_strides)):
            raise TilaLaunchContractError('TILA-MEM-008',
                f'atomic binding {name} requires natural alignment and positive element strides')


def validate_node(node, memory, value_type, consts=None):
    from . import types as TY, dtypes as D, tir as T
    from .dims import equal, free_syms, eval_num
    from .verifier import fail
    def bad(message):
        fail('invalid atomic: ' + message, node.line)
    if memory.access is not TY.READ_WRITE or memory.space is not TY.GLOBAL:
        bad('requires Global ReadWrite')
    if memory.elem not in (D.i32, D.u32, D.f32):
        bad('unsupported dtype')
    if node.order is not Relaxed or node.scope is not GPU:
        bad('unsupported order/scope')
    def shape(vt):
        return vt.dims if isinstance(vt, (TY.BlockT, TY.MaskT)) else ()
    def dt(vt):
        return D.bool_ if isinstance(vt, TY.MaskT) else getattr(vt.elem if isinstance(vt, TY.BlockT) else vt, 'dtype', None)
    def same(a, b):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if equal(x, y):
                continue
            syms = free_syms(x) | free_syms(y)
            if syms <= (consts or {}).keys():
                if eval_num(x, consts or {}) != eval_num(y, consts or {}):
                    return False
            elif consts is not None:
                return False
        return True
    access_shape = ()
    if node.buffer is not None:
        if not isinstance(memory, TY.BufferT) or len(node.coords) != len(memory.dims):
            bad('coordinate rank')
        for coord in node.coords:
            vt = value_type(coord)
            if dt(vt) is None or not dt(vt).is_int:
                bad('coordinate dtype')
            if shape(vt):
                if access_shape and not same(access_shape, shape(vt)):
                    bad('coordinate shape')
                access_shape = shape(vt)
    else:
        if node.coords:
            bad('Ptr cannot have Buffer coordinates')
        access_shape = shape(value_type(node.ptr))
    if len(access_shape) > 1 or not same(shape(node.vt), access_shape) or dt(node.vt) is not memory.elem:
        bad('result dtype/shape')
    vtype = value_type(node.value)
    if dt(vtype) is not memory.elem or (shape(vtype) and not same(shape(vtype), access_shape)):
        bad('value dtype/shape')
    if isinstance(node.value, T.TLit) and memory.elem.is_int:
        from .numeric import limits
        lo, hi = limits(memory.elem)
        if type(node.value.value) is not int or not lo <= node.value.value <= hi:
            bad('integer literal must fit the element dtype')
    if node.mask is not None:
        mtype = value_type(node.mask)
        if dt(mtype) is not D.bool_ or (shape(mtype) and not same(shape(mtype), access_shape)):
            bad('mask dtype/shape')
