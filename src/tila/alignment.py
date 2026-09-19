"""Launch-bound byte alignment evidence; never inferred from user assumptions."""
from dataclasses import dataclass
from math import gcd

from .predicates import Origin, CHECKED


@dataclass(frozen=True)
class AlignmentFact:
    parameter: str
    bytes: int
    element_bytes: int
    stride: int

    @property
    def origin(self):
        return Origin(CHECKED, detail=(f"last validated binding: {self.parameter}; "
            f"view data_ptr % {self.bytes} == 0; element_bytes={self.element_bytes}; "
            f"stride={self.stride}; base address only"))


def collect(kernel, tensors, tensor_info):
    """Only explicit declarations on contiguous 1D views yield hints.

    Uses the view's actual pointer (including storage offset). Incidental
    allocator alignment and TKernel.runtime_alignments are not evidence.
    """
    facts = []
    for param in kernel.buffers + kernel.ptr_params:
        alignment = param.vtype.aligned
        if alignment is None or alignment <= 1:
            continue
        dt, shape, strides, pointer = tensor_info(tensors[param.name])
        if dt is not param.vtype.elem or len(shape) != 1 or strides != (1,):
            continue
        if pointer == 0 or pointer % alignment:
            continue
        facts.append(AlignmentFact(param.name, alignment, max(1, dt.bits // 8), 1))
    return tuple(facts)


def derived_byte_alignment(base_bytes, element_bytes, offset_multiple, stride=1):
    """For address base + sizeof(T)*stride*i, with i divisible by M.

    This rule documents units; it does not license emitting a hint for an
    arbitrary offset without its own proof of divisibility.
    """
    return gcd(base_bytes, abs(element_bytes * stride * offset_multiple))
