"""Explicit host-side normalization; these helpers are not kernel intrinsics."""

import numpy as np

from .errors import TilaError


_INTEGER_TYPES = (int, np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)
_ACCEPTED = "Python int; numpy int8/int16/int32/int64/uint8/uint16/uint32/uint64"


def host_int(value) -> int:
    """Return an exact Python int from an explicitly supported integer scalar.

    Preserve the mathematical value, including arbitrary Python integers and
    uint64's full range. Shapes, grids, refinements and scalar dtypes remain
    subject to their normal use-site validation. No coercion protocol is used
    until the exact concrete type has passed the whitelist.
    """
    value_type = type(value)
    if not any(value_type is accepted for accepted in _INTEGER_TYPES):
        # Do not format the rejected value: even __repr__ may execute user code.
        module = type.__getattribute__(value_type, "__module__")
        name = type.__getattribute__(value_type, "__qualname__")
        raise TilaError(
            "TILA-TYPE-037", "host_int requires a supported exact integer type",
            details=[f"found host type: {module}.{name}", f"accepted: {_ACCEPTED}"],
            fixes=["pass a Python int or a listed NumPy integer scalar; "
                   "bool, floats, arrays and user subclasses are not accepted"],
        )
    return int(value)
