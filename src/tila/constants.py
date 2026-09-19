"""ADR-015 direct round-to-nearest, ties-to-even from exact source values."""
import math
import struct

from .errors import TilaError

FORMATS = {"f16": (5, 10), "bf16": (8, 7), "f32": (8, 23), "f64": (11, 52)}


def rounded_bits(value, dtype, loc=None):
    if dtype.name not in FORMATS or (type(value) is not int and type(value) is not float):
        raise TilaError("TILA-CONST-011", "constant requires an exact int/float source and f16/bf16/f32/f64 target", loc)
    if type(value) is float and not math.isfinite(value):
        raise TilaError("TILA-NUM-002", "constant source must be finite", loc)
    eb, fb = FORMATS[dtype.name]
    sign = int(value < 0 or type(value) is float and math.copysign(1, value) < 0)
    sign_bits = sign << (eb + fb)
    n, d = value.as_integer_ratio() if type(value) is float else (value, 1)
    n = abs(n)
    if not n:
        return sign_bits
    bias = (1 << (eb - 1)) - 1
    emin, emax = 1 - bias, bias
    e = n.bit_length() - d.bit_length()
    below = n < d << e if e >= 0 else n << -e < d
    if below:
        e -= 1
    if e > emax:
        raise TilaError("TILA-NUM-002", f"constant overflows {dtype.name}", loc)
    quantum = max(e, emin) - fb
    numerator, denominator = (n, d << quantum) if quantum >= 0 else (n << -quantum, d)
    q, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and q & 1):
        q += 1
    if q < 1 << fb:
        return sign_bits | q
    exponent = q.bit_length() - 1 + quantum
    if exponent > emax:
        raise TilaError("TILA-NUM-002", f"constant rounds to infinity in {dtype.name}", loc)
    significand = q >> (q.bit_length() - 1 - fb)
    return sign_bits | ((exponent + bias) << fb) | (significand - (1 << fb))


def float_value(dtype, bits):
    """Decode a finite target value exactly into binary64 for diagnostics."""
    if dtype.name == "bf16":
        return struct.unpack(">f", (bits << 16).to_bytes(4, "big"))[0]
    fmt, size = {"f16": (">e", 2), "f32": (">f", 4), "f64": (">d", 8)}[dtype.name]
    return struct.unpack(fmt, bits.to_bytes(size, "big"))[0]
