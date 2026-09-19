"""ADR-007 integer semantics and conservative, per-launch arithmetic gates.

This is a bounded interval validator, not the M2 SMT solver. It never obtains
numeric facts by reading tensor contents or by executing a kernel.
"""
from dataclasses import fields, is_dataclass
import math
import operator

import numpy as np

from . import dtypes as D, tir as T, types as TY
from .errors import Loc, TilaError


def dtype(vt):
    if isinstance(vt, TY.BlockT):
        vt = vt.elem
    return vt.dtype if isinstance(vt, TY.ScalarT) else None


def limits(dt):
    return D._INT_RANGE[dt]


def wrap(value, dt):
    """Integer cast/add/sub/mul use low bits, independent of NumPy promotion."""
    modulus = 1 << dt.bits
    a = np.asarray(value).astype(object)
    bits = a % modulus
    if dt.kind == "int":
        bits = np.where(bits >= modulus // 2, bits - modulus, bits)
    result = np.asarray(bits, dtype=dt.np_dtype)
    return result[()] if result.ndim == 0 else result


def integer_binary(op, left, right, dt):
    a = np.asarray(wrap(left, dt)).astype(object)
    b = np.asarray(wrap(right, dt)).astype(object)
    if op in ("//", "%") and np.any(b == 0):
        raise TilaError("TILA-NUM-001", "integer division by zero")
    if op in ("<<", ">>") and np.any((b < 0) | (b >= dt.bits)):
        raise TilaError("TILA-NUM-001", "shift count is outside the dtype width")
    fn = {"+": operator.add, "-": operator.sub, "*": operator.mul,
          "//": operator.floordiv, "%": operator.mod,
          "<<": operator.lshift, ">>": operator.rshift,
          "&": operator.and_, "|": operator.or_, "^": operator.xor}[op]
    return wrap(fn(a, b), dt)


class Validator:
    def __init__(self, tk, consts, scalars=None, grid=None):
        self.tk, self.grid = tk, grid
        self.launch = scalars is not None
        self.env = {k: (v, v) for k, v in consts.items()}
        self.pending = []
        self.line = 0
        for param in tk.scalars:
            if scalars is not None and param.name in scalars:
                v = scalars[param.name]
                self.env[param.name] = (self.float_interval((v, v), param.dtype)
                                        if param.dtype.is_float else (v, v))
            else:
                self.env[param.name] = None

    def require(self, valid, message, *, definite=False):
        if valid:
            return
        if not self.launch and not definite:
            self.pending.append(message)
            return
        raise TilaError("TILA-NUM-001", message, Loc(self.line),
                        fixes=["use a wider index dtype or constrain arithmetic inputs; "
                               "data-dependent domains need an explicit safe construction"])

    def fits(self, iv, dt):
        return iv is not None and limits(dt)[0] <= iv[0] <= iv[1] <= limits(dt)[1]

    def expr(self, x):
        if x is None:
            return None
        if isinstance(x, T.TLit):
            if x.dtype and x.dtype.is_float:
                return self.float_interval((x.value, x.value), x.dtype)
            return (x.value, x.value)
        if isinstance(x, T.TName):
            return self.env.get(x.name)
        dt = dtype(getattr(x, "vt", None))
        if isinstance(x, T.TPid):
            return (0, max(0, self.grid[x.axis] - 1)) if self.grid else None
        if isinstance(x, T.TNumPrograms):
            return (self.grid[x.axis],) * 2 if self.grid else None
        if isinstance(x, T.TArange):
            end = self.expr(x.end)
            valid = (end is not None and end[0] == end[1] and
                     -(1 << 31) <= x.start < end[0] <= (1 << 31) - 1)
            self.require(valid, "arange bounds must fit i32 and start < end", definite=end is not None)
            return (x.start, end[1] - 1) if valid else None
        if isinstance(x, T.TBin):
            op = x.op
            a = self.expr(x.left)
            if op in ("and", "or") and a is not None and a[0] == a[1]:
                if (op == "and" and not a[0]) or (op == "or" and a[0]):
                    return (bool(a[0]),) * 2
            b = self.expr(x.right)
            operand_dt = x.operand_dtype or dt
            if operand_dt and operand_dt.is_int and not x.staged:
                for iv in (a, b):
                    self.require(self.fits(iv, operand_dt),
                                 f"integer operands must fit {operand_dt.name} before arithmetic",
                                 definite=iv is not None)
            if op in ("//", "%"):
                self.require(b is not None and (b[1] < 0 or b[0] > 0),
                             "cannot prove integer divisor is nonzero", definite=b is not None)
            if op in ("<<", ">>"):
                self.require(b is not None and 0 <= b[0] <= b[1] < dt.bits,
                             "cannot prove 0 <= shift count < dtype width", definite=b is not None)
            iv = None
            if a is not None and b is not None:
                if op == "+": iv = (a[0] + b[0], a[1] + b[1])
                elif op == "-": iv = (a[0] - b[1], a[1] - b[0])
                elif op == "*":
                    products = [u * v for u in a for v in b]
                    iv = (min(products), max(products))
                elif op == "//" and (b[1] < 0 or b[0] > 0):
                    values = [u // v for u in a for v in b]
                    iv = (min(values), max(values))
                elif op == "%" and (b[1] < 0 or b[0] > 0):
                    iv = (0, b[1] - 1) if b[0] > 0 else (b[0] + 1, 0)
                elif op in ("<", "<=", ">", ">=", "==", "!="):
                    if a[0] == a[1] and b[0] == b[1]:
                        fn = {"<": operator.lt, "<=": operator.le,
                              ">": operator.gt, ">=": operator.ge,
                              "==": operator.eq, "!=": operator.ne}[op]
                        iv = (fn(a[0], b[0]),) * 2
                elif op == "&" and b[0] >= 0:
                    iv = (0, b[1])
                elif op in ("and", "or") and a[0] == a[1] and b[0] == b[1]:
                    iv = ((bool(a[0]) and bool(b[0])) if op == "and"
                          else (bool(a[0]) or bool(b[0])),) * 2
            if x.checked_index:
                self.require(self.fits(iv, dt),
                             f"cannot prove intermediate index '{op}' does not overflow {dt.name}",
                             definite=iv is not None)
                if iv is None:
                    return None
            if dt and dt.is_int and not x.staged:
                if iv is not None and iv[0] == iv[1]:
                    return (int(wrap(iv[0], dt)),) * 2
                return iv if self.fits(iv, dt) else limits(dt)
            if iv is not None and dt and dt.is_float:
                return self.float_interval(iv, dt)
            return iv
        if isinstance(x, T.TUna):
            a = self.expr(x.operand)
            iv = (-a[1], -a[0]) if a is not None and x.op == "-" else None
            if x.checked_index:
                self.require(self.fits(iv, dt), "index negation may overflow", definite=iv is not None)
                if iv is None:
                    return None
            if dt and dt.is_int and not x.staged:
                return iv if self.fits(iv, dt) else limits(dt)
            return iv
        if isinstance(x, T.TCast):
            a = self.expr(x.operand)
            if dt is D.bool_:
                return (bool(a[0]),) * 2 if a is not None and a[0] == a[1] else (0, 1)
            if dt.is_int:
                source_dt = self.operand_dtype(x.operand)
                if source_dt and source_dt.is_float:
                    valid = (a is not None and all(math.isfinite(v) for v in a)
                             and limits(dt)[0] <= math.trunc(a[0]) <=
                             math.trunc(a[1]) <= limits(dt)[1])
                    self.require(valid, "float-to-integer cast needs finite, representable values",
                                 definite=a is not None)
                    return (math.trunc(a[0]), math.trunc(a[1])) if valid else limits(dt)
                if a is not None and a[0] == a[1]:
                    return (int(wrap(a[0], dt)),) * 2
                return a if self.fits(a, dt) else limits(dt)
            return self.float_interval(a, dt) if a is not None and dt.is_float else a
        if isinstance(x, (T.TExpand, T.TReshape)):
            return self.expr(x.operand)
        if isinstance(x, T.TWhere):
            self.expr(x.cond)
            a, b = self.expr(x.a), self.expr(x.b)
            return (min(a[0], b[0]), max(a[1], b[1])) if a and b else None
        if isinstance(x, T.TStore) and x.buffer is not None:
            element = next(b.vtype.elem for b in self.tk.buffers if b.name == x.buffer)
            value = self.expr(x.value)
            if element.is_int:
                self.require(self.fits(value, element),
                             f"stored integer must fit {element.name}; use an explicit cast",
                             definite=value is not None)
        # Visit memory coordinates, masks, reduction operands, etc. No tensor reads.
        if is_dataclass(x):
            for f in fields(x):
                value = getattr(x, f.name)
                if isinstance(value, T.TOperand): self.expr(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, T.TOperand): self.expr(item)
        return limits(dt) if dt and dt.is_int else None

    @staticmethod
    def float_interval(iv, dt):
        np_dt = dt.np_dtype
        if np_dt is None:
            if dt is not D.bf16:
                return None
            import ml_dtypes
            np_dt = ml_dtypes.bfloat16
        with np.errstate(over="ignore", invalid="ignore"):
            result = tuple(float(np.asarray(v, dtype=np_dt)) for v in iv)
        return result if all(math.isfinite(v) for v in result) else None

    def operand_dtype(self, x):
        if isinstance(x, T.TLit): return x.dtype
        if isinstance(x, T.TName): return self.types.get(x.name)
        return dtype(getattr(x, "vt", None))

    def stmts(self, body):
        for s in body:
            self.line = getattr(s, "line", self.line)
            if isinstance(s, T.TAssign):
                self.env[s.name] = self.expr(s.value)
                self.types[s.name] = self.operand_dtype(s.value)
            elif isinstance(s, T.TFor):
                start = self.expr(s.start) if s.start is not None else (0, 0)
                end, step = self.expr(s.end), self.expr(s.step)
                valid = (start is not None and end is not None and step is not None
                         and self.fits(start, D.i32) and self.fits(end, D.i32)
                         and 0 < step[0] <= step[1] <= (1 << 31) - 1)
                self.require(valid, "range start/end/positive step must fit i32")
                # i64 induction in lowering prevents the final increment overflowing i32.
                before = dict(self.env)
                # Loop-carried values may change; do not use their first iteration value.
                for name in assigned_names(s.body) & before.keys():
                    dt = self.types.get(name)
                    self.env[name] = limits(dt) if dt and dt.is_int else None
                self.env[s.var] = (start[0], max(start[0], end[1] - 1)) if valid else None
                self.types[s.var] = D.i32
                self.stmts(s.body)
                for name in assigned_names(s.body):
                    if name in before:
                        dt = self.types.get(name)
                        self.env[name] = limits(dt) if dt and dt.is_int else None
                self.env.pop(s.var, None)
            elif isinstance(s, (T.TIf, T.TStaticIf)):
                cond = self.expr(s.cond)
                if cond is not None and cond[0] == cond[1]:
                    self.stmts(s.then_body if cond[0] else s.else_body)
                else:
                    before = dict(self.env)
                    self.stmts(s.then_body)
                    left = dict(self.env)
                    self.env = dict(before)
                    self.stmts(s.else_body)
                    self.env = {k: (min(left[k][0], v[0]), max(left[k][1], v[1]))
                                if v is not None and left.get(k) is not None else None
                                for k, v in self.env.items() if k in left}
            elif isinstance(s, T.TReturn):
                return
            else:
                self.expr(s)

    def run(self):
        self.types = {s.name: s.dtype for s in self.tk.scalars}
        self.stmts(self.tk.body)
        return self.pending


def assigned_names(body):
    names = set()
    for s in body:
        if isinstance(s, T.TAssign): names.add(s.name)
        elif isinstance(s, (T.TIf, T.TStaticIf)):
            names |= assigned_names(s.then_body) | assigned_names(s.else_body)
        elif isinstance(s, T.TFor): names |= assigned_names(s.body)
    return names


def validate(tk, consts, scalars=None, grid=None):
    return Validator(tk, consts, scalars, grid).run()
