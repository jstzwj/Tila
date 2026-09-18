"""DimExpr：符号维表达式（docs/type-system.md §4）。

Shape 的每个维度是一个 Presburger 子集内的符号表达式：
    Const | Sym | + | − | *Const | //Const | ceildiv | %Const | min | max
除数/乘数必须是编译期常量，保证可判定性。
等价判定 = 规范化后结构相等（§4.3 的语法层）。
"""

from __future__ import annotations

from dataclasses import dataclass


class DimExpr:
    __slots__ = ()

    def __str__(self):
        return canon(self)

    def __add__(self, other):  return Add(self, _wrap(other))
    def __radd__(self, other): return Add(_wrap(other), self)
    def __sub__(self, other):  return Sub(self, _wrap(other))
    def __rsub__(self, other): return Sub(_wrap(other), self)
    def __mul__(self, other):  return Mul(self, _wrap(other))
    def __rmul__(self, other): return Mul(_wrap(other), self)
    def __floordiv__(self, other): return FloorDiv(self, _wrap(other))
    def __mod__(self, other):  return Mod(self, _wrap(other))


def _wrap(v):
    if isinstance(v, DimExpr):
        return v
    if isinstance(v, int):
        return Cst(v)
    raise TypeError(f"not a DimExpr operand: {v!r}")


@dataclass(frozen=True)
class Cst(DimExpr):
    value: int


@dataclass(frozen=True)
class Sym(DimExpr):
    name: str


@dataclass(frozen=True)
class Add(DimExpr):
    left: DimExpr
    right: DimExpr


@dataclass(frozen=True)
class Sub(DimExpr):
    left: DimExpr
    right: DimExpr


@dataclass(frozen=True)
class Mul(DimExpr):
    left: DimExpr
    right: DimExpr


@dataclass(frozen=True)
class FloorDiv(DimExpr):
    left: DimExpr
    right: DimExpr


@dataclass(frozen=True)
class CeilDiv(DimExpr):
    left: DimExpr
    right: DimExpr


@dataclass(frozen=True)
class Mod(DimExpr):
    left: DimExpr
    right: DimExpr


@dataclass(frozen=True)
class Min(DimExpr):
    left: DimExpr
    right: DimExpr


@dataclass(frozen=True)
class Max(DimExpr):
    left: DimExpr
    right: DimExpr


def Dim(name: str) -> Sym:
    """用户接口：N = ti.Dim("N")（运行期维符号）。"""
    if not name.isidentifier():
        raise TypeError(f"Dim name must be an identifier, got {name!r}")
    return Sym(name)


# ---------------------------------------------------------------------------
# 规范化与等价
# ---------------------------------------------------------------------------

def int_value(e: DimExpr):
    """全常量表达式的值；非常量返回 None。"""
    if isinstance(e, Cst):
        return e.value
    if isinstance(e, Sym):
        return None
    if isinstance(e, (Add, Sub, Mul)):
        a, b = int_value(e.left), int_value(e.right)  # type: ignore[attr-defined]
        if a is None or b is None:
            return None
        if isinstance(e, Add):
            return a + b
        if isinstance(e, Sub):
            return a - b
        return a * b
    if isinstance(e, (FloorDiv, CeilDiv, Mod)):
        a, b = int_value(e.left), int_value(e.right)  # type: ignore[attr-defined]
        if a is None or b is None or b == 0:
            return None
        if isinstance(e, FloorDiv):
            return a // b
        if isinstance(e, CeilDiv):
            return -((-a) // b)
        return a % b
    if isinstance(e, (Min, Max)):
        a, b = int_value(e.left), int_value(e.right)  # type: ignore[attr-defined]
        if a is None or b is None:
            return None
        return min(a, b) if isinstance(e, Min) else max(a, b)
    return None


def _term_key(e: DimExpr) -> str | None:
    """线性项的 key（Sym 或 const·Sym）；非线性/不透明项返回完整 canon。"""
    if isinstance(e, Sym):
        return e.name
    if isinstance(e, Mul):
        lc, rc = int_value(e.left), int_value(e.right)  # type: ignore[attr-defined]
        if lc is not None and isinstance(e.right, Sym):  # type: ignore[attr-defined]
            return f"{lc}*{e.right.name}" if lc != 1 else e.right.name  # type: ignore[attr-defined]
        if rc is not None and isinstance(e.left, Sym):
            return f"{rc}*{e.left.name}" if rc != 1 else e.left.name
    return None


def _atom_key(e: DimExpr) -> str:
    """不透明原子的规范键：只递归子节点（保证终止）；交换律算子排序子键；
    线性子节点（+ − ×）用 canon 规范化（折叠常量算术）。"""
    if isinstance(e, Sym):
        return e.name
    if isinstance(e, Cst):
        return str(e.value)
    if isinstance(e, (Add, Sub, Mul)):
        lk, rk = canon(e.left), canon(e.right)   # 线性化 + 常量折叠
        if isinstance(e, Sub) and rk == "0":
            return lk
        if isinstance(e, Add) and rk == "0":
            return lk
        if isinstance(e, Mul) and lk == "1":
            return rk
        if isinstance(e, Mul) and rk == "1":
            return lk
    else:
        lk, rk = _atom_key(e.left), _atom_key(e.right)  # type: ignore[attr-defined]
    tag = {Add: "+", Sub: "-", Mul: "*", FloorDiv: "//", CeilDiv: "ceildiv",
           Mod: "%", Min: "min", Max: "max"}[type(e)]
    if type(e) in (Add, Mul, Min, Max) and rk < lk:
        lk, rk = rk, lk
    return f"({lk} {tag} {rk})"


def _collect(e: DimExpr) -> tuple[dict[str, int], int]:
    """把表达式收进 ({term_key: coef}, const)。不透明原子以 atom key 为 key。"""
    if isinstance(e, Cst):
        return {}, e.value
    if isinstance(e, Sym):
        return {e.name: 1}, 0
    if isinstance(e, (Add, Sub)):
        lt, lc = _collect(e.left)
        rt, rc = _collect(e.right)
        if isinstance(e, Sub):
            rt = {k: -c for k, c in rt.items()}
            rc = -rc
        merged = dict(lt)
        for k, c in rt.items():
            merged[k] = merged.get(k, 0) + c
        merged = {k: c for k, c in merged.items() if c != 0}
        return merged, lc + rc
    if isinstance(e, Mul):
        lc = int_value(e.left)
        if lc is not None:
            rt, rc = _collect(e.right)
            if rc == 0 and rt:
                return {k: lc * c for k, c in rt.items()}, 0
        rc2 = int_value(e.right)
        if rc2 is not None:
            lt, lc2 = _collect(e.left)
            if lc2 == 0 and lt:
                return {k: rc2 * c for k, c in lt.items()}, 0
    key = _term_key(e)
    if key is not None:
        return {key: 1}, 0
    return {_atom_key(e): 1}, 0


def canon(e: DimExpr) -> str:
    """规范形式字符串：常量折叠 + 线性项排序。等价 ⇔ canon 相等。"""
    terms, const = _collect(e)
    parts = []
    for k in sorted(terms):
        c = terms[k]
        if c == 0:
            continue
        parts.append(k if c == 1 else (f"{c}*{k}" if c > 0 else f"{-c}*{k}"))
    if const != 0 or not parts:
        parts.append(str(const))
    return " + ".join(parts)


def equal(a: DimExpr, b: DimExpr) -> bool:
    if a == b:
        return True
    return canon(a) == canon(b)


def eval_num(e: DimExpr, env: dict[str, int]) -> int:
    """数值求值（launcher 侧；符号必须全部有值）。"""
    if isinstance(e, Cst):
        return e.value
    if isinstance(e, Sym):
        if e.name not in env:
            raise KeyError(f"no numeric binding for symbol {e.name}")
        return env[e.name]
    if isinstance(e, Add):
        return eval_num(e.left, env) + eval_num(e.right, env)
    if isinstance(e, Sub):
        return eval_num(e.left, env) - eval_num(e.right, env)
    if isinstance(e, Mul):
        return eval_num(e.left, env) * eval_num(e.right, env)
    if isinstance(e, FloorDiv):
        return eval_num(e.left, env) // eval_num(e.right, env)
    if isinstance(e, CeilDiv):
        a, b = eval_num(e.left, env), eval_num(e.right, env)
        return -((-a) // b)
    if isinstance(e, Mod):
        return eval_num(e.left, env) % eval_num(e.right, env)
    if isinstance(e, Min):
        return min(eval_num(e.left, env), eval_num(e.right, env))
    if isinstance(e, Max):
        return max(eval_num(e.left, env), eval_num(e.right, env))
    raise TypeError(f"cannot evaluate {e!r}")


def free_syms(e: DimExpr) -> set[str]:
    if isinstance(e, Cst):
        return set()
    if isinstance(e, Sym):
        return {e.name}
    if isinstance(e, (Add, Sub, Mul, FloorDiv, CeilDiv, Mod, Min, Max)):
        return free_syms(e.left) | free_syms(e.right)  # type: ignore[attr-defined]
    return set()


def ceildiv(a, b):
    return CeilDiv(_wrap(a), _wrap(b))


def rewrite_syms(e: DimExpr, mapping: dict[str, DimExpr]) -> DimExpr:
    """符号替换（特化期把 Const 符号代入数值）。"""
    if isinstance(e, Cst):
        return e
    if isinstance(e, Sym):
        return mapping.get(e.name, e)
    if isinstance(e, Add):
        return rewrite_syms(e.left, mapping) + rewrite_syms(e.right, mapping)
    if isinstance(e, Sub):
        return rewrite_syms(e.left, mapping) - rewrite_syms(e.right, mapping)
    if isinstance(e, Mul):
        return rewrite_syms(e.left, mapping) * rewrite_syms(e.right, mapping)
    if isinstance(e, FloorDiv):
        return FloorDiv(rewrite_syms(e.left, mapping), rewrite_syms(e.right, mapping))
    if isinstance(e, CeilDiv):
        return CeilDiv(rewrite_syms(e.left, mapping), rewrite_syms(e.right, mapping))
    if isinstance(e, Mod):
        return Mod(rewrite_syms(e.left, mapping), rewrite_syms(e.right, mapping))
    if isinstance(e, Min):
        return Min(rewrite_syms(e.left, mapping), rewrite_syms(e.right, mapping))
    if isinstance(e, Max):
        return Max(rewrite_syms(e.left, mapping), rewrite_syms(e.right, mapping))
    return e
