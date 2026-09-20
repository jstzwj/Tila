"""Fail-closed TIR/backend boundary. Does not replace typing or safety proofs."""
import math
import keyword

from . import dtypes as D, tir as T, types as TY
from .dims import eval_num, free_syms
from .errors import Loc, TilaError
from .target import SUPPORTED


NODE_NAMES = frozenset((
    "TAtomicAdd", "TAtomicStmt",
    "TName", "TLit", "TAssign", "TStore", "TAssume", "TIf", "TStaticIf", "TFor", "TReturn",
    "TBin", "TUna", "TCast", "TConstant", "TArange", "TPid", "TNumPrograms", "TZeros",
    "TReshape", "TExpand", "TWhere", "TDot", "TReduce", "TLoad", "TBufPtr", "TPAdd"))
NODES = frozenset(getattr(T, name) for name in NODE_NAMES)
BIN_OPS = frozenset(("+", "-", "*", "/", "//", "%", "<<", ">>", "&", "|", "^",
                     "==", "!=", "<", "<=", ">", ">=", "and", "or"))
UNA_OPS = frozenset(("+", "-", "~", "not", "exp", "exp2", "any", "all"))


def fail(message, line=0):
    raise TilaError("TILA-TARGET-009", message, Loc(line),
                    fixes=["检查 TIR 与 target 支持范围；非法内部节点请提交最小复现"])


def verify(kernel, consts=None, capability=SUPPORTED):
    try:
        _verify(kernel, consts, capability)
    except (AttributeError, TypeError, ValueError, KeyError, ArithmeticError, RecursionError) as exc:
        fail(f"malformed TIR ({type(exc).__name__})")


def _verify(kernel, consts=None, capability=SUPPORTED):
    """Exact node kinds, operand structure and supported dtype/shape/ops.

    Symbolic tile dimensions defer until Const specialization. This validates
    backend admissibility, not SSA dominance, bounds or race freedom.
    """
    if type(kernel) is not T.TKernel:
        fail("expected TKernel")
    specialized = consts is not None
    fp8_supported = capability is not None and capability.fp8_storage
    consts = {} if consts is None else consts
    buffers = {p.name: p.vtype for p in kernel.buffers}
    pointers = {p.name: p.vtype for p in kernel.ptr_params}
    params = kernel.buffers + kernel.ptr_params + kernel.scalars + kernel.consts
    if len({p.name for p in params}) != len(params):
        fail("duplicate kernel parameter")
    for p in kernel.buffers + kernel.ptr_params:
        if p.vtype.elem not in D.ALL.values() or p.vtype.space is not TY.GLOBAL:
            fail("unsupported memory dtype or address space")
        if p.vtype.elem.kind == "float_storage" and not fp8_supported:
            fail("FP8 storage/cast is not validated on this target; use f16/bf16/f32")
    for p in kernel.scalars:
        if p.dtype not in D.ARITH_DTYPES + (D.bool_,):
            fail("unsupported scalar ABI dtype")

    def identifier(name, line=0):
        if type(name) is not str or not name.isidentifier() or keyword.iskeyword(name):
            fail("invalid TIR identifier", line)
    identifier(kernel.name)
    for p in params:
        identifier(p.name)

    def vt_of(operand):
        if isinstance(operand, T.TLit) and operand.dtype is not None:
            return TY.ScalarT(operand.dtype)
        if isinstance(operand, T.TExpr):
            return operand.vt
        if isinstance(operand, T.TName):
            return operand.definition.vtype if operand.definition is not None else None
        return None

    def value_type(vt, line):
        if type(vt) is TY.ScalarT:
            if vt.dtype not in D.ALL.values():
                fail("unknown scalar dtype", line)
            if vt.dtype.kind == "float_storage" and not fp8_supported:
                fail("FP8 storage/cast is not validated on this target; use f16/bf16/f32", line)
        elif type(vt) is TY.PtrT:
            if vt.elem not in D.ALL.values() or vt.space is not TY.GLOBAL:
                fail("unsupported pointer type", line)
        elif type(vt) in (TY.BlockT, TY.MaskT):
            if type(vt) is TY.BlockT:
                value_type(vt.elem, line)
            sizes = []
            for dim in vt.dims:
                if not free_syms(dim) <= consts.keys():
                    if specialized and capability is not None:
                        fail("unresolved target tile dimension", line)
                    continue
                size = eval_num(dim, consts)
                if capability is not None and (type(size) is not int or size <= 0 or size & (size - 1)):
                    fail("Triton tile dimensions must be positive powers of two", line)
                sizes.append(size)
            if capability is not None and math.prod(sizes) > capability.max_tile_elements:
                fail("tile exceeds target element limit", line)
        else:
            fail("unsupported TIR value type", line)

    active = set()
    def visit(node, line=0):
        if type(node) not in NODES:
            fail(f"unsupported TIR node {type(node).__name__}", line)
        if id(node) in active:
            fail("cyclic TIR operand graph", line)
        active.add(id(node))
        line = getattr(node, "line", line)
        if isinstance(node, (T.TName, T.TAssign)):
            identifier(node.name, line)
        if isinstance(node, T.TFor):
            identifier(node.var, line)
        if isinstance(node, T.TExpr):
            value_type(node.vt, line)
        if isinstance(node, T.TLit):
            if type(node.value) not in (int, float, bool) or (node.dtype is not None and node.dtype not in D.ALL.values()):
                fail("unsupported literal payload", line)
        if isinstance(node, T.TBin) and node.op not in BIN_OPS:
            fail("unsupported binary operation", line)
        if isinstance(node, T.TUna) and node.op not in UNA_OPS:
            fail("unsupported unary operation", line)
        if isinstance(node, (T.TPid, T.TNumPrograms)):
            if type(node.axis) is not int or node.axis not in (0, 1, 2):
                fail("program axis must be 0, 1 or 2", line)
        if isinstance(node, T.TReduce):
            if node.op not in ("sum", "max") or node.input_dtype not in D.ARITH_DTYPES:
                fail("unsupported reduction", line)
            if type(node.axis) is not int or node.axis < 0:
                fail("invalid reduction axis", line)
            operand_type = vt_of(node.operand)
            if isinstance(operand_type, TY.BlockT) and node.axis >= len(operand_type.dims):
                fail("reduction axis exceeds operand rank", line)
        if isinstance(node, T.TCast):
            if node.dtype not in D.ALL.values():
                fail("unknown cast dtype", line)
            vt = node.vt.elem if isinstance(node.vt, TY.BlockT) else node.vt
            if not isinstance(vt, TY.ScalarT) or vt.dtype is not node.dtype:
                fail("cast result type disagrees with target dtype", line)
        if isinstance(node, T.TDot):
            dt = node.vt.elem.dtype if isinstance(node.vt, TY.BlockT) else None
            if dt is None or dt.name not in (capability or SUPPORTED).dot_outputs:
                fail("unsupported dot output dtype", line)
            for operand in (node.a, node.b):
                vt = vt_of(operand)
                if vt is not None and (not isinstance(vt, TY.BlockT) or len(vt.dims) != 2 or vt.elem.dtype.name not in (capability or SUPPORTED).dot_inputs):
                    fail("unsupported dot input dtype or rank", line)
            left = vt_of(node.a)
            if capability is not None and isinstance(left, TY.BlockT) and free_syms(left.dims[1]) <= consts.keys():
                if eval_num(left.dims[1], consts) < capability.min_dot_k:
                    fail("dot K is below target minimum 16", line)
        if isinstance(node, T.TConstant):
            if node.dtype not in D.FLOAT_DTYPES or type(node.bits) is not int or not 0 <= node.bits < 1 << node.dtype.bits:
                fail("invalid typed constant payload", line)
        if isinstance(node, (T.TLoad, T.TStore, T.TAtomicAdd)):
            if (node.buffer is None) == (node.ptr is None):
                fail("memory node must use exactly one Buffer or Ptr", line)
            if node.buffer is not None:
                if node.buffer not in buffers or len(node.coords) != len(buffers[node.buffer].dims):
                    fail("invalid buffer or coordinate rank", line)
        if isinstance(node, T.TAtomicStmt) and type(node.value) is not T.TAtomicAdd:
            fail('atomic statement requires atomic_add', line)
        if isinstance(node, T.TAtomicAdd):
            from .atomic import validate_node
            memory = buffers[node.buffer] if node.buffer is not None else vt_of(node.ptr)
            if isinstance(memory, TY.BlockT):
                memory = memory.elem
            validate_node(node, memory, vt_of, consts if specialized else None)
            if capability is not None and (
                    memory.elem.name not in capability.atomic_add_dtypes or
                    node.order.value not in capability.atomic_orders or
                    node.scope.value not in capability.atomic_scopes):
                raise TilaError('TILA-TARGET-012', 'atomic_add configuration is not supported by target', Loc(line))
        # Walk only semantic child fields; metadata types/dimensions are handled
        # separately. Require children even when raw objects replaced operands.
        required = {
            T.TAtomicAdd: ('value',), T.TAtomicStmt: ('value',),
            T.TAssign: ("value",), T.TBin: ("left", "right"), T.TUna: ("operand",),
            T.TCast: ("operand",), T.TArange: ("end",), T.TReshape: ("operand",),
            T.TExpand: ("operand",), T.TWhere: ("cond", "a", "b"), T.TDot: ("a", "b"),
            T.TReduce: ("operand",), T.TPAdd: ("ptr", "offset"), T.TStore: ("value",),
            T.TAssume: ("pred",), T.TIf: ("cond",), T.TStaticIf: ("cond",),
            T.TFor: ("end", "step"),
        }.get(type(node), ())
        optional = ("mask", "other", "ptr", "acc", "start")
        for name in required:
            child = getattr(node, name)
            if not isinstance(child, T.TOperand):
                fail(f"{name} requires a TIR operand", line)
            visit(child, line)
        for name in optional:
            child = getattr(node, name, None)
            if name not in required and child is not None and not (name == "start" and isinstance(node, T.TArange)):
                if not isinstance(child, T.TOperand):
                    fail(f"{name} requires a TIR operand", line)
                visit(child, line)
        for name in ("coords", "shape", "body", "then_body", "else_body"):
            for child in getattr(node, name, ()):
                if name in ("body", "then_body", "else_body") and not isinstance(child, T.TStmt):
                    fail("control-flow body requires statements", line)
                if name in ("coords", "shape") and not isinstance(child, T.TOperand):
                    fail(f"{name} requires TIR operands", line)
                visit(child, line)
        active.remove(id(node))

    for statement in kernel.body:
        if not isinstance(statement, T.TStmt):
            fail("kernel body requires statements")
        visit(statement)
    from .effect_ir import verify_effects
    verify_effects(kernel)
