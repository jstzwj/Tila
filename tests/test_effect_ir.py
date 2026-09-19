"""Instruction ownership and definition-point integrity, not race proofs."""
from dataclasses import replace, FrozenInstanceError

import numpy as np
import pytest
import tila as ti
from tila import tir as T, types as TY, dtypes as D
from tila.effect_ir import OPERANDS, bind_effects, verify_effects
from tila.verifier import verify, NODES
from tila.errors import TilaError


@ti.jit
def nested(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite]):
    a = ti.load(x, 0)
    b = a + a
    ti.store(x, 1, ti.load(x, 2) + ti.load(x, 3))
    ti.unsafe_store(x, 4, ti.unsafe_load(x, 5))


def fresh():
    return ti.jit(nested.fn).tk


def test_all_sites_nested_unsafe_and_value_reuse():
    accesses = verify_effects(nested.tk)
    assert [a.effect.kind for a in accesses] == ['Read', 'Read', 'Read', 'Write', 'Read', 'Write']
    assert len({a.effect.site_id for a in accesses}) == 6
    assert accesses[1].effect.location == accesses[2].effect.location
    assert all(a.effect.element_dtype is D.i32 for a in accesses)
    assert all(a.effect.region_id == nested.tk.buffers[0].region_id for a in accesses)
    assert sum(a.unsafe for a in accesses) == 2
    x = np.arange(8, dtype=np.int32)
    nested[(1,)](x)
    assert x[1] == 5 and x[4] == 5
    with pytest.raises(FrozenInstanceError):
        accesses[0].effect.kind = 'Write'


@pytest.mark.parametrize('field,value', [
    ('kind', 'Write'), ('site_id', 'duplicate'),
    ('region_id', TY.ParamRegion(0, 'wrong')), ('element_dtype', D.f32),
    ('address_space', TY.SHARED), ('location', T.EffectLocation(900)),
])
def test_corrupted_effect_rejected(field, value):
    kernel = fresh()
    access = verify_effects(kernel)[0]
    access.effect = replace(access.effect, **{field: value})
    with pytest.raises(TilaError, match='effect metadata'):
        verify(kernel)


def test_missing_duplicate_and_shared_memory_rejected():
    for mutation in ('missing', 'duplicate', 'shared'):
        kernel = fresh()
        accesses = verify_effects(kernel)
        if mutation == 'missing':
            accesses[0].effect = None
        elif mutation == 'duplicate':
            accesses[1].effect = accesses[0].effect
        else:
            expr = kernel.body[2].value
            expr.right = expr.left
        with pytest.raises(TilaError):
            verify(kernel)


def test_definition_rebinding_and_old_type_table_ignored():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite]):
        i = 0
        a = ti.load(x, i)
        i = 1
        ti.store(x, i, a)
    first, second = verify_effects(kernel.tk)
    old, new = first.coords[0].definition, second.coords[0].definition
    assert old.site != new.site and old.name == new.name == 'i'
    kernel.tk.types['i'] = TY.ScalarT(D.f64)
    verify_effects(kernel.tk)
    second.coords[0].definition = old
    with pytest.raises(TilaError, match='definition metadata'):
        verify(kernel.tk)


def test_pointer_source_and_branch_loop_references():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.ReadWrite], flag: ti.bool):
        p = x.ptr
        step = ti.program_id(0)
        if flag:
            p = p + step
        else:
            p = p + step
        for j in ti.range(0, 2):
            ti.unsafe_store(p, ti.unsafe_load(p))
            p = p + step
    accesses = verify_effects(kernel.tk)
    assert len(accesses) == 2
    assert all(a.effect.region_id == kernel.tk.buffers[0].region_id for a in accesses)
    assert all(a.ptr.definition.kind == 'loop' for a in accesses)
    assert accesses[0].ptr.definition == accesses[1].ptr.definition


def test_false_outer_mask_and_where_keep_operand_reads():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly]):
        a = ti.load(x, 0, mask=False, other=ti.load(x, 1))
        b = ti.where(True, ti.load(x, 2), ti.load(x, 3))
    assert len(verify_effects(kernel.tk)) == 4


def test_exhaustive_coverage_and_unknown_rejected():
    assert set(OPERANDS) == NODES
    class Future(T.TExpr):
        pass
    kernel = fresh()
    kernel.body[0].value = Future(TY.ScalarT(D.i32))
    with pytest.raises(TilaError, match='Future'):
        bind_effects(kernel)
    with pytest.raises(TilaError, match='Future'):
        verify(kernel)


def test_deterministic_and_verification_does_not_repair():
    original = verify_effects(nested.tk)
    copied = fresh()
    assert [a.effect for a in verify_effects(copied)] == [a.effect for a in original]
    copied.body[0].value.effect = None
    with pytest.raises(TilaError):
        verify(copied)
    assert copied.body[0].value.effect is None


def test_regions_do_not_merge_equal_extents():
    @ti.jit
    def kernel(a: ti.ReadPtr[ti.i32, 8], b: ti.WritePtr[ti.i32, 8]):
        ti.store(b, ti.load(a))
    read, write = verify_effects(kernel.tk)
    assert read.effect.region_id != write.effect.region_id
    assert kernel.tk.aliases[0].relation is TY.AliasRelation.MAY_ALIAS


def test_early_return_selects_surviving_definition():
    @ti.jit
    def kernel(x: ti.Buffer[ti.i32, (8,), ti.ReadOnly], flag: ti.bool):
        i = 0
        if flag:
            return
        else:
            i = 1
        a = ti.load(x, i)
    ref = verify_effects(kernel.tk)[0].coords[0].definition
    assert ref.kind == 'definition' and '/else/' in ref.site


def test_corrupt_load_type_and_missing_name_reference_rejected():
    kernel = fresh()
    kernel.body[0].value.vt = TY.ScalarT(D.f32)
    with pytest.raises(TilaError, match='dtype disagrees'):
        verify(kernel)
    kernel = fresh()
    kernel.body[1].value.left.definition = None
    with pytest.raises(TilaError, match='definition metadata'):
        verify(kernel)
