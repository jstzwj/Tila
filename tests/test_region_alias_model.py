"""ADR-002 / M1-02：RegionId、Extent、Effect 与 Alias 相互独立。"""

import numpy as np

import tila as ti
from tila import types as TY
from tila.facts import Obligation


N = ti.Dim("N")


def test_structured_region_variants_and_default_alias_lattice():
    p0 = TY.ParamRegion(0, "x")
    p1 = TY.ParamRegion(1, "y")
    i0 = TY.InternalRegion(0)
    i1 = TY.InternalRegion(1)

    assert TY.default_alias_relation(p0, p0) is TY.AliasRelation.MUST_ALIAS
    assert TY.default_alias_relation(p0, p1) is TY.AliasRelation.MAY_ALIAS
    assert TY.default_alias_relation(i0, i1) is TY.AliasRelation.NO_ALIAS
    assert TY.default_alias_relation(TY.UNKNOWN_REGION, p0) is \
        TY.AliasRelation.MAY_ALIAS


def test_equal_extents_do_not_merge_regions_and_effects_are_structured():
    @ti.jit
    def copy(src: ti.ReadPtr[ti.f32, N],
             dst: ti.WritePtr[ti.f32, N],
             BLOCK: ti.Const[int] = 8):
        offsets = ti.arange(0, BLOCK)
        mask = offsets < N
        values = ti.load(src + offsets, mask=mask)
        ti.store(dst + offsets, values, mask=mask)

    src_region = copy.tk.ptr_params[0].vtype.region_id
    dst_region = copy.tk.ptr_params[1].vtype.region_id
    assert src_region == TY.ParamRegion(0, "src")
    assert dst_region == TY.ParamRegion(1, "dst")
    assert src_region != dst_region
    assert all(p.vtype.extent == TY.LinearExtent(N)
               for p in copy.tk.ptr_params)
    assert [(e.op, e.region_id) for e in copy.tk.effects] == [
        ("Read", src_region), ("Write", dst_region),
    ]
    assert copy.tk.aliases[0].relation is TY.AliasRelation.MAY_ALIAS


def test_bounds_obligation_has_extent_but_no_region_identity_field():
    ob = Obligation("load", "ptr:p", None, None, N, [])
    assert ob.extent == N
    assert ob.source == "ptr:p"
    assert not hasattr(ob, "region")
    assert not hasattr(ob, "bound")


def test_buffer_ptr_and_derived_pointer_inherit_buffer_region():
    @ti.jit
    def derive(x: ti.Buffer[ti.f32, (8,), ti.ReadOnly]):
        p = x.ptr
        q = p + ti.arange(0, 8)

    region = derive.tk.buffers[0].region_id
    assert region == TY.BufferRegion(0, "x")
    assert derive.tk.types["p"].region_id == region
    assert derive.tk.types["q"].elem.region_id == region
    assert derive.tk.types["q"].elem.extent == derive.tk.types["p"].extent


def test_launch_alias_relation_upgrades_without_rewriting_region_ids():
    @ti.jit
    def pair(a: ti.ReadPtr[ti.f32, 4], b: ti.ReadPtr[ti.f32, 4]):
        pass

    left_region = pair.tk.ptr_params[0].vtype.region_id
    right_region = pair.tk.ptr_params[1].vtype.region_id
    assert pair.tk.aliases[0].relation is TY.AliasRelation.MAY_ALIAS

    shared = np.zeros(4, dtype=np.float32)
    pair[(1,)](shared, shared)
    same = pair.tk.runtime_aliases[0]
    assert same.relation is TY.AliasRelation.MUST_ALIAS
    assert (same.left, same.right) == (left_region, right_region)

    pair[(1,)](np.zeros(4, dtype=np.float32),
               np.zeros(4, dtype=np.float32))
    assert pair.tk.runtime_aliases[0].relation is TY.AliasRelation.NO_ALIAS
    assert pair.tk.ptr_params[0].vtype.region_id == left_region
    assert pair.tk.ptr_params[1].vtype.region_id == right_region


def test_overlapping_runtime_views_remain_may_alias():
    @ti.jit
    def pair(a: ti.ReadPtr[ti.f32, 4], b: ti.ReadPtr[ti.f32, 4]):
        pass

    base = np.zeros(6, dtype=np.float32)
    pair[(1,)](base[:4], base[2:6])
    assert pair.tk.runtime_aliases[0].relation is TY.AliasRelation.MAY_ALIAS


def test_dump_and_explain_show_regions_effects_aliases_and_extent_separately():
    @ti.jit
    def read(p: ti.ReadPtr[ti.f32, 8]):
        values = ti.load(p + ti.arange(0, 8))

    dump = read.tk.dump()
    explain = read.explain()
    assert "region=ParamRegion[0:p]" in dump
    assert "effect Read[ParamRegion[0:p]]" in dump
    assert "obligation load[ptr:p] flat: 0 <=" in dump
    assert "< 8" in dump
    assert "effects:" in explain and "Read[ParamRegion[0:p]]" in explain
    assert "aliases:" in explain
