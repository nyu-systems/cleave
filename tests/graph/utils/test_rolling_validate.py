"""What the codegen refuses about stabilized exps and rolling state (CPU).

One hand-built rolling tier (``tests/graph/_rolling_tier.py``) that passes, and one mutation
of it per rule. Each mutation would otherwise emit a kernel that compiles and computes
something else: a plain exp where a max was asked for, a snapshot taken after the update,
a max that restarts every tile.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.codegen import TilelangCodeGen
from cleave.ir.graph import Iterator
from cleave.ir.graph_utils import edit_graph
from cleave.ir.ops import Copy, Exp, Stabilize, wire_input

from tests.graph._rolling_tier import ROWS, STEP, build_rolling_tier, schedule_rolling_tier


def _tier(**kw):
    t = build_rolling_tier(**kw)
    schedule_rolling_tier(t)
    return t


def _refused(t, match):
    with pytest.raises(AssertionError, match=match):
        TilelangCodeGen().gen(t.main)


def _row(extent=ROWS):
    return Iterator(lambda i: ("0",), shape=(extent,), tile_size=(extent,))


def test_the_unmodified_tier_is_accepted():
    assert "ukernel_rescale_1dims_rr" in TilelangCodeGen().gen(_tier().main)


# V1 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("where", ["loop", "main"])
def test_an_exp_still_marked_for_stabilization_is_refused(where):
    t = _tier()
    record = Stabilize(axes=(-1,), scaled=(("rowsum_0", 1),))
    if where == "loop":
        t.p.stabilize = record
    else:  # an in-tile softmax that nothing rewrote
        marked = Exp((64, STEP), name="exp_1", n_batch_dims=0)
        marked.stabilize = record
        with edit_graph(t.main):
            wire_input(t.div, marked, 0)
            wire_input(marked, t.out, 0)
            t.main.nodes.insert(t.main.nodes.index(t.out), marked)
            t.main.tmap[marked] = t.main.tmap[t.div]
            t.main.tinfo_map[marked] = ("t_exp_1", "r")
    _refused(t, "marked for stabilization but was not rewritten")


# V2 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["lag_nodes", "accum_scale"])
def test_rolling_state_is_refused_in_the_main_graph(field):
    t = _tier()
    setattr(t.main, field, [t.div] if field == "lag_nodes" else {t.div: t.div})
    _refused(t, "state carried across loop iterations")


def test_a_lag_node_must_be_a_node_of_the_loop():
    t = _tier()
    t.inner.lag_nodes.append(t.div)
    _refused(t, "lag node 'div_0' is not in `nodes`")


def test_a_lag_node_has_exactly_one_input():
    t = _tier()
    t.inner.lag_nodes.append(t.step)  # reads m_prev and m
    _refused(t, "lag node 'exp_0_max_step' must be a compute node with exactly one input")


def test_a_lag_node_is_not_itself_carried():
    t = _tier()
    t.inner.lag_nodes.append(t.rowsum)
    _refused(t, "lag node 'rowsum_0' is a per-iteration snapshot, not carried state")


def test_a_lag_node_is_not_prefetched_or_windowed():
    t = _tier()
    t.inner.prefetch_nodes.append(t.m_prev)
    _refused(t, "lag node 'exp_0_max_prev' cannot also be prefetched or windowed")


def test_a_lag_node_snapshots_an_accumulator():
    t = _tier()
    t.inner.lag_nodes.append(t.p)  # reads the shifted scores, which have no previous value
    _refused(t, "lag node 'exp_0' snapshots 'exp_0_shifted', which is not an accumulator")


def test_a_snapshot_of_a_max_that_is_not_carried_is_refused():
    """``carry`` forgotten: the lag node would copy a buffer that is refilled every tile."""
    t = _tier(carry=False)
    _refused(t, "lag node 'exp_0_max_prev' snapshots 'exp_0_max', which is not an accumulator")


# V3 ------------------------------------------------------------------------------------
def test_only_an_accumulator_is_rescaled():
    t = _tier()
    t.inner.accum_scale[t.p] = t.alpha
    _refused(t, "rescaled node 'exp_0' is not an accumulator of this graph")


def test_a_factor_is_a_compute_node_of_the_loop():
    t = _tier()
    t.inner.accum_scale[t.rowsum] = t.div
    _refused(t, "rescale factor 'div_0' of 'rowsum_0' is not a compute node of this graph")


@pytest.mark.parametrize("factor", ["psum", "m_prev"])
def test_a_factor_is_a_per_iteration_value(factor):
    t = _tier()
    t.inner.accum_scale[t.rowsum] = getattr(t, factor)
    _refused(t, "must be a per-iteration value, not an accumulator or a lag node")


def test_a_factor_is_not_computed_from_the_accumulator_it_scales():
    t = _tier()
    t.inner.accum_scale[t.m] = t.alpha  # alpha = exp(m_prev - m)
    _refused(t, "rescale factor 'exp_0_alpha' is computed from 'exp_0_max'")


@pytest.mark.parametrize("node", ["alpha", "rowsum"])
def test_factor_and_accumulator_are_float32(node):
    t = _tier()
    getattr(t, node).out_dtype = "float16"
    _refused(t, "rescale of 'rowsum_0' by 'exp_0_alpha': both must be float32")


def test_the_factor_tile_leads_the_accumulator_tile():
    t = _tier()
    t.inner.tmap[t.alpha] = _row(STEP)  # one value per key, not per row
    _refused(t, r"rescale factor 'exp_0_alpha' tile \[64\] is not a leading slice of "
                r"accumulator 'rowsum_0' tile \[16\]")


def test_an_all_unit_factor_tile_is_refused():
    t = _tier()
    t.inner.tmap[t.alpha] = _row(1)
    _refused(t, r"rescale factor 'exp_0_alpha' tile \[\] is not a leading slice")


def test_a_rescaled_accumulator_tile_has_no_unit_extent():
    t = _tier()
    t.inner.tmap[t.psum] = Iterator(lambda i: ("0", "0"), shape=(ROWS, 1), tile_size=(ROWS, 1))
    _refused(t, r"rescaled accumulator 'psum_0' has a unit extent in its tile \(16, 1\)")


# V4 ------------------------------------------------------------------------------------
def test_a_rescaled_accumulator_starts_at_zero():
    t = _tier()
    t.rowsum.accum_init = 1.0
    _refused(t, "rescaled accumulator 'rowsum_0' starts at 1.0")


# V5 ------------------------------------------------------------------------------------
def test_a_finite_start_value_needs_a_float32_accumulator():
    t = _tier()
    t.m.out_dtype = "float16"  # -FLT_MAX rounds to -inf there
    _refused(t, "accumulator 'exp_0_max' starts at -3.4028234663852886e\\+38, a float32 value, "
                "but is stored as float16")


def test_the_identity_of_max_is_fine_in_any_dtype():
    """``-inf`` (the class default of ReduceMax) is exact in float16; only a finite
    start value is tied to float32."""
    t = _tier()
    del t.m.accum_init  # back to the class default
    t.m.out_dtype = "float16"
    t.m_prev.dtype = t.step.dtype = t.shifted.dtype = "float16"
    assert "T.fill(t_exp_0_max, -T.infinity(" in TilelangCodeGen().gen(t.main)


# V6 ------------------------------------------------------------------------------------
def test_a_partial_sum_is_not_read_inside_its_loop():
    t = _tier()
    peek = Copy((ROWS,), name="peek", n_batch_dims=0)
    with edit_graph(t.main):
        wire_input(t.rowsum, peek, 0)
        t.inner.nodes.append(peek)
        t.inner.tmap[peek] = _row()
        t.inner.tinfo_map[peek] = ("t_peek", "r")
    _refused(t, r"accumulator 'rowsum_0' is read by \['peek'\] inside the loop that "
                "accumulates it")


def test_a_running_max_may_be_read_inside_its_loop():
    t = _tier()
    assert type(t.m).running_max is True
    readers = sorted(e.dst_node.name for e in t.m.outputs)
    assert readers == ["exp_0_max_prev", "exp_0_max_step", "exp_0_shifted"]
    TilelangCodeGen().gen(t.main)
    # ... which is a property of the class, not of the instance.
    t = _tier()
    t.m.running_max = False
    _refused(t, r"accumulator 'exp_0_max' is read by \['exp_0_max_step', 'exp_0_shifted'\] "
                "inside the loop")


# V7 ------------------------------------------------------------------------------------
def test_a_max_over_a_slice_is_refused():
    """The silent wrong kernel: a max the loop reads is not an accumulator by inference, so
    it restarts on every tile while the sums it normalises accumulate over all of them."""
    t = _tier(declare=False, carry=False)
    assert t.m not in t.inner.accum_nodes
    _refused(t, "'exp_0_max' takes a max over a slice: the loop gives it 64 of 256 elements")


def test_a_max_over_the_whole_axis_is_fine_in_a_loop():
    t = _tier(declare=False, carry=False)
    inner_in = t.inner.binding[t.outer_in.name]
    # the loop tiles another axis; every iteration sees the whole reduced axis
    t.inner.tmap[inner_in] = Iterator(
        lambda i: ("0", "0"), shape=(ROWS, STEP), tile_size=(ROWS, STEP))
    assert "ukernel_reduce_max_axis1_rr_clear" in TilelangCodeGen().gen(t.main)
