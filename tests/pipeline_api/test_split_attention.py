"""Static split attention through the pipeline: the split records on the searched graphs, and
the partial -> combine Kernel against the Program's own reference.

Until the records existed no static split-attention kernel could be used: a partial never
validated (its reference was laid out by matching extents, which raises when ``S_Q == D``
and silently compares transposed when the split extent equals the heads), and the chained
Kernel was unrunnable (the combine's inputs were paired with the partial's outputs by node
order) and would have handed the partial's split-first outputs over by reshape.

``test_search_*`` run the superoptimizer on CPU, ``test_gpu_*`` compile kernels (root
conftest marks).
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.graph_utils import boundary_signature
from cleave.pipeline.kernel import stages_for
from cleave.pipeline.plans import lower
from cleave.pipeline.report import validation_kind
from cleave.runtime.compile import build_candidate
from cleave.runtime.evaluate import CandidateResult
from cleave.runtime.reference import output_order, placeholder_order, stored_shape
from cleave.runtime.validate import compare_outputs
from cleave.schedule import apply_tiling
from tests._tolerances import tolerance_for
from tests.frontend import _programs as programs

TOLERANCE = tolerance_for("attention")
FACTORS = (4, 8, 16, 32)
ROWSUM, VALUE = ((1, 0, 2), 1), ((1, 0, 2, 3), 0)  # split axis first; the row sum padded
LOGICAL = (((0, 1, 2), 1), ((0, 1, 2, 3), 0))
UNLOOPED = (((0, 1, 2), 0), ((0, 1, 2, 3), 0))  # neither sum looped: stored as computed


def mha(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    """demo/mha_decode.py: the fixed-shift attention of ``_programs`` with one head dim."""
    return programs.attention.fn(q, k, v)


KINDS = {
    "mha": (mha, dict(H=32, SQ=64, SKV=4096, D=64)),
    "gqa": (programs.attention, dict(HQ=32, HKV=8, SQ=4096, SKV=4096, D=64)),
    "mqa": (programs.attention, dict(HQ=32, HKV=1, SQ=4096, SKV=4096, D=64)),
}
SPLIT = P.SuperoptConfig(split=("SKV",), max_ops=14)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _shape(node):
    return tuple(int(s) for s in node.get_shape(0))


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_search_split_records_on_searched_graphs(kind, cache_root):
    fn, shapes = KINDS[kind]
    heads = shapes.get("HQ", shapes.get("H"))
    prog = trace(fn, shapes, cl.f16)
    split = P.superoptimize(prog, SPLIT).filter(lambda g: g.is_split)
    assert len(split) >= 1

    # Record B: on every lowered pair the two sides of the cut are the same tensors.
    for graph in split:
        for factor in FACTORS:
            partial, combine = lower(graph, factor)
            crossing = boundary_signature(output_order(partial))
            assert crossing == boundary_signature(placeholder_order(combine))
            assert crossing == (
                (0, "", (heads, factor, shapes["SQ"]), "float16"),
                (1, "", (heads, factor, shapes["SQ"], shapes["D"]), "float16"))
            # the Program's inputs are the placeholders in declaration order
            assert sorted(ph.boundary.index for ph in placeholder_order(partial)) == [0, 1, 2]

    # Record A: the stored shape of every output of every tiling follows from its record.
    # The four best partial tilings of every graph and factor, and the combine tilings down
    # to the first ones that have a kernel: the best-ranked combine tilings take one
    # partial per step or a single row per block, which the build refuses.
    plans = P.schedule(split, P.ScheduleConfig(topk=32, split_k=FACTORS))
    assert not plans.refused, plans.summary()
    built = {}  # (variant, role) -> the first plan with a kernel, and its result
    layouts_seen = {}
    for plan in plans:
        if plan.rank >= 4 and (plan.role != "combine" or (plan.variant, "combine") in built):
            continue
        try:
            scheduled = apply_tiling(plan.tile)
        except Exception:  # noqa: BLE001 - a tiling apply_tiling refuses has no kernel
            continue
        graph = plan.tile.flat_graph
        for flat, out in zip(output_order(graph), output_order(scheduled)):
            assert out.name == flat.name
            assert stored_shape(_shape(flat), scheduled.stored_layouts[out.name]) == tuple(
                int(s) for s in scheduled.tmap[out].shape), plan.label
        try:
            code, layouts = build_candidate(plan.tile)
        except ScheduleRejected:  # a tiling the build refuses (a unit extent) has no kernel
            continue
        layouts_seen.setdefault((plan.split_factor, plan.role), set()).add(layouts)
        built.setdefault((plan.variant, plan.role), (plan, CandidateResult(
            plan.tile, "ok", code=code, stored_layouts=layouts)))
        if plan.role == "partial":  # bound to the Program inputs in declaration order
            by_name = {ph.name: ph for ph in placeholder_order(graph)}
            assert [by_name[name].boundary.index for name in plan.inputs] == [0, 1, 2]
    for factor in FACTORS:
        # A looped partial stores the split axis first -- also where that leaves the shape
        # as it was (split == heads). GQA with split == H_KV keeps the logical layout: the
        # layout depends on the extents, which is why it is recorded and not derived.
        expected = LOGICAL if (kind, factor) == ("gqa", 8) else (ROWSUM, VALUE)
        assert layouts_seen[(factor, "partial")] == {expected}
        assert layouts_seen[(factor, "combine")] == {(((0, 1, 2), 0),)}

    # The hand-off: every combine input is fed from the partial output that is the same
    # tensor, with the layout that output is stored in.
    for factor in FACTORS:
        variant = f"split-k={factor}"
        (partial_plan, partial_run), combine = built[(variant, "partial")], built[(variant, "combine")]
        stages, problem = stages_for([(partial_plan, partial_run), combine], prog)
        assert problem is None
        assert [(p.source, p.index) for p in stages[0].params] == [
            ("input", partial_plan.inputs.index(p.name)) for p in stages[0].params]
        outputs = output_order(partial_plan.tile.flat_graph)
        phs = {ph.name: ph for ph in placeholder_order(combine[0].tile.flat_graph)}
        assert len(stages[1].params) == len(phs) == 2
        for param in stages[1].params:
            ph, fed_from = phs[param.name], outputs[param.index]
            assert param.source == "stage" and fed_from.boundary == ph.boundary
            assert param.shape == _shape(ph) == _shape(fed_from)
            assert param.layout == partial_run.stored_layouts[param.index]


# ------------------------------------------ partials that take a whole split in one step

SMALL = dict(H=4, SQ=64, SKV=1024, D=64)


def _whole_split_plans(prog, factors=(4, 8)):
    """The combines, and the partial plans of one split graph whose row sum and value
    matmul both take the whole per-split KV extent in one step: neither is looped."""
    graphs = P.superoptimize(prog, SPLIT)
    plans = P.schedule(graphs.filter(lambda g: g.is_split),
                       P.ScheduleConfig(topk=32, split_k=factors))
    first = min(p.graph_index for p in plans if p.role == "partial")

    def whole(plan):
        steps = sorted(int(x) for v in plan.tile.rstep_map.values() for x in v.values())
        return steps[-2:] == [SMALL["SKV"] // plan.split_factor] * 2

    return plans.filter(lambda p: p.role == "combine" or (p.graph_index == first and whole(p)))


def test_search_partials_that_take_a_whole_split_in_one_step_build(cache_root):
    """With both sums un-looped the row sum stays in the block's own tier, next to the
    value matmul. It is not an ancestor of the canonical (value) output, so nothing said
    which launch axis walks it and ``apply_tiling`` refused every such plan ("Expected
    exactly one launch-axis mapping for iter dim 0, got []"): 8 of the 32 partial plans of
    each split graph, at either factor."""
    prog = trace(mha, SMALL, cl.f16)
    plans = _whole_split_plans(prog)
    for factor in (4, 8):
        partials = [p for p in plans if p.role == "partial" and p.split_factor == factor]
        assert len(partials) == 8
        for plan in partials:
            scheduled = apply_tiling(plan.tile)
            code, layouts = build_candidate(plan.tile)
            assert layouts == UNLOOPED
            flat = output_order(plan.tile.flat_graph)
            for out, layout in zip(flat, layouts):
                assert stored_shape(_shape(out), layout) == tuple(
                    int(x) for x in scheduled.tmap[out].shape)
            # The kernel's outputs are its trailing arguments, row sum first -- although
            # the row sum is finished before the value matmul's V is reached.
            signature = code[code.index("def main_kernel("):code.index("with T.Kernel")]
            names = [line.split(":")[0].strip() for line in signature.splitlines()[1:]
                     if ": T.Tensor(" in line]
            assert names[-2:] == [out.name for out in flat], names
            assert sorted(names[:3]) == sorted(ph.name for ph in placeholder_order(
                plan.tile.flat_graph))


# ------------------------------------------------------------------ GPU

def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")


def _split_plans(prog, topk, factors):
    graphs = P.superoptimize(prog, SPLIT)
    plans = P.schedule(graphs.filter(lambda g: g.is_split),
                       P.ScheduleConfig(topk=topk, split_k=factors))
    # every split graph of this program lowers to the same partial: one of them is enough
    first = min(p.graph_index for p in plans if p.role == "partial")
    return plans.filter(lambda p: p.role == "combine" or (p.graph_index == first and p.rank < 8))


def _partials_all_validate(report, factor):
    partials = [r for r in report if r.role == "partial" and r.variant == f"split-k={factor}"]
    failed = [(r.label, r.error_type, validation_kind(r.error))
              for r in partials if not r.ok and r.stage == "validate"]
    assert not failed, failed  # used to be all of them
    ok = [r for r in partials if r.ok]
    assert ok, [(r.label, r.error_type) for r in partials]
    for r in ok:
        assert len(r.measured) == 2 and r.run.stored_layouts == (ROWSUM, VALUE)
        assert all(m.rel <= TOLERANCE.rtol for m in r.measured)
    return ok


def test_gpu_chained_split_attention_matches_the_program_reference(cache_root):
    """H=4, S_Q=64, D=64, S_KV=1024: split 4 equals the heads (equal shapes for both layouts)
    and S_Q == D makes split 8 ambiguous by extent, so both failures are on the table."""
    _gpu()
    prog = trace(mha, dict(H=4, SQ=64, SKV=1024, D=64), cl.f16)
    # the combine tilings that compile rank in the twenties
    plans = _split_plans(prog, topk=32, factors=(4, 8))
    report = P.evaluate(plans, P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                                            tolerance=TOLERANCE))
    for factor in (4, 8):
        variant = f"split-k={factor}"
        _partials_all_validate(report, factor)
        assert any(r.ok for r in report if r.role == "combine" and r.variant == variant)

        kernel = report.best(variant)
        assert kernel.unrunnable is None and kernel.variant == variant
        assert [s.role for s in kernel.stages] == ["partial", "combine"]
        partial, combine = kernel.kernels
        assert [p.dims for p in partial.outputs] == [(factor, 4, 64, 1), (factor, 4, 64, 64)]
        assert sorted((p.shape, p.layout) for p in kernel.stages[1].params) == [
            ((4, factor, 64), ROWSUM), ((4, factor, 64, 64), VALUE)]
        for seed in (1, 2, 3):
            q, k, v = prog.random_inputs(seed=seed)
            reference = prog.reference(q, k, v)
            (measured,) = compare_outputs([kernel(q, k, v)], reference, TOLERANCE,
                                          what=f"chained {variant} seed {seed}")
            assert 0 < measured.rel <= TOLERANCE.rtol
            # by name too, and in a different order: K and V have one shape
            assert torch.equal(kernel(v=v, q=q, k=k), kernel(q, k, v))


def test_gpu_split_attention_partials_validate_at_the_demo_shapes(cache_root):
    """demo/mha_decode.py (H=32, S_Q=64, D=64, S_KV=4096, split 16 and 32): at 16 every
    partial used to fail with "Ambiguous output layout reconciliation", at 32 (== heads)
    every partial was compared transposed and failed as a mismatch."""
    _gpu()
    prog = trace(mha, dict(H=32, SQ=64, SKV=4096, D=64), cl.f16)
    plans = _split_plans(prog, topk=8, factors=(16, 32)).filter(lambda p: p.role == "partial")
    report = P.evaluate(plans, P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                                            tolerance=TOLERANCE))
    for factor in (16, 32):
        ok = _partials_all_validate(report, factor)
        assert {tuple(p.dims for p in r.kernel.outputs) for r in ok} == {
            ((factor, 32, 64, 1), (factor, 32, 64, 64))}


def test_gpu_partials_that_take_a_whole_split_in_one_step_validate_and_chain(cache_root):
    """The plans ``apply_tiling`` used to refuse: they validate in the layout they are
    stored in (the logical one: nothing comes out of a loop), and chained with the combine
    they compute the program."""
    _gpu()
    prog = trace(mha, SMALL, cl.f16)
    report = P.evaluate(_whole_split_plans(prog), P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                                                               tolerance=TOLERANCE))
    for factor in (4, 8):
        variant = f"split-k={factor}"
        partials = [r for r in report if r.role == "partial" and r.variant == variant]
        assert len(partials) == 8
        failed = [(r.label, r.error_type, validation_kind(r.error))
                  for r in partials if not r.ok and r.stage == "validate"]
        assert not failed, failed
        ok = [r for r in partials if r.ok]
        assert ok, [(r.label, r.stage, r.error_type) for r in partials]
        for r in ok:
            assert len(r.measured) == 2 and r.run.stored_layouts == UNLOOPED
            assert all(m.rel <= TOLERANCE.rtol for m in r.measured)
            assert [p.dims for p in r.kernel.outputs] == [(4, factor, 64), (4, factor, 64, 64)]

        kernel = report.best(variant)
        assert kernel.unrunnable is None
        assert [s.role for s in kernel.stages] == ["partial", "combine"]
        assert sorted((p.shape, p.layout) for p in kernel.stages[1].params) == [
            ((4, factor, 64), UNLOOPED[0]), ((4, factor, 64, 64), UNLOOPED[1])]
        for seed in (1, 2, 3):
            q, k, v = prog.random_inputs(seed=seed)
            (measured,) = compare_outputs([kernel(q, k, v)], prog.reference(q, k, v), TOLERANCE,
                                          what=f"chained {variant} seed {seed}")
            assert 0 < measured.rel <= TOLERANCE.rtol
