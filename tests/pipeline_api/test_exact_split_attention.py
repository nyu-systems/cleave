"""The exact softmax across a split, through the pipeline and on a GPU.

``cl.softmax`` attention split over the keys (the split-KV of FlashDecoding): each partial
kernel returns the statistics of its own split, ``(m_i, l_i, o_i)`` with ``l_i`` and
``o_i`` relative to that split's own max ``m_i``, and the combine kernel weights them by
``exp(m_i - max_i m_i)``. Three things are run here, at the decode shapes of
``demo/{mha,gqa,mqa}_decode.py`` with 64 queries:

* the partial alone and the combine alone, each against the reference of its own lowered
  graph, on normal inputs and on every adversarial softmax case (the build holds a program
  whose softmax is exact to all of them: logits x100, rising and falling along the softmax
  axis, a spike on its last entry, constant rows, a common offset). For the combine the
  softmax axis is the split axis: maxima that rise along the splits, one split that
  dominates;
* the chained ``Kernel`` against the Program's own reference, on normal inputs and on
  every adversarial case, and on flat rows over values with a common offset, where a
  split's own sums are past float16;
* a combine fed by hand: maxima spread over hundreds of units, and a split that saw no key
  (it returns the start value of its carried max and sums of zero).

``test_search_*`` run the superoptimizer on CPU, ``test_gpu_*`` compile kernels (root
conftest marks).
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.graph_utils import boundary_signature
from cleave.ir.ops import NEG_FLT_MAX
from cleave.pipeline.plans import bind_inputs, lower
from cleave.pipeline.report import validation_kind
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.inputs import ADVERSARIAL, apply_distribution
from cleave.runtime.reference import (execute, order_inputs_for_kernel, output_order,
                                      output_roles, placeholder_order)
from cleave.runtime.validate import compare_outputs
from cleave.schedule import apply_tiling
from cleave.schedule.stabilize import sites
from tests._tolerances import tolerance_for

TOLERANCE = tolerance_for("exact_split_attention")  # by boundary role: "" and "shift"
EVERY_CASE = ("normal",) + ADVERSARIAL


def mha(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def grouped(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
            v: cl.Tensor["HKV", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


# kind -> (program, shapes, split factors): demo/{mha,gqa,mqa}_decode.py with 64 queries
DECODE = {
    "mha": (mha, dict(H=32, SQ=64, SKV=4096, D=64), (16, 32)),
    "gqa": (grouped, dict(HQ=32, HKV=8, SQ=64, SKV=4096, D=64), (4, 8, 16)),
    "mqa": (grouped, dict(HQ=32, HKV=1, SQ=64, SKV=4096, D=64), (4, 8, 16)),
}
SPLIT = P.SuperoptConfig(split=("SKV",), max_ops=14)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")


def _case(plan):
    (site,) = sites(apply_tiling(plan.tile))
    return site.case


def _steps(plan):
    return sorted({int(x) for v in plan.tile.rstep_map.values() for x in (v or {}).values()})


def _split_plans(prog, factors):
    """Per factor: partial plans of the first split graph (every split graph of these
    programs lowers to the same partial) that carry a running max and that take a split in
    one step, on 128 threads, and combine plans of both kinds. Among the combines, the
    tilings that had no right kernel while their loop was pipelined: two partials of 8 or
    16 rows a step (their kernel faulted) and of 64 rows a step (wrong values)."""
    plans = P.schedule(P.superoptimize(prog, SPLIT).filter(lambda g: g.is_split),
                       P.ScheduleConfig(topk=1000, split_k=factors))
    assert not [r for r in plans.refused if r.role != "combine"], plans.summary()
    first = min(p.graph_index for p in plans if p.role == "partial")
    keep = []
    for factor in factors:
        mine = [p for p in plans if p.split_factor == factor]
        partials = [p for p in mine if p.role == "partial" and p.graph_index == first
                    and p.n_threads == 128]
        combines = [p for p in mine if p.role == "combine"]
        for case, pool in (("rolling", partials), ("in-tile", partials),
                           ("rolling", combines), ("in-tile", combines)):
            keep += [p for p in pool if _case(p) == case][:2]
        keep += [p for p in combines if _steps(p) == [2]
                 and list(p.tile.output_tile)[1] in (8, 16, 64)]
    chosen = {id(p) for p in keep}
    return plans.filter(lambda p: id(p) in chosen)


def _what(case, normal=""):
    """How a comparison names its inputs in the validation log: an adversarial case under
    the name validation gives it, so that it is held to the declared bound and is no input
    to it (``tests/_measure_tolerances.py``)."""
    return normal if case == "normal" else f"{case} inputs"


def _program_inputs(prog, case, seed):
    """The Program's inputs in one of the softmax input cases: its random inputs, shaped
    the way validation shapes the inputs of the lowered graph."""
    inputs = prog.random_inputs(seed=seed)
    if case == "normal":
        return inputs
    (ir,) = lower(GraphSet.identity(prog)[0])
    by_name = dict(zip(bind_inputs(ir, prog), inputs))
    order = [p.name for p in placeholder_order(ir)]
    shaped = dict(zip(order, apply_distribution(ir, tuple(by_name[n] for n in order), case)))
    return [shaped[n] for n in bind_inputs(ir, prog)]


@pytest.mark.parametrize("kind", sorted(DECODE))
def test_search_every_split_graph_lowers_to_an_exact_pair(kind, cache_root):
    """CPU: the searched split graphs of the exact program. Each lowers, at every factor,
    to a partial that returns its shift first and a combine that reads the same three
    tensors; a split the block holds whole is served in place, a looped one by a running
    max, and the plans carry what the hand-off needs."""
    fn, shapes, factors = DECODE[kind]
    heads = shapes.get("HQ", shapes.get("H"))
    prog = trace(fn, shapes, cl.f16)
    split = P.superoptimize(prog, SPLIT).filter(lambda g: g.is_split)
    assert len(split) >= 1
    for graph in split:
        for factor in factors:
            partial, combine = lower(graph, factor)
            crossing = boundary_signature(output_order(partial))
            assert crossing == boundary_signature(placeholder_order(combine)) == (
                (0, "", (heads, factor, shapes["SQ"]), "float32"),
                (1, "", (heads, factor, shapes["SQ"], shapes["D"]), "float32"),
                (2, "shift", (heads, factor, shapes["SQ"]), "float32"))
            assert output_roles(partial) == ("shift", "", "") and output_roles(combine) == ("",)
            assert bind_inputs(partial, prog) is not None
    plans = _split_plans(prog, factors)
    for factor in factors:
        mine = [p for p in plans if p.split_factor == factor]
        cases = {(p.role, _case(p)) for p in mine}
        # a split of 1024 keys (4 splits) does not fit one step; any other does
        whole = {("partial", "in-tile")} if shapes["SKV"] // factor <= 256 else set()
        assert cases == {("partial", "rolling"), ("combine", "rolling"),
                         ("combine", "in-tile")} | whole, (factor, cases)


# What a split's sums come to relative to its own max: name -> (shapes, split factor, the
# common offset of the values). Flat rows (every score equal) weight every key by 1, so a
# split's row sum is the number of its keys and its value sum that many times the values.
OVERFLOW = {
    "value_sums": (dict(H=2, SQ=4, SKV=4096, D=8), 16, 300.0),  # 256 keys x 300
    "row_sums": (dict(H=2, SQ=4, SKV=262144, D=8), 4, 0.0),  # 65536 keys
}


@pytest.mark.parametrize("name", sorted(OVERFLOW))
def test_search_the_sums_of_a_split_cross_the_cut_in_float32(name, cache_root, monkeypatch):
    """CPU, reference level: the pair computes the program where a split's own sums are
    past what the program's dtype holds. They crossed the cut in float16: the partial
    handed over ``inf``, the combine returned ``inf`` (or 0, where only the row sums
    overflow), and each stage still agreed with its own reference, which was cast the same
    way. The solo kernel divides in float32 before it casts; so does the pair now."""
    from cleave.runtime.reference import build_inputs_and_reference
    from cleave.runtime.validate import Tolerance
    from tests._softmax import chained_reference

    # (References and made-up tensors, no kernel: nothing here belongs in a measurement.)
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)
    shapes, factor, offset = OVERFLOW[name]
    prog = trace(mha, shapes, cl.f16)
    split = P.superoptimize(prog, SPLIT).filter(lambda g: g.is_split)
    assert len(split) >= 1
    q, k, v = prog.random_inputs(seed=0, device="cpu")
    q, v = torch.zeros_like(q), (v.float() + offset).to(v.dtype)
    (want,) = prog.reference(q, k, v)
    assert torch.isfinite(want).all()
    for graph in split:
        partial, combine = lower(graph, factor)
        crossing, (out,) = chained_reference(partial, combine, prog, [q, k, v])
        assert [str(x.dtype) for x in crossing] == ["torch.float32"] * 3
        assert all(torch.isfinite(x).all() for x in crossing)
        # ... and float16 would not hold them
        assert max(float(x.abs().max()) for x in crossing[:2]) > torch.finfo(torch.float16).max
        compare_outputs([out], [want], Tolerance(rtol=5e-3))
        # the inputs validation makes for the partial: on constant rows its reference is finite
        _, reference, _ = build_inputs_and_reference(partial, "float16", device="cpu",
                                                     distribution="constant")
        assert all(torch.isfinite(r).all() for r in reference)


@pytest.mark.parametrize("kind", sorted(DECODE))
def test_gpu_exact_split_attention(kind, cache_root):
    _gpu()
    fn, shapes, factors = DECODE[kind]
    heads = shapes.get("HQ", shapes.get("H"))
    prog = trace(fn, shapes, cl.f16)
    plans = _split_plans(prog, factors)
    report = P.evaluate(plans, P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                                            tolerance=TOLERANCE))

    # Each stage alone, against the reference of its own graph, on every input case. No
    # kernel that compiled is wrong, on any of them.
    assert all(r.cases == EVERY_CASE and not r.cases_note for r in report)
    failed = [(r.label, r.error_type, validation_kind(r.error), str(r.error)[-300:])
              for r in report if not r.ok and r.stage in ("validate", "bench", "timeout", "crash")]
    assert not failed, failed
    for factor in factors:
        variant = f"split-k={factor}"
        ok = [r for r in report if r.ok and r.variant == variant]
        served = {(r.role, _case(r.plan)) for r in ok}
        assert {("partial", "rolling"), ("combine", "rolling"), ("combine", "in-tile")} <= served
        for r in ok:
            roles = output_roles(r.plan.tile.flat_graph)
            assert roles == (("shift", "", "") if r.role == "partial" else ("",))
            assert len(r.measured) == len(roles)
            for role, measured in zip(roles, r.measured):
                assert measured.rel <= TOLERANCE[role].rtol, (r.label, role, measured)
        # the loops that had no right kernel while they were pipelined
        once_wrong = [r for r in ok if r.role == "combine" and _steps(r.plan) == [2]
                      and list(r.plan.tile.output_tile)[1] in (8, 16, 64)]
        assert once_wrong and all("for i0 in T.serial(" in r.source for r in once_wrong)

        # The chained Kernel against the Program, on normal inputs and on every adversarial
        # case: finite, and within the bound of the program's own outputs.
        kernel = report.best(variant)
        assert kernel.unrunnable is None and kernel.variant == variant
        assert [s.role for s in kernel.stages] == ["partial", "combine"]
        partial, combine = kernel.kernels
        assert [p.dtype for p in partial.outputs] == ["float32", "float32", "float32"]
        assert sorted(p.shape for p in kernel.stages[1].params) == [
            (heads, factor, 64), (heads, factor, 64), (heads, factor, 64, 64)]
        assert all(p.source == "stage" and p.layout is not None for p in kernel.stages[1].params)
        for case in EVERY_CASE:
            for seed in (1, 2):
                inputs = _program_inputs(prog, case, seed)
                reference = prog.reference(*inputs)
                assert torch.isfinite(reference[0]).all()
                (measured,) = compare_outputs([kernel(*inputs)], reference, TOLERANCE[""],
                                              what=_what(case, f"chained {variant}"))
                assert measured.rel <= TOLERANCE[""].rtol

        # Flat rows over values with a common offset. Relative to its own max a split's
        # sums are then the number of its keys times the values: more than float16 holds,
        # where the program's result (the mean of the values) is nowhere near it. The sums
        # cross the cut in float32. (Constant logits: logged as that case, held to the
        # bound and no input to it.)
        keys = shapes["SKV"] // factor
        offset = 16.0 * round(1.25 * float(torch.finfo(torch.float16).max) / keys / 16.0)
        assert keys * (offset - 8.0) > torch.finfo(torch.float16).max
        q, k, v = prog.random_inputs(seed=3)
        flat = [torch.zeros_like(q), k, (v.float() + offset).to(v.dtype)]
        reference = prog.reference(*flat)
        assert torch.isfinite(reference[0]).all()
        (measured,) = compare_outputs([kernel(*flat)], reference, TOLERANCE[""],
                                      what=_what("constant", f"chained {variant}"))
        assert measured.rel <= TOLERANCE[""].rtol

        # The combine fed by hand: maxima spread over hundreds of units, and one split
        # that saw no key at all.
        best = min((r for r in ok if r.role == "combine"), key=lambda r: r.latency_us)
        graph = best.plan.tile.flat_graph
        compiled = best.kernel.compile()
        gen = torch.Generator().manual_seed(factor)
        l = torch.rand((heads, factor, 64), generator=gen) + 0.5
        o = torch.randn((heads, factor, 64, 64), generator=gen)
        m = torch.randn((heads, factor, 64), generator=gen) * 100.0
        l[:, 1], o[:, 1], m[:, 1] = 0.0, 0.0, NEG_FLT_MAX
        by_index = {0: l, 1: o, 2: m}
        feed = {p.name: by_index[p.boundary.index].cuda() for p in placeholder_order(graph)}
        out = compiled(*order_inputs_for_kernel(feed, compiled))
        assert torch.isfinite(out).all()
        compare_outputs([out], execute(graph, list(feed.values())), TOLERANCE[""],
                        what=_what("large_logits"))
        # ... the same answer as without that split
        keep = [i for i in range(factor) if i != 1]
        lk, ok_, mk = (feed[p.name][:, keep].float() for p in sorted(
            placeholder_order(graph), key=lambda p: p.boundary.index))
        w = torch.exp(mk - mk.amax(1, keepdim=True))
        want = (w[..., None] * ok_).sum(1) / (w * lk).sum(1)[..., None]
        compare_outputs([out], [want.to(torch.float16)], TOLERANCE[""],
                        what=_what("large_logits"))


def test_gpu_a_combine_loop_that_was_overwritten_in_flight_is_right_on_every_launch(cache_root):
    """Two partials of 64 rows a step, 8 partials: while this loop was pipelined its value
    partials were reloaded before their last use (columns 24 and up of every row came from
    a later iteration, a different count of them on every launch). As a plain loop every
    launch gives the same, right, values."""
    _gpu()
    fn, shapes, _ = DECODE["gqa"]
    prog = trace(fn, shapes, cl.f16)
    plans = _split_plans(prog, (8,)).filter(
        lambda p: p.role == "combine" and _steps(p) == [2]
        and list(p.tile.output_tile)[1] == 64)
    assert len(plans) == 1
    report = P.evaluate(plans, P.EvalConfig(warmup=2, iters=20, max_parallel=1,
                                            tolerance=TOLERANCE))
    (result,) = report.results
    assert result.ok, result.error
    graph = result.plan.tile.flat_graph
    compiled = result.kernel.compile()
    from cleave.runtime.reference import build_inputs_and_reference

    inputs_by_name, (reference,), _ = build_inputs_and_reference(graph, "float16")
    inputs = order_inputs_for_kernel(inputs_by_name, compiled)
    first = compiled(*inputs).clone()
    compare_outputs([first], [reference], TOLERANCE[""], what="first launch")
    for _ in range(50):
        assert torch.equal(compiled(*inputs), first)
