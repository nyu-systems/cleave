"""The exact softmax across a split built from the loop state, through the pipeline and on a GPU.

``cl.softmax`` attention split over the keys (the split-KV of FlashDecoding). The search
returns no split graph for a softmax with its max, so the pair is built from the loop
state of the flash graph (``pipeline.plans.state_pair``): each partial returns the
accumulators of its own split and the max they are relative to, ``(o_i, l_i, m_i)``, and
the combine takes ``M = max_i m_i`` and repairs each by the proven rule,
``o_i * exp(m_i - M)``, before it sums over the splits and divides. Both stages are flat
graphs with a max in front of their sums, so each loops wherever its derivation lets it:
the partial over the keys of its split, the combine over the splits. Three things are run
here, at the decode shapes of ``demo/{mha,gqa,mqa}_decode.py`` with 64 queries:

* the partial alone and the combine alone, each against the reference of its own graph,
  on normal inputs and on every adversarial softmax case. For the combine the softmax axis
  is the split axis: maxima that rise along the splits, one split that dominates. Every
  combine tiling that takes all splits in one tile is among them;
* the chained ``Kernel`` against the Program's own reference, on normal inputs and on
  every adversarial case, and on flat rows over values with a common offset, where a
  split's own sums are past float16;
* a combine fed by hand: maxima spread over hundreds of units, and a split that saw no key
  (it returns the start value of its carried max and sums of zero).

On the CPU the pair is held to the solo graph at the reference level for every demo factor
(normal, large_logits, offset). ``test_search_*`` run the superoptimizer on CPU,
``test_gpu_*`` compile kernels (root conftest marks).
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
from cleave.runtime.inputs import ADVERSARIAL, adversarial_cases, apply_distribution
from cleave.runtime.reference import (execute, make_inputs, order_inputs_for_kernel,
                                      output_order, output_roles, placeholder_order)
from cleave.runtime.validate import compare_outputs
from cleave.schedule import apply_tiling
from cleave.schedule.streaming import sites
from tests._explicit_softmax import chained_reference
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
SPLIT = P.SuperoptConfig(split=("SKV",))  # searched unsplit, at the budget of no split


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")


def _case(plan):
    """``"looped"`` where the tiling carries the stage's max across a loop over the dim it
    reduces (the keys of a split, the splits), ``"one tile"`` where a tile holds that dim."""
    return "looped" if sites(apply_tiling(plan.tile)) else "one tile"


def _steps(plan):
    return sorted({int(x) for v in plan.tile.rstep_map.values() for x in (v or {}).values()})


def _flash(graphs):
    (graph,) = [g for g in graphs if g.summary().endswith("sum>matmul>div")]
    return graph


def _split_plans(prog, factors):
    """Per factor: partial plans on 128 threads, two that loop over the keys of a split
    and two that take a split in one tile (where a tile holds one), two combines that loop
    over the splits, EVERY combine that takes the splits in one tile, and the combines that
    step two partials at a time over 8, 16 or 64 rows (under the hand-derived pass, the
    loops that had no right kernel while they were pipelined)."""
    graphs = P.superoptimize(prog, SPLIT)
    flash = _flash(graphs)
    plans = P.schedule(graphs, P.ScheduleConfig(topk=1000, split_k=factors))
    assert not [r for r in plans.refused if r.graph_index == flash.index], plans.summary()
    keep = []
    for factor in factors:
        mine = [p for p in plans if p.split_factor == factor]
        partials = [p for p in mine if p.role == "partial" and p.n_threads == 128]
        combines = [p for p in mine if p.role == "combine"]
        keep += [p for p in partials if _case(p) == "looped"][:2]
        keep += [p for p in partials if _case(p) == "one tile"][:2]
        keep += [p for p in combines if _case(p) == "looped"][:2]
        keep += [p for p in combines if _case(p) == "one tile"]
        keep += [p for p in combines if _steps(p) == [2]
                 and list(p.tile.output_tile)[1] in (8, 16, 64)]
    chosen = {id(p) for p in keep}
    return plans.filter(lambda p: id(p) in chosen)


def _what(case, normal=""):
    """How a comparison names its inputs in the validation log: an adversarial case under
    the name validation gives it, so that it is held to the declared bound and is no input
    to it (``tests/_measure_tolerances.py``)."""
    return normal if case == "normal" else f"{case} inputs"


def _program_inputs(prog, case, seed, device="cuda"):
    """The Program's inputs in one of the softmax input cases: its random inputs, shaped
    the way validation shapes the inputs of the lowered graph."""
    inputs = prog.random_inputs(seed=seed, device=device)
    if case == "normal":
        return inputs
    (ir,) = lower(GraphSet.identity(prog)[0])
    by_name = dict(zip(bind_inputs(ir, prog), inputs))
    order = [p.name for p in placeholder_order(ir)]
    shaped = dict(zip(order, apply_distribution(ir, tuple(by_name[n] for n in order), case)))
    return [shaped[n] for n in bind_inputs(ir, prog)]


@pytest.mark.parametrize("kind", sorted(DECODE))
def test_search_every_factor_builds_an_exact_pair_from_the_loop_state(kind, cache_root):
    """CPU: the flash graph lowers, at every factor, to a partial that returns its max
    last (role "shift") behind its two sums and a combine that reads the same three
    tensors; both are held to every adversarial case. The partial loops over the keys of a
    split (and takes a split in one tile where a tile holds 256 keys), the combine loops
    over the splits in a plain loop or takes them in one tile."""
    fn, shapes, factors = DECODE[kind]
    heads = shapes.get("HQ", shapes.get("H"))
    prog = trace(fn, shapes, cl.f16)
    graph = _flash(P.superoptimize(prog, SPLIT))
    for factor in factors:
        partial, combine = lower(graph, factor)
        crossing = boundary_signature(output_order(partial))
        assert crossing == boundary_signature(placeholder_order(combine)) == (
            (0, "", (heads, factor, shapes["SQ"], shapes["D"]), "float32"),
            (1, "", (heads, factor, shapes["SQ"]), "float32"),
            (2, "shift", (heads, factor, shapes["SQ"]), "float32"))
        assert output_roles(partial) == ("shift", "", "") and output_roles(combine) == ("",)
        assert bind_inputs(partial, prog) is not None
        assert adversarial_cases(partial) == adversarial_cases(combine) == ADVERSARIAL
    plans = _split_plans(prog, factors)
    for factor in factors:
        mine = [p for p in plans if p.split_factor == factor]
        cases = {(p.role, _case(p)) for p in mine}
        whole = {("partial", "one tile")} if shapes["SKV"] // factor <= 256 else set()
        assert cases == {("partial", "looped"), ("combine", "looped"),
                         ("combine", "one tile")} | whole, (factor, cases)
        for p in mine:
            if p.role == "combine" and _case(p) == "looped":
                tiled = apply_tiling(p.tile)
                (site,) = sites(tiled)
                assert site.block == factor and site.step < factor
                assert {t.loop_kind for t, o in _tiers(tiled) if o is not None} == {"serial"}


def _tiers(tiled):
    from cleave.ir.graph_utils import graph_tiers

    return graph_tiers(tiled)


@pytest.mark.parametrize("kind", sorted(DECODE))
def test_search_partial_then_combine_is_the_solo_graph(kind, cache_root, monkeypatch):
    """CPU, reference level: the partial of every demo factor, handed to
    the combine by boundary record, computes the solo graph on normal inputs, on logits
    x100 and on a common offset of -1e4."""
    # (References on the CPU, no kernel: nothing here belongs in a measurement.)
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)
    fn, shapes, factors = DECODE[kind]
    prog = trace(fn, shapes, cl.f16)
    graph = _flash(P.superoptimize(prog, SPLIT))
    (solo,) = lower(graph)
    names = bind_inputs(solo, prog)
    for case in ("normal", "large_logits", "offset"):
        inputs = make_inputs(solo, "float16", seed=0, device="cpu", distribution=case)
        (want,) = execute(solo, inputs)
        by_name = dict(zip([p.name for p in placeholder_order(solo)], inputs))
        program_inputs = [by_name[n] for n in names]
        for factor in factors:
            partial, combine = lower(graph, factor)
            crossing, (got,) = chained_reference(partial, combine, prog, program_inputs)
            assert [str(x.dtype) for x in crossing] == ["torch.float32"] * 3
            assert torch.isfinite(got).all() and got.shape == want.shape
            compare_outputs([got], [want], TOLERANCE[""], what=f"{case} inputs")


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
    past what the program's dtype holds. They cross the cut in float32 (in float16 the
    partial would hand over ``inf``), and the inputs validation makes for the partial on
    constant rows give it a finite reference."""
    from cleave.runtime.reference import build_inputs_and_reference
    from cleave.runtime.validate import Tolerance

    # (References and made-up tensors, no kernel: nothing here belongs in a measurement.)
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)
    shapes, factor, offset = OVERFLOW[name]
    prog = trace(mha, shapes, cl.f16)
    graph = _flash(P.superoptimize(prog, SPLIT))
    q, k, v = prog.random_inputs(seed=0, device="cpu")
    q, v = torch.zeros_like(q), (v.float() + offset).to(v.dtype)
    (want,) = prog.reference(q, k, v)
    assert torch.isfinite(want).all()
    partial, combine = lower(graph, factor)
    crossing, (out,) = chained_reference(partial, combine, prog, [q, k, v])
    assert [str(x.dtype) for x in crossing] == ["torch.float32"] * 3
    assert all(torch.isfinite(x).all() for x in crossing)
    # ... and float16 would not hold them
    assert max(float(x.abs().max()) for x in crossing[:2]) > torch.finfo(torch.float16).max
    compare_outputs([out], [want], Tolerance(rtol=5e-3))
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
    # kernel that compiled is wrong, on any of them; every combine that takes the splits
    # in one tile validates.
    assert all(r.cases == EVERY_CASE and not r.cases_note for r in report)
    failed = [(r.label, r.error_type, validation_kind(r.error), str(r.error)[-300:])
              for r in report if not r.ok and r.stage in ("validate", "bench", "timeout", "crash")]
    assert not failed, failed
    one_tile = [r for r in report if r.role == "combine" and _case(r.plan) == "one tile"]
    assert one_tile and all(r.ok for r in one_tile), [
        (r.label, r.error_type, str(r.error)[-300:]) for r in one_tile if not r.ok]
    for factor in factors:
        variant = f"split-k={factor}"
        ok = [r for r in report if r.ok and r.variant == variant]
        served = {(r.role, _case(r.plan)) for r in ok}
        assert {("partial", "looped"), ("combine", "looped"), ("combine", "one tile")} <= served
        for r in ok:
            roles = output_roles(r.plan.tile.flat_graph)
            assert roles == (("shift", "", "") if r.role == "partial" else ("",))
            assert len(r.measured) == len(roles)
            for role, measured in zip(roles, r.measured):
                assert measured.rel <= TOLERANCE[role].rtol, (r.label, role, measured)
        # the loops over the splits are plain loops: they weight the partials they load
        stepped = [r for r in ok if r.role == "combine" and _steps(r.plan) == [2]
                   and list(r.plan.tile.output_tile)[1] in (8, 16, 64)]
        assert stepped and all("for i0 in T.serial(" in r.source for r in stepped)

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
        o = torch.randn((heads, factor, 64, 64), generator=gen)
        l = torch.rand((heads, factor, 64), generator=gen) + 0.5
        m = torch.randn((heads, factor, 64), generator=gen) * 100.0
        l[:, 1], o[:, 1], m[:, 1] = 0.0, 0.0, NEG_FLT_MAX
        by_index = {0: o, 1: l, 2: m}
        feed = {p.name: by_index[p.boundary.index].cuda() for p in placeholder_order(graph)}
        out = compiled(*order_inputs_for_kernel(feed, compiled))
        assert torch.isfinite(out).all()
        compare_outputs([out], execute(graph, list(feed.values())), TOLERANCE[""],
                        what=_what("large_logits"))
        # ... the same answer as without that split
        keep = [i for i in range(factor) if i != 1]
        ok_, lk, mk = (by_index[i][:, keep].float() for i in (0, 1, 2))
        w = torch.exp(mk - mk.amax(1, keepdim=True))
        want = (w[..., None] * ok_).sum(1) / (w * lk).sum(1)[..., None]
        compare_outputs([out], [want.to(torch.float16).cuda()], TOLERANCE[""],
                        what=_what("large_logits"))


def test_gpu_a_combine_loop_over_the_splits_is_right_on_every_launch(cache_root):
    """Two partials of 64 rows a step, 8 partials (under the hand-derived pass, while such a
    loop was pipelined its value partials were reloaded before their last use, a different
    count of them on every launch). As a plain loop every launch gives the same, right,
    values."""
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
    assert "for i0 in T.serial(" in result.source
    graph = result.plan.tile.flat_graph
    compiled = result.kernel.compile()
    from cleave.runtime.reference import build_inputs_and_reference

    inputs_by_name, (reference,), _ = build_inputs_and_reference(graph, "float16")
    inputs = order_inputs_for_kernel(inputs_by_name, compiled)
    first = compiled(*inputs).clone()
    compare_outputs([first], [reference], TOLERANCE[""], what="first launch")
    for _ in range(50):
        assert torch.equal(compiled(*inputs), first)
