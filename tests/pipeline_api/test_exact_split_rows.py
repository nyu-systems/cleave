"""The exact softmax of an input across a split built from the loop state, on a GPU.

``cl.softmax(x) @ v`` split over the softmax axis. Unlike attention, whose scores a kernel
computes, the logits here are an input: the partial reads them from storage, through the
view of the split, and so does the max it carries and returns. Three things are pinned:

* every tiling the policy proposes for the partial is scheduled, the ones that loop over
  the keys of a split carrying the max (in 1119 they were all refused once, as "handed one
  element of the axis it reduces per step", and again in this branch, as "the dim it
  reduces is not in the tile of its operand": the tile of the input was asked for the dim
  number of the split tensor), at the key steps the solo graph is offered;
* the partial is held to every adversarial softmax case, like the solo graph and the
  combine of the same program (its logits are still an input, read through the split);
* on a GPU: the partial alone and the combine alone on normal inputs and on every
  adversarial case, the chained ``Kernel`` against the Program's own reference, and a
  looped partial launched again and again. Its loop is pipelined and reads the slices of
  the logits it loads with ops that are not matmuls; every launch returns the same bits.

``test_search_*`` run the superoptimizer on CPU, ``test_gpu_*`` compile kernels (root
conftest marks).
"""

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.graph_utils import boundary_signature
from cleave.pipeline.plans import bind_inputs, lower
from cleave.pipeline.report import validation_kind
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate
from cleave.runtime.inputs import ADVERSARIAL, adversarial_cases, apply_distribution
from cleave.runtime.reference import (build_inputs_and_reference, order_inputs_for_kernel,
                                      output_order, output_roles, placeholder_order,
                                      reconcile_output_layout)
from cleave.runtime.validate import compare_outputs, tolerances_of
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.streaming import sites
from cleave.schedule.tiling import schedule_kernel
from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for

TOLERANCE = tolerance_for("exact_split_rows")  # by boundary role: "" and "shift"
EVERY_CASE = ("normal",) + ADVERSARIAL
SHAPES = dict(R=64, C=1024, D=64)
FACTORS = (4, 8)  # splits of 256 and of 128 keys
SPLIT = P.SuperoptConfig(split=("C",))  # searched unsplit, at the budget of no split


def rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    return cl.softmax(x, dim=-1) @ v


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")


def _case(plan):
    return "looped" if sites(apply_tiling(plan.tile)) else "one tile"


def _key_step(cand):
    (step,) = {int(x) for v in cand.rstep_map.values() for x in (v or {}).values()}
    return step


def _flash(prog):
    (graph,) = [g for g in P.superoptimize(prog, SPLIT) if g.summary().endswith("sum>matmul>div")]
    return graph


def _split_plans(prog):
    """Per factor: partial plans on 128 threads, two that loop over the keys of a split
    and two that take a split in one tile, and combine plans of both kinds."""
    graphs = P.superoptimize(prog, SPLIT)
    flash = [g for g in graphs if g.summary().endswith("sum>matmul>div")][0]
    plans = P.schedule(graphs, P.ScheduleConfig(topk=1000, split_k=FACTORS))
    assert not [r for r in plans.refused if r.graph_index == flash.index], plans.summary()
    keep = []
    for factor in FACTORS:
        mine = [p for p in plans if p.split_factor == factor]
        partials = [p for p in mine if p.role == "partial" and p.n_threads == 128]
        combines = [p for p in mine if p.role == "combine"]
        for case, pool in (("looped", partials), ("one tile", partials),
                           ("looped", combines), ("one tile", combines)):
            keep += [p for p in pool if _case(p) == case][:2]
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


def test_search_the_partial_loops_over_the_keys_of_its_split(cache_root):
    """CPU: the pair of the flash graph, at both factors. The partial returns its max last
    (role "shift") and the combine reads the same three tensors; both are held to every
    adversarial case. Every tiling of the partial is scheduled: the max carried where the
    keys of a split are looped (those kernels go through TileLang), the max in place where
    a step holds the split; the key steps are the solo graph's, up to a split's keys."""
    prog = trace(rows, SHAPES, cl.f16)
    graph = _flash(prog)
    (solo,) = lower(graph)
    solo_steps = sorted({_key_step(c) for c in search_tiling(solo, topk=1000)})
    assert solo_steps == [64, 128, 256]
    for factor in FACTORS:
        keys = SHAPES["C"] // factor
        partial, combine = lower(graph, factor)
        crossing = boundary_signature(output_order(partial))
        assert crossing == boundary_signature(placeholder_order(combine)) == (
            (0, "", (factor, SHAPES["R"], SHAPES["D"]), "float32"),
            (1, "", (factor, SHAPES["R"]), "float32"),
            (2, "shift", (factor, SHAPES["R"]), "float32"))
        assert output_roles(partial) == ("shift", "", "") and output_roles(combine) == ("",)
        assert bind_inputs(partial, prog) is not None
        assert adversarial_cases(partial) == ADVERSARIAL
        assert adversarial_cases(combine) == ADVERSARIAL

        served = Counter()
        lowered = 0
        for cand in search_tiling(partial, topk=1000):
            step = _key_step(cand)
            tiled = schedule_kernel(cand)  # none is refused
            found = sites(tiled)
            if step < keys:
                (site,) = found
                assert (site.step, site.block) == (step, keys)
                code, _ = build_candidate(cand)
                assert code.count("T.reduce_max(") == 1  # the carried max, returned
                assert "-3.4028234663852886e38" in code
                lower_to_cuda(code)
                lowered += 1
            else:
                assert not found  # a step holds the split: the flat graph's own max
            served[step] += 1
        assert sorted(served) == [s for s in solo_steps if s <= keys] and lowered

    plans = _split_plans(prog)
    for factor in FACTORS:
        cases = {(p.role, _case(p)) for p in plans if p.split_factor == factor}
        assert cases == {("partial", "looped"), ("partial", "one tile"),
                         ("combine", "looped"), ("combine", "one tile")}, (factor, cases)


def test_gpu_exact_split_rows(cache_root):
    _gpu()
    prog = trace(rows, SHAPES, cl.f16)
    plans = _split_plans(prog)
    report = P.evaluate(plans, P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                                            tolerance=TOLERANCE))

    # Each stage alone, against the reference of its own graph, on every input case: the
    # partial as well. No kernel that compiled is wrong, on any of them.
    assert all(r.cases == EVERY_CASE and not r.cases_note for r in report)
    failed = [(r.label, r.error_type, validation_kind(r.error), str(r.error)[-300:])
              for r in report if not r.ok and r.stage in ("validate", "bench", "timeout", "crash")]
    assert not failed, failed
    for factor in FACTORS:
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

        # The chained Kernel against the Program, on normal inputs and on every
        # adversarial case: finite, and within the bound of the program's own outputs.
        kernel = report.best(variant)
        assert kernel.unrunnable is None and kernel.variant == variant
        assert [s.role for s in kernel.stages] == ["partial", "combine"]
        assert [p.dtype for p in kernel.kernels[0].outputs] == ["float32", "float32", "float32"]
        for case in EVERY_CASE:
            for seed in (1, 2):
                inputs = _program_inputs(prog, case, seed)
                reference = prog.reference(*inputs)
                assert torch.isfinite(reference[0]).all()
                (measured,) = compare_outputs([kernel(*inputs)], reference, TOLERANCE[""],
                                              what=_what(case, f"chained {variant}"))
                assert measured.rel <= TOLERANCE[""].rtol

    # A looped partial is a pipelined loop whose logits are slices it loads, read by the
    # max, the subtraction and the exp before the value matmul sees anything. Launched
    # again and again it returns the bits of its first launch, which are the reference's.
    looped = [r for r in report if r.ok and r.role == "partial" and _case(r.plan) == "looped"]
    assert looped and all("for i0 in T.Pipelined(" in r.source for r in looped)
    for result in looped[:2]:
        graph = result.plan.tile.flat_graph
        compiled = result.kernel.compile()
        roles = output_roles(graph)
        for case in ("normal", "increasing"):
            inputs_by_name, reference, _ = build_inputs_and_reference(graph, "float16",
                                                                      distribution=case)
            inputs = order_inputs_for_kernel(inputs_by_name, compiled)
            first = [t.clone() for t in compiled(*inputs)]
            reference = [reconcile_output_layout(ref, act.shape, layout) for ref, act, layout
                         in zip(reference, first, result.run.stored_layouts)]
            compare_outputs(first, reference, tolerances_of(TOLERANCE, roles),
                            what=_what(case, "first launch"), roles=roles)
            for _ in range(50):
                again = compiled(*inputs)
                assert all(torch.equal(a, b) for a, b in zip(again, first))
