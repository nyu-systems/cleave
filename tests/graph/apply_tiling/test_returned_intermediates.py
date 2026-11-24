"""A value that is returned and also read by another op is searched, built and validated.

``y = x @ w; return y, y @ v`` and ``y = x @ w; return y, y * 2`` had no tiling. The
policy's shared-memory model keeps a block for a value that an op reads and gives it back
once every reader has run -- and it counted the value's output node as a reader. An output
node is not one of the nodes the model walks, so it never ran, the block was never given
back, and the model's closing ``assert len(block_map) == 0`` ended the search of the whole
graph while it rated its first tile, with a bare ``AssertionError``. Only the pipeline's
schedule stage turned that into a refusal, and its reason was the exception's name:
``ScheduleRejected: 'AssertionError: '``.

A returned value is stored, not held: its output node is no reader the block waits for
(the same condition under which the model allocates a block in the first place). The
programs get the tilings of their single-output neighbours, ``(x @ w) @ v`` and ``x @ w``,
at the same modelled shared memory or more, and their kernels validate.
"""

import re
from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave.core.errors import CompileError, ScheduleRejected, SmemExceeded
from cleave.frontend import trace
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate
from cleave.schedule import search_tiling

from tests._tolerances import tolerance_for


def product_and_chain(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"], v: cl.Tensor["N", "P"]):
    """The first product is returned and is the second one's operand."""
    y = x @ w
    return y, y @ v


def product_and_double(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    """The product is returned next to an elementwise reader of it."""
    y = x @ w
    return y, y * 2.0


def chain(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"], v: cl.Tensor["N", "P"]):
    return (x @ w) @ v


def product(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


SIZES = dict(M=1024, K=256, N=64, P=64)
# program -> (its single-output neighbour, tilings, tilings apply_tiling builds). The chain
# has no 64-row tile over 256 threads: the second matmul reads its first operand from the
# first one's accumulator, which TileLang lays out for fewer than 16 rows a warp only one
# way (``TCPolicy._register_operand_layout_conflict``; 16 such tilings, 4 of which built and
# then failed to compile).
RETURNED = {
    "product_and_chain": (product_and_chain, chain, 80, 20),
    "product_and_double": (product_and_double, product, 64, 64),
}


def _ir(fn):
    sizes = {name: SIZES[name] for name in "MKNP"[: fn.__code__.co_argcount + 1]}
    (graph,) = list(GraphSet.identity(trace(fn, sizes, cl.f16)))
    (ir,) = lower(graph)
    return ir


def _key(cand):
    return (tuple(cand.output_tile), cand.n_threads,
            tuple(int(step) for steps in cand.rstep_map.values() for step in steps.values()))


def _built(fn):
    """The tilings of ``fn`` that ``apply_tiling`` builds, and why it refuses the others
    (the reason with its numbers left out)."""
    built, refused = [], Counter()
    for cand in search_tiling(_ir(fn), topk=1000):
        try:
            build_candidate(cand)
        except ScheduleRejected as refusal:
            refused[re.sub(r"\d+", "N", str(refusal))] += 1
        else:
            built.append(cand)
    return built, refused


@pytest.mark.parametrize("name", sorted(RETURNED))
def test_a_returned_value_with_another_reader_has_the_tilings_of_its_neighbour(name):
    """The search ended with ``AssertionError`` at the first tile it rated."""
    fn, alone, tilings, _ = RETURNED[name]
    returned = {_key(c): c for c in search_tiling(_ir(fn), topk=1000)}
    neighbour = {_key(c): c for c in search_tiling(_ir(alone), topk=1000)}
    assert len(returned) == tilings and set(returned) == set(neighbour)
    # The returned value costs no shared memory by being returned: where its other reader
    # is the second matmul it is held as the chain holds it, and next to an elementwise
    # reader it is held where the plain product is stored straight from its registers.
    extra = {int(returned[k].smem_cost - neighbour[k].smem_cost) for k in returned}
    assert min(extra) == 0
    assert (max(extra) == 0) == (name == "product_and_chain")


@pytest.mark.parametrize("name", sorted(RETURNED))
def test_the_pipeline_schedules_a_returned_value_with_another_reader(name):
    """The schedule stage had no plan and one refusal: "AssertionError: "."""
    from cleave import pipeline as P

    fn = RETURNED[name][0]
    sizes = {dim: SIZES[dim] for dim in "MKNP"[: fn.__code__.co_argcount + 1]}
    plans = P.schedule(GraphSet.identity(trace(fn, sizes, cl.f16)), P.ScheduleConfig(topk=8))
    assert len(list(plans)) == 8
    assert not any("AssertionError" in str(r.error) for r in plans.refused)


@pytest.mark.parametrize("name", sorted(RETURNED))
def test_the_tilings_that_are_not_built_are_typed_refusals(name):
    """Returning the first product is not free for the tiler: a tiling that loops the
    second matmul would read the returned value inside the loop and store it outside,
    and one whose grid no single launch axis indexes has no place for it. Both are
    refusals with their reason; nothing else stops a tiling."""
    fn, _, tilings, built = RETURNED[name]
    cands, refused = _built(fn)
    assert len(cands) == built and sum(refused.values()) == tilings - built
    assert refused == ({} if name == "product_and_double" else {
        "mma_N is computed per iteration of GraphNode_mma_N but also read by OutputNode "
        "outside it": 40,
        "Expected exactly one launch-axis mapping for iter dim N, got [N].": 20,
    })


def _validate(fn, tolerance):
    """GPU: every tiling of ``fn`` that is built, at the bound the program declares."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to compile and run candidates.")
    from cleave.runtime.evaluate import run_candidates

    cands, _ = _built(fn)
    results = run_candidates(cands, "float16", max_parallel=8, tolerance=tolerance)
    # A tiling that does not validate fails before its kernel runs, in one of the two ways
    # the single-output neighbours' tilings do: TileLang finds no layout (a 256-thread chain
    # on 64-row tiles did not, before the policy stopped proposing it), or the kernel asks
    # for more shared memory than the device has.
    failed = Counter(type(r.error) for r in results if not r.ok)
    assert set(failed) <= {CompileError, SmemExceeded}, [
        str(r.error)[:300] for r in results if not r.ok
        and type(r.error) not in (CompileError, SmemExceeded)]
    return sum(r.ok for r in results), failed, results


def test_gpu_the_returned_product_and_the_chain_validate():
    ok, failed, results = _validate(product_and_chain, tolerance_for("product_and_chain"))
    assert ok == 20 and not failed


def test_gpu_the_returned_product_and_its_double_validate():
    ok, failed, _ = _validate(product_and_double, tolerance_for("product_and_double"))
    assert ok == 58 and failed == {SmemExceeded: 6}
