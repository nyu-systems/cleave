"""GPU: kernels with an output summed across CTAs (``T.atomic_add`` into fp32).

The runtime zeroes the fp32 buffer before every launch and casts it to the output dtype
after (``runtime.compile.InterOutputKernel``). Atomics make the result order-dependent, so
nothing here compares two runs bit for bit: every result is held to the reference with
the operator's declared tolerance.
"""

import pytest

from cleave import pipeline as P
from cleave.core.settings import Settings
from cleave.pipeline import GraphSet
from cleave.runtime.compile import (
    InterOutputKernel,
    _compile,
    _load_main_kernel_from_code,
    build_candidate_code,
)
from cleave.runtime.reference import graph_out_indices, order_inputs_for_kernel, placeholder_order
from cleave.runtime.timer import mean_launch_ms
from cleave.runtime.validate import compare_outputs
from tests._tolerances import tolerance_for
from tests.codegen.test_inter_codegen import candidates, contexts, inter_pair

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

TOLERANCE = tolerance_for("inter_pair")


@pytest.fixture(scope="module")
def program():
    return inter_pair()


@pytest.fixture(scope="module")
def inputs(program):
    return program.random_inputs(seed=Settings.from_env().test_seed or 0)


@pytest.fixture(scope="module")
def reference(program, inputs):
    return program.reference(*inputs)


@pytest.fixture(scope="module", params=[("intra", "inter"), ("inter", "intra")], ids="-".join)
def kernel(request, program):
    """The first candidate of each assignment that TileLang compiles, compiled as the
    pipeline compiles it. (Some tilings of either assignment hit TileLang's fragment
    layout conflicts, as they do in single-output kernels; that is not what this tests.)"""
    tried = []
    for cand in candidates(program):
        if contexts(cand) != request.param:
            continue
        code = build_candidate_code(cand)
        try:
            k = _compile(_load_main_kernel_from_code(code), graph_out_indices(cand.flat_graph))
        except Exception as e:  # noqa: BLE001 - a TileLang compile failure of this tiling
            tried.append(f"{type(e).__name__}")
            if len(tried) == 8:
                raise
            continue
        assert isinstance(k, InterOutputKernel)
        return k, cand
    raise AssertionError(f"no candidate of {request.param} compiled: {tried}")


def _args(program, inputs, kernel, cand):
    """Program inputs in the kernel's parameter order (placeholder k = program input k)."""
    by_position = {f"PlaceHolder_input_{i + 1}": x for i, x in enumerate(inputs)}
    feed = {ph.name: by_position[ph.name] for ph in placeholder_order(cand.flat_graph)}
    return order_inputs_for_kernel(feed, kernel)


def test_outputs_match_reference_in_their_dtype(program, inputs, reference, kernel):
    k, cand = kernel
    outs = k(*_args(program, inputs, k, cand))
    assert [o.dtype for o in outs] == [torch.float16, torch.float16]
    compare_outputs(outs, reference, TOLERANCE, what="inter_pair")


def test_every_launch_starts_from_zero(program, inputs, reference, kernel):
    # Timing replays one CUDA graph on the same buffers: an un-zeroed accumulator would
    # come back as a multiple of the result. Call, time (100 replays), call again.
    k, cand = kernel
    args = _args(program, inputs, k, cand)
    first = k(*args)
    assert mean_launch_ms(k, args, warmup=2, iters=100) > 0
    second = k(*args)
    for outs in (first, second):
        compare_outputs(outs, reference, TOLERANCE, what="inter_pair")


def test_graph_replays_zero_the_buffer(program, inputs, reference, kernel):
    # The outputs a captured call returns are rewritten by each replay: after 20
    # replays they still hold one launch's sums.
    k, cand = kernel
    args = _args(program, inputs, k, cand)
    k(*args)  # compile-time and lazy initialisation outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        k(*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        captured = k(*args)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    compare_outputs(captured, reference, TOLERANCE, what="inter_pair")


def test_pipeline_validates_and_returns_declared_dtypes(program, inputs, reference, monkeypatch, tmp_path):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    plans = P.schedule(GraphSet.identity(program), P.ScheduleConfig(topk=4))
    assert {p.label.split(" acc=")[1].split(" ")[0] for p in plans} <= {
        "[intra,inter]", "[inter,intra]"}
    report = P.evaluate(plans, P.EvalConfig(warmup=2, iters=20, max_parallel=1, tolerance=TOLERANCE))
    assert report.correct, report.summary()
    best = report.best()
    assert "INTER_OUTPUTS = {" in best.source and best.latency_us > 0
    outs = best(*inputs)
    assert [o.dtype for o in outs] == [torch.float16, torch.float16]
    compare_outputs(outs, reference, TOLERANCE, what="inter_pair")
