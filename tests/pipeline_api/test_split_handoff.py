"""CPU: the partial -> combine hand-off of a split Kernel.

A combine's inputs are the partial's outputs. Which is which comes from the boundary record
``to_ir`` writes (``Boundary``), not from the order of either node list; how the partial's
kernel stores each one comes from the layout its tiling recorded, which ``Kernel.__call__``
undoes before the combine reads it. Both used to be guessed (node order, guarded by element
count; reshape), which made every split-attention Kernel unrunnable and would have scrambled
it had it run.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from cleave.core.errors import CleaveError
from cleave.ir.graph import Boundary, OutputNode, PlaceHolderNode
from cleave.ir.graph_utils import boundary_order
from cleave.lower.process import to_bundles
from cleave.pipeline.build import BuiltKernel, built_kernels
from cleave.pipeline.kernel import Kernel, Param, boundary_of, load, stages_for
from cleave.pipeline.report import Report, Result
from cleave.runtime.evaluate import CandidateResult
from cleave.runtime.reference import (
    execute,
    make_inputs,
    output_order,
    placeholder_order,
    to_stored,
)

STRUCTURE = json.loads((Path(__file__).resolve().parents[1] / "integration" / "graph_structures"
                        / "gqa" / "before_split_gqa_graph_structure_valid_1.graph").read_text())
# The tensors of this fixture: Q (HQ, SQ, D), K and V (HKV, SKV, D), with symbols
# 4 = HKV, 8 = HQ (and SKV / split), 12 = SQ, 16 = SKV, 20 = D.
HQ, HKV, SQ, D = 8, 4, 64, 64
# What a looped partial stores: the split axis first, the row sum padded to rank 4.
ROWSUM, VALUE = ((1, 0, 2), 1), ((1, 0, 2, 3), 0)


def _lower(factor, dtype="float32"):
    symbols = {4: HKV, 8: HQ, 12: SQ, 16: factor * HQ, 20: D}
    ((partial, combine),) = to_bundles([STRUCTURE], symbol_map=symbols, split_factors=[factor],
                                       dtype=dtype, accum_dtype="float32", out_dtype=dtype)
    return partial, combine


def _plan(graph, role, label=None, variant="split-k=8"):
    inputs = None
    if role == "partial":  # the Program's inputs are the placeholders in declaration order
        phs = placeholder_order(graph)
        inputs = tuple(ph.name for ph in boundary_order(phs) or phs)
    return SimpleNamespace(tile=SimpleNamespace(flat_graph=graph), role=role, variant=variant,
                           label=label or f"g0 {variant} {role}", inputs=inputs, program=None,
                           graph_index=0)


def _run(layouts=None, latency_ms=1.0, code="def main_kernel(): pass\n"):
    return CandidateResult(None, "ok", code=code, latency_ms=latency_ms, stored_layouts=layouts)


def _shape(node):
    return tuple(int(s) for s in node.get_shape(0))


def _stages(partial, combine, layouts=(ROWSUM, VALUE)):
    return stages_for([(_plan(partial, "partial"), _run(layouts)),
                       (_plan(combine, "combine"), _run())], None)


@pytest.mark.parametrize("factor", [8, 4])  # 8 == HQ: both layouts have one shape
def test_combine_inputs_are_paired_by_boundary_record(factor):
    partial, combine = _lower(factor)
    outputs = output_order(partial)
    assert [_shape(o) for o in outputs] == [(HQ, factor, SQ), (HQ, factor, SQ, D)]
    # The combine reads the value partial first and the row sum second: its node list is
    # not in boundary order, and the old pairing went by that list.
    listed = [n for n in combine.nodes if isinstance(n, PlaceHolderNode)]
    assert [ph.boundary.index for ph in listed] == [1, 0]

    stages, problem = _stages(partial, combine)
    assert problem is None
    assert [s.role for s in stages] == ["partial", "combine"]
    assert [(p.source, p.index, p.layout) for p in stages[0].params] == [
        ("input", 0, None), ("input", 1, None), ("input", 2, None)]
    by_name = {p.name: p for p in stages[1].params}
    assert [p.name for p in stages[1].params] == [ph.name for ph in placeholder_order(combine)]
    for ph in listed:
        param = by_name[ph.name]
        fed_from = outputs[param.index]
        assert param.source == "stage" and fed_from.boundary == ph.boundary
        assert param.shape == _shape(ph) == _shape(fed_from)
        assert param.layout == (ROWSUM, VALUE)[param.index]


def test_a_combine_that_reads_other_tensors_is_refused():
    partial, combine = _lower(8)
    # another split factor: the shapes differ
    _, problem = _stages(partial, _lower(4)[1])
    assert "the combine reads" in problem and "(8, 4, 64)" in problem and "(8, 8, 64)" in problem
    # the same shapes in another dtype
    _, problem = _stages(partial, _lower(8, dtype="float16")[1])
    assert "float16" in problem and "float32" in problem
    # the same shape and dtype under another role: not the tensor the partial writes there
    renamed = combine.clone()
    for ph in placeholder_order(renamed):
        if ph.boundary.index == 0:
            ph.boundary = Boundary(0, "shift")
    _, problem = _stages(partial, renamed)
    assert "#0:shift" in problem
    # one tensor more on one side
    fewer = combine.clone()
    extra = [ph for ph in placeholder_order(fewer) if ph.boundary.index == 1][0]
    extra.boundary = Boundary(2)
    stages, problem = _stages(partial, fewer)
    assert problem is not None
    assert {p.index for p in stages[1].params} == {0, -1}  # nothing to feed input #2 from
    # a combine with no partial before it
    _, problem = stages_for([(_plan(combine, "combine"), _run())], None)
    assert "needs the partial stage" in problem


def test_graphs_without_records_keep_the_positional_pairing():
    """Hand-built graphs carry no records: they get the pairing there was before, by the
    combine's node order, guarded by element count."""
    partial, combine = (g.clone() for g in _lower(8))
    for graph in (partial, combine):
        for node in graph.nodes:
            if isinstance(node, (PlaceHolderNode, OutputNode)):
                node.boundary = None
    assert boundary_of(_plan(partial, "partial")) is None
    stages, problem = _stages(partial, combine, layouts=None)
    assert problem == "g0 split-k=8 combine: combine inputs do not line up with the partial kernel's outputs"
    assert [(p.index, p.layout) for p in stages[1].params] == [(0, None), (1, None)]


class _FakeCompiled:
    """Stands in for a compiled TileLang kernel: named input buffers, then outputs."""

    def __init__(self, input_names, n_out, fn):
        names = list(input_names) + [f"out{i}" for i in range(n_out)]
        params = list(range(len(names)))
        buffer_map = {p: SimpleNamespace(name=n) for p, n in zip(params, names)}
        self.adapter = SimpleNamespace(
            prim_func=SimpleNamespace(params=params, buffer_map=buffer_map),
            result_idx=params[len(input_names):])
        self.fn, self.seen = fn, []

    def __call__(self, *args):
        self.seen.append([tuple(a.shape) for a in args])
        return self.fn(*args)


def _chained(factor, layouts):
    """A split Kernel whose two stages are the graphs' own torch references: the partial
    returns its outputs as a kernel stores them, the combine reads logical tensors."""
    partial, combine = _lower(factor)
    stages, problem = _stages(partial, combine, layouts)
    assert problem is None
    phs = placeholder_order(partial)

    def run_partial(*args):
        return tuple(to_stored(out, layout).contiguous()
                     for out, layout in zip(execute(partial, list(args)), (ROWSUM, VALUE)))

    def run_combine(*args):
        for arg, ph in zip(args, placeholder_order(combine)):
            assert tuple(arg.shape) == _shape(ph) and arg.is_contiguous()
        return execute(combine, list(args))[0]

    kernel = Kernel(stages, name="split", input_names=["q", "k", "v"],
                    input_shapes=[_shape(ph) for ph in boundary_order(phs)],
                    output_shapes=[(HQ, SQ, D)], variant=f"split-k={factor}")
    kernel._compiled = [
        _FakeCompiled([ph.name for ph in phs], 2, run_partial),
        _FakeCompiled([ph.name for ph in placeholder_order(combine)], 1, run_combine),
    ]
    inputs = dict(zip([ph.name for ph in phs], make_inputs(partial, "float32", seed=0, device="cpu")))
    ordered = [inputs[ph.name] for ph in boundary_order(phs)]
    # what the pair computes when every boundary tensor reaches the input it belongs to
    logical = {o.boundary.index: t for o, t in zip(output_order(partial),
                                                   execute(partial, list(inputs.values())))}
    want = execute(combine, [logical[ph.boundary.index] for ph in placeholder_order(combine)])[0]
    return kernel, ordered, want


@pytest.mark.parametrize("factor", [8, 4])
def test_kernel_call_hands_the_partial_outputs_over_in_the_logical_layout(factor):
    kernel, inputs, want = _chained(factor, (ROWSUM, VALUE))
    got = kernel(*inputs)
    assert got.shape == (HQ, SQ, D) and torch.equal(got, want)
    assert kernel._compiled[0].seen == [[(HQ, SQ, D), (HKV, factor * HQ, D), (HKV, factor * HQ, D)]]
    assert sorted(kernel._compiled[1].seen[0]) == [(HQ, factor, SQ), (HQ, factor, SQ, D)]


def test_a_reshape_hand_off_would_scramble_the_partial_outputs():
    """Why the layout is carried: with split == HQ the stored and the logical shapes are
    equal, so without the record the combine gets the tensors transposed and sums the
    wrong axis. (This is what every split-first partial did before the record.)"""
    kernel, inputs, want = _chained(8, None)
    assert all(p.layout is None for p in kernel.stages[1].params)
    got = kernel(*inputs)
    assert got.shape == want.shape and not torch.allclose(got, want, rtol=1e-2, atol=1e-3)


def test_stage_layouts_survive_save_and_load(tmp_path):
    partial, combine = _lower(8)
    stages, _ = _stages(partial, combine)
    kernel = Kernel(stages, name="split", input_names=["q", "k", "v"],
                    input_shapes=[(HQ, SQ, D), (HKV, 64, D), (HKV, 64, D)],
                    output_shapes=[(HQ, SQ, D)], variant="split-k=8")
    saved = json.loads((kernel.save(tmp_path / "k") / "kernel.json").read_text())
    assert [q["layout"] for q in saved["stages"][1]["params"]] == [[[1, 0, 2, 3], 0], [[1, 0, 2], 1]]
    loaded = load(tmp_path / "k")
    assert [s.params for s in loaded.stages] == [s.params for s in kernel.stages]
    assert loaded.stages[1].params[0].layout == ((1, 0, 2, 3), 0)  # tuples again, not lists
    assert loaded.stages[0].params[0] == Param(stages[0].params[0].name, "input", 0, (HQ, SQ, D))


def test_built_kernel_reports_the_dims_the_kernel_declares():
    partial, _ = _lower(8)
    plan = _plan(partial, "partial")
    (built,) = built_kernels([plan], [_run((ROWSUM, VALUE))])
    assert [p.dims for p in built.outputs] == [(8, HQ, SQ, 1), (8, HQ, SQ, D)]
    assert [p.dims for p in built.inputs] == [_shape(ph) for ph in placeholder_order(partial)]
    # without a record the lowered graph's logical dims are all there is
    (logical,) = built_kernels([plan], [_run(None)])
    assert [p.dims for p in logical.outputs] == [(HQ, 8, SQ), (HQ, 8, SQ, D)]
    with pytest.raises(ValueError, match="1 stored layouts"):
        BuiltKernel.from_plan(plan, "src", stored_layouts=(ROWSUM,))


def _result(graph, role, latency_ms, label):
    return Result(_plan(graph, role, label), _run((ROWSUM, VALUE), latency_ms))


def test_variants_pair_a_partial_only_with_a_combine_of_its_own_boundary():
    """The combine of a factor is searched once and shared, so the fastest partial and the
    fastest combine can come from different cuts; each validates alone on random inputs."""
    partial, combine = _lower(8)
    other_partial, other_combine = _lower(8, dtype="float16")  # same shapes, another dtype

    report = Report([
        _result(other_partial, "partial", 1.0, "fast partial of another cut"),
        _result(partial, "partial", 3.0, "slow partial"),
        _result(partial, "partial", 2.0, "partial"),
        _result(combine, "combine", 0.5, "combine"),
        _result(combine, "combine", 0.7, "slow combine"),
    ])
    (variant,) = report.variants().values()
    assert [m.label for m in variant.members] == ["partial", "combine"]
    assert variant.latency_us == pytest.approx(2500.0)
    kernel = report.best("split-k=8")
    assert kernel.unrunnable == "split kernel without Program output shapes"  # no Program here
    assert [p.layout for p in kernel.stages[1].params] == [VALUE, ROWSUM]

    # both cuts complete: the faster pair wins as a whole
    both = Report(report.results + [_result(other_combine, "combine", 0.1, "other combine")])
    assert [m.label for m in both.variants()["split-k=8"].members] == [
        "fast partial of another cut", "other combine"]

    # nothing that fits together: no variant, and best() says so
    apart = Report([_result(other_partial, "partial", 1.0, "p"), _result(combine, "combine", 0.5, "c")])
    assert apart.variants() == {}
    with pytest.raises(CleaveError, match="no split-k=8 compiled, validated and timed"):
        apart.best("split-k=8")


def test_variants_without_records_pair_as_before():
    plans = [SimpleNamespace(label=f"p{i}", variant=v, role=r)
             for i, (v, r) in enumerate([("solo", "main"), ("solo", "main"),
                                         ("split-k=4", "partial"), ("split-k=4", "combine"),
                                         ("split-k=8", "partial")])]
    latencies = [2.0, 1.0, 0.4, 0.3, 0.1]
    report = Report([Result(p, _run(None, ms)) for p, ms in zip(plans, latencies)])
    variants = report.variants()
    assert list(variants) == ["solo", "split-k=4"]  # split-k=8 has no combine
    assert [m.label for m in variants["solo"].members] == ["p1"]
    assert [m.label for m in variants["split-k=4"].members] == ["p2", "p3"]
    assert variants["split-k=4"].latency_us == pytest.approx(700.0)
