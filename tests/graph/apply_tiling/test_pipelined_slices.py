"""A loop is pipelined only where every tensor tile it transfers fills whole 128-byte buffers.

``T.Pipelined`` lets TileLang load the slices of an iteration as asynchronous bulk
transfers into one shared-memory buffer per pipeline stage. It lays those buffers out
back to back, and a bulk transfer needs its destination aligned to 128 bytes: with a
slice of 64 bytes the buffer of the second stage starts 64 bytes into the first one's
block, and the kernel faults when it is launched (``CUDA error: misaligned address``; in
the kernel log one ``Xid 13 ... Misaligned Address`` line per multiprocessor, then the
channel of the process is torn down).

Seven tilings of the test fixtures were such kernels, all on blocks of 16 or 8 elements:
two partials of 16 or of 8 row sums per iteration of a split-KV combine (64 and 32
bytes), and two partials of a matmul split-K combine on tiles ``[1, 16]``, ``[2, 8]`` and
``[1, 8]``. The split decode demos launch sixteen of them per run. They lowered, so
nothing short of a launch found them, and each launch is a fault in the driver.

Such a loop is emitted as a plain loop now, like the loops TileLang cannot pipeline at
all (``schedule.tiling._pipeline_one_loop``): it computes the same thing and validates.

Every slice here is a tensor tile: a part of its tensor that is not one run of consecutive
elements. A slice that is one run is loaded as a flat range, which needs 16 bytes only, and
stays in its pipeline at any size (``test_split_combine_placement.py``:
``test_a_slice_that_is_one_run_of_its_tensor_keeps_the_pipeline`` and its GPU test).
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.runtime.compile import build_candidate_code

from tests._graphs import (
    batched_matmul_splitk_combine_graph,
    matmul_splitk_combine_graph,
    mha_split_combine_graph,
)
from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

MHA = (mha_split_combine_graph, (32, 4, 4096, 64))
MATMUL = (matmul_splitk_combine_graph, (4, 128, 128))
BATCHED = (batched_matmul_splitk_combine_graph, (2, 4, 128, 128))

# name -> (fixture, output tile, steps, bytes of each slice the loop transfers per iteration)
MISALIGNED = {
    # both sums in one loop: the value partials are wide, the row sums are not
    "mha_16_rows_one_loop": (MHA, [1, 16, 64], dict(value_combine_0=2, rowsum_combine_0=2),
                             (64, 4096)),
    "mha_8_rows_one_loop": (MHA, [1, 8, 64], dict(value_combine_0=2, rowsum_combine_0=2),
                            (32, 2048)),
    # only the row sum looped
    "mha_16_rows_rowsum": (MHA, [1, 16, 64], dict(value_combine_0=4, rowsum_combine_0=2), (64,)),
    "mha_8_rows_rowsum": (MHA, [1, 8, 64], dict(value_combine_0=4, rowsum_combine_0=2), (32,)),
    "matmul_1x16": (MATMUL, [1, 16], dict(reduction_0=2), (64,)),
    "matmul_2x8": (MATMUL, [2, 8], dict(reduction_0=2), (64,)),
    "matmul_1x8": (MATMUL, [1, 8], dict(reduction_0=2), (32,)),
    # the same three with a batch axis in front
    "batched_1x16": (BATCHED, [1, 1, 16], dict(reduction_0=2), (64,)),
    "batched_2x8": (BATCHED, [1, 2, 8], dict(reduction_0=2), (64,)),
    "batched_1x8": (BATCHED, [1, 1, 8], dict(reduction_0=2), (32,)),
}
# The neighbours: every slice is a multiple of 128 bytes, and the loop keeps its pipeline.
ALIGNED = {
    "mha_32_rows_one_loop": (MHA, [1, 32, 64], dict(value_combine_0=2, rowsum_combine_0=2),
                             (128, 8192)),
    "mha_32_rows_rowsum": (MHA, [1, 32, 64], dict(value_combine_0=4, rowsum_combine_0=2), (128,)),
    "mha_128_rows_one_loop": (MHA, [1, 128, 64], dict(value_combine_0=2, rowsum_combine_0=2),
                              (512, 32768)),
    "matmul_1x32": (MATMUL, [1, 32], dict(reduction_0=2), (128,)),
    "matmul_2x16": (MATMUL, [2, 16], dict(reduction_0=2), (128,)),
    "matmul_4x8": (MATMUL, [4, 8], dict(reduction_0=2), (128,)),
    "batched_2x32": (BATCHED, [1, 2, 32], dict(reduction_0=2), (256,)),
    # four row sums a partial are under 16 bytes a row and are not transferred at all; the
    # value partials next to them are, in whole buffers
    "mha_4_rows_one_loop": (MHA, [1, 4, 64], dict(value_combine_0=2, rowsum_combine_0=2),
                            (1024,)),
}


def _candidate(name):
    (build, dims), tile, steps, _ = {**MISALIGNED, **ALIGNED}[name]
    graph = build(*dims)
    rstep = {
        node.name: {node.raxis[0].var.name: steps[node.name]} if node.name in steps else {}
        for node in graph.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    return build_tiled_candidate(graph, tile, rstep)


def _loop_kinds(code):
    return re.findall(r"for i0 in T\.(\w+)\(", code)


def _transferred(name):
    """Bytes per iteration of each slice of the candidate's loop that TileLang transfers:
    the input tiles the loop cuts, 16 bytes a row or wider."""
    from cleave.ir.graph import GraphNode, PlaceHolderNode
    from cleave.schedule import apply_tiling

    scheduled = apply_tiling(_candidate(name))
    (loop,) = [n for n in scheduled.nodes if isinstance(n, GraphNode)]
    sizes = []
    for node in loop.graph.nodes:
        if not isinstance(node, PlaceHolderNode):
            continue
        it = loop.graph.tmap[node]
        if tuple(it.tile_size) == tuple(it.shape):
            continue  # handed down whole
        element = int(node.get_dtype(0).bits) // 8
        if int(it.tile_size[-1]) * element >= 16:
            size = element
            for extent in it.tile_size:
                size *= int(extent)
            sizes.append(size)
    return tuple(sorted(sizes))


@pytest.mark.parametrize("name", sorted(MISALIGNED))
def test_a_loop_that_would_transfer_part_of_a_buffer_is_not_pipelined(name):
    assert _transferred(name) == MISALIGNED[name][3]
    assert any(size % 128 for size in MISALIGNED[name][3])
    code = build_candidate_code(_candidate(name))
    assert _loop_kinds(code) == ["serial"]


@pytest.mark.parametrize("name", sorted(ALIGNED))
def test_a_loop_that_transfers_whole_buffers_keeps_its_pipeline(name):
    assert _transferred(name) == ALIGNED[name][3]
    assert not any(size % 128 for size in ALIGNED[name][3])
    code = build_candidate_code(_candidate(name))
    assert _loop_kinds(code) == ["Pipelined"]


@pytest.mark.parametrize("name", ["mha_16_rows_one_loop", "mha_8_rows_rowsum", "matmul_2x8",
                                  "batched_1x16"])
def test_the_plain_loop_lowers(name):
    assert "main_kernel" in lower_to_cuda(build_candidate_code(_candidate(name)))


@pytest.mark.parametrize("name", sorted(MISALIGNED))
def test_gpu_the_plain_loop_validates(name):
    """GPU: the kernels that faulted at launch with a misaligned address."""
    (build, _), _, _, _ = MISALIGNED[name]
    assert_candidate_compiles_and_runs(_candidate(name), tolerance_for(build))
