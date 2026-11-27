"""A matmul never reads its first operand from a fragment on a block two warp groups tall.

TileLang 0.1.13 issues a Hopper tensor-core matmul per group of four warps, each
instruction computing 64 rows. With the first operand in a fragment, it lays that fragment
out with all the warps of the block side by side and repeats that for the next 64 rows of
each group, and the result with the rows of one group together. On 256 threads (two
groups) and tiles of 256 rows or more the two disagree on which rows a group holds, and
the instruction multiplies what its registers hold: a 256-row tile came out with rows
64..127 and 128..191 exchanged, a 512-row tile with six of its eight 64-row bands in the
wrong place. Nothing failed before validation: six tilings of the RMSNorm split-K
partial were built, lowered and launched, and were off by twice the row maximum; six of
SwiGLU the same, reported as a non-finite output because the one entry the float16
reference overflows on landed in another row.

The fragment is there because the operand has a second reader in the loop (``x`` under
``x @ w`` next to ``sum(x * x)``, or under two matmuls). Codegen now keeps the operand
of such a matmul in shared memory, where the gemm addresses the rows of each group
itself: an input that only matmuls read is loaded there in the first place, an input with
another reader is read from the shared buffer its slice was loaded through, and a computed
operand is staged.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.codegen import passes
from cleave.codegen.passes import register_operand_is_misplaced, wgmma_row_warps
from cleave.ir.graph import PlaceHolderNode
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import Matmul
from cleave.runtime.compile import CompilerPassOptions, build_candidate_code
from cleave.schedule.tiling import schedule_kernel

from tests._graphs import rmsnorm_matmul_split_partial_graph, swiglu_graph
from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

# The dims of the end2end cases, and the twelve tilings that were wrong at 256 threads:
# (output tile, step of both reductions).
RMSNORM_DIMS = (4, 1024, 6144, 1024)
SWIGLU_DIMS = (4096, 14336, 4096)
RMSNORM_WRONG = [([1, 256, 32], 32), ([1, 256, 64], 32), ([1, 256, 64], 64),
                 ([1, 256, 128], 32), ([1, 512, 32], 32), ([1, 512, 64], 32)]
SWIGLU_WRONG = [([256, 32], 32), ([256, 32], 64), ([256, 64], 32), ([256, 64], 64),
                ([512, 32], 32), ([512, 32], 64)]


def _candidate(build, dims, tile, step, n_threads):
    graph = build(*dims)
    rstep = {
        node.name: {node.raxis[0].var.name: step} if len(node.raxis) else {}
        for node in graph.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    cand = build_tiled_candidate(graph, tile, rstep)
    cand.n_threads = n_threads
    return cand


def _rmsnorm(tile, step, n_threads=256):
    return _candidate(rmsnorm_matmul_split_partial_graph, RMSNORM_DIMS, tile, step, n_threads)


def _swiglu(tile, step, n_threads=256):
    return _candidate(swiglu_graph, SWIGLU_DIMS, tile, step, n_threads)


def _annotated(cand):
    """The scheduled graph after the passes the build runs before it emits a kernel."""
    graph = schedule_kernel(cand)
    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tinfo(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    return passes.annotate_fragment_layout_conflicts(graph)


def _first_operands(graph):
    """``{matmul: (scope its first operand is loaded in, whether it is staged)}``."""
    found = {}
    for tier, _ in graph_tiers(graph):
        for node in tier.nodes:
            if isinstance(node, Matmul):
                scope = tier.tinfo_map[node.inputs[0].src_node][1]
                found[node.name] = (scope, 0 in node.stage_fragment_operands_in_shared)
    return found


@pytest.mark.parametrize("policy, n_threads, rows, row_warps", [
    ("rows", 128, 512, (4,)),       # one warp group: four warps, whatever the tile
    ("rows", 256, 64, (4,)),        # 64 rows hold four warps of 16 rows, two columns of them
    ("rows", 256, 128, (8,)),
    ("rows", 256, 512, (8,)),
    ("cols", 256, 512, (4,)),       # as few row warps as divide the block
    ("even", 256, 256, (4, 8)),     # chosen by a ratio: either may be it
    ("rows", 64, 256, ()),          # not whole warp groups: no warp-group gemm
    ("rows", 256, 32, ()),          # fewer rows than one instruction computes
])
def test_the_warps_tilelang_lays_along_the_rows(policy, n_threads, rows, row_warps):
    assert wgmma_row_warps(policy, rows, 32, n_threads) == row_warps


@pytest.mark.parametrize("policy, n_threads, rows, misplaced", [
    ("rows", 128, 256, False),   # one group
    ("rows", 128, 512, False),
    ("rows", 256, 64, False),
    ("rows", 256, 128, False),   # two groups, one instruction each
    ("rows", 256, 256, True),    # two groups, two instructions each
    ("rows", 256, 512, True),
    ("cols", 256, 512, False),   # the two groups sit side by side along the columns
    ("even", 256, 256, True),    # it may be laid out like "rows"
    ("even", 256, 128, False),
])
def test_which_tiles_tilelang_multiplies_on_the_wrong_rows(policy, n_threads, rows, misplaced):
    assert register_operand_is_misplaced(policy, rows, 32, n_threads) is misplaced


@pytest.mark.parametrize("tile, step", RMSNORM_WRONG)
def test_an_input_with_another_reader_is_staged_for_the_matmul(tile, step):
    """``x`` feeds the matmul and the sum of squares: the square keeps its fragment, and
    the matmul reads the slice in the shared buffer it was loaded through on its way to
    that fragment. (It used to get a copy of the fragment in a second shared buffer.)"""
    graph = _annotated(_rmsnorm(tile, step))
    assert _first_operands(graph) == {"mma_0": ("r", True)}
    code = build_candidate_code(_rmsnorm(tile, step))
    assert "T.copy(PlaceHolder_A, PlaceHolder_A_intermediate)" in code
    assert "T.copy(PlaceHolder_A_intermediate, PlaceHolder_AA)" in code
    assert "(PlaceHolder_A_intermediate, PlaceHolder_BB, t_mma_0)" in code
    assert "PlaceHolder_AA_intermediate" not in code
    # the loop keeps three shared buffers: the two operands and the staged result
    assert code.count("T.alloc_shared(") == 3


@pytest.mark.parametrize("tile, step", SWIGLU_WRONG)
def test_an_input_only_matmuls_read_is_loaded_into_shared_memory(tile, step):
    """``x`` feeds two matmuls and nothing else: there is no fragment to stage."""
    graph = _annotated(_swiglu(tile, step))
    assert _first_operands(graph) == {"mma_up": ("s", False), "mma_gate": ("s", False)}
    code = build_candidate_code(_swiglu(tile, step))
    assert "PlaceHolder_AA = T.alloc_shared(" in code
    assert "PlaceHolder_AA = T.alloc_fragment(" not in code
    assert "ukernel_mma_ssr(PlaceHolder_AA, " in code and "ukernel_mma_rsr" not in code


@pytest.mark.parametrize("make, tile, n_threads", [
    (_rmsnorm, [1, 256, 32], 128),   # one warp group
    (_rmsnorm, [1, 128, 32], 256),   # one instruction per group
    (_rmsnorm, [1, 64, 64], 256),
    (_swiglu, [512, 32], 128),
    (_swiglu, [128, 64], 256),
])
def test_a_fragment_tilelang_reads_right_is_left_alone(make, tile, n_threads):
    """Every other tiling keeps the kernel it had: the operand stays in its fragment."""
    graph = _annotated(make(tile, 32, n_threads))
    assert set(_first_operands(graph).values()) == {("r", False)}
    code = build_candidate_code(make(tile, 32, n_threads))
    assert "ukernel_mma_rsr(PlaceHolder_AA, " in code
    assert "PlaceHolder_AA_intermediate" not in code


def test_the_mla_pass_keeps_the_operand_out_of_the_fragment():
    """``annotate_mla_accumulation_layout_conflict`` resets every matmul's staging before
    it sets its own; the rule holds behind it too."""
    cand = _rmsnorm([1, 256, 32], 32)
    graph = passes.annotate_mla_accumulation_layout_conflict(_annotated(cand))
    assert _first_operands(graph) == {"mma_0": ("r", True)}
    code = build_candidate_code(cand, CompilerPassOptions(mla_decode_pass=True))
    assert "(PlaceHolder_A_intermediate, PlaceHolder_BB, t_mma_0)" in code


def test_a_main_tier_input_is_never_a_fragment():
    """Only a loop tier loads an input into a fragment; the pass leaves the rest alone."""
    graph = _annotated(_swiglu([256, 32], 32))
    for node, (_, scope) in graph.tinfo_map.items():
        if isinstance(node, PlaceHolderNode):
            assert scope == "g"


@pytest.mark.parametrize("make, tile, step", [(_rmsnorm, [1, 256, 32], 32),
                                              (_swiglu, [512, 32], 64)])
def test_the_kernel_lowers(make, tile, step):
    assert "main_kernel" in lower_to_cuda(build_candidate_code(make(tile, step)))


@pytest.mark.parametrize("tile, step", RMSNORM_WRONG)
def test_gpu_rmsnorm_partial_on_two_warp_groups_validates(tile, step):
    """GPU: the six kernels whose matmul output was off by twice the row maximum."""
    assert_candidate_compiles_and_runs(
        _rmsnorm(tile, step), tolerance_for(rmsnorm_matmul_split_partial_graph))


@pytest.mark.parametrize("tile, step", SWIGLU_WRONG)
def test_gpu_swiglu_on_two_warp_groups_validates(tile, step):
    """GPU: the six kernels that failed as a non-finite output. ``[512, 32]`` at step 64
    exceeded shared memory while its input was staged from a fragment."""
    assert_candidate_compiles_and_runs(_swiglu(tile, step), tolerance_for(swiglu_graph))
