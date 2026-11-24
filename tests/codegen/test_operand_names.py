"""A gemm's shared operands keep their own names, so each gets the alignment its swizzle needs.

A TileLang macro that rebinds an argument (``A_ptr = A``) renames the caller's buffer, and
TileLang 0.1.13 keys the shared-memory alignment its op lowering records
(``tl.smem_alignment_map``) by buffer name, then treats that entry as exact
(``merge_shared_memory_allocations.cc``). When every gemm's A operand was renamed ``A_ptr``,
the 512 bytes a TMA-loaded A operand with a 64B swizzle recorded were applied to an operand
staged from registers with a 128B swizzle, which needs 1024 (the wgmma descriptor assumes a
1024-aligned base). Behind a few small staged vectors it landed at 512 mod 1024, and the
stmatrix writes and the wgmma reads disagreed on the swizzle: dK and dV of the attention
backward on 256 threads came out wrong for ten tilings, which the GPU validation caught.

CPU only: the kernels are lowered for an offline sm_90a target and the offset of every
buffer behind a 128B-swizzle wgmma descriptor is read off the CUDA source.
"""

import re

import pytest

pytest.importorskip("tilelang")

from cleave.ir.ops import Matmul
from cleave.pipeline import GraphSet, ScheduleConfig, schedule
from cleave.runtime.compile import build_candidate_code
from tests._lowering import lower_to_cuda
from tests.lower import _attention_bwd as A

REF = dict(H=32, SQ=4096, SKV=4096, D=64)
_OFFSET = re.compile(r"void\* ([A-Za-z0-9_]+) = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\)")
_DESCRIPTOR = re.compile(
    r"initialize_wgmma_descriptor<(\d+), \d+, \d+>\([a-z_0-9]+, \(&\(\(\([a-z_0-9]+\*\)([A-Za-z_0-9]+)\)")


def swizzled_128b_offsets(cuda: str) -> dict:
    """``{buffer: offset in dynamic shared memory}`` of every buffer a 128B-swizzle wgmma
    descriptor (layout type 1) points at."""
    offsets = {name: int(off) for name, off in _OFFSET.findall(cuda)}
    return {buf: offsets[buf] for layout, buf in _DESCRIPTOR.findall(cuda)
            if layout == "1" and buf in offsets}


@pytest.mark.parametrize("scopes", ["ssr", "rsr", "srr", "rrr"])
def test_a_gemm_reads_a_shared_or_register_operand_under_its_own_name(scopes):
    mma = Matmul((64, 64, 64), dtype="float16", out_dtype="float32", name="mma_0",
                 transpose_A=True)
    code = mma.gen_macro(mem_scope=scopes, clear=False)
    assert "A_ptr" not in code and "B_ptr" not in code, code
    assert re.search(r"T\.gemm\(\s*A,\s*B,\s*C_ptr,", code), code


def test_a_gemm_stages_a_global_operand_in_a_buffer_of_its_own():
    mma = Matmul((64, 64, 64), dtype="float16", out_dtype="float32", name="mma_0")
    code = mma.gen_macro(mem_scope="ggr", clear=True)
    assert "A_ptr = T.alloc_shared" in code and "T.copy(A, A_ptr)" in code
    assert re.search(r"T\.gemm\(\s*A_ptr,\s*B_ptr,\s*C_ptr,", code), code


def _wide(name):
    """The plans of ``name`` with a [1, 64, 64] output tile on 256 threads, one per step of
    the loop over the queries (dkdv, dkdv_o) or all of them (fused_o)."""
    plans = list(schedule(GraphSet.identity(A.program(name, REF)), ScheduleConfig(topk=1000)))
    wide = [p for p in plans if p.n_threads == 256 and list(p.output_tile) == [1, 64, 64]]
    if name == "fused_o":
        return wide
    by_step = {}
    for p in wide:
        step = [int(x) for v in p.tile.rstep_map.values() for x in v.values()][-1]
        by_step.setdefault(step, p)
    assert sorted(by_step) == [32, 64, 128], sorted(by_step)
    return list(by_step.values())


@pytest.mark.parametrize("name", ["dkdv", "dkdv_o", "fused_o"])
def test_every_128b_swizzled_gemm_operand_is_1024_aligned(name):
    """The tilings that validated wrong (dkdv and dkdv_o at every query step, fused_o's two
    with dK and dV accumulated in the block): every 128B-swizzle operand of their wgmmas
    starts on a 1024-byte boundary of shared memory."""
    plans = _wide(name)
    assert plans
    for plan in plans:
        code = build_candidate_code(plan.tile)
        offsets = swizzled_128b_offsets(lower_to_cuda(code))
        assert offsets, plan.label
        misaligned = {b: off for b, off in offsets.items() if off % 1024}
        assert not misaligned, f"{plan.label}: {misaligned}"
