"""What a tensor-core matmul multiplies is declared on the node and said in the report (CPU).

A matmul is a tensor-core instruction, and the instruction decides what it reads of an
operand. For a float32 program TileLang issues the TF32 instruction: the operands stay
float32 buffers, nothing is cast to half precision, and each operand is read with the
leading 10 fraction bits of its significand -- the other 13 are dropped inside the
instruction. The sums are float32. Such a kernel is right to about 1e-3 of the row
maximum against the float32 product of its inputs (1.05e-3 measured on a GPU for a
512 x 1024 x 512 matmul, and 4.3e-6 once the inputs have no bits to drop), where a
float32 output would otherwise be held to 2e-5.

Nothing said so: not the kernel, not the report, not the error of a compile whose every
candidate missed a float32 bound. ``Matmul.tensor_core_format`` declares the format, and
the report and the compiled ``Kernel`` carry it wherever it is not the program's dtype.
"""

import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave.core.errors import CleaveError, CompileError, OutputMismatch
from cleave.frontend import trace
from cleave.ir.ops import Matmul
from cleave.pipeline.plans import lower
from cleave.pipeline.report import Report, Result, matmul_operands
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate_code
from cleave.runtime.evaluate import CandidateResult
from cleave.runtime.validate import MeasuredError, Tolerance
from cleave.schedule import search_tiling

from tests._lowering import lower_to_cuda

SIZES = dict(M=512, K=1024, N=512)
TF32_NOTE = ("matmul operands multiplied as tf32 (11 significant bits): the result is not the "
             "product at the program's dtype")


def matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


def _lowered(dtype):
    (graph,) = list(GraphSet.identity(trace(matmul, SIZES, dtype)))
    (ir,) = lower(graph)
    return ir


@pytest.mark.parametrize("dtype, fmt, bits", [("float16", "float16", 11),
                                              ("bfloat16", "bfloat16", 8),
                                              ("float32", "tf32", 11)])
def test_a_matmul_declares_the_format_it_multiplies_in(dtype, fmt, bits):
    node = Matmul((64, 32, 32), dtype=dtype, out_dtype="float32")
    assert node.tensor_core_format == fmt
    assert Matmul.SIGNIFICANT_BITS[fmt] == bits


def test_a_float32_program_keeps_float32_operands_and_multiplies_them_as_tf32():
    """What the code does, end to end: the lowered matmul reads float32 operands, the
    kernel allocates float32 buffers only, and the instruction TileLang issues for it is
    the TF32 one. (A float16 program gets the warp-group instruction on float16.)"""
    ir = _lowered(cl.f32)
    (node,) = [n for n in ir.nodes if isinstance(n, Matmul)]
    assert (node.dtype, node.out_dtype, node.tensor_core_format) == ("float32", "float32", "tf32")
    assert [str(e.src_node.get_dtype(e.src_id)) for e in node.inputs] == ["float32", "float32"]

    code = build_candidate_code(search_tiling(ir, topk=4)[0])
    assert set(re.findall(r"T\.alloc_\w+\([^\"]*\"(\w+)\"", code)) == {"float32"}
    instructions = set(re.findall(r"tl::(mma_sync|wgmma_\w+)<([^,]+), ([^,]+), ([^,]+),",
                                  lower_to_cuda(code)))
    assert instructions == {("mma_sync", "tl::DataType::kTensorFloat32",
                             "tl::DataType::kTensorFloat32", "tl::DataType::kFloat32")}

    half = build_candidate_code(search_tiling(_lowered(cl.f16), topk=4)[0])
    instructions = set(re.findall(r"tl::(mma_sync|wgmma_\w+)<([^,]+), ([^,]+), ([^,]+),",
                                  lower_to_cuda(half)))
    assert instructions == {("wgmma_ss", "tl::DataType::kFloat16", "tl::DataType::kFloat16",
                             "tl::DataType::kFloat32")}


def test_only_a_reduced_format_is_reported():
    assert matmul_operands(_lowered(cl.f32)) == (("tf32", 11),)
    assert matmul_operands(_lowered(cl.f16)) == ()
    assert matmul_operands(None) == ()


def _result(i, graph, *, error=None, stage=None, measured=None, latency_ms=None):
    plan = SimpleNamespace(label=f"p{i}", variant="solo", role="main",
                           tile=SimpleNamespace(flat_graph=graph))
    run = CandidateResult(None, "failed" if error else "ok", stage, error, latency_ms=latency_ms,
                          measured=measured)
    return Result(plan, run)


def _m(rel):
    return (MeasuredError(rel, 0.0),)


def test_the_report_says_what_a_float32_matmul_multiplied():
    ir = _lowered(cl.f32)
    report = Report([_result(0, ir, measured=_m(1.05e-3), latency_ms=1.0),
                     _result(1, ir, measured=_m(1.05e-3), latency_ms=2.0)],
                    tolerance=Tolerance(rtol=5e-3))
    assert report.matmul_operands() == (("tf32", 11),)
    (validate,) = [text for stage, text, _ in report.stage_lines() if stage == "validate"]
    assert validate.endswith(TF32_NOTE)
    validation = json.loads(json.dumps(report.to_dict()))["validation"]
    assert validation["matmul_operands"] == [{"format": "tf32", "significant_bits": 11}]


def test_a_report_of_16_bit_matmuls_says_nothing_about_them():
    ir = _lowered(cl.f16)
    report = Report([_result(0, ir, measured=_m(7e-4), latency_ms=1.0)],
                    tolerance=Tolerance(rtol=5e-3))
    assert report.matmul_operands() == ()
    assert "matmul operands" not in report.summary()
    assert "matmul_operands" not in report.to_dict()["validation"]


def test_a_compile_that_missed_its_bound_names_the_format():
    """Every kernel of a float32 matmul measures about 1e-3. Held to a float32 bound they
    all fail as mismatches, and the error of the compile used to stop at the count."""
    ir = _lowered(cl.f32)
    mismatch = [_result(i, ir, error=OutputMismatch("x"), stage="validate", measured=_m(1.05e-3))
                for i in range(3)]
    with pytest.raises(CleaveError) as failed:
        Report(mismatch, tolerance=Tolerance(rtol=1e-5)).best()
    assert str(failed.value) == (
        "no kernel compiled, validated and timed (3 candidates; failures: 3x OutputMismatch; "
        + TF32_NOTE + ")")
    # nothing was compared: the format is not what failed
    crashed = [_result(0, ir, error=CompileError("x"), stage="compile")]
    with pytest.raises(CleaveError) as failed:
        Report(crashed, tolerance=Tolerance(rtol=1e-5)).best()
    assert "tf32" not in str(failed.value)
    # 16-bit operands: a mismatch is a mismatch
    half = [_result(0, _lowered(cl.f16), error=OutputMismatch("x"), stage="validate",
                    measured=_m(0.5))]
    with pytest.raises(CleaveError) as failed:
        Report(half, tolerance=Tolerance(rtol=5e-3)).best()
    assert "matmul operands" not in str(failed.value)
