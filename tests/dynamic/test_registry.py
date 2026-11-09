import os
import subprocess
import sys
from pathlib import Path

import pytest

from cleave.dynamic import OPERATORS, DynamicOperator
from cleave.dynamic.dense import (
    KV_PROXY,
    flash_attention_sd_graph,
    mla_attention_sd_graph,
    tiled_dense_attention,
)
from cleave.dynamic.workloads import WORKLOADS
from cleave.dynamic.paging import MLASpec, PagingRefusal, PagingSpec, apply_paging, can_page
from cleave.schedule import apply_tiling, search_tiling

REPO = Path(__file__).resolve().parents[2]


def test_registry_imports_without_tests_on_path(tmp_path):
    keep = [p for p in sys.path if p and Path(p).resolve() != REPO]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(keep)}
    script = (
        "import sys, cleave.dynamic as d;"
        "assert not [m for m in sys.modules if m == 'tests' or m.startswith('tests.')];"
        "assert d.EXPECTED_KERNELS_DIR.is_dir();"
        "print(len(d.OPERATORS))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip().splitlines()[-1] == str(len(OPERATORS))


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_every_operator_is_complete(name):
    op = OPERATORS[name]
    assert isinstance(op, DynamicOperator) and op.name == name
    op.spec.validate()
    assert op.workloads == WORKLOADS[name] and op.workloads
    assert callable(op.reference)
    assert op.expected_kernel_path.is_file()
    assert op.tolerance.rtol > 0
    if op.is_mla:
        assert op.scale is not None


def _dense(spec, rows):
    if isinstance(spec, MLASpec):
        return mla_attention_sd_graph(rows, KV_PROXY, 512, 64, dtype="bfloat16", out_dtype="bfloat16")
    return flash_attention_sd_graph(rows, KV_PROXY, 128, dtype="bfloat16", out_dtype="bfloat16")


def _pinned(spec):
    """A grid of pinned tilings, which reaches the tiling gates rather than the recognizer."""
    out = []
    for rows in (16, 32, 64):
        for kv_tile in (32, 64):
            for threads in (128, 256):
                if isinstance(spec, MLASpec):
                    steps = {"mma_0": {"v2": 512}, "mma_pe": {"v2": 64},
                             "mma_1": {"v2": kv_tile}, "rowsum_0": {"v1": kv_tile}}
                    width = 512
                else:
                    steps, width = None, 128
                try:
                    out.append(tiled_dense_attention(
                        _dense(spec, rows), rows, width, kv_tile, steps=steps, n_threads=threads
                    ))
                except Exception:  # noqa: BLE001 - not a paging question
                    continue
    return out


def _searched(spec, *, keep=8):
    """Up to ``keep`` searched tilings ``can_page`` accepts and ``keep`` it refuses."""
    rows = 16 if isinstance(spec, MLASpec) or "decode" in spec.kind else 64
    yes, no = [], []
    for cand in search_tiling(_dense(spec, rows), topk=1000):
        try:
            graph = apply_tiling(cand)
        except Exception:  # noqa: BLE001
            continue
        (yes if can_page(graph, spec) else no).append(graph)
    return yes[:keep] + no[:: max(1, len(no) // keep)][:keep]


@pytest.mark.parametrize(
    "name",
    [
        "gqa_paged_decode_h32_kv8_d128_ps1",
        "gqa_paged_decode_h32_kv8_d128_ps64",
        "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
        "gqa_ragged_prefill_causal_h32_kv8_d128",
        "mla_paged_decode_h16_ckv512_kpe64_ps1",
        "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
    ],
)
def test_can_page_agrees_with_apply_paging(name):
    spec = OPERATORS[name].spec
    verdicts = []
    for graph in _pinned(spec) + _searched(spec):
        predicted = can_page(graph, spec)
        try:
            apply_paging(graph, spec)
            built = True
        except PagingRefusal:
            built = False
        assert predicted == built
        verdicts.append(built)
    # Both answers occur, so the agreement is not vacuous.
    assert any(verdicts) and not all(verdicts), f"{sum(verdicts)}/{len(verdicts)}"


def test_refusals_are_typed_assertion_errors():
    with pytest.raises(PagingRefusal) as info:
        PagingSpec(num_qo_heads=32, num_kv_heads=1).validate()
    assert isinstance(info.value, AssertionError)
