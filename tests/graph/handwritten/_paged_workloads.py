from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from tests.graph.handwritten import _paged_reference as paged_reference

# Workload axes captured from NVIDIA SOL-ExecBench, one tuple per distinct line of
# flashinfer-trace's workloads/<family>/<operator>.jsonl. Head counts, head_dim and page
# size are fixed by the operator name, so only the varying axes appear here. ``num_pages``
# is the whole cache pool; a workload only reads the pages its index list names, and
# ``num_kv_indices`` is how many that is. Per-request lengths are NOT part of the capture
# -- the trace keeps them in separate blobs -- so they are synthesized as a deterministic
# partition of the captured totals.

DECODE_AXES = ("batch", "num_pages", "num_kv_indices")
PAGED_PREFILL_AXES = ("batch", "total_q", "num_kv_indices", "num_pages")
RAGGED_AXES = ("batch", "total_q", "total_kv")

WORKLOADS = {
    "gqa_paged_decode_h32_kv4_d128_ps1": (
        (1, 8, 7),
        (1, 10, 9),
        (1, 12, 11),
        (1, 15, 14),
        (1, 17, 2),
        (1, 57, 40),
        (1, 67, 50),
        (1, 81, 64),
        (1, 191, 141),
        (1, 302, 252),
        (1, 412, 362),
        (1, 486, 436),
        (1, 596, 546),
        (1, 9317, 72),
        (1, 9332, 87),
        (1, 9347, 102),
        (16, 1070, 1020),
        (16, 2158, 2108),
        (16, 3246, 3196),
        (16, 4334, 4284),
        (16, 5422, 5372),
        (16, 6510, 6460),
        (16, 7598, 7548),
        (16, 8686, 8636),
        (16, 24732, 15463),
        (16, 25820, 16551),
        (16, 26908, 17639),
        (16, 27996, 18727),
        (16, 29084, 19815),
        (16, 30172, 20903),
        (16, 31260, 21991),
        (16, 32348, 23079),
        (64, 28831, 28815),
        (64, 31007, 30991),
        (64, 33183, 33167),
        (64, 35359, 35343),
        (64, 37535, 37519),
        (64, 39711, 39695),
        (64, 41887, 41871),
        (64, 44063, 44047),
        (64, 46303, 46287),
        (64, 48479, 48463),
        (64, 50655, 50639),
        (64, 52831, 52815),
        (64, 55007, 54991),
        (64, 57183, 57167),
        (64, 59359, 59343),
        (64, 61535, 61519),
    ),
    "gqa_paged_decode_h32_kv8_d128_ps1": (
        (1, 11, 10),
        (1, 15, 14),
        (1, 18, 2),
        (1, 55, 38),
        (1, 59, 42),
        (1, 63, 46),
        (1, 67, 50),
        (1, 71, 54),
        (1, 74, 57),
        (1, 78, 61),
        (1, 82, 65),
        (1, 223, 173),
        (1, 406, 356),
        (1, 597, 547),
        (1, 9316, 73),
        (1, 9341, 98),
        (16, 1069, 1034),
        (16, 1220, 1193),
        (16, 1396, 1369),
        (16, 1556, 1529),
        (16, 1732, 1705),
        (16, 1892, 1865),
        (16, 2052, 2025),
        (16, 2212, 2185),
        (16, 2372, 2345),
        (16, 2548, 2521),
        (16, 2708, 2681),
        (16, 2868, 2841),
        (16, 3044, 3017),
        (16, 4333, 4298),
        (16, 30163, 20911),
        (16, 81390, 12942),
        (64, 60071, 50902),
        (64, 62605, 53334),
        (64, 63053, 53782),
        (64, 63437, 54166),
        (64, 63821, 54550),
        (64, 64205, 54934),
        (64, 64653, 55382),
        (64, 65037, 55766),
        (64, 65421, 56150),
        (64, 65805, 56534),
        (64, 66253, 56982),
        (64, 66637, 57366),
        (64, 67021, 57750),
        (64, 67405, 58134),
        (64, 67853, 58582),
        (64, 68237, 58966),
    ),
    "gqa_paged_decode_h32_kv8_d128_ps64": (
        (4, 349, 212),
        (4, 389, 232),
        (4, 398, 268),
        (8, 839, 528),
        (8, 902, 576),
        (8, 910, 640),
        (16, 1686, 1033),
        (16, 1735, 1145),
        (16, 1794, 1257),
        (31, 15233, 14145),
        (32, 15114, 13783),
        (32, 15178, 14103),
        (63, 29569, 27392),
        (64, 29189, 26582),
        (64, 29441, 27222),
        (127, 52481, 48120),
        (128, 51792, 46414),
        (128, 52033, 47438),
    ),
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": (
        (1, 1, 33, 50),
        (1, 2, 2, 15615),
        (1, 5, 1, 683),
        (1, 13, 2, 62875),
        (1, 16, 2, 86),
        (1, 34, 2, 0),
        (1, 42, 2, 15898),
        (1, 70, 2, 0),
        (1, 178, 2, 61699),
        (1, 398, 2, 16732),
        (1, 1200, 3, 444449),
        (1, 1210, 1, 176707),
        (1, 16384, 2, 27915),
        (2, 14, 2, 184865),
        (3, 80, 4, 15898),
        (4, 126, 11, 138584),
        (8, 10143, 37, 553157),
        (15, 14823, 16, 9254),
        (18, 7938, 16403, 291614),
        (22, 5404, 28, 224501),
        (26, 4061, 47, 401913),
        (27, 15344, 48, 556622),
        (28, 12383, 1, 85),
        (28, 15783, 25, 62875),
        (35, 9145, 39, 114143),
        (42, 15910, 55, 224323),
        (51, 16189, 57, 274149),
        (61, 9823, 133, 403995),
        (62, 11763, 148, 612942),
        (63, 8342, 88, 181724),
        (63, 9277, 117, 514213),
        (63, 13108, 94, 139067),
        (63, 14263, 142, 626709),
    ),
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": (
        (1, 1, 34, 51),
        (1, 2, 3, 13469),
        (1, 5, 2, 682),
        (1, 16, 3, 87),
        (1, 17, 2, 552310),
        (1, 30, 1, 2),
        (1, 34, 1, 2),
        (1, 42, 3, 13719),
        (1, 178, 3, 59402),
        (1, 398, 3, 14547),
        (1, 1171, 4, 439455),
        (1, 2473, 2, 176551),
        (1, 10447, 1, 2),
        (1, 16384, 3, 25634),
        (2, 14, 4, 181871),
        (2, 78, 4, 13719),
        (3, 123, 12, 135698),
        (3, 202, 7, 174706),
        (4, 70, 13, 222053),
        (4, 134, 10, 383616),
        (7, 7419, 25, 270692),
        (7, 10393, 17, 395233),
        (10, 4950, 25, 296392),
        (11, 8981, 16, 20585),
        (12, 1286, 35, 178800),
        (12, 7009, 30, 70149),
        (16, 6102, 41, 256226),
        (17, 3478, 2560, 716370),
        (20, 2002, 67, 508530),
        (26, 7786, 74, 397053),
        (28, 13515, 28, 2),
        (30, 4803, 66, 111721),
        (32, 12734, 33, 86),
        (38, 7140, 97, 399096),
        (40, 8140, 104, 136181),
        (41, 8298, 143, 605498),
        (43, 6558, 101, 175078),
        (52, 10749, 99, 20585),
    ),
    "gqa_ragged_prefill_causal_h32_kv4_d128": (
        (1, 1, 1),
        (1, 6, 6),
        (1, 34, 34),
        (3, 172, 172),
        (15, 969, 969),
        (27, 12251, 12251),
        (33, 16294, 16294),
    ),
    "gqa_ragged_prefill_causal_h32_kv8_d128": (
        (1, 1, 1),
        (1, 7, 7),
        (1, 18, 18),
        (1, 35, 35),
        (1, 71, 71),
        (2, 81, 81),
        (5, 92, 92),
        (10, 12571, 12571),
        (15, 982, 982),
        (25, 13557, 13557),
        (36, 12845, 12845),
    ),
    "mla_paged_decode_h16_ckv512_kpe64_ps1": (
        (1, 989669, 8),
        (1, 989669, 108),
        (1, 989669, 208),
        (1, 989669, 308),
        (1, 989669, 408),
        (1, 989669, 508),
        (1, 989669, 608),
        (1, 989669, 708),
        (1, 989669, 808),
        (1, 989669, 1008),
        (1, 989669, 1108),
        (1, 989669, 1208),
        (1, 989669, 1908),
        (1, 989669, 2408),
        (1, 989669, 2708),
        (16, 989669, 457),
        (16, 989669, 1857),
        (16, 989669, 2757),
        (16, 989669, 3557),
        (16, 989669, 4357),
        (16, 989669, 5057),
        (16, 989669, 5857),
        (16, 989669, 6657),
        (16, 989669, 7257),
        (16, 989669, 8057),
        (16, 989669, 8857),
        (16, 989669, 9657),
        (16, 989669, 10857),
        (16, 989669, 12857),
        (16, 989669, 14857),
        (16, 989669, 17257),
        (64, 989669, 9945),
        (64, 989669, 16345),
        (64, 989669, 22745),
        (64, 989669, 27545),
        (64, 989669, 30745),
        (64, 989669, 33945),
        (64, 989669, 37145),
        (64, 989669, 40345),
        (64, 989669, 44845),
        (64, 989669, 48045),
        (64, 989669, 51245),
        (64, 989669, 54445),
        (64, 989669, 57645),
        (64, 989669, 62345),
        (64, 989669, 68745),
        (64, 989669, 75145),
    ),
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": (
        (1, 1, 34, 989669),
        (1, 3, 5, 989669),
        (1, 5, 7, 989669),
        (1, 8, 12, 989669),
        (1, 10, 12, 989669),
        (1, 13, 14, 989669),
        (1, 15, 18, 989669),
        (1, 17, 19, 989669),
        (1, 29, 34, 989669),
        (1, 33, 34, 989669),
        (1, 43, 46, 989669),
        (1, 58, 60, 989669),
        (1, 96, 98, 989669),
        (1, 199, 203, 989669),
        (1, 287, 288, 989669),
        (1, 376, 381, 989669),
        (1, 1028, 1038, 989669),
        (1, 16384, 16387, 989669),
        (2, 2, 53, 989669),
        (2, 26, 32, 989669),
        (2, 123, 185, 989669),
        (2, 10870, 10875, 989669),
        (3, 69, 90, 989669),
        (3, 1187, 1205, 989669),
        (3, 3024, 3029, 989669),
        (4, 4, 121, 989669),
        (4, 52, 67, 989669),
        (4, 138, 151, 989669),
        (4, 805, 814, 989669),
        (5, 473, 491, 989669),
        (6, 6, 109, 989669),
        (11, 6053, 6091, 989669),
        (18, 15883, 15937, 989669),
        (20, 3842, 3916, 989669),
        (22, 22, 17759, 989669),
        (26, 15092, 15187, 989669),
        (28, 1954, 2044, 989669),
        (56, 8987, 14390, 989669),
    ),
}


# `OPERATORS` below is the canonical list of the shipped paged operators and the dims each
# one's name fixes -- what `cases` needs and the name does not carry in machine-readable
# form. It lives next to `cases` because a caller who needs one needs the other: the perf
# harness and the demos both read it here rather than keeping their own copy. (The nine
# numeric tests deliberately do not: each passes its own `dims` literal, because it also
# hands that tuple to its schedule's `get_handwritten_scheduled_graph`. That the two agree
# is checked by `tests/graph/apply_paging/test_expected_kernels.py`.)
@dataclass(frozen=True)
class OperatorSpec:
    dims: tuple
    page_size: int = 1
    # The MLA references are only valid at the scale their kernel bakes in, so it is read
    # from the schedule module rather than defaulted.
    scale_from_schedule: bool = False


OPERATORS: dict[str, OperatorSpec] = {
    # num_qo_heads, num_kv_heads, head_dim
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": OperatorSpec((32, 8, 128)),
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": OperatorSpec((32, 4, 128)),
    "gqa_ragged_prefill_causal_h32_kv8_d128": OperatorSpec((32, 8, 128)),
    "gqa_ragged_prefill_causal_h32_kv4_d128": OperatorSpec((32, 4, 128)),
    "gqa_paged_decode_h32_kv8_d128_ps1": OperatorSpec((32, 8, 128)),
    "gqa_paged_decode_h32_kv4_d128_ps1": OperatorSpec((32, 4, 128)),
    # num_qo_heads, num_kv_heads, head_dim, page_size
    "gqa_paged_decode_h32_kv8_d128_ps64": OperatorSpec((32, 8, 128, 64), page_size=64),
    # num_qo_heads, head_dim_ckv, head_dim_kpe
    "mla_paged_decode_h16_ckv512_kpe64_ps1": OperatorSpec(
        (16, 512, 64), scale_from_schedule=True
    ),
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": OperatorSpec(
        (16, 512, 64), scale_from_schedule=True
    ),
}


def keyed_workloads(operator, *, keep: int = 0):
    """A zero-arg factory yielding ``(workload_key, inputs, reference)``.

    What ``cleave.util.autotune`` consumes: it never learns which operators exist, so the
    axes-to-key mapping and the sampling live here, next to the captured axes they come from.
    ``keep=0`` is every captured workload; ``keep>0`` is a stratified sample of that many --
    the same selection ``tests/perf`` times, since ``WORKLOADS`` is ordered smallest-first
    and a prefix would sample only the degenerate shapes.
    """
    from cleave.util.autotune import sampled, workload_key

    axes_list = list(WORKLOADS[operator])
    wanted = set(sampled(range(len(axes_list)), keep))

    def factory():
        for index, (inputs, reference) in enumerate(workloads(operator)):
            if index not in wanted:
                del inputs, reference
                continue
            yield workload_key(axes_list[index]), inputs, reference

    return factory


def workloads(operator):
    """``cases`` for ``operator``, with the dims and page size its name fixes."""
    spec = OPERATORS[operator]
    kwargs = {}
    if spec.page_size != 1:
        kwargs["page_size"] = spec.page_size
    if spec.scale_from_schedule:
        import importlib

        schedule = importlib.import_module(
            f"tests.graph.handwritten.handwritten_schedules.{operator}"
        )
        kwargs["scale"] = schedule.SM_SCALE
    return cases(operator, spec.dims, **kwargs)


def _rng(operator, axes):
    return random.Random(zlib.crc32(repr((operator, axes)).encode()))


def _positive(total, parts, rng):
    # ``total`` split into exactly ``parts`` positive lengths.
    cuts = sorted(rng.sample(range(1, total), parts - 1)) if parts > 1 else []
    bounds = [0, *cuts, total]
    return [b - a for a, b in zip(bounds, bounds[1:])]


def _nonnegative(total, parts, rng):
    cuts = sorted(rng.randrange(total + 1) for _ in range(parts - 1))
    bounds = [0, *cuts, total]
    return [b - a for a, b in zip(bounds, bounds[1:])]


def _tokens(pages, page_size, rng):
    # A page count becomes a token count: every page full but the last, which is partial.
    return [
        (n - 1) * page_size
        + (page_size if page_size == 1 else rng.randint(1, page_size))
        for n in pages
    ]


def _decode_lengths(operator, axes, page_size):
    batch, pool, n_idx = axes
    rng = _rng(operator, axes)
    batch = min(batch, n_idx)
    kv_lens = _tokens(_positive(n_idx, batch, rng), page_size, rng)
    return batch, max(pool, n_idx), n_idx, kv_lens


def _prefill_lengths(operator, axes):
    batch, total_q, n_idx = axes[0], axes[1], axes[2]
    rng = _rng(operator, axes)
    batch = min(batch, total_q, n_idx)
    q_lens = _positive(total_q, batch, rng)
    if n_idx >= total_q:
        # Real prefill attends over the request's own tokens plus its cached prefix, so
        # give every request at least its query length and spread the surplus.
        extra = _nonnegative(n_idx - total_q, batch, rng)
        kv_lens = [q + e for q, e in zip(q_lens, extra)]
    else:
        # Captured lines where the KV window is shorter than the query run do occur; those
        # leave fully-masked rows, which are defined to be 0 in the output and -inf in the
        # lse -- not NaN, which SOL's own sanity check rejects outright.
        kv_lens = _positive(n_idx, batch, rng)
    return batch, q_lens, kv_lens


def cases(operator, dims, page_size=1, scale=None):
    """(inputs, expected outputs) for each captured workload, built one at a time."""
    for axes in WORKLOADS[operator]:
        if operator.startswith("mla_paged_decode"):
            batch, pool, n_idx, kv_lens = _decode_lengths(operator, axes, page_size)
            wl = (batch, *dims, pool, n_idx)
            inputs = paged_reference.mla_paged_decode_inputs(wl, kv_lens, scale=scale)
            expected = paged_reference.mla_paged_decode_reference(wl, kv_lens)
        elif operator.startswith("gqa_paged_decode"):
            batch, pool, n_idx, kv_lens = _decode_lengths(operator, axes, page_size)
            wl = (batch, *dims[:3], pool, n_idx)
            inputs = paged_reference.gqa_paged_decode_inputs(
                wl, kv_lens, page_size=page_size, scale=scale
            )
            expected = paged_reference.gqa_paged_decode_reference(
                wl, kv_lens, page_size=page_size
            )
        else:
            batch, q_lens, kv_lens = _prefill_lengths(operator, axes)
            total_q, n_idx = sum(q_lens), sum(kv_lens)
            if operator.startswith("gqa_ragged"):
                wl = (*dims, total_q, n_idx)
                inputs = paged_reference.gqa_ragged_prefill_inputs(
                    wl, q_lens, kv_lens, scale=scale
                )
                expected = paged_reference.gqa_ragged_prefill_reference(
                    wl, q_lens, kv_lens
                )
            elif operator.startswith("mla_paged_prefill"):
                wl = (*dims, max(axes[3], n_idx), n_idx, total_q)
                inputs = paged_reference.mla_paged_prefill_inputs(
                    wl, q_lens, kv_lens, scale=scale
                )
                expected = paged_reference.mla_paged_prefill_reference(
                    wl, q_lens, kv_lens
                )
            else:
                wl = (*dims, max(axes[3], n_idx), n_idx, total_q)
                inputs = paged_reference.gqa_paged_prefill_inputs(
                    wl, q_lens, kv_lens, scale=scale
                )
                expected = paged_reference.gqa_paged_prefill_reference(
                    wl, q_lens, kv_lens
                )
        yield inputs, expected(*inputs)
