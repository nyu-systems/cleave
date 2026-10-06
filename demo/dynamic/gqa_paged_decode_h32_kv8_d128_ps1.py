"""Grouped-query paged decode: 32 query heads over 8 kv heads, head dim 128, one token per page.

    python demo/dynamic/gqa_paged_decode_h32_kv8_d128_ps1.py

The spec and the dense tiling come from the shared registry the tests pin, so what this
runs is what ``tests/graph/apply_paging/test_expected_kernels.py`` asserts reproduces the
operator's tuned kernel.
"""

import argparse
import sys
from pathlib import Path

# Run as a script, so the repository root has to be importable before `demo.` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.dynamic._driver import run_registered

OPERATOR = "gqa_paged_decode_h32_kv8_d128_ps1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk", type=int, default=400)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--skip-workloads", action="store_true")
    parser.add_argument(
        "--dump-best",
        default=None,
        help="write the winning kernel to DIR/<operator>.py, for bench_dynamic.py --cleave-dir",
    )
    parser.add_argument(
        "--max-workloads",
        type=int,
        default=6,
        help="how many captured workloads to TIME each candidate on; every candidate is "
             "validated on all of them regardless",
    )
    parser.add_argument(
        "--split",
        default="",
        help="comma-separated split-KV factors to search as well (e.g. 4,8,16); a "
             "decode operator then also gets partial+combine kernel pairs as candidates, "
             "and --dump-best writes the best single kernel to <operator>.nosplit.py",
    )
    args = parser.parse_args()

    run_registered(
        OPERATOR,
        topk=args.topk,
        checkpoint=args.checkpoint,
        workloads=not args.skip_workloads,
        max_workloads=args.max_workloads,
        dump_best=args.dump_best,
        split_factors=[int(s) for s in args.split.split(",") if s.strip()],
    )


if __name__ == "__main__":
    main()
