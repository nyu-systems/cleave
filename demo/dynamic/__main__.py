"""Search, validate and time one dynamic attention operator through the frontend.

    python -m demo.dynamic <operator> [--topk N] [--split 4,8,16] ...
    python -m demo.dynamic --list

Each operator is a preset of a family demo (``demo/dynamic/<family>.py``); its spec is
derived from the traced function and validated on the registry's captured workloads.
"""

import argparse

from cleave.dynamic import OPERATORS

FAMILIES = ("gqa_paged_decode", "mla_paged_decode", "gqa_paged_prefill", "gqa_ragged_prefill",
            "mla_paged_prefill")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m demo.dynamic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("operator", nargs="?", choices=sorted(OPERATORS), metavar="operator")
    parser.add_argument("--list", action="store_true", help="list the operators and exit")
    parser.add_argument("--topk", type=int, default=400)
    parser.add_argument("--skip-workloads", action="store_true")
    parser.add_argument("--dump-best", default=None,
                        help="write the winning kernel to DIR/<operator>.py")
    parser.add_argument(
        "--max-workloads", type=int, default=6,
        help="how many captured workloads to TIME each candidate on; every candidate is "
        "validated on all of them regardless",
    )
    parser.add_argument(
        "--split", default="",
        help="comma-separated split-KV factors to search as well (e.g. 4,8,16); a decode "
        "operator then also gets partial+combine kernel pairs and a split/single dispatch",
    )
    parser.add_argument(
        "--filter-before-topk", action="store_true",
        help="drop tilings the access layer refuses before --topk truncates the ranking",
    )
    args = parser.parse_args(argv)
    if args.list or args.operator is None:
        print("\n".join(sorted(OPERATORS)))
        return
    module, preset = family_of(args.operator)
    flags = ["--topk", str(args.topk), "--max-workloads", str(args.max_workloads)]
    flags += ["--split", args.split] if args.split else []
    flags += ["--dump-best", args.dump_best] if args.dump_best else []
    flags += ["--skip-workloads"] * args.skip_workloads
    flags += ["--filter-before-topk"] * args.filter_before_topk
    return module.main(flags, preset=preset)


def family_of(operator):
    """``(family demo module, preset)`` that defines ``operator``."""
    import importlib

    for family in FAMILIES:
        module = importlib.import_module(f"demo.dynamic.{family}")
        for preset in module.PRESETS:
            if module.build(preset)[0].__name__ == operator:
                return module, preset
    raise KeyError(f"no frontend demo defines {operator!r}")


if __name__ == "__main__":
    main()
