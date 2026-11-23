"""The search of every existing program explores exactly as many states as it did.

A change to the native search that only means to add something (a new op, a new kind of input)
must leave the searches of the programs that do not use it alone. The number of states a search
explores (``[Search] Total states explored: N`` on stderr) is the finest cheap witness of that:
it moves with any change of the candidate ops, of their order, of the pattern pruning, of the
reducibility flags or of the verifier. The counts below are the demos' searches at their own
configs (the costliest demo searches, 170k-390k states each, are left out: ``mha_decode`` with
its split covers that path), pinned at 9af4dd3. They do not depend on the process nor on the
thread count (checked with 1 and 8 threads, in fresh processes, before pinning).

Each case runs in a fresh interpreter on the CPU, without a cache, with 4 search threads.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

CHILD = textwrap.dedent("""
    import dataclasses, runpy, sys

    root, name, *argv = sys.argv[1:]
    sys.path.insert(0, root + "/demo")
    import cleave.examples as examples
    from cleave.pipeline import superoptimize

    seen = {}

    class Stop(Exception):
        pass

    def fake_compile(program, *, superopt, schedule, eval, **kw):
        seen.update(program=program, so=superopt)
        raise Stop

    examples._compile = fake_compile
    examples._target = lambda: "cpu"
    sys.argv = [f"{root}/demo/{name}.py", *argv]
    try:
        runpy.run_path(f"{root}/demo/{name}.py", run_name="__main__")
    except Stop:
        pass
    program, so = seen["program"], seen["so"]
    graphs = superoptimize(program, dataclasses.replace(so, cache=False, threads=4))
    print("RESULT", so.resolve_max_ops(program), len(graphs), flush=True)
""")

# id: (demo, argv, max_ops, explored states, graphs found)
CASES = {
    "matmul": ("matmul", [], 13, 3, 1),
    "matmul-split-K": ("matmul", ["--split", "K"], 13, 1352, 2),
    "rmsnorm_matmul": ("rmsnorm_matmul", [], 13, 9851, 2),
    "swiglu": ("swiglu", [], 13, 3818, 1),
    "lora": ("lora", [], 13, 4150, 4),
    "matmul_rowsum": ("matmul_rowsum", [], 13, 2717, 4),
    "matmul_matmul": ("matmul_matmul", [], 13, 614, 4),
    "matmul_matmul-split": ("matmul_matmul", ["--preset", "split"], 8, 12426, 12),
    "mha_decode": ("mha_decode", [], 14, 210646, 30),
    "mha_decode-nosplit": ("mha_decode", ["--split", "none"], 14, 1292, 2),
    "gqa_decode-nosplit": ("gqa_decode", ["--split", "none"], 14, 1080, 2),
    "mha_prefill": ("mha_prefill", [], 12, 467, 2),
    "gqa_prefill": ("gqa_prefill", [], 13, 722, 2),
    "mla_prefill": ("mla_prefill", [], 12, 467, 2),
    "mla_decode": ("mla_decode", [], 13, 2231, 2),
    "qknorm": ("qknorm", [], 26, 5151, 2),
    "mha_decode_exact": ("mha_decode_exact", [], 14, 833, 2),
}


@pytest.mark.parametrize("case", list(CASES))
def test_search_explores_the_pinned_number_of_states(tmp_path, case):
    demo, argv, max_ops, states, found = CASES[case]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(tmp_path))
    proc = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), demo, *argv], cwd=tmp_path,
                          capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = [ln.split()[1:] for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    explored = re.findall(r"\[Search\] Total states explored: (\d+)", proc.stderr)
    assert len(result) == 1 and len(explored) == 1, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert [int(x) for x in result[0]] == [max_ops, found]
    assert int(explored[0]) == states
