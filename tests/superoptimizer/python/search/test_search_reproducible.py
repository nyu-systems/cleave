"""One program is one search problem, whatever the process did before, and one result list.

Two things used to depend on the process (PB9):

* the symbols of a program's extent groups followed the iteration order of tables hashed on
  tensor guids, and guids come from a process-wide counter. A program built second in a
  process got other symbols than when it was built first, so its checkpoint key changed and
  its checkpoint was ignored ("key mismatch; re-searching");
* the searched graphs were listed in the order the search threads finished, so two fresh
  searches of one program numbered the graphs differently (plan labels ``g1`` / ``g5`` / ``g7``).

Every case runs in fresh interpreters (the guid counter is per process) on the CPU, with its
own empty cache.
"""

import json
import os
import subprocess
import sys
import textwrap

PRELUDE = """
import json, sys

import cleave.lang as cl
import cleave.superoptimizer as csopt
from cleave.frontend import trace
from cleave.pipeline import ScheduleConfig, SuperoptConfig, schedule, superoptimize


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    e = cl.exp((q @ k.mT) / 8.0 + (-6.25))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def matmul(a: cl.Tensor["M", "K"], b: cl.Tensor["K", "N"]):
    return a @ b


def chain(q: cl.Tensor["BH", "SQ", "D"], k: cl.Tensor["BH", "SKV", "D"], v: cl.Tensor["BH", "SKV", "D"]):
    return (q @ k.mT) @ v


def P():
    return trace(attention, dict(H=32, SQ=4096, SKV=2048, D=64), cl.f16)


def Q(i=0):
    return trace(matmul, dict(M=128 + i, K=4096, N=128), cl.f16)


def symbols(program):
    symbolic, symbol_map = csopt.create_symbolic_tile_graph(program.to_tile_graph(split=()))
    return dict(symbol_map=sorted(symbol_map.items()), key=symbolic.checkpoint_key())


def listing(graphs):
    # A searched graph without its tensor guids (process-wide numbers): each tensor is named
    # by the position of its definition.
    out = []
    for g in graphs:
        index, ops = {}, []
        for op in g.structure:
            ins = [index[t["guid"]] for t in op["input_tensors"]]
            for t in op["output_tensors"]:
                index[t["guid"]] = len(index)
            ops.append([op["op_type"], ins, [t["dim"] for t in op["output_tensors"]]])
        out.append([g.index, g.kind, ops])
    return out
"""


def _child(body, *argv, cache):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(cache))
    proc = subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(body), *argv],
                          capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, proc.stdout[-3000:]
    return json.loads(lines[0][len("RESULT "):]), proc.stdout


def test_symbols_do_not_depend_on_what_the_process_built_before(tmp_path):
    seen, _ = _child(
        """
        seen = []
        for n in range(6):
            for i in range(n):
                symbols(Q(i))  # other programs: each moves the guid counter
            seen.append(symbols(P()))
        print("RESULT " + json.dumps(seen))
        """,
        cache=tmp_path,
    )
    # seen[0] is the program as the first thing a process builds
    assert sorted(extent for _, extent in seen[0]["symbol_map"]) == [32, 64, 64, 2048, 4096]
    assert [s["key"] for s in seen] == [seen[0]["key"]] * 6
    assert [s["symbol_map"] for s in seen] == [seen[0]["symbol_map"]] * 6


def test_a_program_searched_second_in_a_process_hits_its_checkpoint(tmp_path):
    search = """
        cfg = SuperoptConfig(threads=8)
        seen = []
        for name in sys.argv[1]:
            graphs = superoptimize(Q() if name == "q" else P(), cfg)
            if name == "p":
                seen.append(dict(from_cache=graphs.from_cache, graphs=len(graphs),
                                 key=open(graphs.checkpoint + ".key").read().strip()))
        print("RESULT " + json.dumps(seen))
        """
    (first,), _ = _child(search, "p", cache=tmp_path)
    assert not first["from_cache"] and first["graphs"] > 0
    # ... then in one process, after one to four other searches and right after itself
    seen, out = _child(search, "qpqpqpqpp", cache=tmp_path)
    assert "key mismatch" not in out
    assert seen == [dict(first, from_cache=True)] * 5


def test_two_fresh_searches_list_the_same_graphs_in_the_same_order(tmp_path):
    # 12 graphs, with and without split ops, found by 16 threads.
    search = """
        program = trace(chain, dict(BH=32, SQ=64, SKV=4096, D=64), cl.f16)
        graphs = superoptimize(program, SuperoptConfig(max_ops=8, split=("SKV",), threads=int(sys.argv[1])))
        assert not graphs.from_cache
        plans = schedule(graphs, ScheduleConfig(topk=2, split_k=(4,)))
        print("RESULT " + json.dumps(dict(graphs=listing(graphs), labels=[p.label for p in plans])))
        """
    first, *others = [_child(search, threads, cache=tmp_path / f"cache{i}")[0]
                      for i, threads in enumerate(("16", "16", "16", "1"))]
    assert [g[0] for g in first["graphs"]] == list(range(12))
    assert len({label.split()[0] for label in first["labels"]}) > 4  # "g<index> ..." of several graphs
    # the same list in every process, and it is the one a single thread walking the search
    # tree produces
    for other in others:
        assert [g[1] for g in other["graphs"]] == [g[1] for g in first["graphs"]]  # the kind at each index
        assert other["labels"] == first["labels"]
        assert other["graphs"] == first["graphs"]
