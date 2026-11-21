"""One program is one search problem, whatever the process did before, and one result list.

Two things used to depend on the process (PB9):

* the symbols of a program's extent groups followed the iteration order of tables hashed on
  tensor guids, and guids come from a counter that the whole process draws from, the threads
  of a running search included. A program built second in a process, or while another thread
  searched, got other symbols than when it was built first, so its checkpoint key changed and
  its checkpoint was ignored ("key mismatch; re-searching");
* the searched graphs were listed in the order the search threads finished, so two fresh
  searches of one program numbered the graphs differently (plan labels ``g1`` / ``g5`` / ``g7``).

Every case runs in fresh interpreters (the guid counter is per process) on the CPU, with its
own empty cache.

These tests check the compiled search (``cleave.superoptimizer.core``). They fail with a
library built from other native sources than this tree's, for example a copied one.
"""

import functools
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

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


class Interleaved:
    # A tile graph that creates next(gaps) tensors of another graph before each of its own.
    def __init__(self, graph, gaps):
        self.graph, self.gaps, self.other = graph, gaps, new_tile_graph()

    def __getattr__(self, name):
        method = getattr(self.graph, name)
        if name == "create_tiled_output":  # defines no tensor
            return method

        def build(*args, **kwargs):
            for _ in range(next(self.gaps)):
                self.other.create_tiled_input((2, 2))
            return method(*args, **kwargs)

        return build


new_tile_graph = csopt.new_tile_graph


def symbols(program, gaps=None):
    # gaps[i]: how many foreign tensors to create right before the program's i-th tensor
    if gaps is not None:
        csopt.new_tile_graph = lambda **kw: Interleaved(new_tile_graph(**kw), iter(gaps))
    try:
        tile = program.to_tile_graph(split=())
    finally:
        csopt.new_tile_graph = new_tile_graph
    tile = getattr(tile, "graph", tile)
    guids = [t["guid"] for op in tile.get_graph_structure() for t in op["output_tensors"]]
    symbolic, symbol_map = csopt.create_symbolic_tile_graph(tile)
    return dict(symbol_map=sorted(symbol_map.items()), key=symbolic.checkpoint_key(),
                guids=[g - guids[0] for g in guids])


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


def _native(test):
    """On a failure, name the library the test ran against."""
    @functools.wraps(test)
    def run(tmp_path):
        try:
            test(tmp_path)
        except AssertionError as e:
            import cleave.superoptimizer.core as core

            raise AssertionError(
                f"{e}\n\nThis test checks the native search, loaded from {core.__file__}. If that "
                "file was not built from this tree's src/ and include/ (a copied or older build), "
                "rebuild it before reading the failure.") from e
    return run


def _child(body, *argv, cache):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(cache))
    proc = subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(body), *argv],
                          capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, proc.stdout[-3000:]
    return json.loads(lines[0][len("RESULT "):]), proc.stdout


@_native
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


@_native
def test_symbols_do_not_depend_on_tensors_created_in_between(tmp_path):
    # A search running in another thread takes guids while a program is built, so the guids of
    # the program's tensors are no longer consecutive. Here the gaps are made on purpose.
    seen, _ = _child(
        """
        import random
        n = len(symbols(P())["guids"])
        rng = random.Random(0)
        patterns = [[0] * n]
        patterns += [[int(i == j) for i in range(n)] for j in range(1, n)]  # one gap, at each place
        patterns += [[1] * n, list(range(n))]
        patterns += [[rng.randrange(40) for _ in range(n)] for _ in range(8)]
        print("RESULT " + json.dumps([symbols(P(), gaps) for gaps in patterns]))
        """,
        cache=tmp_path,
    )
    fresh, *gapped = seen
    assert fresh["guids"] == list(range(10))
    assert len(gapped) == 19 and all(s["guids"] != fresh["guids"] for s in gapped)
    assert [s["symbol_map"] for s in gapped] == [fresh["symbol_map"]] * 19
    assert [s["key"] for s in gapped] == [fresh["key"]] * 19


@_native
def test_symbols_do_not_depend_on_a_search_running_in_another_thread(tmp_path):
    # The same, with the real cause: search_tile_graph releases the GIL and its threads create
    # tensors all the time.
    seen, _ = _child(
        """
        import threading, time
        fresh = symbols(P())
        program = trace(chain, dict(BH=32, SQ=64, SKV=4096, D=64), cl.f16)
        config = SuperoptConfig(max_ops=8, threads=4, cache=False)
        # The first search imports modules; a thread that does so while this one spins waits
        # for the interpreter lock at every file it reads, so import them here.
        superoptimize(program, config)
        stop = threading.Event()

        def search():
            while not stop.is_set():
                superoptimize(program, config)

        thread = threading.Thread(target=search)
        thread.start()
        builds, gapped, different = 0, 0, []
        try:
            deadline = time.monotonic() + 120
            while gapped < 20 and time.monotonic() < deadline:
                got = symbols(P())
                builds += 1
                gapped += got["guids"] != fresh["guids"]
                if (got["symbol_map"], got["key"]) != (fresh["symbol_map"], fresh["key"]):
                    different.append(got)
        finally:
            stop.set()
            thread.join()
        print("RESULT " + json.dumps(dict(builds=builds, gapped=gapped, different=different[:3])))
        """,
        cache=tmp_path,
    )
    assert seen["gapped"] == 20, seen  # the search did take guids in the middle of 20 builds
    assert seen["different"] == []


@_native
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


@_native
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
    # The same list in every process and for every thread count. The list is in depth-first
    # order of the search tree, which the search establishes by sorting once it is done; no run
    # is the reference for it (the old library returned another order on one thread as well),
    # so the one-thread search checks that the thread count does not matter, and nothing more.
    for other in others:
        assert [g[1] for g in other["graphs"]] == [g[1] for g in first["graphs"]]  # the kind at each index
        assert other["labels"] == first["labels"]
        assert other["graphs"] == first["graphs"]


@_native
def test_paged_plan_labels_name_the_same_graphs_in_every_process(tmp_path):
    # The paged scheduler puts the same graph index in its labels ("g0 solo kv=32 ...").
    plan = """
        sys.path.insert(0, sys.argv[2])  # the demo package
        from cleave.frontend.paged import trace_paged
        from cleave.pipeline.dynamic import schedule_paged
        from demo.dynamic.gqa_paged_decode import build

        fn, hints = build("kv8")
        attention = trace_paged(fn, hints=hints).attention()
        graphs = superoptimize(attention.dense_body(), SuperoptConfig(threads=int(sys.argv[1])))
        assert not graphs.from_cache
        plans = schedule_paged(graphs, attention.paging_spec(), ScheduleConfig(topk=4),
                               filter_before_topk=True)
        print("RESULT " + json.dumps(dict(graphs=listing(graphs), labels=[p.label for p in plans])))
        """
    first, other = [_child(plan, threads, str(ROOT), cache=tmp_path / f"cache{threads}")[0]
                    for threads in ("16", "1")]
    assert len(first["graphs"]) == 2 and len(first["labels"]) == 4
    assert all(label.split()[0] in ("g0", "g1") and label.split()[1] == "solo"
               for label in first["labels"]), first["labels"]
    assert other == first
