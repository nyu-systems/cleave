"""Traced/built programs emit the recorded csopt builder calls (``tests/demo/snapshots/builder_calls.json``)."""

import contextlib
import json

import pytest

import cleave.superoptimizer as csopt
from cleave.superoptimizer.core import DTensor

from ..demo._snapshot import SNAPSHOT, record, updating
from . import _programs as P

_METHODS = ("create_tiled_input", "create_tiled_output", "matmul", "add", "mul", "div", "exp",
            "sqrt", "silu", "reduce_sum")


class _Abort(Exception):
    pass


def _norm_split_enabled(x):
    return bool(x) if isinstance(x, bool) else any(x)


@contextlib.contextmanager
def spy(monkeypatch):
    log = []
    orig_new = csopt.new_tile_graph

    def new_tile_graph(split_enabled=False, symbolic=False):
        log.append(("new_tile_graph", _norm_split_enabled(split_enabled), symbolic))
        return orig_new(split_enabled=split_enabled, symbolic=symbolic)

    def symbolic(graph):
        raise _Abort

    with monkeypatch.context() as m:
        m.setattr(csopt, "new_tile_graph", new_tile_graph)
        m.setattr(csopt, "create_symbolic_tile_graph", symbolic)
        for name in _METHODS:
            orig = getattr(csopt.TileGraph, name)

            def wrapped(self, *args, _orig=orig, _name=name, **kw):
                out = _orig(self, *args, **kw)
                log.append((_name, args, kw, out))
                return out

            m.setattr(csopt.TileGraph, name, wrapped)
        yield log


def normalized(log):
    ids = {}

    def norm(x):
        if isinstance(x, DTensor):
            return f"t{ids[x.guid]}"
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return float(x)  # the C++ builder stores every scalar as float
        if isinstance(x, tuple):
            return tuple(norm(y) for y in x)
        if type(x).__name__ == "dtype":
            return x.name
        return x

    seq = []
    for rec in log:
        if rec[0] == "new_tile_graph":
            seq.append(rec)
            continue
        m, args, kw, out = rec
        a = [norm(x) for x in args]
        if m == "create_tiled_input":
            a = [a[0], tuple(bool(c) for c in a[1]), a[2] if len(a) > 2 else norm(kw.get("dtype", csopt.core.float16))]
        elif m == "matmul":
            a = (a + [False, False])[:4]
            a[2], a[3] = bool(kw.get("trans_A", a[2])), bool(kw.get("trans_B", a[3]))
        else:
            assert not kw, (m, kw)
        r = (m, tuple(a))
        if isinstance(out, DTensor):
            ids[out.guid] = len(ids)
            r += (f"t{ids[out.guid]}",)
        seq.append(r)
    return json.loads(json.dumps(seq))


def dag_signature(seq):
    """Order-independent structural hash of every value (inputs keyed by creation order)."""
    h, sigs, n_in = {}, [], 0
    for rec in seq:
        if rec[0] == "new_tile_graph":
            s = rec
        elif rec[0] == "create_tiled_input":
            s, n_in = ("in", n_in, rec[1]), n_in + 1
        else:
            s = (rec[0],) + tuple(h.get(a, a) if isinstance(a, str) else a for a in rec[1])
        if len(rec) == 3 and rec[0] != "new_tile_graph":
            h[rec[2]] = str(s)
        sigs.append(str(s))
    return sorted(sigs)


def record_program(monkeypatch, prog, split=()):
    with spy(monkeypatch) as log:
        prog.to_tile_graph(split=split)
    return normalized(log)


CASES = [
    # id, jit fn, shapes, split dims
    ("matmul", P.matmul, dict(M=128, N=128, K=4096), ()),
    ("matmul-split", P.matmul, dict(M=128, N=128, K=4096), ("K",)),
    ("rmsnorm_matmul", P.rmsnorm_matmul, dict(M=4096, N=14336, K=4096), ()),
    ("swiglu", P.swiglu, dict(M=4096, N=14336, K=4096), ()),
    ("lora", P.lora, dict(M=16, N=4096, K=4096, R=16), ()),
    ("mha_decode", P.attention, dict(HQ=32, HKV=32, SQ=64, SKV=4096, D=64), ()),
    ("mha_decode-split", P.attention, dict(HQ=32, HKV=32, SQ=64, SKV=4096, D=64), ("SKV",)),
    ("gqa_decode", P.attention, dict(HQ=32, HKV=8, SQ=4096, SKV=4096, D=64), ()),
    ("gqa_decode-split", P.attention, dict(HQ=32, HKV=8, SQ=4096, SKV=4096, D=64), ("SKV",)),
    ("matmul_rowsum", P.matmul_rowsum, dict(M=4096, N=4096, K=64), ()),
    ("matmul_matmul_bhsd", P.matmul_matmul, dict(BH=32, SQ=4096, SKV=4096, D=64), ()),
    ("qknorm", P.qknorm, dict(HQ=64, HKV=8, SQ=4096, SKV=4096, D=128), ()),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_jit_matches_recorded_builder_calls(monkeypatch, request, case):
    cid, jf, shapes, split = case
    new = record_program(monkeypatch, jf.trace(**shapes), split=split)
    if updating(request):
        record("traces", cid, new)
    assert new == SNAPSHOT["traces"][cid]


def test_every_recorded_trace_has_a_case():
    assert {c[0] for c in CASES} == set(SNAPSHOT["traces"])


def test_qknorm_composite_same_dag(monkeypatch):
    """cl.rms_norm emits q's chain then k's (the demo interleaves them): same DAG, other order."""
    shapes = dict(HQ=64, HKV=8, SQ=4096, SKV=4096, D=128)
    recorded = SNAPSHOT["traces"]["qknorm"]
    new = record_program(monkeypatch, P.qknorm_composite.trace(**shapes))
    assert new != recorded
    assert dag_signature(new) == dag_signature(recorded)


@pytest.mark.parametrize("which", ["lora", "rmsnorm_matmul"])
def test_builder_matches_jit(monkeypatch, which):
    recorded = SNAPSHOT["traces"][which]
    if which == "lora":
        prog, jprog = P.lora_builder(), P.lora.trace(M=16, N=4096, K=4096, R=16)
    else:
        prog = P.rmsnorm_matmul_builder(M=4096, N=14336, K=4096)
        jprog = P.rmsnorm_matmul.trace(M=4096, N=14336, K=4096)
    assert record_program(monkeypatch, prog) == recorded
    assert prog.fingerprint() == jprog.fingerprint()
