"""CPU: the wrapper that owns an inter output's fp32 buffer (``InterOutputKernel``).

Per call it hands the kernel freshly zeroed fp32 buffers, after the inputs, and returns
every output in declaration order and in its declared dtype. ``tests/runtime/
test_inter_outputs.py`` runs real kernels (GPU).
"""

from types import SimpleNamespace

import pytest

from cleave.runtime.compile import (
    InterOutputKernel,
    _InterAdapter,
    _inter_positions,
    _normalized_out_idx,
    _load_main_kernel_from_code,
    build_candidate_code,
)
from cleave.runtime.reference import graph_out_indices, order_inputs_for_kernel, placeholder_order
from tests.codegen.test_inter_codegen import candidates, contexts, inter_pair

torch = pytest.importorskip("torch")


class FakeKernel:
    """Stands in for TileLang: params ``(a, b, out0, sum1, out2)``; out0/out2 are its own
    results, sum1 is an inter buffer it accumulates ``a + b`` into."""

    def __init__(self):
        self.seen = []
        self.adapter = SimpleNamespace(result_idx=[2, 4], prim_func=None, params="adapter")

    def __call__(self, a, b, sum1):
        self.seen.append(sum1.clone())
        sum1 += a + b
        return [a * 2, b * 3]

    def get_kernel_source(self):
        return "src"


def _wrapped():
    prim_func = SimpleNamespace(params=list(range(5)))
    positions = {3: ((2, 2), torch.float32, torch.float16)}
    return InterOutputKernel(FakeKernel(), prim_func, [2, 3, 4], positions)


def test_buffers_are_zeroed_every_call_and_cast():
    k = _wrapped()
    a, b = torch.ones(2, 2, dtype=torch.float16), torch.full((2, 2), 2.0, dtype=torch.float16)
    for _ in range(3):
        out0, sum1, out2 = k(a, b)
        assert sum1.dtype == torch.float16 and torch.equal(sum1, torch.full((2, 2), 3.0).half())
        assert torch.equal(out0, a * 2) and torch.equal(out2, b * 3)
    assert all(torch.count_nonzero(s) == 0 and s.dtype == torch.float32 for s in k.kernel.seen)


def test_looks_like_the_tilelang_kernel():
    k = _wrapped()
    assert k.adapter.result_idx == [2, 3, 4]  # the inter output counts as a result
    assert k.adapter.params == "adapter" and k.get_kernel_source() == "src"


def test_a_kernel_of_inter_outputs_only():
    calls = []

    class Only:
        adapter = SimpleNamespace()

        def __call__(self, x, acc):
            calls.append(acc.clone())
            acc += x
            return None

    k = InterOutputKernel(Only(), SimpleNamespace(params=[0, 1]), [-1],
                          {1: ((3,), torch.float32, torch.float16)})
    x = torch.ones(3)
    assert torch.equal(k(x), torch.ones(3).half()) and torch.equal(k(x), torch.ones(3).half())


def test_positions_come_from_the_generated_module():
    cand = next(c for c in candidates(inter_pair()) if contexts(c) == ("intra", "inter"))
    prim_func = _load_main_kernel_from_code(build_candidate_code(cand))
    names = [prim_func.buffer_map[p].name for p in prim_func.params]
    out_idx = graph_out_indices(cand.flat_graph)
    inter = prim_func._cleave_inter_outputs
    assert list(inter) == [names[out_idx[1]]] and set(inter.values()) == {"float16"}
    assert _inter_positions(prim_func, inter, out_idx) == {
        out_idx[1] % len(names): ((1024, 64), torch.float32, torch.float16)}
    with pytest.raises(ValueError, match="not among out_idx"):
        _inter_positions(prim_func, inter, out_idx[:1])
    # inputs are still bound by name, the inter buffer is not one of them
    k = SimpleNamespace(adapter=_InterAdapter(None, prim_func, _normalized_out_idx(prim_func, out_idx)))
    feed = {ph.name: ph.name for ph in placeholder_order(cand.flat_graph)}
    assert sorted(order_inputs_for_kernel(feed, k)) == sorted(feed)
