import importlib

import pytest

import cleave
import cleave.lang as cl

compile_mod = importlib.import_module("cleave.pipeline.compile")


def test_compile_honors_jit_options(monkeypatch):
    seen = {}

    def spy(program, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(compile_mod, "resolve_configs", spy)

    @cleave.jit(preset="fast", split=cl.Split("K", factors=(4,)))
    def mm(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
        return x @ w

    with pytest.raises(RuntimeError, match="stop"):
        cleave.compile(mm, shapes=dict(M=16, N=16, K=64))
    assert seen["preset"] == "fast"
    assert seen["split"].dim == "K"

    seen.clear()
    with pytest.raises(RuntimeError, match="stop"):
        cleave.compile(mm, shapes=dict(M=16, N=16, K=64), preset="exhaustive", split=False)
    assert seen["preset"] == "exhaustive" and seen["split"] is False
