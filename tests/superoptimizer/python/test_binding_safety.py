"""Binding-level safety: user errors raise (never abort), wrappers keep owners alive,
dtype survives symbolize+search, and stale checkpoints are re-searched."""

import gc
import os
import subprocess
import sys
import textwrap

import pytest

import cleave.superoptimizer as csopt


def _run_isolated(code):
    # A C++ abort would kill pytest itself, so probe in a child interpreter.
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )


@pytest.mark.parametrize(
    "body, exc",
    [
        ("g = csopt.new_kernel_graph(); a = g.new_input((4, 8)); g.matmul(a, a)", "ValueError"),
        ("g = csopt.new_kernel_graph(); a = g.new_input((4, 8)); g.reduce_sum(a, 7)", "ValueError"),
        ("g = csopt.new_kernel_graph(); a = g.new_input((4, 8)); g.reduce_sum(a, -1)", "ValueError"),
        ("g = csopt.new_kernel_graph(); g.mark_output(None)", "ValueError"),
        ("g = csopt.new_kernel_graph(); g.exp(csopt.DTensor(None))", "ValueError"),
        ("g = csopt.new_tile_graph(); g.create_tiled_input((4, 8), (True, False))", "ValueError"),
        ("g = csopt.new_tile_graph(); g.search_tile_graph('')", "RuntimeError"),
        ("csopt.core.search(None)", "TypeError"),
    ],
)
def test_bad_input_raises_not_aborts(body, exc):
    proc = _run_isolated(
        f"""
        import cleave.superoptimizer as csopt
        try:
            {body}
        except {exc} as e:
            print("RAISED", type(e).__name__)
        else:
            print("NO-RAISE")
        """
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert f"RAISED {exc}" in proc.stdout, proc.stdout + proc.stderr[-2000:]


@pytest.mark.parametrize("factory", ["kernel", "tile"])
def test_more_than_max_dims_value_error(factory):
    dims = (2,) * 6
    if factory == "kernel":
        g = csopt.new_kernel_graph()
        with pytest.raises(ValueError, match="at most 5 dims"):
            g.new_input(dims)
    else:
        g = csopt.new_tile_graph()
        with pytest.raises(ValueError, match="at most 5 dims"):
            g.create_tiled_input(dims)
    # Exactly MAX_TENSOR_DIMS still works.
    ok = (g.new_input((2,) * 5) if factory == "kernel" else g.create_tiled_input((2,) * 5))
    assert ok.num_dims == 5


def test_dtensor_survives_graph_wrapper_gc():
    g = csopt.new_kernel_graph()
    a = g.new_input((4, 8), csopt.float32)
    b = g.exp(a)
    del g
    gc.collect()
    assert b.dim(1) == 8 and b.dtype == csopt.float32

    t = csopt.new_tile_graph()
    x = t.create_tiled_input((4, 8), csopt.bfloat16)
    y = t.exp(x)
    del t, x
    gc.collect()
    assert y.dim(0) == 4 and y.dtype == csopt.bfloat16


def _build_symbolic_exp_rowsum(dtype=csopt.float32):
    concrete = csopt.new_tile_graph()
    X = concrete.create_tiled_input((16, 64), dtype)
    concrete.create_tiled_output(concrete.reduce_sum(concrete.exp(X), 1))
    symbolic, _ = csopt.create_symbolic_tile_graph(concrete)
    config = symbolic.setup_search_config()
    config.search_thread = 4
    return symbolic, config


def test_dtype_after_symbolize_and_search(tmp_path):
    symbolic, config = _build_symbolic_exp_rowsum(csopt.float32)
    graphs = symbolic.search_tile_graph(str(tmp_path / "ckpt.json"), config=config)
    assert graphs
    for g in graphs:
        inputs = g.get_input_dtensors()
        assert inputs and all(t.dtype == csopt.float32 for t in inputs)
        structure = g.get_graph_structure()
        assert structure[0]["output_tensors"][0]["dtype"] == "float32"
    # The checkpoint round-trip keeps dtype too.
    reloaded = symbolic.search_tile_graph(str(tmp_path / "ckpt.json"), config=config)
    assert len(reloaded) == len(graphs)
    assert all(t.dtype == csopt.float32 for g in reloaded for t in g.get_input_dtensors())
    # A searched graph's tensors outlive the result list.
    t = graphs[0].get_input_dtensors()[0]
    del graphs, reloaded
    gc.collect()
    assert t.dtype == csopt.float32


def test_checkpoint_key_mismatch_researches(tmp_path, capsys):
    ckpt = tmp_path / "ckpt.json"
    key_path = tmp_path / "ckpt.json.key"
    symbolic, config = _build_symbolic_exp_rowsum()
    n = len(symbolic.search_tile_graph(str(ckpt), config=config))
    assert n > 0 and ckpt.exists()
    key = key_path.read_text().strip()
    assert key == symbolic.checkpoint_key(config=config)
    capsys.readouterr()

    # Matching key: the (deliberately emptied) checkpoint is loaded as-is.
    ckpt.write_text("[]")
    assert symbolic.search_tile_graph(str(ckpt), config=config) == []
    assert "stale checkpoint" not in capsys.readouterr().out

    # Mismatched key: ignored, re-searched, checkpoint and key rewritten.
    key_path.write_text("0" * 64 + "\n")
    again = symbolic.search_tile_graph(str(ckpt), config=config)
    assert len(again) == n
    assert "stale checkpoint" in capsys.readouterr().out
    assert key_path.read_text().strip() == key
    assert ckpt.read_text() != "[]"

    # Missing key file is treated as a mismatch.
    key_path.unlink()
    ckpt.write_text("[]")
    assert len(symbolic.search_tile_graph(str(ckpt), config=config)) == n

    # Key depends on config and on the build.
    other = symbolic.setup_search_config()
    other.max_num_kernel_graph_op = config.max_num_kernel_graph_op + 1
    assert symbolic.checkpoint_key(config=other) != key
    assert len(csopt.core._build_id()) == 16
