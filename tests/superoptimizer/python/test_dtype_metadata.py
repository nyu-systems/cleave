import cleave.superoptimizer as csopt


def test_dtype_names_use_scheduler_conventions():
    assert str(csopt.float16) == "float16"
    assert str(csopt.bfloat16) == "bfloat16"
    assert str(csopt.float32) == "float32"
    assert str(csopt.float64) == "float64"
    assert csopt.dtype.is_dtype("float32")
    assert not csopt.dtype.is_dtype("fp32")


def test_kernel_graph_dtype_metadata_preserved():
    graph = csopt.new_kernel_graph()

    a = graph.new_input((2, 3), csopt.float32)
    b = graph.exp(a)
    c = graph.add(b, b)

    assert a.dtype == csopt.float32
    assert b.dtype == csopt.float32
    assert c.dtype == csopt.float32

    operators = graph.cygraph.get_graph_structure()
    assert operators[0]["output_tensors"][0]["dtype"] == "float32"
    assert operators[-1]["output_tensors"][0]["dtype"] == "float32"


def test_tile_graph_dtype_metadata_float16_default():
    graph = csopt.new_tile_graph()

    a = graph.create_tiled_input((2, 3))
    b = graph.exp(a)

    assert a.dtype == csopt.float16
    assert b.dtype == csopt.float16
    assert str(a.dtype) == "float16"


def test_tile_graph_dtype_metadata_provided():
    graph = csopt.new_tile_graph(split_enabled=True)

    a = graph.create_tiled_input((2, 3), csopt.float32)
    b = graph.create_tiled_input((2, 3), (False, True), dtype=csopt.bfloat16)
    c = graph.exp(a)
    d = graph.exp(b)

    assert a.dtype == csopt.float32
    assert c.dtype == csopt.float32
    assert b.dtype == csopt.bfloat16
    assert d.dtype == csopt.bfloat16
    assert str(b.dtype) == "bfloat16"
