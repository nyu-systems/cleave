import cleave.superoptimizer as csopt


def _ops(g):
    return [(op["op_type"], op.get("scalar"), op.get("scalar_on_left")) for op in g.get_graph_structure()]


def test_tile_sub_scalar_is_add_sugar():
    g = csopt.new_tile_graph()
    x = g.create_tiled_input((4, 8))
    g.create_tiled_output(g.sub(x, 2.0))
    g.create_tiled_output(g.sub(2.0, x))

    ref = csopt.new_tile_graph()
    r = ref.create_tiled_input((4, 8))
    ref.create_tiled_output(ref.add(r, -2.0))
    ref.create_tiled_output(ref.add(ref.mul(r, -1.0), 2.0))
    assert _ops(g) == _ops(ref)
    assert all(op != "kn_sub_op" for op, _, _ in _ops(g))


def test_tile_sub_tensors_stays_sub():
    g = csopt.new_tile_graph()
    x, y = g.create_tiled_input((4, 8)), g.create_tiled_input((4, 8))
    g.create_tiled_output(g.sub(x, y))
    assert [op for op, _, _ in _ops(g)].count("kn_sub_op") == 1
