import cleave.superoptimizer as csopt

# Empty checkpoint names bypass checkpoint loading and force a fresh search.
NO_CHECKPOINT = ""
EXPECTED_VALID_GRAPHS = 30


def _split_enabled(*can_split_vectors):
    return any(any(can_split) for can_split in can_split_vectors)


def build_gqa_flash_decode():
    split_qk = False
    split_pv = True

    q_can_split = (False, False, False, split_qk)
    k_can_split = (False, False, split_pv, split_qk)
    v_can_split = (False, False, split_pv, False)

    concrete_graph = csopt.new_tile_graph(
        split_enabled=_split_enabled(q_can_split, k_can_split, v_can_split),
    )
    ho, hi_q, hi_kv, s, d = 1, 32, 8, 4096, 64

    Q = concrete_graph.create_tiled_input((ho, hi_q, s, d), q_can_split)
    K = concrete_graph.create_tiled_input((ho, hi_kv, s, d), k_can_split)
    V = concrete_graph.create_tiled_input((ho, hi_kv, s, d), v_can_split)

    A = concrete_graph.matmul(Q, K, trans_B=True)
    A = concrete_graph.div(A, (d**0.5))
    A = concrete_graph.add(A, -50 / (d**0.5))
    E = concrete_graph.exp(A)
    S = concrete_graph.reduction(E, 3)
    D = concrete_graph.div(E, S)
    O = concrete_graph.matmul(D, V)
    concrete_graph.create_tiled_output(O)

    symbolic_graph, symbol_map = csopt.create_symbolic_tile_graph(concrete_graph)
    assert set(symbol_map.values()) == {ho, hi_q, hi_kv, s, d}

    config = symbolic_graph.setup_search_config()
    config.search_thread = 16
    config.max_num_kernel_graph_op = 14

    return symbolic_graph.search_tile_graph(NO_CHECKPOINT, False, True, config)


def test_symbolic_gqa_flash_decode_valid_graph_count():
    assert len(build_gqa_flash_decode()) == EXPECTED_VALID_GRAPHS
