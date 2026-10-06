import cleave.superoptimizer as csopt

# Empty checkpoint names bypass checkpoint loading and force a fresh search.
NO_CHECKPOINT = ""
EXPECTED_VALID_GRAPHS = 2


def build_mha_prefill_bhsd():
    split_qk = False
    split_pv = False

    graph = csopt.new_tile_graph(
        split_enabled=(split_qk or split_pv),
        symbolic=True,
    )
    n_symbols = 5
    b, h, s_q, s_kv, hd = (i * 2 for i in range(2, n_symbols * 2 + 1, 2))
    norm_factor = (hd**0.5)

    q_can_split = (False, False, split_qk)
    k_can_split = (False, split_pv, split_qk)
    v_can_split = (False, split_pv, False)
    Q = graph.create_tiled_input((b*h, s_q, hd), q_can_split)
    K = graph.create_tiled_input((b*h, s_kv, hd), k_can_split)
    V = graph.create_tiled_input((b*h, s_kv, hd), v_can_split)

    A = graph.matmul(Q, K, trans_B=True)
    A = graph.div(A, norm_factor)
    A = graph.add(A, -50 / norm_factor)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.create_tiled_output(O)

    config = graph.setup_search_config()
    config.search_thread = 16

    return graph.search_tile_graph(NO_CHECKPOINT, False, True, config)


def test_mha_prefill_bhsd_valid_graph_count():
    assert len(build_mha_prefill_bhsd()) == EXPECTED_VALID_GRAPHS
