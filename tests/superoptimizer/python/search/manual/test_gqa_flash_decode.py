import cleave.superoptimizer as csopt

# Empty checkpoint names bypass checkpoint loading and force a fresh search.
NO_CHECKPOINT = ""
EXPECTED_VALID_GRAPHS = 30


def build_gqa_flash_decode():
    split_qk = False
    split_pv = True

    graph = csopt.new_tile_graph(
        split_enabled=(split_qk or split_pv),
        symbolic=True,
    )
    # ho, hi_q, hi_kv, s, d = 5, 2, 1, 16, 8 # s will collide with d on split
    ho, hi_q, hi_kv, s, d = 1, 12, 3, 10, 20

    q_can_split = (False, False, False, split_qk)
    k_can_split = (False, False, split_pv, split_qk)
    v_can_split = (False, False, split_pv, False)
    Q = graph.create_tiled_input((ho, hi_q, s, d), q_can_split)
    K = graph.create_tiled_input((ho, hi_kv, s, d), k_can_split)
    V = graph.create_tiled_input((ho, hi_kv, s, d), v_can_split)

    A = graph.matmul(Q, K, trans_B=True)
    A = graph.div(A, (d**0.5))
    A = graph.add(A, -50 / (d**0.5))
    E = graph.exp(A)
    S = graph.reduction(E, 3)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.create_tiled_output(O)

    config = graph.setup_search_config()
    config.search_thread = 16
    config.max_num_kernel_graph_op = 14

    return graph.search_tile_graph(NO_CHECKPOINT, False, True, config)


def test_gqa_flash_decode_valid_graph_count():
    assert len(build_gqa_flash_decode()) == EXPECTED_VALID_GRAPHS
