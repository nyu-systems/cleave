DEFAULT_MAIN_GRAPH_THREADS = 128


def ensure_main_graph_threads(graph):
    if isinstance(graph, (list, tuple)):
        for item in graph:
            ensure_main_graph_threads(item)
        return graph

    if not hasattr(graph, "n_threads"):
        graph.n_threads = DEFAULT_MAIN_GRAPH_THREADS
    return graph
