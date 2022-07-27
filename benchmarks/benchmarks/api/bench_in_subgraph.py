import time
import dgl
import torch
import numpy as np
import dgl.function as fn


from .. import utils


@utils.benchmark('time')
@utils.parametrize('graph_name', ['livejournal', 'reddit'])
@utils.parametrize('format', ['csc'])  # coo is not supported
@utils.parametrize('seed_nodes_num', [200, 5000, 20000])
def track_time(graph_name, format, seed_nodes_num):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)

    seed_nodes = np.random.randint(0, graph.num_nodes(), seed_nodes_num)

    # dry run
    for i in range(3):
        dgl.in_subgraph(graph, seed_nodes)

    # timing
    with utils.Timer() as t:
        for i in range(3):
            dgl.in_subgraph(graph, seed_nodes)

    return t.elapsed_secs / 3
