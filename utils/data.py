from networkx import generators
import torch
from .graph_generators import gen_erdos_renyi, gen_grid2D, gen_ladder, gen_tree, gen_barabasi_albert
from .algorithms import bfs_step
from .transforms import add_source
import numpy as np

def generator(algo,graph_type,n_nodes,n_graphs, seed=72):
    items = []

    if graph_type == "erdos renyi":
        generator = gen_erdos_renyi
    if graph_type == "grid":
        generator = gen_grid2D
    if graph_type == "ladder":
        generator = gen_ladder
    if graph_type == "tree":
        generator = gen_tree
    if graph_type == "barabasi albert":
        generator = gen_barabasi_albert

    if algo == "bfs":
        algo_step = bfs_step

    random_state = np.random.RandomState(seed)

    for i in range(n_graphs):

        graph = generator(n=n_nodes, seed=random_state)
        terminated = 0

        if algo == "bfs":
            graph = add_source(graph)

        item = [(graph,torch.Tensor([terminated]))]
        while not terminated:
            graph, terminated = bfs_step(graph)
            item.append((graph, torch.Tensor([terminated])))

        items.append(item)
    
    return items

