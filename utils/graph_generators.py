from networkx.generators.random_graphs import fast_gnp_random_graph, barabasi_albert_graph
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.classic import ladder_graph
from networkx.generators.trees import random_tree
from torch_geometric.utils import from_networkx
import torch_geometric
import torch
import math
import random

def gen_erdos_renyi(n=20,seed=72):
    """
    Function to generate an Erdos Renyi graph with n nodes
    """
    p = min(
        math.log(n,2)/n,
        0.5
    )
    G = fast_gnp_random_graph(n,p,seed=seed)
    graph = from_networkx(G)
    graph.x = torch.tensor([[0]]*n)
    graph.edge_index = torch_geometric.utils.add_self_loops(graph.edge_index)[0]
    return graph

def gen_barabasi_albert(n=20,seed=72):
    """
    Function to generate an Barabasi Albert graph with n nodes
    """
    G = barabasi_albert_graph(n,random.choice([4,5]),seed=seed)
    graph = from_networkx(G)
    graph.x = torch.tensor([[0]]*n)
    graph.edge_index = torch_geometric.utils.add_self_loops(graph.edge_index)[0]
    return graph

def gen_grid2D(n=(5,4),seed=72):
    """
    Function to generate a 2D grid graph with n nodes
    """
    assert isinstance(n,tuple), "First input should be a tuple"

    dim1,dim2 = n

    G = grid_2d_graph(dim1,dim2)
    graph = from_networkx(G)
    graph.x = torch.tensor([[0]]*dim1*dim2)
    graph.edge_index = torch_geometric.utils.add_self_loops(graph.edge_index)[0]
    return graph

def gen_ladder(n=20,seed=72):
    """
    Function to generate a ladder graph with n nodes
    """
    assert n%2==0, "number of nodes cannot build a ladder graph"
    k = n//2
    G = ladder_graph(k)
    graph = from_networkx(G)
    graph.x = torch.tensor([[0]]*n)
    graph.edge_index = torch_geometric.utils.add_self_loops(graph.edge_index)[0]
    return graph

def gen_tree(n=20,seed=72):
    """
    Function to generate a tree that follows a Prüfer sequence with n nodes
    """
    G = random_tree(n)
    graph = from_networkx(G)
    graph.x = torch.tensor([[0]]*n)
    graph.edge_index = torch_geometric.utils.add_self_loops(graph.edge_index)[0]
    return graph
