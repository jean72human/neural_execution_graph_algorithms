import random
import networkx
import numpy as np

def add_weights(graph):
    """
    Take in a networkx graph and gives it edge weights
    """
    weights = np.random.uniform(low=0.2, high=1.0, size=len(graph.edges))
    edge_weights = {edge: weights[i] for i, edge in enumerate(graph.edges())}
    networkx.set_edge_attributes(graph, edge_weights, 'edge_weight')
    return graph

def add_source(data):
    """
    Randomly sets a node attribute to 1. That node will serve as source  
    """
    data.x[ random.randrange(0,data.x.size(0)-1), : ] = 1
    return data

def add_source_bf(data):
    """
    Randomly sets a node to 0 and all the other as 1e10. That node will serve as source
    """
    data.x[:,:] = 1e10
    data.x[ random.randrange(0,data.x.size(0)-1), : ] = 0
    return data