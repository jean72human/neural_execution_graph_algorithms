import torch 
from copy import deepcopy

def bfs_step(original_data):
    data = deepcopy(original_data)
    u,v = data.edge_index
    terminated = False
    neighbors = v[[index in torch.nonzero(data.x)[:,0] for index in u]]
    data.x[neighbors, :] = 1
    if min(data.x[v[[index in torch.nonzero(data.x)[:,0] for index in u]], :] == 1):
        terminated = True
    return data, int(terminated)