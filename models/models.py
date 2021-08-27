import torch
from torch import nn
from torch_geometric import nn as geo_nn
from .base_model import BaseModel
from .layers import MPNN

class GATNeuralAlgorithm(BaseModel):
    def __init__(self, hidden_dim=32, last_layer_activation = "sigmoid"):
        super().__init__(hidden_dim=hidden_dim, last_layer_activation = "sigmoid")

    def _get_layer(self, dim_1, dim_2):
        return geo_nn.GATConv(dim_1, dim_2).jittable()

class MPNNsumNeuralAlgorithm(BaseModel):
    def __init__(self, hidden_dim=32, last_layer_activation = "sigmoid"):
        super().__init__(hidden_dim=hidden_dim, last_layer_activation = "sigmoid")

    def _get_layer(self, dim_1, dim_2):
        return MPNN(dim_1, dim_2, aggr="add").jittable()

class MPNNmaxNeuralAlgorithm(BaseModel):
    def __init__(self, hidden_dim=32, last_layer_activation = "sigmoid"):
        super().__init__(hidden_dim=hidden_dim, last_layer_activation = "sigmoid")

    def _get_layer(self, dim_1, dim_2):
        return MPNN(dim_1, dim_2, aggr="max").jittable()

class MPNNmeanNeuralAlgorithm(BaseModel):
    def __init__(self, hidden_dim=32, last_layer_activation = "sigmoid"):
        super().__init__(hidden_dim=hidden_dim, last_layer_activation = "sigmoid")

    def _get_layer(self, dim_1, dim_2):
        return MPNN(dim_1, dim_2, aggr="mean").jittable()
