import torch
from torch import nn
from torch.nn.modules.pooling import MaxPool1d
import torch_geometric.nn as geo_nn

class BaseModel(nn.Module):
    def __init__(self, hidden_dim,last_layer_activation):
        super(BaseModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(hidden_dim+1,hidden_dim)
        self.processor = self._get_layer(hidden_dim,hidden_dim)
        self.decoder = nn.Linear(2*hidden_dim,1)

        self.termination = geo_nn.Sequential("x, edge_index",[
                (lambda x: nn.functional.adaptive_max_pool1d(x.transpose(0,1).unsqueeze(0),1).squeeze(), "x -> x"),
                nn.Linear(2*hidden_dim,1)
        ])

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if last_layer_activation=="sigmoid":
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.ReLU()


    def forward(self, x, edge_index, h=None, edge_features=None):
        if h is None:
            h = torch.zeros_like(x).repeat(1,self.hidden_dim)
        z = self.encoder(
            torch.cat([x,h], dim=1)
        )
        h = self.activation(self.processor(z,edge_index))
        y = self.last_activation(self.decoder(
            torch.cat([h,z], dim=1)
        ))

        h_bar = h.sum(0, keepdim = True).repeat(h.size(0),1)
        h_bar = h_bar/h.size(0)

        t = self.sigmoid(self.termination(
            torch.cat([h,h_bar], dim=1),
            edge_index
        ))

        return y, h, t

    def _get_layer(self, dim_1, dim_2):
        raise NotImplementedError