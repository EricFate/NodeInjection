import torch.nn as nn
import torch
import torch.nn.functional as F
import tqdm
from pygcn.layers import GraphConvolution
from torch_geometric.nn import SGConv, SAGEConv, GCNConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        n_layers = 3
        in_channels = 64
        hidden_channels = 64
        out_channels = 64
        class_num = 18
        bias = True
        normalize = True

        self.embedding = nn.Sequential(nn.Linear(100, in_channels),
                                       nn.ReLU(), )

        self.layers.append(
            SAGEConv(in_channels, hidden_channels, normalize=normalize, bias=bias))
        # hidden layers
        for i in range(n_layers - 1):
            if i != n_layers - 2:
                self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=normalize, bias=bias))
            else:
                self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=normalize, bias=bias))
        # output layer
        self.out_layer = SAGEConv(out_channels, class_num, normalize=normalize, bias=bias)
        # activation None

    def forward(self, x, edge_index, edge_weight):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.20, training=self.training)
        x = self.out_layer(x, edge_index)
        return x
