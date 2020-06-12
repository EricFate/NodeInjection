from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T_co
import torch as t
import torch.sparse as sp

from pygcn.layers import GraphConvolution
from torch_geometric.nn import SGConv


class AdjMlp(nn.Module):

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.weight = nn.Parameter(t.FloatTensor(in_features, out_features))

    def forward(self, adj, size):
        val = t.ones(size)
        x = sp.FloatTensor(adj, val, size=(size, size))
        x = sp.mm(x, self.weight)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        return x


class SGCNModel(nn.Module):
    def __init__(self, K, input_size, hidden_size, pre_proj_num, after_proj_num, class_num):
        super(SGCNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.pre_proj_layers = nn.ModuleList()
        self.pre_proj_layers.append(nn.Linear(input_size, hidden_size))
        for i in range(pre_proj_num):
            self.pre_proj_layers.append(nn.Linear(hidden_size, hidden_size))
            self.pre_proj_layers.append(nn.ReLU())
            self.pre_proj_layers.append(nn.BatchNorm1d(hidden_size))
        self.conv = SGConv(hidden_size, hidden_size)
        self.after_proj_layers = nn.ModuleList()
        for i in range(after_proj_num):
            self.after_proj_layers.append(nn.Linear(hidden_size, hidden_size))
            self.after_proj_layers.append(nn.ReLU())
            self.after_proj_layers.append(nn.BatchNorm1d(hidden_size))

        self.output_layer = nn.Linear(hidden_size, class_num)

    def forward(self, x, edge_index, edge_weight=None, feature=False):
        for proj in self.pre_proj_layers:
            x = proj(x)
        # x1 = x
        c = self.conv(x, edge_index, edge_weight)
        x = F.relu(c)
        c = self.conv(x, edge_index, edge_weight)
        x = F.relu(c)
        # + x1
        for proj in self.after_proj_layers:
            x = proj(x)
        if feature:
            return x
        x = self.output_layer(x)
        return x
