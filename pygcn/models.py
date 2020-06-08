import torch.nn as nn
import torch
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
from torch_geometric.nn import SGConv, SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


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

    def forward(self, x, edge_index, edge_weight=None):
        for proj in self.pre_proj_layers:
            x = proj(x)
        x = F.relu(self.conv(x, edge_index))

        for proj in self.after_proj_layers:
            x = proj(x)
        x = self.output_layer(x)
        return x


class SGCNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SGCNet, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_size))
        self.conv1 = SGConv(hidden_size, 128)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        # self.conv2 = SGConv(128, 128)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        # self.conv3 = SGConv(128, 128)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, 18)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        # x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        # x, edge_index, _, _, _, _ = self.pool1(x, edge_index)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(self.conv2(x, edge_index))

        # x, edge_index, _, _, _, _ = self.pool2(x, edge_index)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(self.conv3(x, edge_index))

        # x, edge_index, _, _, _, _ = self.pool3(x, edge_index)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x))

        return x
