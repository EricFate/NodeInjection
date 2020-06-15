import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from torch.nn.parameter import Parameter
from torch_geometric.nn import SGConv, SAGEConv, GCNConv
from tqdm import tqdm

from pygcn.layers import GraphConvolution


class LinearSurrogate(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(t.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def forward(self, x, adj):
        support = t.mm(x, self.weight)
        output = t.spmm(adj, support)
        z = t.spmm(adj, output)
        return z

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    @property
    def W(self):
        return self.weight


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


class SGCNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SGCNet, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(hidden_size))
        self.conv1 = SGConv(hidden_size, hidden_size)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        # self.conv2 = SGConv(128, 128)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        # self.conv3 = SGConv(128, 128)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = t.nn.Linear(hidden_size, 64)
        self.lin2 = t.nn.Linear(64, 32)
        self.lin3 = t.nn.Linear(32, 18)
        self.bn1 = t.nn.BatchNorm1d(64)
        self.bn2 = t.nn.BatchNorm1d(32)
        self.act1 = t.nn.ReLU()
        self.act2 = t.nn.ReLU()

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

        x = t.sigmoid(self.lin3(x))

        return x


class GraphCN(nn.Module):
    def __init__(self):
        super(GraphCN, self).__init__()
        self.layers = nn.ModuleList()
        n_layers = 3
        in_channels = 100
        hidden_channels = 64
        class_num = 18
        bias = True
        # self.embedding = nn.Sequential(nn.Linear(100, in_channels),
        #                                nn.ReLU(), )

        self.layers.append(
            GCNConv(in_channels, hidden_channels, normalize=True, bias=bias))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels, normalize=True, bias=bias))
        # output layer
        self.out_layer = GCNConv(hidden_channels, class_num, normalize=True, bias=bias)
        # activation None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.25, training=self.training)
        x = self.out_layer(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        n_layers = 4
        in_channels = 128
        hidden_channels = 80
        out_channels = 80
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.20, training=self.training)
        x = self.out_layer(x, edge_index)
        return x


class GraphCluster(nn.Module):
    def __init__(self):
        super(GraphCluster, self).__init__()
        n_layers = 7
        in_channels = 128
        hidden_channels = 64
        class_num = 18
        bias = True
        normalize = False
        # self.embedding = nn.Sequential(nn.Linear(100, in_channels),
        #                                nn.ReLU(), )

        self.layers.append(
            SAGEConv(in_channels, hidden_channels, normalize=normalize, bias=bias))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=normalize, bias=bias))
        # output layer
        self.out_layer = SAGEConv(hidden_channels, class_num, normalize=normalize, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.25, training=self.training)
        x = self.out_layer(x, edge_index)
        return x

    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.cuda()
                x = x_all[n_id].cuda()
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = t.cat(xs, dim=0)

        pbar.close()

        return x_all