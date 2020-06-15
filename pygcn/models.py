import torch.nn as nn
import torch
import torch.nn.functional as F
import tqdm
from pygcn.layers import GraphConvolution
from torch_geometric.nn import SGConv, SAGEConv, GCNConv
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
        self.conv1 = SGConv(hidden_size, hidden_size)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        # self.conv2 = SGConv(128, 128)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        # self.conv3 = SGConv(128, 128)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(hidden_size, 64)
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
        in_channels = 80
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

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
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

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all
