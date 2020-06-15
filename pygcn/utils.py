import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="/home/amax/data/kdd_cup/", dataset="kddcup"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    import pickle as pk
    with open(path + 'experimental_adj.pkl', 'rb') as f:
        adj = pk.load(f)
    with open(path + 'experimental_features.pkl', 'rb') as f:
        features = pk.load(f)
    with open(path + 'experimental_train.pkl', 'rb') as f:
        labels = pk.load(f)

    # features = normalize(features, False)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    total = labels.shape[0]
    idx = np.arange(total)
    np.random.shuffle(idx)
    train_frac = 0.9
    val_frac = 0.95
    train_div = int(total * train_frac)
    val_div = int(total * val_frac)
    idx_train = idx[:train_div]
    idx_val = idx[train_div:val_div]
    idx_test = idx[val_div:]
    #
    # features = torch.FloatTensor(features)
    # labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def convert_to_coo(adj):
    coo_adj = adj.tocoo()
    edge_weight = np.ones_like(coo_adj.data)
    edge_index = np.stack([coo_adj.row, coo_adj.col])
    return edge_index, edge_weight


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range * 100


def normalization_neg(data):
    _range = np.max(data) - np.min(data)
    return ((data - np.min(data)) / _range * 100) - 50


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def get_data(cuda=True, use_adj=False, adv=False):
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    data_len = len(labels)
    # features = normalization_neg(features)
    edge_index, edge_weight = convert_to_coo(adj)

    # features_min = np.min(features, axis=0)
    # features_max = np.min(features, axis=0)
    # print(features_min)
    # print(features_max)
    # print(np.mean(features, axis=0))
    # print(np.var(features,axis=0))
    if use_adj:
        data = Data(x=(torch.from_numpy(features)).float(), edge_index=(torch.from_numpy(adj.toarray())).long(),
                    y=(torch.from_numpy(labels)).long())
    else:
        data = Data(x=(torch.from_numpy(features)).float(), edge_index=(torch.from_numpy(edge_index)).long(),
                    y=(torch.from_numpy(labels)).long(), edge_weight=(torch.from_numpy(edge_weight)).long())
    data.train_mask = idx_train
    data.val_mask = idx_val
    data.test_mask = idx_test
    # if adv:
    #     adv_feature = torch.rand((500, 100)) * 100 - 50
    #     data.x = torch.cat((data.x, adv_feature), dim=0)
    #     # adv_adj_1 = torch.zeros((500, adj.size()[0])).long()
    #     # adv_adj_2 = torch.zeros((500 + adj.size()[0], 500)).long()
    #     # data.edge_index = torch.cat((data.edge_index, adv_adj_1), dim=0)
    #     # data.edge_index = torch.cat((data.edge_index, adv_adj_2), dim=1)
    #     for i in range(500):
    #         adv_feature = torch.rand((500, 100)) * 100 - 50
    #         data.x = torch.cat((data.x, adv_feature), dim=0)
    #         edge_1 = (torch.ones(1, 100) * (data_len + i)).long()
    #         edge_2 = (torch.rand(1, 100) * data_len).long()
    #         edges = torch.cat((edge_1, edge_2), dim=0)
    #         data.edge_index = torch.cat((data.edge_index, edges), dim=1)
    #     data.y = torch.cat((data.y, torch.zeros(500).long()))
    #     data.adv_mask = (torch.LongTensor(np.arange(data_len, 500))).cuda()

    if cuda:
        data.x = data.x.cuda()
        data.edge_index = data.edge_index.cuda()
        data.edge_weight = data.edge_weight.cuda()
        data.y = data.y.cuda()
        data.train_mask = data.train_mask.cuda()
        data.val_mask = data.val_mask.cuda()
        data.test_mask = data.test_mask.cuda()
    return data


class Dataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None):
        super(Dataset, self).__init__(root=None, transform=transform, pre_transform=pre_transform)
        data, idx_train, idx_val, idx_test = get_data()
        self.data, self.slices = self.collate([data])
        self.train_mask = idx_train.cuda()
        self.val_mask = idx_val.cuda()
        self.test_mask = idx_test.cuda()


# def load_data(path="./data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     features = normalize(features, False)
#     adj = normalize(adj + sp.eye(adj.shape[0]))
#
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)
#
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx, symmetric=True):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if symmetric:
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    else:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
