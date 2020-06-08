import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
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

    features = normalize(features, False)
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


def get_data():
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    edge_index, edge_weight = convert_to_coo(adj)
    data = Data(x=(torch.from_numpy(features)).float().cuda(), edge_index=(torch.from_numpy(edge_index)).long().cuda())
    return data, labels, idx_train, idx_val, idx_test


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
