from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch

import torch.nn.functional as F

import torch.optim as optim

from pygcn.utils import load_data, accuracy, convert_to_coo, get_data, Dataset
from pygcn.models import GraphSAGE
from pygcn.trainer import GcnTrainer
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=48, help='Random seed.')
parser.add_argument('--gpu', type=str, default='1', help='gpu number.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


def train(model, optimizer, features, edge_index, edge_weight, labels, epoch, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index, edge_weight)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def model_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # Model and optimizer
    model = SGCNModel(K=2, input_size=100,
                      hidden_size=32, class_num=18, pre_proj_num=2, after_proj_num=2)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     args.epochs,
    #     eta_min=0  # a tuning parameter
    # )

    edge_index, edge_weight = convert_to_coo(adj)
    if args.cuda:
        model.cuda()
        features = torch.from_numpy(features).cuda().float()
        edge_index = torch.from_numpy(edge_index).cuda().long()
        edge_weight = torch.from_numpy(edge_weight).cuda()
        labels = torch.from_numpy(labels).cuda().long()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, features, edge_index, edge_weight, labels, epoch, idx_train, idx_val)
        # lr_scheduler.step()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    model_test()


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data = get_data()
    # model = GraphCN()
    model = GraphSAGE()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = GcnTrainer(model, optimizer)
    trainer.train(data, 3000)
    trainer.test(data)
