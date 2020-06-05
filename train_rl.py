from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import time
import torch

from env import GCNEnv
from pygcn.utils import load_data, convert_to_coo

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--gpu', type=str, default='1', help='gpu number.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

if __name__ == '__main__':

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    edge_index, edge_weight = convert_to_coo(adj)
    # add adversarial node
    features = np.concatenate([features, np.zeros(500, 100)], axis=0)

    if args.cuda:
        features = torch.from_numpy(features).cuda().float()
        edge_index = torch.from_numpy(edge_index).cuda().long()
        edge_weight = torch.from_numpy(edge_weight).cuda()
        labels = torch.from_numpy(labels).cuda().long()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()

    env = GCNEnv(args, args.epochs, idx_train, idx_val, labels, features=features, edge_index=edge_index,
                 edge_weight=edge_weight)

    # ----------------- rl code ------------------ #

    # ----------------- rl code ------------------ #

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
