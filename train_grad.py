from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import time
import torch

from utils.utils import count_acc
from pygcn.utils import load_data, convert_to_coo
from pygcn.models import SGCNModel
from grad.attack import GradAttack
from rl.agent import Agent
from rl.env import GCNEnv

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='whether or not normalize feature.')
parser.add_argument('--attack_train', action='store_true', default=False,
                    help='whether or not normalize feature.')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='whether or not normalize feature.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--gpu', type=str, default='0,1', help='gpu number.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--adv_feature_epoch', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--adv_per_iter', type=int, default=25,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--num_adv', type=int, default=500,
                    help='Number of hidden units.')
parser.add_argument('--dqn_hidden', type=int, default=64,
                    help='Number of hidden units of dqn.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--init_feature', type=str, default=None, help='gpu number.')

if __name__ == '__main__':

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(args.gpu)
    args.device = torch.device('cpu')
    # args.device = device = torch.device('cuda:0' if args.cuda else 'cpu')
    args.device1 = torch.device('cuda:1' if args.cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    edge_index, edge_weight = convert_to_coo(adj)
    # add adversarial node
    # features = np.concatenate([features, np.zeros(500, 100)], axis=0)

    features = torch.from_numpy(features).to(args.device).float()
    edge_index = torch.from_numpy(edge_index).to(args.device).long()
    edge_weight = torch.from_numpy(edge_weight).to(args.device).float()
    labels = torch.from_numpy(labels).to(args.device).long()
    # idx_train = idx_train.to(args.device1)
    # idx_val = idx_val.to(args.device1)
    # idx_test = idx_test.to(args.device1)

    # Train model
    t_total = time.time()
    model = SGCNModel(K=2, input_size=100,
                      hidden_size=args.hidden, class_num=18, pre_proj_num=2, after_proj_num=2).to(args.device)
    model.load_state_dict(torch.load('./saved/gcn.pth'))
    model.eval()
    if not args.attack_train:
        idx = np.arange(labels.size(0), features.size(0))
        with torch.no_grad():
            f = features.to(args.device1)
            e_i = edge_index.to(args.device1)
            e_w = edge_weight.to(args.device1)
            tmp_model = model.to(args.device1)
            logits = tmp_model(f, e_i, e_w)
            # logits = model(features, edge_index, edge_weight)
            train_logits = logits[idx]
            lb = torch.argmax(train_logits, dim=1)
            # .to(args.device)
            print(count_acc(logits[idx], lb))
        lb = lb.to(args.device)
        model = model.to(args.device)
    else:
        idx = np.arange(len(labels))
        lb = labels
    attack = GradAttack(args, model)
    attack.attack(features, edge_index, edge_weight, lb, idx)
