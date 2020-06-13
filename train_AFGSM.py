from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import time
import torch

from utils.utils import count_acc
from pygcn.utils import load_data, convert_to_coo
from pygcn.models import LinearSurrogate
from rl.agent import Agent
from rl.env import GCNEnv
from afgsm.attack import AFGSM

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--gpu', type=str, default='0,1', help='gpu number.')
parser.add_argument('--retrain_epoch', type=int, default=500,
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

if __name__ == '__main__':

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(args.gpu)
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # add adversarial node
    # features = np.concatenate([features, np.zeros(500, 100)], axis=0)

    # Train model
    t_total = time.time()
    model = LinearSurrogate(100, 18).to(args.device)

    attacker = AFGSM(args)
    attacker.attack(model, features, adj, labels, np.arange(len(labels)))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
