from __future__ import division
from __future__ import print_function
import torch as t

import time
import argparse
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import sys
from pygcn.utils import load_data, accuracy, convert_to_coo, normalize
from pygcn.models import SGCNModel
import os
from utils.utils import count_acc, freeze_model
import copy


class GCNEnv:
    def __init__(self, args, model, labels, adv_start, features, edge_weight, edge_index):
        self.args = args
        self.features = features
        self.edge_weight = edge_weight
        self.edge_index = edge_index
        self.adv_weight = None
        self.init_edge_index = edge_index.clone()
        self.adv_features = t.autograd.Variable(t.randn((args.num_adv, 100))).to(args.device1).requires_grad_(True)
        self.idx = t.arange(len(labels)).to(args.device1).data
        self.adv_start = adv_start
        model.eval()
        # freeze_model(model)
        self.model = model

        self.labels = labels
        self.update_state(self.features, self.edge_index, self.edge_weight)
        self.init_acc = self.pre_acc = self.test_acc()
        # self.init_acc = self.pre_acc = 0.6
        self.reward = 0
        self.count = np.zeros(args.num_adv).astype(np.int)
        self.actions1 = np.arange(args.num_adv)

    def reset(self):
        self.edge_index = self.init_edge_index.clone()
        self.adv_features = t.randn((self.args.num_adv, 100)).to(self.args.device1).requires_grad_(True)
        self.reward = 0

    def update_state(self, features, edge_index, edge_weight):
        self.state = edge_index

    def copy_state(self):
        return self.state.clone()

    def step(self, a_node, node, reward=True):
        '''
        :param a_node: adversarial节点
        :param node: 另一个节点
        :return:
        '''
        print('connect %d and %d' % (a_node, node))
        self.count[a_node] += 1
        if node > self.adv_start:
            self.count[node - self.adv_start] += 1
        new_edge = t.LongTensor([[a_node + self.adv_start, node], [node, a_node + self.adv_start]]).to(
            self.args.device1)
        self.edge_index = t.cat([self.edge_index, new_edge], dim=1)
        if reward or self.isTerminal():
            self.adversarial_train()
            acc = self.test_acc()
            self.reward = self.pre_acc - acc
            self.pre_acc = acc
        self.update_state(self.features, self.edge_index, self.edge_weight)
        print('reward is %.4f' % self.reward)
        return self.state, self.reward

    def compute_reward(self):
        self.adversarial_train()
        acc = self.test_acc()
        return self.init_acc - acc

    @t.no_grad()
    def test_acc(self):
        features = t.cat([self.features, self.adv_features], dim=0)
        # features = self.features
        # features = normalize(features)
        if self.adv_weight is None:
            edge_weight = self.edge_weight
        else:
            edge_weight = t.cat([self.edge_weight, self.adv_weight], dim=0)
        logits = self.model(features, self.edge_index, edge_weight)
        acc = count_acc(logits[self.idx], self.labels)
        t.cuda.empty_cache()
        return acc

    def adversarial_train(self):
        new = self.edge_index.size(1) - len(self.edge_weight)
        # self.adv_weight = t.autograd.Variable(t.randn(new)).to(self.args.device1).requires_grad_(True)
        self.adv_weight = t.ones(new).to(self.args.device1)
        opt = optim.Adam((self.adv_features,), lr=1000)
        features = t.cat([self.features, self.adv_features], dim=0)
        edge_weight = t.cat([self.edge_weight, self.adv_weight], dim=0)
        # t.nn.init.normal_(self.adv_features)
        # adv_feat = t.zeros((self.args.num_adv, 100)).to(self.args.device1)
        # features = t.autograd.Variable(t.cat([self.features, adv_feat], dim=0)).to(self.args.device1).requires_grad_(
        #     True)
        # features = normalize(features)
        for i in range(25):
            logits = self.model(features, self.edge_index, edge_weight)
            loss = - F.cross_entropy(logits[self.idx], self.labels)
            opt.zero_grad()
            # t.cuda.empty_cache()
            loss.backward()
            # opt.step()
            self.adv_features.data.add_(0.1, t.sign(self.adv_features.grad.data))
            # print(count_acc(logits[self.idx], self.labels))
        t.cuda.empty_cache()

    def valid_adv(self):
        return self.actions1[self.count < 100]

    def isTerminal(self):
        return np.all(self.count >= 100)
