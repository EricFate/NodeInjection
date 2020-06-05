from __future__ import division
from __future__ import print_function
import torch as t

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, convert_to_coo
from pygcn.models import SGCNModel
import os


class GCNEnv:
    def __init__(self, args, epoch, idx_train, idx_val, labels, **kwargs):
        self.args = args
        self.attributes = ['features', 'edge_index', 'edge_weight']
        self.epoch = epoch
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.labels = labels
        for k in self.attributes:
            assert k in kwargs
            v = kwargs[k]
            setattr(self, k, v)
            setattr(self, 'init_%s' % k, v.copy())

    def reset(self):
        for k in self.attributes:
            setattr(self, k, getattr(self, 'init_%s' % k).copy())

    def step(self, a_node, node, weight, feature):
        '''
        :param a_node: adversarial节点
        :param node: 另一个节点
        :param weight: 添加边的权重
        :param feature: adversarial节点的feature
        :return:
        '''
        new_edge = t.LongTensor([[a_node, node], [node, a_node]]).to(self.args.device)
        self.edge_index = t.cat([self.edge_index, new_edge], dim=1)
        new_weight = t.FloatTensor([weight, weight]).to(self.args.device)
        self.edge_weight = t.cat([self.edge_weight, new_weight], dim=0)
        self.features[a_node] = feature
        model = SGCNModel(K=2, input_size=100,
                          hidden_size=32, class_num=18, pre_proj_num=2, after_proj_num=2).to(self.args.device)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)
        for epoch in range(self.args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[self.idx_train], self.labels[self.idx_train])
            acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            optimizer.step()

            # loss_val = F.cross_entropy(output[self.idx_val], self.labels[self.idx_val])
            acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        state = self.struct2vec()
        return 1 - (acc_val + acc_val), state

    # TODO : struct2vec function
    def struct2vec(self):
        return torch.zeros(120)
