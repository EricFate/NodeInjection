from __future__ import print_function

import os
import sys
import numpy as np
import torch
# import networkx as nx
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
from pygcn.models import SGCNModel
from rl.dqn import DQN
import itertools as it


# from .q_net_node import QNetNode, NStepQNetNode, node_greedy_actions
# from .node_attack_common import load_base_model, NodeAttakEnv, init_setup

# sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
# from common.cmd_args import cmd_args

# sys.path.append('%s/../node_classification' % os.path.dirname(os.path.realpath(__file__)))
# from node_classification.node_utils import run_test, load_raw_graph

# sys.path.append('%s/../graph_attack' % os.path.dirname(os.path.realpath(__file__)))
# from graph_attack.nstep_replay_mem import NstepReplayMem

class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.occ = set()
        self.avail = set(range(memory_size))

    def add(self, record):
        i = self.avail.pop()
        self.memory[i] = record
        self.occ.add(i)

    def sample(self):
        i = self.occ.pop()
        record = self.memory[i]
        self.memory[i] = None
        self.avail.add(i)
        return record


class Agent(object):
    def __init__(self, args, env, target_dict, adv_start):
        self.mem_pool = Memory(memory_size=500000)
        self.env = env
        self.args = args
        self.target_dict = target_dict
        self.actions1 = np.arange(args.num_adv)
        self.actions2 = np.arange(len(target_dict))
        self.adv_start = adv_start

        # self.net = QNetNode(features, labels, list_action_space)
        # self.old_net = QNetNode(features, labels, list_action_space)
        self.encoder = SGCNModel(K=2, input_size=100,
                                 hidden_size=64, class_num=18, pre_proj_num=2, after_proj_num=2).to(args.device)
        self.dqn1 = DQN(args.hidden, args.dqn_hidden, args.num_adv).to(args.device)
        self.dqn1_ = DQN(args.hidden, args.dqn_hidden, args.num_adv).to(args.device)
        self.dqn2 = DQN(2 * args.hidden, args.dqn_hidden, len(target_dict)).to(args.device)
        self.dqn2_ = DQN(2 * args.hidden, args.dqn_hidden, len(target_dict)).to(args.device)

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 100000
        self.burn_in = 10
        self.step = 0
        self.pos = 0
        self.best_eval = None
        self.take_snapshot()

    def take_snapshot(self):
        self.dqn1_.load_state_dict(self.dqn1.state_dict())
        self.dqn2_.load_state_dict(self.dqn2.state_dict())

    def make_actions(self, greedy=False):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                                      * (self.eps_step - max(0., self.step)) / self.eps_step)
        valid_adv = self.env.valid_adv()
        if random.random() < self.eps and not greedy:
            a_1 = np.random.choice(self.actions1)
            while a_1 not in valid_adv:
                a_1 = np.random.choice(self.actions1)
            a_2 = np.random.choice(self.actions2)
            while a_1 + self.adv_start == self.target_dict[a_2]:
                a_2 = np.random.choice(self.actions2)
        else:
            cur_state = self.env.state
            with torch.no_grad():
                embeddings = self.encoder(*cur_state)
            graph_emb = torch.mean(embeddings, dim=1)
            # get a_1
            a_1, _ = self.dqn1.greedy_action(graph_emb, valid_adv)
            emb = torch.cat([graph_emb, embeddings[a_1]])
            a_2, _ = self.dqn2.greedy_action(emb)

        return a_1, a_2

    def run_simulation(self):
        self.env.reset()
        while not self.env.isTerminal():
            a_1, a_2 = self.make_actions()
            s_t = self.env.copy_state()

            _, r = self.env.step(a_1, a_2,reward=False)
            s_tp = self.env.copy_state()
            self.mem_pool.add((s_t, a_1, a_2, s_tp, r))

    def eval(self):
        while not self.env.isTerminal():
            a_1, a_2 = self.make_actions(greedy=True)
            self.env.step(a_1, a_2, reward=False)

        acc = self.env.compute_reward()
        print('\033[93m average test: acc %.5f\033[0m' % (acc))
        # if cmd_args.phase == 'train' and self.best_eval is None or acc < self.best_eval:
        #     print('----saving to best attacker since this is the best attack rate so far.----')
        #     torch.save(self.net.state_dict(), cmd_args.save_dir + '/epoch-best.model')
        #     with open(cmd_args.save_dir + '/epoch-best.txt', 'w') as f:
        #         f.write('%.4f\n' % acc)
        #     with open(cmd_args.save_dir + '/attack_solution.txt', 'w') as f:
        #         for i in range(len(self.idx_meta)):
        #             f.write('%d: [' % self.idx_meta[i])
        #             for e in self.env.modified_list[i].directed_edges:
        #                 f.write('(%d %d)' % e)
        #             f.write('] succ: %d\n' % (self.env.binary_rewards[i]))
        #     self.best_eval = acc

    def train(self):
        pbar = tqdm(range(self.burn_in), unit='batch')
        for p in pbar:
            self.run_simulation()

        pbar = tqdm(range(self.args.num_steps), unit='steps')
        optimizer = optim.Adam(it.chain(self.encoder.parameters(),
                                        self.dqn1.parameters(),
                                        self.dqn2.parameters()), lr=self.args.learning_rate)

        for self.step in pbar:

            self.run_simulation()

            if self.step % 123 == 0:
                self.take_snapshot()
            if self.step % 500 == 0:
                self.eval()

            s_t, a_1, a_2, s_tp, r = self.mem_pool.sample()
            s_t_emb = self.encoder(*s_t)
            graph_emb = torch.mean(s_t_emb, dim=0)
            q_sa_1 = self.dqn1(graph_emb)[a_1]
            _, q1_t_plus_1 = self.dqn1_.greedy_action(s_t_emb)
            emb_2 = torch.cat([graph_emb, s_t_emb[a_1]])
            q_sa_2 = self.dqn1(emb_2)[a_2]
            _, q2_t_plus_1 = self.dqn2_.greedy_action(emb_2)

            target_1 = r + q1_t_plus_1
            target_2 = r + q2_t_plus_1

            # q_sa = torch.cat(q_sa, dim=0)
            loss = F.mse_loss(q_sa_1, target_1) + F.mse_loss(q_sa_2, target_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('eps: %.5f, loss: %0.5f' % (self.eps, loss.item()))
