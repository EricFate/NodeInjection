from __future__ import print_function

import torch
# import networkx as nx
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear_h = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.linear_h(x))
        x = self.linear_out(x)
        return x

    @torch.no_grad()
    def greedy_action(self, x, valid_actions=None):
        x = self.forward(x)
        if valid_actions is not None:
            val, idx = torch.max(x[valid_actions])
            return valid_actions[idx], val
        else:
            val, idx = torch.max(x)
            return idx, val
