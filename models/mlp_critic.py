import torch.nn as nn
import torch.nn.functional as F
from utils.math import *

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(50,50), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers_v = nn.ModuleList()
        self.bn_layers_v = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers_v.append(nn.Linear(last_dim, nh))
            self.bn_layers_v.append(nn.BatchNorm1d(nh, momentum=0.5))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        # # self.value_head.weight.data.mul_(0.1)
        # self.value_head.weight.data.mul_(1.0)
        # self.value_head.bias.data.mul_(0.0)

        set_init(self.affine_layers_v)
        set_init([self.value_head])

    def forward(self, x):
        self.eval()

        for affine,bn in self.affine_layers_v, self.bn_layers_v:
            x = affine(x)
            x = bn(x)
            x = self.activation(x)

        value = self.value_head(x)
        return value
