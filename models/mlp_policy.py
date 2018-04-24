import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(50,50), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers_p = nn.ModuleList()
        self.bn_layers_p = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers_p.append(nn.Linear(last_dim, nh))
            self.bn_layers_p.append(nn.BatchNorm1d(nh, momentum=0.5))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        # self.action_mean.weight.data.mul_(0.1)
        # # self.action_mean.weight.data.mul_(1.0)
        # self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

        set_init(self.affine_layers_p)
        set_init([self.action_mean])

    def forward(self, x):
        self.eval()

        for affine, bn in self.affine_layers_p, self.bn_layers_p:
            x = affine(x)
            x = bn(x)
            x = self.activation(x)

        action_mean = F.tanh(self.action_mean(x))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = 2.0*torch.normal(action_mean, action_std)
        action.clamp(-2., 2.)

        return action.data

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}


