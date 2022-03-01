import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions


## Coupling layers
class Coupling_layer_NICE(nn.Module):
    def __init__(self, input_dim, n_layers, mask_type, hidden_dim=1024, device="cpu"):
        super(Coupling_layer_NICE, self).__init__()

        self.mask = self.get_mask(input_dim, mask_type).to(device)

        a = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)]
        for i in range(n_layers - 2):
            a += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1)]

        a += [nn.Linear(hidden_dim, input_dim)]
        self.a = nn.Sequential(*a)

    def forward(self, x, reverse=False):
        if reverse:
            x = x.view(x.shape[0], -1)
            h1, h2 = x * self.mask, x * (1 - self.mask)

            m = self.a(h1) * (1 - self.mask)
            h2 = h2 - m

            x = h1 + h2

            return x.view(x.shape)

        else:
            z = x.view(x.shape[0], -1)
            h1, h2 = z * self.mask, z * (1 - self.mask)

            m = self.a(h1) * (1 - self.mask)
            h2 = h2 + m

            z = h1 + h2

            return z.view(x.shape), 0

    def get_mask(self, input_dim, mask_type):
        self.mask = torch.zeros(input_dim)
        if mask_type == 0:
            self.mask[::2] = 1
        elif mask_type == 1:
            self.mask[1::2] = 1
        return self.mask.view(1, -1).float()


## Models
class NICE(nn.Module):
    def __init__(self, input_dim, n_layers, n_couplings, hidden_dim):
        super(NICE, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim

        self.layers = []
        for i in range(n_couplings):
            self.layers += [
                Coupling_layer_NICE(
                    input_dim, n_layers, i % 2, hidden_dim, device
                ).float()
            ]

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, reverse=False):
        if reverse:
            z = x.view(-1, self.input_dim).float()
            for layer in self.layers[::-1]:
                z = layer(z, reverse=True)
            return z, 0
        else:
            logdetJ_ac = 0
            x = x.view(-1, self.input_dim).float()

            for layer in self.layers:
                x, logdetJ = layer(x)
                logdetJ_ac += logdetJ

            return x, logdetJ_ac
