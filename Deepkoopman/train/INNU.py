import numpy as np
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
from sklearn import datasets
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from collections import OrderedDict

torch.set_default_dtype(torch.float64)  # use double precision numbers

class Affine_Coupling(nn.Module):
    def __init__(self, mask, inn_lays):
        super(Affine_Coupling, self).__init__()
        tnsl_layers = inn_lays["tnsl_layers"]
        cmpd_layers = inn_lays["cmpd_layers"]
        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad=False)

        TLayers = OrderedDict()
        for layer_i in range(len(tnsl_layers) - 1):
            TLayers["linear_{}".format(layer_i)] = nn.Linear(
                tnsl_layers[layer_i], tnsl_layers[layer_i + 1]
            )
            if layer_i != len(tnsl_layers) - 2:
                TLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.tnsl_net = nn.Sequential(TLayers)

        CLayers = OrderedDict()
        for layer_i in range(len(cmpd_layers) - 1):
            CLayers["linear_{}".format(layer_i)] = nn.Linear(
                cmpd_layers[layer_i], cmpd_layers[layer_i + 1]
            )
            if layer_i != len(cmpd_layers) - 2:
                CLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.cmpd_net = nn.Sequential(CLayers)
  

    def forward(self, z, bu):
        v1 = z * self.mask + self.tnsl_net((1 - self.mask) * z) + bu * self.mask
        z_plus = v1 + z * (1 - self.mask) + self.cmpd_net(v1) + bu * (1 - self.mask)
        return z_plus

    def inverse(self, zp, bu):
        u2 = zp * (1 - self.mask) - self.cmpd_net(zp * self.mask) - bu * (1 - self.mask)
        z = u2 + zp * self.mask - self.tnsl_net(u2) - bu * self.mask
        return z
