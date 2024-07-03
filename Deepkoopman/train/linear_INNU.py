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
        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.fix_idx = torch.nonzero(mask).squeeze()
        self.change_idx = torch.nonzero(1 - mask).squeeze()
        if self.change_idx.shape==torch.Size([]):
            self.change_idx = self.change_idx.unsqueeze(0)
        tnsl_layers = inn_lays["tnsl_layers"]
        cmpd_layers = inn_lays["cmpd_layers"]


        TLayers = OrderedDict()
        for layer_i in range(len(tnsl_layers) - 1):
            TLayers["linear_{}".format(layer_i)] = nn.Linear(
                tnsl_layers[layer_i], tnsl_layers[layer_i + 1], bias=False
            )
        self.tnsl_net = nn.Sequential(TLayers)

        CLayers = OrderedDict()
        for layer_i in range(len(cmpd_layers) - 1):
            CLayers["linear_{}".format(layer_i)] = nn.Linear(
                cmpd_layers[layer_i], cmpd_layers[layer_i + 1],    bias=False)
        self.cmpd_net = nn.Sequential(CLayers)

        self.T1 = nn.Linear(tnsl_layers[-1], tnsl_layers[-1],bias=False)
        self.T2 = nn.Linear(cmpd_layers[-1], cmpd_layers[-1],bias=False)
        init.eye_(self.T1.weight)
        init.eye_(self.T2.weight)
  

    def forward(self, z, bu):
        z1 = z[:,self.fix_idx]
        z2 = z[:,self.change_idx]
        v1 = self.T1(z1) + self.tnsl_net(z2) + bu[:,self.fix_idx]
        v2 = self.T2(z2) + self.cmpd_net(v1) + bu[:,self.change_idx]
        zp = torch.zeros_like(z)
        zp[:,self.fix_idx] = v1
        zp[:,self.change_idx] = v2
        return zp   

    def inverse(self, zp, bu):
        z1 = zp[:,self.fix_idx]
        z2 = zp[:,self.change_idx]
        z2_p = z2 - self.cmpd_net(z1) - bu[:,self.change_idx]
        u2 = torch.linalg.solve(self.T2.weight.T, z2_p, left=False)
        z1_p= z1 - self.tnsl_net(u2) - bu[:,self.fix_idx]
        u1 = torch.linalg.solve(self.T1.weight.T, z1_p, left=False)
        z = torch.zeros_like(zp)
        z[:,self.fix_idx] = u1
        z[:,self.change_idx] = u2
        return z
    
    # def forward(self, z, bu):
    #     z1 = z[:,self.fix_idx]
    #     z2 = z[:,self.change_idx]
    #     v1 = z1 + self.tnsl_net(z2) + bu[:,self.fix_idx]
    #     v2 = z2 + self.cmpd_net(v1) + bu[:,self.change_idx]
    #     zp = torch.zeros_like(z)
    #     zp[:,self.fix_idx] = v1
    #     zp[:,self.change_idx] = v2
    #     return zp

    # def inverse(self, zp, bu):
    #     z1 = zp[:,self.fix_idx]
    #     z2 = zp[:,self.change_idx]
    #     u2 = z2 - self.cmpd_net(z1) - bu[:,self.change_idx]
    #     u1 = z1 - self.tnsl_net(u2) - bu[:,self.fix_idx]
    #     z = torch.zeros_like(zp)
    #     z[:,self.fix_idx] = u1
    #     z[:,self.change_idx] = u2
    #     return z

