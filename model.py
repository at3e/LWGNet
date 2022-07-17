#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 02:11:33 2020

@author: at3ee
"""
import torch
import torch.nn as nn
from utils import init_obj_field
from layers import forwardLayerWF
from subLayers import gainLayer, complexConv, Ctanh

class reconModel(nn.Module):
    def __init__(self, opts, device):
        super(reconModel, self).__init__()
        self.device = device
        self.CTF = opts['P']
        self.N_obj = opts["N_obj"]
        self.nlayers = opts['numLayers']
        self.init_val = [0.01, 0.015, 0.02, 0.04, 0.05]
        self.forwardLayer1 = nn.ModuleList([forwardLayerWF(opts, self.CTF, self.device).to(self.device) for _ in range(self.nlayers)])
        self.convLayer1 = nn.ModuleList([complexConv(in_channels=opts['channels'], out_channels=128, kernel_size=(3, 3), padding=1, bias=False).to(self.device) for i in range(self.nlayers)])
        self.convLayer2 = nn.ModuleList([complexConv(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, bias=False).to(self.device) for i in range(self.nlayers)])
        self.convLayer3 = nn.ModuleList([complexConv(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False).to(self.device) for i in range(self.nlayers)])
        self.Lin1 = nn.ModuleList([gainLayer(in_features=32, out_features=1, init_val=self.init_val[i]).to(self.device) for i in range(self.nlayers)])

    def forward(self, I, Ns):

        cen0 = tuple(n//2 for n in self.N_obj)
        x_init = init_obj_field(I[:, 0, :, :], scale=2)
        x = torch.tensor(x_init, dtype = torch.cfloat, device=self.device)
        
        for layer, conv1, conv2, conv3, gain1 in zip(self.forwardLayer1, self.convLayer1, self.convLayer2, self.convLayer3, self.Lin1):
            phi = x.to(self.device)
            dx = layer(phi, I.to(self.device), Ns.squeeze())
            dx = conv1(dx)
            dx = Ctanh(dx)
            dx = conv2(dx)
            dx = Ctanh(dx)
            dx = conv3(dx)
            dx = Ctanh(dx)
            x = x.unsqueeze(1) - gain1(dx)

            x = x.squeeze(1)

        x = x.unsqueeze(1)

        return x
