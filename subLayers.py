#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class complexConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False, groups=1, is_norm=True):
        
        super(complexConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.f_real = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=self.padding, bias =bias, groups=groups)
        self.f_imag = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=self.padding, bias =bias, groups=groups)
        self.norm_r = nn.InstanceNorm2d(self.out_channels)
        self.norm_i = nn.InstanceNorm2d(self.out_channels)
        self.is_norm = is_norm
        
    def forward(self, x):
        b = x.shape[0]
        
        x_real = x.real 
        x_imag = x.imag
        out_real = self.f_real(x_real)
        out_imag = self.f_imag(x_imag) 
        if self.is_norm == True:
            out = torch.complex(self.norm_r(out_real), self.norm_i(out_imag))
        else:
            out = torch.complex(out_real, out_imag)

        return out


class gainLayer(nn.Module):

    def __init__(self, in_features, out_features, init_val):
        
        super(gainLayer, self).__init__()
        self.gain_r = nn.Linear(in_features, out_features, False)
        nn.init.constant_(self.gain_r.weight, init_val)
        self.gain_i = nn.Linear(in_features, out_features, False)
        nn.init.constant_(self.gain_i.weight, 0) 

    def forward(self,x):
        x_real = x.real.permute(0,2,3,1)
        x_imag = x.imag.permute(0,2,3,1)
        out_real = self.gain_r(x_real).permute(0,3,1,2) -  self.gain_i(x_imag).permute(0,3,1,2)
        out_imag = self.gain_r(x_imag).permute(0,3,1,2) +  self.gain_i(x_real).permute(0,3,1,2)

        return torch.complex(out_real, out_imag)

def Ctanh(x):
    x_r = x.real
    x_i = x.imag
    return torch.complex(F.tanh(x_r), F.tanh(x_i))

