#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : MaskBatchNorm.py
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from scipy import io
__all__ = ['MaskAnchorNorm']


def min_len(tensor):
    #tensor: [T,B,C]
    tokens = (tensor!=0).all(dim=-1)
    length = tokens.sum(dim=0).long()
    return length.min()


class MaskAnchorNorm(nn.Module):
    """
    An implementation of masked batch normalization, used for testing the numerical
    stability.
    """
    __count = 0
    def update_cnt(self):
        MaskAnchorNorm.__count += 1
    def __init__(self, num_features, eps=1e-5, momentum=0.05, affine=True, position=0,
        num=1, with_seq = False, prefix = 'None'):
        super().__init__() 
        self.id = self.__count
        self.prefix = prefix
        self.position = position
        self.num = num
        #print(self.id)
        self.update_cnt()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def extra_repr(self):
        return '{num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None, is_encoder=False):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        T, B, C = input.shape
        num = min(min_len(input)-self.position,self.num)
        #input = input.contiguous()
        if self.training:
            data = input[self.position:self.position+num,:,:]
            var, mean = torch.var_mean(data,dim=(0,1))
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            mean, var = self.running_mean, self.running_var
        input = (input - mean)/torch.sqrt(var.clamp(self.eps))
        if self.affine:
            output =  input*self.weight + self.bias
        else:
            output = input
        return output