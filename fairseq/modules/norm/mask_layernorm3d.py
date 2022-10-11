#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : MaskBatchNorm.py
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
import numpy as np
from scipy import io
__all__ = ['MaskLayerNorm3d']


class MaskLayerNorm3d(nn.Module):
    """
    An implementation of masked batch normalization, used for testing the numerical
    stability.
    """
    __count = 0
    def update_cnt(self):
        MaskLayerNorm3d.__count += 1
    def __init__(self, num_features, eps=1e-5, \
        affine=True, with_seq = False, prefix = 'None'):
        super().__init__() 
        self.id = self.__count
        self.prefix = prefix
        self.update_cnt()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def extra_repr(self):
        return '{num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        #print(input.shape)#torch.Size([26, 2, 128])
        T, B, C = input.shape
        #input = input.contiguous()
        # construct the mask_input, size to be (BxL) x C: L is the real length here
        # Compute the statistics
        mean = input.mean(dim=2,keepdim=True)
        #var = ((input-mean) ** 2).mean(dim=2,keepdim=True)+self.eps
        var = input.var(dim=2,keepdim=True)+self.eps
        output = (input - mean) / torch.sqrt(var)
        if self.affine:
            output =  output*self.weight + self.bias
        return output