#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : MaskBatchNorm.py
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['MaskIdentity']

class MaskIdentityNorm(nn.Module):
    """
    """

    def __init__(self, num_features,affine=False):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        
    def extra_repr(self):
        return '{num_features},' \
               'affine={affine}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None, is_encoder=False):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        return input