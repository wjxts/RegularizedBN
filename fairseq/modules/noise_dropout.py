# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)


class NoiseDropout(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, inplace: bool = False):
        if self.training:
            coef = (2*torch.rand_like(x)-1).to(x)
            coef *= self.alpha
            x = x*(1+coef)
            return x
        else:
            return x
