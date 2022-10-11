# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


def parse_activation(activation_type):
    args = activation_type.split("_")
    return args

def ActivationSelect(activation_type):
    args = parse_activation(activation_type)
    activation_type = args[0]
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=float(args[1]))
        #default 0.01
    elif activation_type == "prelu":
        return nn.PReLU(init=float(args[1]))
        #default 0.25
    elif activation_type =='celu':
        return nn.CELU(alpha=float(args[1]))
        #default 1
    elif activation_type =='gelu':
        return nn.GELU()
        #no parameter 
    elif activation_type =='elu':
        return nn.ELU(alpha=float(args[1]))
        #default 1
    elif activation_type =='identity':
        return nn.Identity()
    elif activation_type =='sigmoid':
        return nn.Sigmoid()
    elif activation_type =='silu':
        #torch: 1.10才有
        return nn.SiLU()
    elif activation_type =='tanh':
        return nn.Tanh()
    else:
        print("error ActivationSelect!")
        exit()
