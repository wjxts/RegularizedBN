# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from .fc.wn import CWN
from .fc.conv import Conv1d
from .fc.dropout_fc import DropoutFC
from .fc.oni_fc import ONI_Linear
def parse_fc(fc_type):
    args = fc_type.split("_")
    return args
def FcSelect(input_dim, output_dim, fc_type, layer_id=-1):
    args = parse_fc(fc_type)
    fc_type = args[0]
    #print(args);exit()
    if fc_type == "nnlinear":
        return nn.Linear(input_dim, output_dim)
    elif fc_type == "linear":
        if len(args)==1:
            return Linear(input_dim, output_dim)
        if len(args)==2:
            return Linear(input_dim, output_dim, scale=float(args[1]))
        if len(args)==3:
            return Linear(input_dim, output_dim, scale=float(args[1]), zero_bias=int(args[2]))
        else:
            print("error len linear!")
    elif fc_type == "dpfc":
        if args[-1]=='select':
            if args[-2].find(str(layer_id))>=0:
                return DropoutFC(input_dim, output_dim, dropout=float(args[1]),scale=float(args[2]))
            else:
                return DropoutFC(input_dim, output_dim, dropout=0.0,scale=float(args[2]))
        if len(args)==1:
            return DropoutFC(input_dim, output_dim, dropout=float(args[1]))
        if len(args)==2:
            return DropoutFC(input_dim, output_dim, dropout=float(args[1]), scale=float(args[2]))
        else:
            print("error len dpfc!")
    elif fc_type == "onifc":
        if args[-1]=='select':
            if args[-2].find(str(layer_id))>=0:
                return ONI_Linear(input_dim, output_dim)
            else:
                return Linear(input_dim, output_dim, scale=float(args[1]))
                
        if len(args)==1:
            return Linear(input_dim, output_dim, scale=float(args[1]))
        else:
            print("error len dpfc!")

    elif fc_type == "conv1d":
        return Conv1d(input_dim,output_dim,kernel_size=int(args[1]))
    elif fc_type == "wn":
        return CWN(input_dim, output_dim, iscenter=int(args[1]), Nscale=float(args[2]), adjustScale=int(args[3]))
    else:
        print("error FcSelect!")
        exit()
    #elif norm_type == 'power':
    #    return MaskPowerNorm(embed_dim, group_num=head_num, warmup_iters=warmup_updates)
    
def Linear(in_features, out_features, scale=1.0, zero_bias=True):
    m = nn.Linear(in_features, out_features, bias=True)
    nn.init.xavier_uniform_(m.weight,gain=scale)
    if zero_bias:
        nn.init.constant_(m.bias, 0.0)
    return m