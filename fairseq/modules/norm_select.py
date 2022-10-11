# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .norm.mask_layernorm3d import MaskLayerNorm3d
from .norm.mask_batchnorm3d import MaskBatchNorm3d
from .norm.mask_powernorm3d import MaskPowerNorm3d
from .norm.mask_groupnorm import GroupNorm
from .norm.mask_groupscale import MaskGroupScale
from .norm.mask_identity import MaskIdentityNorm
from torch import nn

def parse_norm(norm):
    args = norm.split("_")
    return args
def NormSelect(norm_type, embed_dim, layer_id=-1, prefix='None'):
    #print(norm_type)
    args = parse_norm(norm_type)
    norm_type = args[0] 
    if norm_type == "layer":
        if len(args)==1:
            print("nn.layernorm")
            return nn.LayerNorm(embed_dim)
            #return MaskLayerNorm3d(embed_dim)
        elif len(args)==2:
            return MaskLayerNorm3d(embed_dim, affine=int(args[1]))
        else:
            return MaskLayerNorm3d(embed_dim, affine=int(args[1]), square=int(args[2]))
    elif norm_type == "identity":
        return MaskIdentityNorm(embed_dim)
    elif norm_type == "group":
        #assert len(args)==6, "wrong groupnorm argument!"
        return GroupNorm(embed_dim, num_groups=int(args[1]))
        #return GroupNorm(embed_dim, num_groups=int(args[1]), affine=int(args[2]) ,subtract_type=args[3], robust_mean=int(args[4]),robust_std=int(args[5]))
    elif norm_type == "power":
        if args[-1]=='select':
            if args[-2].find(str(layer_id))>=0:
                return MaskPowerNorm3d(embed_dim, prefix=prefix,  penalty_var=float(args[1]))
            else:
                return MaskLayerNorm3d(embed_dim, affine=1)
        if len(args)==1:
            return MaskPowerNorm3d(embed_dim, prefix=prefix)
        elif len(args)==2:
            return MaskPowerNorm3d(embed_dim, prefix=prefix,  penalty_var=float(args[1]))

    elif norm_type == "batch":
        # return MaskBatchNorm(embed_dim)
        if len(args)==3:
            return MaskBatchNorm3d(embed_dim, affine=int(args[1]), with_seq=int(args[2]), prefix=prefix)
        elif len(args)==4:
            return MaskBatchNorm3d(embed_dim, penalty_type=args[1], penalty_mean=float(args[2]), penalty_var=float(args[3]), prefix=prefix)
        else:
            print("error len BN!")
            
    elif norm_type == "groupscale":
        print("groupscale")
        return MaskGroupScale(embed_dim, group_num=8)
    else:
        print("error NormSelect!")
        exit()
    #elif norm_type == 'power':
    #    return MaskPowerNorm(embed_dim, group_num=head_num, warmup_iters=warmup_updates)
    
