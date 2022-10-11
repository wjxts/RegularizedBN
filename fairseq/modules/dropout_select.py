# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from .noise_dropout import NoiseDropout

def parse_dropout(dropout_type):
    args = dropout_type.split("_")
    return args

def DropoutSelect(dropout_type):
    args = parse_dropout(dropout_type)
    dropout_type = args[0]
    if dropout_type == "hard":
        return nn.Dropout(p=args[1])
    elif dropout_type == "noise":
        return NoiseDropout(alpha=args[1])
        
    else:
        print("error DropoutSelect!")
        exit()
