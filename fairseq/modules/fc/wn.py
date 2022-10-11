import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable

__all__ = ['CWN']

#  norm funcitons--------------------------------


class CWNorm(torch.nn.Module):
    def forward(self, weight):
        weight_ = weight.view(weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_CWN = weight_ / norm
        return weight_CWN.view(weight.size())

#默认scale是1.414，可能是由于relu
class CWN(torch.nn.Linear):
    def __init__(self, in_features, out_features, iscenter=True, bias=True,
                 NScale=1, adjustScale=False, *args, **kwargs):
        super(CWN, self).__init__(in_features, out_features, bias)
        print('CWN:---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = CWNorm()
        if NScale<0:
            self.scale_ = torch.norm(self.weight.data,dim=1,keepdim=True)
        else:
            self.scale_ = torch.ones(out_features, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out

class MultiHeadCWN(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_heads=8, 
                 NScale=1, adjustScale=False, *args, **kwargs):
        super(MultiHeadCWN, self).__init__(in_features, out_features, bias)
        print('MultiHeadCWN:---NScale:', NScale, '---adjust:', adjustScale)
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_features, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        #self.weight = self.weight.reshape(self.in_features*self.num_heads, self.out_features//self.num_heads)
        #self.weight = self.weight.view(self.in_features*self.num_heads, self.out_features//self.num_heads)
        weight_q = self.weight_normalization(self.weight)
        #weight_q = weight_q.reshape(self.in_features, self.out_features)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out

if __name__ == '__main__':
    cwn_ = CWNorm()
    print(cwn_)
    w_ = torch.randn(4, 3, 2)
    w_.requires_grad_()
    y_ = cwn_(w_)
    z_ = y_.view(w_.size(0), -1)
    print(z_.norm(dim=1))

    y_.sum().backward()
    print('w grad', w_.grad.size())