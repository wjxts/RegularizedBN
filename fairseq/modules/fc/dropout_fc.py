import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['DropoutFC']


class DropoutFC(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0, scale=1.0):
        super(DropoutFC, self).__init__(in_features, out_features, bias)
        print('DropoutFC dropout:{}, scale:{}'.format(dropout,scale))
        self.weight_dropout_module = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.weight,gain=scale)
        nn.init.constant_(self.bias, 0.0)
    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_dropout_module(self.weight)
        out = F.linear(input_f, weight_q, self.bias)
        return out


if __name__ == '__main__':
    m = DropoutFC(2,4)
    w_ = torch.randn(5, 2)
    w_.requires_grad_()
    y_ = m(w_)
    #z_ = y_.view(w_.size(0), -1)
    #print(z_.norm(dim=1))

    y_.sum().backward()
    print('w grad', w_.grad.size())