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
__all__ = ['MaskPowerNorm3d']


class MaskPowerNorm3d(nn.Module):
    """
    An implementation of masked batch normalization, used for testing the numerical
    stability.
    """
    __count = 0
    def update_cnt(self):
        MaskPowerNorm3d.__count += 1
    def __init__(self, num_features, eps=1e-5, momentum=0.05, \
        affine=True, track_running_stats=True, sync_bn=True, process_group=None, \
        with_seq = True, prefix = 'None', weight_affine=0, penalty_var=0, \
        penalty_type='diff', learn_alpha=False, alpha=1):
        super().__init__()
        self.id = self.__count
        self.prefix = prefix
        #print(self.id)
        self.update_cnt()
        self.interval = 1100  #standard: 1100
        #self.save_interval = 20
        self.batchnum = 0
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum #default0.05
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.maxlen = 600
        self.with_seq = with_seq
        self.learn_alpha = learn_alpha
        if self.learn_alpha:
            self.alpha = Parameter(torch.Tensor(1))
            nn.init.constant_(self.alpha, alpha)
        self.weight_affine = weight_affine
        self.penalty_var = penalty_var
        self.penalty_type = penalty_type
        assert self.penalty_type in ['diff','mag','reldiffnorm','reldiffcosine','symkl'], "wrong penalty type for BN!"
        #self.running_mean_list = []
        #self.running_var_list = []
        #self.batch_mean_list = []
        #self.batch_var_list = []
       
        self.batch_var_norm = []
        self.running_var_norm = []
        self.diff_var_norm = []
        self.grad_proj_list = []
        self.file = 'statistics/{}/bn_{}'.format(self.prefix,self.id)

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.sync_bn = sync_bn
        # gpu_size is set through DistributedDataParallel initialization. This is to ensure that SyncBatchNorm is used
        # under supported condition (single GPU per process)
        self.process_group = process_group
        self.ddp_gpu_size = 4
        self.reset_parameters()
    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def record_forward(self):
        diff_var_data = (self.running_var-self.batch_var).data
        self.diff_var_norm.append(diff_var_data.norm().cpu().numpy().item())
        self.batch_var_norm.append(self.batch_var.norm().cpu().numpy().item())
        self.running_var_norm.append(self.running_var.norm().cpu().numpy().item())
        if self.batchnum%self.interval==0:
            savefile = "{}_forward_{}.mat".format(self.file,self.batchnum//self.interval)
            d = {}
            diff_var = np.array(self.diff_var_norm)
            batch_var = np.array(self.batch_var_norm)
            running_var = np.array(self.running_var_norm)

            d['diff_var'] = diff_var
            d['running_var'] = running_var
            d['batch_var'] = batch_var
            io.savemat(savefile, d)
            self.batch_var_norm = []
            self.running_var_norm = []
            self.diff_var_norm = []

    def backward_hook(self,grad):
        #B, T, C
        grad_proj = ((grad*self.mask)*self.x).sum(dim=(0,1))/self.sum_size**0.5
        #self.grad_mean_list.append(grad_mean.data.cpu().reshape([1,-1]))
        #self.grad_proj_list.append(grad_proj.data.cpu().reshape([1,-1]))
        self.grad_proj_list.append(grad_proj.norm().cpu().numpy().item())
        #print(grad_mean.shape,grad_proj.shape);exit()
        if self.batchnum%self.interval==0:
            savefile = "{}_backward_{}.mat".format(self.file,self.batchnum//self.interval)
            d = {}
            #grad_mean_arr = torch.cat(self.grad_mean_list,dim=0)
            #grad_proj_arr = torch.cat(self.grad_proj_list,dim=0)
            d['grad_proj'] = np.array(self.grad_proj_list)
            from scipy import io
            io.savemat(savefile, d)
            self.grad_proj_list = []

    def loss(self):
        loss = 0
        assert self.training==True, "wrongly adding BN inconsistent loss!"
        if self.penalty_var==0:
            return loss
        if self.penalty_type=='diff':
            loss = self.loss_diff(loss)
        if self.penalty_type=='mag':
            loss = self.loss_magnitude(loss)
        if self.penalty_type=='reldiffnorm':
            loss = self.loss_rel_diff_norm(loss)
        if self.penalty_type=='reldiffcosine':
            loss = self.loss_rel_diff_cosine(loss)
        if self.penalty_type=='symkl':
            loss = self.sym_kl_loss(loss)

        return loss

    def loss_diff(self, loss):
        if self.weight_affine:
            loss += self.penalty_var*((torch.abs(self.running_var-self.batch_var))*self.weight.detach()).sum()
            #print(loss) #loss: 初始十几，训了一小会，零点几,1,2的居多，偶尔有9
        else:
            loss += self.penalty_var*(torch.abs(self.running_var-self.batch_var)).sum()
        return loss

    def loss_rel_diff_norm(self, loss):
        loss += self.penalty_var*(torch.abs(self.running_var-self.batch_var)/(self.running_var.norm(p=1)).sum()+1)
        return loss

    def loss_rel_diff_cosine(self, loss):
        loss -= self.penalty_var*(self.running_var*self.batch_var).sum()/self.batch_var.norm()/self.running_var.norm()
        #loss -= self.penalty_var*torch.sqrt((self.running_var*self.batch_var)).sum()/torch.sqrt(self.batch_var.sum()*self.running_var.sum())
        return loss

    def loss_magnitude(self, loss):
        loss += self.penalty_var*(torch.abs(self.batch_var)).sum()
        return loss

    def sym_kl_loss(self,loss):
        item1 = self.running_var/(self.batch_var+self.eps)+self.batch_var/(self.running_var+self.eps)
        loss += self.penalty_mean*(item1.sum())
        return loss

    def forward(self, input, pad_mask=None, is_encoder=False, update_run=True):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """

        T, B, C = input.shape
        input = input.contiguous()
        #print(input.shape) #21, 192, 512
        input = input.permute(1,0,2) #T,B,C-->B,T,C

        #print(pad_mask.shape,input.shape)#torch.Size([192, 21]) torch.Size([192, 21, 512])
        #print(~pad_mask);exit()#true: 有mask, false:无mask
        if pad_mask is not None:
            mask = 1-pad_mask.unsqueeze(dim=-1).type_as(input)
        else:
            mask = torch.ones(B,T,1).cuda()
            #if not self.training:
            #    mask = torch.ones(B,T,1).cuda()
            #else:
            #    print("error bn pad mask!")
            #    print(self.id)
            #    print(pad_mask)
            #    exit()

        input = input*mask
        # Compute the sum and square-sum.
        sum_size = mask.sum()
        #print(sum_size,input.size(0)*input.size(1),sum_size/input.size(0)/input.size(1));exit() #4032
        input_ssum = (input**2).sum(dim=(0,1),keepdim=True)

        
        # # Compute the statistics
        if self.training:

            self.batchnum += 1
            self.mask = mask
            self.sum_size = sum_size #相当于求统计量的batch size
            bias_var = input_ssum/sum_size
                
            #用于计算running statistics
            self.batch_var = bias_var.squeeze()
            with torch.no_grad():
                stat_var = bias_var.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * stat_var.data
                self.record_forward()
        else:
            bias_var = self.running_var
        input = (input)/torch.sqrt(bias_var.clamp(self.eps))

        if self.training:
            #input.register_hook(self.backward_hook)
            pass
        if self.training:
            self.x = input.data*mask
        if self.affine:
            output = input*self.weight + self.bias
        else:
            output = input
        output = output.permute(1, 0, 2).contiguous() #B,T,C-->T,B,C
        return output