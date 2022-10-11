import torch
from torch import nn
import torch.nn.functional as F
class Conv1d(nn.Conv1d):
	def __init__(self,in_channels, out_channels, kernel_size=3, stride=1):
		self.padding = (kernel_size-1)//2
		self.stride = stride
		super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,padding=self.padding)
	def extra_repr(self):
		return 'in_channels{in_channels}, out_channels{out_channels},kernel_size{kernel_size}'.format(**self.__dict__)
	def forward(self,x):
		#T,B,C
		x = x.permute(1,2,0) #T,B,C-->B,C,T
		x = F.conv1d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
		x = x.permute(2,0,1) #B,C,T-->T,B,C
		return x
		