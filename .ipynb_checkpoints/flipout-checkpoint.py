import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math

def out_sign(x):
    return x * torch.empty(x.shape, requires_grad = False, device = x.device).uniform_(-1, 1).sign()

class Conv2d_flipout(nn.Module):
    def __init__(self, in_features, out_features, bias=False, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2d_flipout, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight_mu = nn.Parameter(torch.empty([self.out_features, self.in_features] + self.kernel_size).normal_(0, 1/self.out_features))
        self.weight_logvar = nn.Parameter(torch.ones([self.out_features, self.in_features] + self.kernel_size) *
                                          (- torch.empty(1).fill_(self.out_features).log()))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        weight_noise = torch.empty(self.weight_mu.shape, requires_grad = False, device = self.weight_mu.device).normal_(0,1)
        
        in_sign = torch.empty(x.shape, requires_grad = False, device = self.weight_mu.device).uniform_(-1,1).sign()
        
        x = F.conv2d(
            x,
            self.weight_mu,
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups) + out_sign(F.conv2d(
            in_sign * x,
            weight_noise * self.weight_mu * self.weight_logvar.div(2).exp(),
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups))
        
        return x, (self.weight_mu.pow(2) - self.weight_logvar + self.weight_logvar.exp() - 1).mean()/2
    
class Linear_flipout(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_flipout, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty([self.out_features, self.in_features]).normal_(0, 1/self.out_features))
        self.weight_logvar = nn.Parameter(torch.ones([self.out_features, self.in_features]) *
                                          (- torch.empty(1).fill_(self.out_features).log()))
    
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight_noise = torch.empty(self.weight_mu.shape, requires_grad = False,  device = self.weight_mu.device).normal_(0,1)
        
        in_sign = torch.empty(x.shape, requires_grad = False, device = self.weight_mu.device).uniform_(-1,1).sign()
        
        x = F.linear(x, self.weight_mu, bias=self.bias) + out_sign(F.linear(in_sign * x, weight_noise * self.weight_mu * self.weight_logvar.div(2).exp()))
        
        return x, (self.weight_mu.pow(2) - self.weight_logvar + self.weight_logvar.exp() - 1).mean()/2
