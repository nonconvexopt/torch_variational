import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils import mul_sign



class Conv2d_flipout(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=False, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2d_flipout, self).__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight_mu = nn.Parameter(
            torch.empty(
                [out_features, in_features] + self.kernel_size
            ).normal_(0, 1/out_features)
        )
        
        self.weight_logvar = nn.Parameter(
            torch.ones(
                [out_features, in_features] + self.kernel_size
            ) * (- torch.empty(1).fill_(out_features).log()))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        
    def kld(self):
        return (self.weight_mu.pow(2) - self.weight_logvar + self.weight_logvar.exp() - 1).mean().div(2)
            
    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        weight_noise = torch.empty(self.weight_mu.shape, requires_grad = False, device = self.weight_mu.device).normal_(0,1)
        
        x = F.conv2d(
            x,
            self.weight_mu,
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups) + mul_sign(F.conv2d(
            mul_sign(x),
            weight_noise * self.weight_mu * self.weight_logvar.div(2).exp(),
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups))
        
        return x, self.kld()
    
class Linear_flipout(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True) -> None:
        super(Linear_flipout, self).__init__()
        
        self.weight_mu = nn.Parameter(torch.empty([out_features, in_features]).normal_(0, 1/out_features))
        self.weight_logvar = nn.Parameter(torch.ones([out_features, in_features]) *
                                          (- torch.empty(1).fill_(out_features).log()))
    
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def kld(self) -> torch.Tensor:
        return (self.weight_mu.pow(2) - self.weight_logvar + self.weight_logvar.exp() - 1).mean().div(2)

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        weight_noise = torch.empty(self.weight_mu.shape, requires_grad = False,  device = self.weight_mu.device).normal_(0,1)
        x = F.linear(x, self.weight_mu, bias=self.bias) + mul_sign(F.linear(mul_sign(x), weight_noise * self.weight_mu * self.weight_logvar.div(2).exp()))
        
        return x, self.kld()