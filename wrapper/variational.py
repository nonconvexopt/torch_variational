import sys
import torch
import torch.nn as nn

sys.path.append('../')
from util import mul_sign


registered_modules = [
    #Linear Layers
    nn.Linear,
    nn.Bilinear,
    nn.LazyLinear,
    
    #Convolutional Layers
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.LazyConv1d,
    nn.LazyConv2d,
    nn.LazyConv3d,
    nn.LazyConvTranspose1d,
    nn.LazyConvTranspose2d,
    nn.LazyConvTranspose3d,
    
    #Recurrent Layers
        #TBA
    
    #Transformer Layers
        #TBA
]


#I considered bias deterministic because of several reasons:
#1. Bias in fully-connected layer and Convolutional layer
#   has been known to exhibits little advantages when considered stochastic.
#2. Bias gets information from gradient at multiple sources compared to weight
#3. 

class Variational_Flipout(nn.Module):
    def __init__(self, module: nn.Module, weight_multiplcative_variance = True):
        super(Variational_Flipout, self).__init__()
        """
        Wrapper class for existing torch modules.
        Use multiplicative noise in weight space to make layer stochastic.
        """
        
        #assert True in [isinstance(module, m) for m in registered_modules]

        self.weight_mean = module
        self.weight_logvar = nn.Parameter(self.weight_mean.weight.data.clone().detach().fill_(0))
        self.weight_multiplcative_variance = weight_multiplcative_variance
        
    def forward(self, x) -> torch.Tensor:
        weight = self.weight_mean.weight.data
        bias = self.weight_mean.bias
        self.weight_mean.bias = None
        if self.weight_multiplcative_variance:
            self.weight_mean.weight.data = (
                self.weight_mean.weight.data
                * self.weight_logvar.div(2).exp()
                * torch.randn(self.weight_logvar.shape, device = self.weight_logvar.device)
            )
        else:
            self.weight_mean.weight.data = (
                self.weight_logvar.div(2).exp()
                * torch.randn(self.weight_logvar.shape, device = self.weight_logvar.device)
            )
        noise = mul_sign(self.weight_mean(mul_sign(x)))
        self.weight_mean.weight.data = weight
        self.weight_mean.bias = bias
        mean = self.weight_mean(x)
        
        return mean + noise
      
    def kld(self) -> torch.Tensor:
        #KL(q||p) with respect to Standard Normal p
        return (
            self.weigth_mean.weight.pow(2)
            - self.weight_logvar
            + self.weight_logvar.exp()
            - 1
        ).sum().div(2)
    
    
    
class Variational_LRT(nn.Module):
    def __init__(self, module: nn.Module, stochastic_bias = False):
        super(Variational_LRT, self).__init__()
        """
        Wrapper class for existing torch modules.
        Use multiplicative noise in weight space to make layer stochastic.
        """
        
        assert True in [isinstance(module, m) for m in registered_modules]

        self.module = module
        
        self.weight = module.weight
        self.weight_logvar = nn.Parameter(
            torch.zeros(
                self.weight.shape,
                device = self.weight.device
            )
        )
        
        self.bias = module.bias
        module.bias = None
        
    def kld(self) -> torch.Tensor:
        #KLD with Standard Normal
        return (self.weight.pow(2) - self.weight_logvar + self.weight_logvar.exp() - 1).mean().div(2)
    
    def forward(self, x) -> torch.Tensor:
        
        self.module.weight = self.weight
        mean = self.module(x)
        
        self.module.weight = self.weight * self.weight_logvar.exp()
        var = self.module(x.pow(2))
        
        output = mean + var.sqrt() * torch.randn(var.shape, device = self.bias.device, requires_grad = False)
        
        if self.bias:
            output += self.bias
        
        return output