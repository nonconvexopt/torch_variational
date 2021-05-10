import torch
import torch.nn as nn

from utils import mul_sign


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
    
    
class Variational_Flipout(nn.Module):
    def __init__(self, module: nn.Module):
        super(Variational_Flipout, self).__init__()
        """
        Wrapper class for existing torch modules.
        Use multiplicative noise to make layer stochastic.
        """
        
        assert True in [isinstance(module, m) for m in registered_modules]

        self.module = module
        
        self.weight = module.weight
        self.weight_logvar = nn.Parameter(
            torch.empty(module.weight.shape, module.weight.device).fill_(1).log()
        )
        
        self.bias = module.bias
        module.bias = None
        
    def kld(self) -> torch.Tensor:
        #KLD with Standard Normal
        return (self.weight.pow(2) - self.weight_logvar + self.weight_logvar.exp() - 1).mean().div(2)
        
    def forward(self, x) -> torch.Tensor:
        
        self.module.weight = self.weight
        mean = self.module(x)
        
        self.module.weight = self.weight \
            * self.module.weight_logvar.div(2).exp() \
            * torch.randn(var.shape, device = self.bias.device, requires_grad = False)
        noise = mul_sign(self.module(mul_sign(x)))
        
        output = mean + noise
        
        if self.bias:
            output += self.bias
        
        return output

    
    
class Variational_LRT(nn.Module):
    def __init__(self, module: nn.Module, stochastic_bias = False):
        super(Variational_LRT, self).__init__()
        """
        Wrapper class for existing torch modules.
        Use multiplicative noise to make layer stochastic.
        """
        
        assert True in [isinstance(module, m) for m in registered_modules]

        self.module = module
        
        self.weight = module.weight
        self.weight_logvar = nn.Parameter(
            torch.empty(module.weight.shape, module.weight.device).fill_( torch.ones([]).log() )
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