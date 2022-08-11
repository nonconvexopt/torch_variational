import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from ..util import mul_sign

linear_args = ()
conv_args = ('stride', 'padding', 'groups', 'dilation')
conv_transposed_args = ('stride', 'padding', 'output_padding', 'groups', 'dilation')

module_to_functional = {
    #Linear Layers
    nn.Linear:[F.linear, linear_args],
    nn.Bilinear:[F.bilinear, linear_args],
    nn.LazyLinear:[F.linear, linear_args],
    
    #Convolutional Layers
    nn.Conv1d:[F.conv1d, conv_args],
    nn.Conv2d:[F.conv2d, conv_args],
    nn.Conv3d:[F.conv3d, conv_args],
    nn.ConvTranspose1d:[F.conv_transpose1d, conv_transposed_args],
    nn.ConvTranspose2d:[F.conv_transpose2d, conv_transposed_args],
    nn.ConvTranspose3d:[F.conv_transpose3d, conv_transposed_args],
    nn.LazyConv1d:[F.conv1d, conv_args],
    nn.LazyConv2d:[F.conv2d, conv_args],
    nn.LazyConv3d:[F.conv3d, conv_args],
    nn.LazyConvTranspose1d:[F.conv_transpose1d, conv_transposed_args],
    nn.LazyConvTranspose2d:[F.conv_transpose2d, conv_transposed_args],
    nn.LazyConvTranspose3d:[F.conv_transpose3d, conv_transposed_args],
    
    #Recurrent Layers
        #TBA
    
    #Transformer Layers
        #TBA
}


#I considered bias deterministic because of several reasons:
#Bias in fully-connected layer and Convolutional layer has been
#known to exhibit little advantages when considered stochastic since
#bias gets information from gradient at multiple sources compared to weight
#and the variance of bias shrinks quickly.

class Variational_Flipout(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        weight_multiplcative_variance = True
    ):
        super(Variational_Flipout, self).__init__()
        """
        Wrapper class for existing torch modules.
        Use multiplicative noise in weight space to make layer stochastic.
        """
        
        assert True in [isinstance(module, m) for m in module_to_functional]

        self.functional, self.arglist = module_to_functional[module.__class__]
        
        self.weight_mean = module.weight
        self.weight_logvar = nn.Parameter(
            torch.full(
                size = self.weight_mean.shape,
                fill_value = math.log(1 / torch.prod(torch.tensor(self.weight_mean.shape[1:]))),
                device = self.weight_mean.device,
                requires_grad = True,
            )
        )
        self.weight_multiplcative_variance = weight_multiplcative_variance
        self.bias = module.bias
        
        self.functional_kwargs = {
            key:value
            for key, value
            in module.__dict__.items()
            if key not in ['input', 'weight', 'bias'] and key in self.arglist
        }

    def forward(self, x) -> torch.Tensor:
        noise_std = torch.randn(
            self.weight_mean.shape,
            device = self.weight_mean.device,
            requires_grad = False,
        ) * self.weight_logvar.div(2).exp()
        
        if self.weight_multiplcative_variance:
            noise_std = noise_std * self.weight_mean
            
        return self.functional(
            input = x,
            weight = self.weight_mean,
            bias = self.bias,
            **self.functional_kwargs,
        ) + mul_sign(
            self.functional(
                input = mul_sign(x),
                weight = noise_std,
                bias = None,
                **self.functional_kwargs,
            )
        )
    
    def kld(self) -> torch.Tensor:
        #KL(q||p) with respect to Standard Normal p
        return (
            self.weight_mean.pow(2)
            - self.weight_logvar
            + self.weight_logvar.exp()
            - 1
        ).mean().div(2)


class Variational_LRT(nn.Module):
    def __init__(self, module: nn.Module, weight_multiplcative_variance = True):
        super(Variational_LRT, self).__init__()
        """
        Wrapper class for existing torch modules.
        Use multiplicative noise in weight space to make layer stochastic.
        """
        
        assert True in [isinstance(module, m) for m in module_to_functional]
        
        self.functional, self.arglist = module_to_functional[module.__class__]
        
        self.weight_mean = module.weight
        self.weight_logvar = nn.Parameter(
            torch.full(
                size = self.weight_mean.shape,
                fill_value = math.log(1 / torch.prod(torch.tensor(self.weight_mean.shape[1:]))),
                #fill_value = math.log(1 / self.weight_mean.shape[1]),
                device = self.weight_mean.device,
                requires_grad = True,
            )
        )
        self.register_buffer(
            'weight_logvar_prior',
            torch.full(
                size = self.weight_mean.shape,
                fill_value = math.log(1 / torch.prod(torch.tensor(self.weight_mean.shape[1:]))),
                #fill_value = math.log(1 / self.weight_mean.shape[1]),
                device = self.weight_mean.device,
                requires_grad = False,
            )
        )

        self.weight_multiplcative_variance = weight_multiplcative_variance
        self.bias = module.bias
        
        self.functional_kwargs = {
            key:value
            for key, value
            in module.__dict__.items()
            if key not in ['input', 'weight', 'bias'] and key in self.arglist
        }
    
    def forward(self, x) -> torch.Tensor:
        x_squared = x.pow(2)
        
        mean = self.functional(
            input = x,
            weight = self.weight_mean,
            bias = None,
            **self.functional_kwargs,
        )
        
        weight_var = self.weight_logvar.exp()
        weight_var_prior = self.weight_logvar_prior.exp()
        if self.weight_multiplcative_variance:
            weight_var = weight_var * self.weight_mean.pow(2)
            weight_var_prior = weight_var_prior * self.weight_mean.pow(2)
        
        var = self.functional(
            input = x_squared,
            weight = weight_var,
            bias = None,
            **self.functional_kwargs,
        )

        var_prior = self.functional(
            input = x_squared,
            weight = weight_var_prior,
            bias = None,
            **self.functional_kwargs,
        )
        
        self._kld = ((var_prior / var).log() + (var + mean.pow(2)) / var_prior).mean().div(2)
        
        out = mean + var.sqrt() * torch.randn(var.shape, device = var.device, requires_grad = False)
        if self.bias is not None:
            out += self.bias.view([1, -1] + [1] * (out.dim() - 2))
        
        return out

    def kld(self) -> torch.Tensor:
        #KL(q||p) with respect to Standard Normal p
        return (
            self.weight_mean.pow(2)
            - self.weight_logvar
            + self.weight_logvar.exp()
            - 1
        ).mean().div(2)