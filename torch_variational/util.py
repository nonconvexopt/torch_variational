import torch
import torch.nn as nn

def mul_sign(x) -> torch.Tensor:
    #Best performance on several experiments
    return x.mul(torch.empty(x.shape, device = x.device).uniform_(-1,1).sign())

def apply_wrapper(model: nn.Module, target, wrapper):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            apply_wrapper(module, target, wrapper)
            
        if isinstance(module, target):
            setattr(model, name, wrapper(module))
            
def kld(model: nn.Module):
    sum_kl = 0.0
    for name, module in model.named_children():
        if hasattr(module, "kld"):
            sum_kl += module.kld()
    return sum_kl