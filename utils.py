import torch

def mul_sign(x) -> torch.Tensor:
    #Best performance on several experiments
    return x.mul(torch.empty(x.shape, device = x.device).uniform_(-1,1).sign())