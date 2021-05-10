import torch
import pytorch_flipout

if __name__ == "__main__":
    layer_conv2d = flipout.Conv2d_flipout(1, 1, bias = False)
    layer_conv2d(torch.randn(1, 1, 1, 1))

    layer_linear = flipout.Linear_flipout(1, 1, bias = True)
    layer_linear(torch.randn(1, 1))
    
    layer_conv2d = flipout.Conv2d_flipout(1, 1, bias = False)
    layer_conv2d(torch.randn(1, 1, 1, 1))

    layer_linear = flipout.Linear_flipout(1, 1, bias = True)
    layer_linear(torch.randn(1, 1))
    
    print("Test Success")