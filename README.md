# pytorch-flipout
Pytorch implementation and Wapper classes for <h5>torch.nn.modules</h5> layers of:<br>
Flipout in "Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches", ICLR 2018<br>
Local Reparameterization Trick in "Variational Dropout and the Local Reparameterization Trick", NIPS 2015<br>

## Dependencies
Pytorch >= 1.0.0

## Available modules
Variationalizers - Wrapper for nn.module class(Supports [Lazy|Standard][Linear|Convolutional] layers.)
- Flipout Wrapper<br>
- Local Reparameterization Wrapper<br>

Stand-alone Flipout layers:
- Conv2d_flipout<br>
- Linear_flipout<br>

## Usage
Pull this repository or just simply copy and paste the codes.

Example usage 1:
```
import pytorch_flipout.flipout
layer = flipout.Linear_flipout(in_features = 10, out_features = 10, bias = True)
output, kld = layer(torch.randn(1, 10))
```

Example usage 2:
```
import pytorch_flipout.Variational
layer = flipout(nn.Linear(in_features = 10, out_features = 10, bias = True))
```

## Derivations
```
$$
q(w_{ij})=N(w_{ij}|\mu_{ij}, \mu_{ij}^2\sigma_{ij}^2) \\
a_{ij} = \sum_{i} x_i w_{ij} \\
q(a_{ij}) = N(a_{ij}|\sum_{i} x_i w_{ij}, \sum_{i} x_i^2 w_{ij}^2)
$$
```

## References
```
[1] @inproceedings{DBLP:conf/iclr/WenVBTG18,
  author    = {Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, Roger B. Grosse},
  title     = {Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches},
  booktitle = {6th International Conference on Learning Representations, {ICLR} 2018,
               Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
  publisher = {OpenReview.net},
  year      = {2018},
  url       = {https://openreview.net/forum?id=rJNpifWAb}
}
```
