# torch_variational
Pytorch implementation and Wapper classes for torch.nn.modules layers of:<br>
- Flipout[1]<br>
- Local Reparameterization Trick[2]<br>

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

Example usage for wrapper classes:
```
import pytorch_flipout.Variational
layer = flipout(nn.Linear(in_features = 10, out_features = 10, bias = True))
```

Example usage for Stand-alone Flipout layers:
```
import pytorch_flipout.flipout
layer = flipout.Linear_flipout(in_features = 10, out_features = 10, bias = True)
output, kld = layer(torch.randn(1, 10))
```

## Derivations
```
$$
q(w_{ij})=N(w_{ij}|\mu_{ij}, \mu_{ij}^2\sigma_{ij}^2) \\
a_{ij} = \sum_{i} x_i w_{ij} \\
q(a_{ij}) = N(a_{ij}|\sum_{i} x_i w_{ij}, \sum_{i} x_i^2 w_{ij}^2)
$$
```
```
<img src="https://render.githubusercontent.com/render/math?math=%24%24%0Aq(w_%7Bij%7D)%3DN(w_%7Bij%7D%7C%5Cmu_%7Bij%7D%2C%20%5Cmu_%7Bij%7D%5E2%5Csigma_%7Bij%7D%5E2)%20%5C%5C%0Aa_%7Bij%7D%20%3D%20%5Csum_%7Bi%7D%20x_i%20w_%7Bij%7D%20%5C%5C%0Aq(a_%7Bij%7D)%20%3D%20N(a_%7Bij%7D%7C%5Csum_%7Bi%7D%20x_i%20w_%7Bij%7D%2C%20%5Csum_%7Bi%7D%20x_i%5E2%20w_%7Bij%7D%5E2)%0A%24%24">
```

## References
```
[1] @inproceedings{DBLP:conf/iclr/WenVBTG18,
  author    = {Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, Roger B. Grosse},
  title     = {Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches},
  booktitle = {6th International Conference on Learning Representations, {ICLR} 2018,
               Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
  year      = {2018},
  url       = {https://openreview.net/forum?id=rJNpifWAb}
}
[2] @inproceedings{NIPS2015_bc731692,
 author = {Kingma, Durk P and Salimans, Tim and Welling, Max},
 title = {Variational Dropout and the Local Reparameterization Trick},
 booktitle = {Advances in Neural Information Processing Systems},
 volume = {28},
 year = {2015}
 url = {https://proceedings.neurips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf},
}
```
