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
Assumed weight multiplicative variances.
### Flipout
![formula](https://render.githubusercontent.com/render/math?math=\Large{W_{ij}=\overline{W_{ij}}\+{\Delta}W_{ij}})<br>
![formula](https://render.githubusercontent.com/render/math?math=\Large{q({\Delta}W_{ij})=N({\Delta}W_{ij}{\mid}0,\overline{W_{ij}}^2\sigma_{ij}^2)})<br>
![formula](https://render.githubusercontent.com/render/math?math=\Large{f\left(x_n\right)=x_n^T\overline{W_{ij}}+\left(\left(x_n^T{\circ}s_n\right){\Delta}W_{ij}\right){\circ}r_n^T}})<br>
### Local Reparameterization Trick
![formula](https://render.githubusercontent.com/render/math?math=\Large{q(W_{ij})=N(W_{ij}\mid\mu_{ij},\mu_{ij}^2\sigma_{ij}^2)})<br>
![formula](https://render.githubusercontent.com/render/math?math=\Large{a_{nj}=\sum_{i}x_{ni}w_{ij}})<br>
![formula](https://render.githubusercontent.com/render/math?math=\Large{q(a_{nj})=N(a_{nj}\mid\sum_{i}x_{ni}W_{ij},\sum_{i}x_{ni}^2W_{ij}^2)})<br>

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
