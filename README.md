# pytorch-flipout
Pytorch implementation of "Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches", ICLR 2018

## Dependencies
Pytorch >= 1.6.0

## Available modules
Conv2d_flipout<br>
Linear_flipout<br>
(To be added)

## Usage
Pull this repository or just simply copy and paste the codes.

Example usage:
<pre><code>import flipout
layer = flipout.Linear_flipout(in_features = 10, out_features = 10, bias = True)</code></pre>


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
