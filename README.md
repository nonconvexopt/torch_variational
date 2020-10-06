# pytorch-flipout
Simple Pytorch implementation of "Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches"

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
  author    = {Yeming Wen and
               Paul Vicol and
               Jimmy Ba and
               Dustin Tran and
               Roger B. Grosse},
  title     = {Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches},
  booktitle = {6th International Conference on Learning Representations, {ICLR} 2018,
               Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
  publisher = {OpenReview.net},
  year      = {2018},
  url       = {https://openreview.net/forum?id=rJNpifWAb},
  timestamp = {Thu, 25 Jul 2019 14:25:49 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/WenVBTG18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
