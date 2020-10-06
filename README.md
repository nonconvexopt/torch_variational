# pytorch-flipout
Simple Pytorch implementation of "Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches"

##Version
Pytorch >= 1.6.0

##Usage
Pull this repository or just simply copy and paste the codes.

Example usage:
<pre><code>import flipout
layer = flipout.Linear_flipout(in_features = 10, out_features = 10, bias = True)</code></pre>


##References
```
[1] @misc{wen2018flipout,
      title={Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches}, 
      author={Yeming Wen and Paul Vicol and Jimmy Ba and Dustin Tran and Roger Grosse},
      year={2018},
      eprint={1803.04386},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
