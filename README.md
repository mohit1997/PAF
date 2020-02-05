# Improved Polynomial Neural Networks with Normalised Activationsactivations

Polynomials, which are widely used to study non-linear systems, have been shown to be extremely useful in analyzing neural networks (NNs). However, the existing methods for training neural networks with polynomial activation functions (PAFs), called as PNNs, are applicable for shallow networks and give a stable performance with quadratic PAFs only. This is due to the optimization issues encountered during training PNNs. We propose a working model for PAFs using a novel normalizing transformation which alleviates the problem of training PNNs with arbitrary degree. Our PAF can be directly used to train shallow PNNs in practice for degrees as high as ten. It can also be utilized to learn multivariate sparse polynomials of small degrees. We also propose a way to train deep CNNs with PAFs which achieve performance similar to deep CNNs with standard activations. Through rigorous experimentation on multiple data sets, we show that PNNs can be effectively trained in practice. This also highlights the potential of the proposed method to support the research on using polynomials to study deep learning.

### Arxiv: [https://arxiv.org/abs/1906.09529](https://arxiv.org/abs/1906.09529)

## Requirements
1. python 2.x/3.x
2. tensorflow 1.8+
3. keras 2.2.2


