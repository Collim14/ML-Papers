# Deepseek mHC implementation

This repository contains my implementation of the mHC paper as outlined by Deepseek https://arxiv.org/pdf/2512.24880

The basic function of the code works by wrapping predefined layer(s) with mHCWrapper, with the entire network wrapped by MHCEntry and MHCExit to transform the initial inputs and final network outputs to be compatible with the mHC implementation.

The mHC implementation is compatible with NNs which vary dimension size with depth, there is also support for static, or dynamic hyper connections as outlined in the Deepseek paper and the original Hyper Connections paper https://arxiv.org/pdf/2409.19606 

There is also support for using Newton-Schulz iteration to orthogonalise H_res rather than using Sinkhorn to get a doubly stochastic H_res. 

Key metrics are tracked such as the gradient norm, the gradients flowing through the hyper connections, the spectral norm of the hyper connections, and an approximation of the spectral norm of the Jacobian of each layer at each iteration. Tracking the latter is motivated by the utility of the mHC over the standard HC method, as the standard HC method was unconstrained its spectral norm was unbounded. For deeper networks, this leads to training instability as the spectral norm tends to explode. In comparison, double stochastic matrices are closed under multiplication, which was a primary motivation for applying the Sinkhorn algorithm to H_res. Tracking the spectral norm of the Jacobian thus allows us to monitor the health of the network with the various implementations HC, mHC, mHC (with Newton Schulz), and standard residual connections. The entropy of H_res is also tracked, with higher entropy meaning higher mixing between the streams.

The models in this repository are a NanoGPT (based off of the original NanoGPT), and a CNN following a standard ResNet.

The implementation supports MPS to run on macs. It currently does have integrated GPU support, I might build a CUDA backend for it.

## Ablations
Planned ablations include comparing the initialisations of the various phi_res,phi_post,phi_pre, b_res,b_post, b_pre, and similarly for the static implementation. Training NanoGPT_mhc on tinyshakespeare (and maybe fineweb), comparing performance to HC, residual connections, and mHC with NS. Similar tests with the CNN on Mnist (and maybe ImageNet), with the primary focus on comparing to the initial ResNets.