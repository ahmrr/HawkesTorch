# HawkesTorch

<p align="center">
  <a href="https://arxiv.org/abs/2604.01342"><img src="https://img.shields.io/badge/arXiv-2604.01342-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/ahmrr/HawkesTorch/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ahmrr/HawkesTorch" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
</p>

A PyTorch library for learning multivariate Hawkes processes, with a focus on **extensibility**, **speed**, and **scalability**.

**[Overview](#overview) · [Quick Start](#quick-start) · [Organization](#library-organization) · [Documentation](#documentation) · [Future Work](#future-work) · [Citation](#citation)**

## Overview

This library is designed around **multivariate linear exponential Hawkes processes**, which have intensity functions of the form:
<!-- $$
\lambda_p(t) = \textcolor{red}{\mu_p(t)} + \sum_{j: t_j < t} \sum_{k=1}^K \textcolor{green}{\alpha_{p, m_j,k}}\textcolor{blue}{\gamma_k} e^{-\textcolor{blue}{\gamma_k}(t - t_j)}
$$ --> 

<div align="center"><img src="svg/mf10O0WwDD.svg"></div>

<br>

| Symbol | Description |
|--------|-------------|
| <!-- $\textcolor{red}{\mu_p(t)}$ --> <img style="transform: translateY(0.1em);" src="svg/2Gxo9ysjtv.svg"> | Base intensity of node <!-- $p$ --> <img style="transform: translateY(0.1em);" src="svg/RlMvXZmEiQ.svg"> at time <!-- $t$ --> <img style="transform: translateY(0.1em);" src="svg/tY7P4yabM8.svg"> (a Poisson process) |
| <!-- $\textcolor{green}{\alpha_{p, q, k}}$ --> <img style="transform: translateY(0.1em);" src="svg/GybAYZiPAJ.svg"> | Influence of node <!-- $q$ --> <img style="transform: translateY(0.1em);" src="svg/g3h1QzBEe4.svg"> on node <!-- $p$ --> <img style="transform: translateY(0.1em);" src="svg/zQIYaR13s2.svg"> on timescale <!-- $k$ --> <img style="transform: translateY(0.1em);" src="svg/cV1Ng2jzd0.svg"> |
| <!-- $\textcolor{blue}{\gamma_k}$ --> <img style="transform: translateY(0.1em);" src="svg/gvYpL9iSbo.svg"> | Decay rate of timescale <!-- $k$ --> <img style="transform: translateY(0.1em);" src="svg/K11wBYJfpf.svg"> |
| <!-- $N$ --> <img style="transform: translateY(0.1em);" src="svg/rbTQjxY4Xi.svg"> | Number of events in the sequence |
| <!-- $M$ --> <img style="transform: translateY(0.1em);" src="svg/HfObpVl1n6.svg"> | Number of dimensions (nodes) in the process |
| <!-- $K$ --> <img style="transform: translateY(0.1em);" src="svg/eiunNoxW0S.svg"> | Number of kernels (timescales of influence) |
| <!-- $t_i \in [0, T]$ --> <img style="transform: translateY(0.1em);" src="svg/YN9gD29LIg.svg"> | Time of the <!-- $i$ --> <img style="transform: translateY(0.1em);" src="svg/QDLHDzWk9m.svg">-th event |
| <!-- $m_i \in \{1,\dots,M\}$ --> <img style="transform: translateY(0.1em);" src="svg/mWfEKHchpf.svg"> | Dimension (node type) of event <!-- $i$ --> <img style="transform: translateY(0.1em);" src="svg/KYXUrFdk8V.svg"> |

<br>

We implement a highly parallel algorithm for fitting Hawkes processes via **exact maximum likelihood estimation**. This allows scaling to massive datasets; e.g., ~1,000 nodes and ~10,000,000 events. Therefore, a GPU or many CPU cores are recommended to realize these speedups.

<br>

<figure align="center">
<img src="./docs/scaling_curves.png">
<figcaption style="text-align: center;">

Per-epoch time (left) and memory usage (right) on an Nvidia A100 GPU with <!-- $K=3$ --> <img style="transform: translateY(0.1em);" src="svg/v2uIzreh1t.svg">.

</figcaption>
</figure>

<br>

See [our paper](https://arxiv.org/abs/2604.01342) for experiment results and more details on the optimizations done for this library.

## Quick Start

For a detailed worked example of simulation, fitting, and evaluation, see `examples/full_example.ipynb`.

### Installation

The main prerequisite is PyTorch, ideally with GPU support, and `matplotlib` to run the plotting code. The `numpy` package is also required to run some of the examples.

Since this library is under active development, it will be updated frequently to add features and fix bugs. We recommend installing via `pip` directly from GitHub so you can easily pull in the latest updates:

```bash
pip install git+https://github.com/ahmrr/HawkesTorch
```

Alternatively, you can add this repo as a submodule to your project if you
want to modify the source.

### Basic Usage

Here is an example of simulating a Hawkes process and fitting a model to it. See `examples/full_example.ipynb` for a more detailed walkthrough.

```python
import torch

from hawkes import models
from hawkes.utils import config

# Simulate data

M = 10  # number of nodes
N = 100000  # number of events
sim_gamma = torch.tensor([1.5]) # decay rate
sim_mu = torch.tensor([0.1] * M) # constant base rate
sim_alpha = torch.rand(1, M, M) * 0.5 # influence (K x M x M)

sim_base_process = models.Poisson(M, mu_init=sim_mu)
sim_model = models.Hawkes(
    gamma=sim_gamma,
    gamma_param=False,
    base_process=sim_base_process,
    alpha_init=sim_alpha,
)

seq = sim_model.simulate(max_events=N)

# Fit model

est_gamma = torch.tensor([1.0]) # initial decay rate

est_base_process = models.Poisson(M)
est_model = models.Hawkes(
    gamma=est_gamma,
    gamma_param=True, # allow learning the decay rate
    base_process=est_base_process,
)

fit_config = config.FitConfig(
    num_steps=4000, # epochs to train for
    monitor_interval=400, # how often to print training progress
    learning_rate=0.1, # learning rate for Adam optimizer
)

stats = est_model.fit(seq, fit_config) # returns dict of training details
```


## Organization

<br>

<figure align="center">
<img src="./docs/class_organization.png">
<figcaption style="text-align: center;">

Arrows indicate inheritance, diamonds indicate composition.

</figcaption>
</figure>

<br>

Most functionality is implemented in three main abstract base classes. Subclasses of these implement specific Hawkes and Poisson parameterizations and penalizations, which you can also extend to your own use cases.

- `HawkesBase`: Provides intensity and likelihood computation, as well as `fit` and `simulate` methods. Subclasses implement custom parameterizations of <!-- $\textcolor{green}{\alpha_{p, q, k}}$ --> <img style="transform: translateY(0.1em);" src="svg/sFgxspH9FI.svg"> and <!-- $\textcolor{blue}{\gamma_k}$ --> <img style="transform: translateY(0.1em);" src="svg/poumHIgalA.svg">, e.g., low-rank, sparse, etc.

- `PoissonBase`: Represents the base rate <!-- $\textcolor{red}{\mu_p(t)}$ --> <img style="transform: translateY(0.1em);" src="svg/Uz7Pz4Dxed.svg"> and is used in `HawkesBase` via the `base_process` attribute. Also provides standalone `fit` and `simulate` methods, if raw Poisson fitting is desired. Subclasses implement custom parameterizations of the time-varying base rate.

- `Penalty`: An abstract `dataclass` that represents a regularization term added to the log-likelihood during fitting. Extend this to create custom penalties beyond those already implemented. Penalties are attached to a model via the `penalization` attribute, which accepts a `dataclass` that stores `Penalty` instances for each parameter.


To implement your own Hawkes or Poisson process, or to add custom penalization, please reference:
- Files in `examples/templates` for templates you can copy and modify. 
- The `HawkesBase`, `PoissonBase`, and `Penalty` base classes for core methods that are already implemented.
- The `Hawkes`, `HawkesLowRank`, `Poisson`, `PoissonPiecewise`, `PoissonFourier`, and `NormPenalty` subclasses for concrete implementation examples.

## Documentation

There will be an [API Reference]() available soon with detailed documentation and examples. In the meantime, please refer to the code, docstrings, examples, and paper.

## Future Work

- [ ] Even faster likelihood computation via a custom CUDA implementation (using CUB `DeviceScan`)

- [x] Piecewise constant `PoissonBase` model (`PoissonPiecewise`)

- [ ] More `PoissonBase` models (e.g., splines, etc.)

- [ ] API documentation 

If you encounter a bug or have a feature request, please feel free to [open an issue](https://github.com/ahmrr/hawkestorch/issues). Suggestions are also much appreciated!

## Citation

If you find this library to be useful in your research, please cite the following [paper](https://arxiv.org/abs/2604.01342):

```bibtex
@article{raza2026hawkestorch,
  title = {Massively Parallel Exact Inference for Hawkes Processes},
  author = {Raza, Ahmer and Smith, Hudson},
  journal = {arXiv preprint arXiv:2604.01342},
  year = 2026,
}
```
