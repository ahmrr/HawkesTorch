# HawkesTorch

A PyTorch library for learning multivariate Hawkes processes on large-scale data.

# Usage

The `hawkes.models` module contains implementations of regular (`HawkesFullRank`), low-rank (`HawkesLowRank`), and upper-triangular (`HawkesLowRank`) Hawkes models, all of which are children of the `HawkesBase` abstract class whose conditional intensity function takes the following form:
$$
\lambda_p(t) = \mu_p + \sum_{j: t_j < t} \sum_{k=1}^K \alpha_{p, m_j}^k\gamma^k e^{-\gamma^k(t - t_j)}
$$

Each class contains a `simulate` method used to simulate an event sequence, and a `fit` method used to train a Hawkes model on event data using optimized intensity calculation. Custom Hawkes models can be implemented as subclasses of the `HawkesBase` class; only the `alpha` and `mu` parameters must be implemented (as well as `gamma`, if the $\gamma^k$ in the above intensity function are parametrized). See the `examples` directory for sample usage; in particular, `examples/fit_simulation.py` contains an example of a full simulation-training-evaluation workflow using the library. 

# Environment

The only dependency for the library itself is `pytorch` alongside a working CUDA installation for GPU acceleration. Using the utility plotting code in `hawkes.utils.plotting` also requires `matplotlib`. 

# Citation

See [our paper]() for more details on the optimizations done for this library.

To cite this project, please use the following BibTeX:

```bibtex
@inproceedings{

}
```