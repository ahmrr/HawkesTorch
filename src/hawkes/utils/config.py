import torch
from dataclasses import dataclass
from collections.abc import Callable


@dataclass
class HawkesFitConfig:
    """
    Dataclass containing hyperparameters and configuration for Hawkes model fitting.

    Attributes:
        num_steps: Number of optimization steps/epochs
        batch_size: Size of each batch during fitting (None for no batching)
        monitor_interval: Print fitting progress every monitor_interval steps
        learning_rate: Learning rate for the Adam optimizer
        l1_penalty: L1 regularization strength for sparsity
        l1_hinge: Rate values below this are regularized using L1 penalty
        nuc_penalty: Nuclear norm regularization strength for low-rank
    """

    num_steps: int = 1000
    batch_size: int = None
    monitor_interval: int = 100
    learning_rate: float = 0.01
    l1_penalty: float = 0.01
    l1_hinge: float = 1
    nuc_penalty: float = 0.01


@dataclass
class HawkesDebugConfig:
    """
    Dataclass containing debug options for the Hawkes models

    Attributes:
        deterministic_sim: Set to make simulated data deterministic (each run with the same params will simulate the exact same data)
        profile_mem_iters: Set to profile the memory usage of some amount of training iterations
        profile_mem_entries: Maximum number of recorded memory entries, if memory profiling is enabled
    """

    deterministic_sim: bool = True
    profile_mem_iters: int = 0
    profile_mem_entries: int = 100000


@dataclass
class Transformation:
    """
    Utility dataclass storing a type of transformation to map the learned parameters in a Hawkes model. Given defaults are EXP and SOFTPLUS.

    Attributes:
        forward: The vectorized transformation function
        inverse: The vectorized inverse of the transformation
    """

    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]


def _softplus_inv(x):
    return x + torch.log(-torch.expm1(-x))


EXP = Transformation(forward=torch.exp, inverse=torch.log)
SOFTPLUS = Transformation(
    forward=torch.nn.functional.softplus,
    inverse=_softplus_inv,
)
