import torch

from typing import Literal
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
class HawkesRuntimeConfig:
    """
    Dataclass containing extra runtime options for the Hawkes models

    Attributes:
        deterministic_sim: Set to make simulated data deterministic (each run with the same params will simulate the exact same data)
        profile_mem_iters: Set to profile the memory usage of some amount of training iterations
        profile_mem_entries: Maximum number of recorded memory entries, if memory profiling is enabled
        check_grad_epsilon: Set to compare gradients against autograd during fitting
        detect_anomalies: Set to enable PyTorch anomaly detection during fitting
        use_autograd_gradients: Use PyTorch autograd to compute gradients instead of custom backward
        intensity_implementation: Which implementation to use for intensity computation (general, sequential, parallel)
    """

    deterministic_sim: bool = False

    profile_mem_iters: int = 0
    profile_mem_entries: int = 100000

    check_grad_epsilon: bool = False
    detect_anomalies: bool = False
    use_autograd_gradients: bool = False

    intensity_implementation: Literal["general", "sequential", "parallel"] = "parallel"
    prefix_scan_implementation: Literal["hillis-steele", "blelloch"] = "blelloch"


@dataclass
class Transformation:
    """
    Utility dataclass storing a type of transformation to map the learned parameters in a Hawkes model. Given defaults are IDENTITY, EXP, and SOFTPLUS.

    Attributes:
        forward: The vectorized transformation function
        inverse: The vectorized inverse of the transformation
    """

    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]


def _identity(x):
    return x


def _softplus_inv(x):
    """Numerically stable softplus inverse that uses a linear approximation for large x"""
    return torch.where(x > 20, x, torch.log(torch.expm1(x)))


IDENTITY = Transformation(
    forward=_identity,
    inverse=_identity,
)
EXP = Transformation(
    forward=torch.exp,
    inverse=torch.log,
)
SOFTPLUS = Transformation(
    forward=torch.nn.functional.softplus,
    inverse=_softplus_inv,
)
