"""
A starting point for implementing custom parameter penalization for Hawkes
and Poisson models, if the given implementations are insufficient.
"""

import torch
from dataclasses import dataclass, field

from hawkes.models import Penalty


@dataclass
class PenaltyExample(Penalty):
    """
    Example of a custom penalty implementation.
    """

    hyperparameter_1: float = 0.0
    hyperparameter_2: float = 0.0

    def compute(self, param: torch.Tensor) -> torch.Tensor:
        """
        Implement the unweighted parameter penalty.

        NOTE: The base class handles weighting, so this should return the raw
        penalty without multiplying by self.weight.

        NOTE: Importantly, the returned value should be a 0-dimensional tensor,
        not a Python float, to ensure autograd gradients work correctly. For
        the same reason, avoid .item() anywhere when computing the penalty.
        """

        return ...
