"""
A starting point for implementing custom Hawkes models. See Hawkes and
HawkesLowRank for complete reference implementations.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from hawkes.models import HawkesBase, PoissonBase, Penalty
from hawkes.utils import config


@dataclass
class HawkesExamplePenalty:
    """
    Optionally define penalties for each learnable nn.Parameter in your model.
    You can just use HawkesPenalty if penalizing alpha and gamma (instead of
    the parameters they are constructed from) is sufficient.
    """

    custom_parameter_1: Penalty | None = None
    custom_parameter_2: Penalty | None = None


class HawkesExample(HawkesBase):
    """
    Custom Hawkes process implementation.
    """

    def __init__(
        self,
        # Add or remove constructor arguments as needed (e.g. default values)
        gamma: torch.Tensor,
        gamma_param: bool,
        base_process: PoissonBase,
        alpha_init: torch.Tensor | float | None = None,
        penalization: HawkesPenalty | None = HawkesExamplePenalty(),
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        """
        Args:
            gamma:          Initial or fixed (if gamma is parametrized or not, respectively) memory values for the exponential decay kernels
            gamma_param:    Whether to parametrize gamma, the exponential kernel memory values
            base_process:   Base Poisson process for the Hawkes model
            alpha_init:     Initial scale for the excitation matrix (if float, all entries initialized to the same value)
            penalization:   HawkesExamplePenalty dataclass with penalty values for custom parameters (if needed)
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            runtime_config: Runtime configuration for debugging and profiling
        """

        if len(gamma.shape) != 1:
            raise ValueError("gamma must be a rank-1 tensor")

        K = len(gamma)

        # Base class handles simulation, fitting, and base process integration
        super().__init__(
            K, gamma_param, base_process, penalization, runtime_config, device
        )

        self.t = transformation

        # Define any custom values and nn.Parameter variables here
        _custom_parameter_1 = self.t.inverse(torch.tensor(0.0))
        _custom_parameter_2 = self.t.inverse(torch.tensor(0.0))

        self._custom_parameter_1 = nn.Parameter(_custom_parameter_1.to(self.device))
        self._custom_parameter_2 = nn.Parameter(_custom_parameter_2.to(self.device))

    @property
    def alpha(self) -> torch.Tensor:
        """Construct alpha from the learnable parameters."""

        _alpha = self.t.forward(...)

        return _alpha

    @alpha.setter
    def alpha(self, value: torch.Tensor | float):
        """Convert the given value of alpha into the learnable parameters."""

        _custom_parameter_1 = self.t.inverse(...)

        self._custom_parameter_1.data = _custom_parameter_1

    @property
    def gamma(self) -> torch.Tensor:
        """
        Construct gamma from the learnable parameters.
        NOTE: If self.gamma_param is False (i.e. gamma is not learnable),
        the returned value should not depend on any learnable parameters
        """

        _gamma = self.t.forward(...)

        return _gamma

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        """Store gamma in the learnable parameters."""

        _custom_parameter_2 = self.t.inverse(...)

        self._custom_parameter_2.data = _custom_parameter_2

    def penalty(self) -> float:
        """
        Optionally override the default penalty method, if your parameters
        require custom penalization that is not covered by the base class.
        """

        return ...
