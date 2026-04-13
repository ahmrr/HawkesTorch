import math
import torch
import typing
import torch.nn as nn

from . import HawkesBase, HawkesPenalty
from ..penalty import Penalty
from ..poisson import PoissonBase
from ...utils import config


class Hawkes(HawkesBase):
    """Standard Hawkes process implementation."""

    def __init__(
        self,
        gamma: torch.Tensor,
        gamma_param: bool,
        base_process: PoissonBase,
        alpha_init: torch.Tensor | float | None = None,
        penalization: HawkesPenalty = HawkesPenalty(),
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        """
        Args:
            M: Number of nodes
            gamma: Initial or fixed (if gamma is parametrized or not, respectively) memory values for the exponential decay kernels
            gamma_param: Whether to parametrize gamma, the exponential kernel memory values
            base_process: Base Poisson process for the Hawkes model
            alpha_init: Initial scale for the excitation matrix (if float, all entries initialized to the same value)
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            runtime_config: Runtime configuration for debugging and profiling
        """

        if len(gamma.shape) != 1:
            raise ValueError("gamma must be a rank-1 tensor")

        K = len(gamma)

        super().__init__(
            K, gamma_param, base_process, penalization, runtime_config, device
        )

        self.t = transformation

        if alpha_init is None:
            alpha_init = 0.01 + 0.09 * torch.rand(K, self.M, self.M).to(self.device)
        elif isinstance(alpha_init, float):
            alpha_init = alpha_init * torch.ones(K, self.M, self.M).to(self.device)

        _gamma = self.t.inverse(gamma).to(self.device)
        _alpha = self.t.inverse(alpha_init).to(self.device)

        self._gamma = nn.Parameter(_gamma) if gamma_param else _gamma
        self._alpha = nn.Parameter(_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.t.forward(self._alpha)

    @alpha.setter
    def alpha(self, value: torch.Tensor | float):
        self._alpha.data = self.t.inverse(
            value * torch.ones_like(self._alpha).data.clone()
        )

    @property
    def gamma(self) -> torch.Tensor:
        return self.t.forward(self._gamma)

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        self._gamma.data = self.t.inverse(value)
