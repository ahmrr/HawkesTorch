import math
import torch
import typing
import torch.nn.functional as F

from . import HawkesBase
from ..utils import config, _torch_scan

torch.set_printoptions(threshold=10_000)


class HawkesFullRank(HawkesBase):
    """Standard full-rank Hawkes process implementation."""

    def __init__(
        self,
        M: int,
        gamma: torch.Tensor,
        init_scale=0.1,
        gamma_param=False,
        transformation=config.SOFTPLUS,
        debug_config=config.HawkesDebugConfig(),
    ):
        """
        Args:
            M: Number of nodes
            gamma: Initial or fixed (if gamma is parametrized or not, respectively) memory values for the exponential decay kernels
            init_scale: Initial scale for all parameters
            gamma_param: Whether to parametrize gamma, the exponential kernel memory values
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            debug_config: Debug configuration settings
        """

        if len(gamma.shape) != 1:
            raise ValueError("gamma must be a rank-1 tensor")

        K = len(gamma)
        self.device = gamma.device

        super().__init__(
            M,
            K,
            device=self.device,
            debug_config=debug_config,
        )

        self.trans = transformation

        if gamma_param:
            self._inv_gamma = torch.nn.Parameter(
                (self.trans.inverse(gamma)).to(self.device)
            )
        else:
            self._inv_gamma = self.trans.inverse(gamma)

        self.init_scale = torch.tensor([init_scale])

        # Initialize low-rank parameters
        self._inv_mu = torch.nn.Parameter(
            (self.trans.inverse(self.init_scale) * torch.ones(M)).to(self.device)
        )
        self._inv_alpha = torch.nn.Parameter(
            (
                self.trans.inverse(self.init_scale) * torch.ones(self.K, self.M, self.M)
            ).to(self.device)
        )

    @property
    def mu(self) -> torch.Tensor:
        return self.trans.forward(self._inv_mu)

    @mu.setter
    def mu(self, value: torch.Tensor | float):
        self._inv_mu.data = self.trans.inverse(
            value * torch.ones_like(self._inv_mu).data.clone()
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.trans.forward(self._inv_alpha)

    @alpha.setter
    def alpha(self, value: torch.Tensor | float):
        self._inv_alpha.data = self.trans.inverse(
            value * torch.ones_like(self._inv_alpha).data.clone()
        )

    @property
    def gamma(self) -> torch.Tensor:
        return self.trans.forward(self._inv_gamma)

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        self._inv_gamma.data = self.trans.inverse(value)
