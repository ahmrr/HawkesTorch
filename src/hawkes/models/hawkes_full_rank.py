import math
import torch
import typing

from . import HawkesBase
from ..utils import config


class HawkesFullRank(HawkesBase):
    """Standard full-rank Hawkes process implementation."""

    def __init__(
        self,
        M: int,
        gamma: torch.Tensor,
        init_scale=0.1,
        gamma_param=False,
        debug_config=config.HawkesDebugConfig(),
    ):
        assert not gamma_param, "parametrized gamma not supported yet"

        assert len(gamma.shape) == 1, "gamma must be a rank-1 tensor"
        self.K = len(gamma)
        self.device = gamma.device

        super().__init__(M, self.K, device=self.device, debug_config=debug_config)

        if gamma_param:
            self._inv_gamma = torch.nn.Parameter(gamma.to(self.device))
            # assert self.K == 1
        else:
            self._inv_gamma = gamma

        self.init_scale = init_scale
        self._log_mu = torch.nn.Parameter(math.log(init_scale) * torch.ones(self.M))
        self._log_alpha = torch.nn.Parameter(
            math.log(init_scale) * torch.ones(self.K, self.M, self.M)
        )

    @property
    def mu(self) -> torch.Tensor:
        return self._log_mu.exp()

    @mu.setter
    def mu(self, value: torch.Tensor | float):
        self._log_mu.data = torch.log(
            value * torch.ones_like(self._log_mu).data.clone()
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self._log_alpha.exp()

    @alpha.setter
    def alpha(self, value: torch.Tensor | float):
        self._log_alpha.data = torch.log(
            value * torch.ones_like(self._log_alpha).data.clone()
        )

    @property
    def gamma(self) -> torch.Tensor:
        return self._inv_gamma

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        self._inv_gamma = value  # TODO: param doesn't work
