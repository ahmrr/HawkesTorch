import math
import torch
import typing

from . import HawkesBase
from ..utils import config


class HawkesUpperTriangular(HawkesBase):
    """
    Hawkes process with upper triangular parameterization where alpha = P @ L @ P^T.
    L is constrained to be upper triangular to address identifiability.
    """

    def __init__(
        self,
        M: int,
        gamma: torch.Tensor,
        rank: int,
        init_scale=0.1,
        gamma_param=False,
        debug_config=config.HawkesDebugConfig(),
    ):
        """
        Args:
            M: Number of event types
            gamma: Kernel decay rates
            rank: Dimension of latent space (R in P @ L @ P^T)
            init_scale: Scale for parameter initialization
        """

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

        self.rank = rank
        self.init_scale = init_scale

        # Initialize parameters
        self._log_mu = torch.nn.Parameter(math.log(init_scale) * torch.ones(self.M))
        self._alpha_diag = torch.nn.Parameter(
            math.log(init_scale) * torch.ones(self.K, self.M)
        )  # Diagonal of alpha
        self._log_P = torch.nn.Parameter(math.log(init_scale) * torch.ones(self.M, rank))
        self._log_L = torch.nn.Parameter(
            math.log(init_scale) * torch.ones(self.K, rank, rank)
        )

    @property
    def P(self) -> torch.Tensor:
        return self._log_P.exp()

    @property
    def L(self) -> torch.Tensor:
        return torch.triu(self._log_L.exp())

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
        # Compute P @ L @ P^T for each kernel
        P = self.P  # avoid computing P twice
        alpha_diag = torch.diag_embed(
            self._alpha_diag.exp(), dim1=-2, dim2=-1
        )  # Shape: (K, M, M)

        alpha_off_diag = torch.einsum(
            "mr,krs,ns->kmn", P, self.L, P
        )  # Shape: (K, M, M)

        # this makes the diagonal elements sensitive to both U/V and alpha_diag
        alpha_diag = alpha_diag * alpha_off_diag

        mask = torch.eye(self.M, device=self._alpha_diag.device).bool()
        alpha = torch.where(
            mask,
            alpha_diag,
            alpha_off_diag,
        )

        return alpha

    @alpha.setter
    def alpha(self, value: float):
        if isinstance(value, float):
            sqrt_sqrt_val = math.sqrt(math.sqrt(value))
            self._log_P.data = torch.log(
                sqrt_sqrt_val * torch.ones_like(self._log_P).data
            )
            self._log_L.data = torch.log(
                sqrt_sqrt_val * torch.ones_like(self._log_L).data
            )

    @property
    def num_params(self) -> int:
        """override num params to subtract lower diagonal of _log_L"""
        p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        p -= self.rank * (self.rank - 1) // 2

        return p

    @property
    def gamma(self) -> torch.Tensor:
        return self._inv_gamma

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        self._inv_gamma = value  # TODO: param doesn't work
