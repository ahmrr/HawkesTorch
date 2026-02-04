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
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
    ):
        """
        Args:
            M: Number of nodes
            gamma: Initial or fixed (if gamma is parametrized or not, respectively) memory values for the exponential decay kernels
            rank: Approximate rank to use for the parametrization of alpha
            init_scale: Initial scale for all parameters
            gamma_param: Whether to parametrize gamma, the exponential kernel memory values
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            runtime_config: Debug configuration settings
        """

        # TODO: fix out-of-memory errors

        if len(gamma.shape) != 1:
            raise ValueError("gamma must be a rank-1 tensor")

        K = len(gamma)
        self.device = gamma.device

        super().__init__(M, K, device=self.device, runtime_config=runtime_config)

        self.t = transformation

        if gamma_param:
            self._inv_gamma = torch.nn.Parameter(
                (self.t.inverse(gamma)).to(self.device)
            )
        else:
            self._inv_gamma = self.t.inverse(gamma)

        self.rank = rank
        self.init_scale = torch.tensor([init_scale])

        # Initialize parameters
        self._inv_mu = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(M)).to(self.device)
        )
        self._inv_alpha_diag = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(K, M)).to(self.device)
        )  # Diagonal of alpha
        self._inv_P = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(M, rank)).to(self.device)
        )
        self._inv_L = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(K, rank, rank)).to(
                self.device
            )
        )

    @property
    def P(self) -> torch.Tensor:
        return self.t.forward(self._inv_P)

    @property
    def L(self) -> torch.Tensor:
        return torch.triu(self.t.forward(self._inv_L))

    @property
    def mu(self) -> torch.Tensor:
        return self.t.forward(self._inv_mu)

    @mu.setter
    def mu(self, value: torch.Tensor | float):
        self._inv_mu.data = self.t.inverse(
            value * torch.ones_like(self._inv_mu).data.clone()
        )

    @property
    def alpha(self) -> torch.Tensor:
        # Compute P @ L @ P^T for each kernel
        P = self.P  # avoid computing P twice
        alpha_diag = torch.diag_embed(
            self.t.forward(self._inv_alpha_diag), dim1=-2, dim2=-1
        )  # Shape: (K, M, M)

        alpha_off_diag = torch.einsum(
            "mr,krs,ns->kmn", P, self.L, P
        )  # Shape: (K, M, M)

        # this makes the diagonal elements sensitive to both U/V and alpha_diag
        alpha_diag = alpha_diag * alpha_off_diag

        mask = torch.eye(self.M, device=self.device).bool()
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
            self._inv_P.data = self.t.inverse(
                sqrt_sqrt_val * torch.ones_like(self._inv_P).data
            )
            self._inv_L.data = self.t.inverse(
                sqrt_sqrt_val * torch.ones_like(self._inv_L).data
            )

    @property
    def num_params(self) -> int:
        """override num params to subtract lower diagonal of _inv_L"""
        p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        p -= self.rank * (self.rank - 1) // 2

        return p

    @property
    def gamma(self) -> torch.Tensor:
        return self.t.forward(self._inv_gamma)

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        self._inv_gamma.data = self.t.inverse(value)
