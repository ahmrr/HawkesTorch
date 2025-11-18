import math
import torch
import typing

from . import HawkesBase
from ..utils import config


class HawkesLowRank(HawkesBase):
    """Low-rank Hawkes process implementation."""

    def __init__(
        self,
        M: int,
        gamma: torch.Tensor,
        rank: int,
        init_scale=0.1,
        gamma_param=False,
        transformation=config.SOFTPLUS,
        debug_config=config.HawkesDebugConfig(),
    ):
        """
        Args:
            M: Number of nodes
            gamma: Initial or fixed (if gamma is parametrized or not, respectively) memory values for the exponential decay kernels
            rank: Approximate rank to use for the parametrization of alpha
            init_scale: Initial scale for all parameters
            gamma_param: Whether to parametrize gamma, the exponential kernel memory values
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            debug_config: Debug configuration settings
        """

        if len(gamma.shape) != 1:
            raise ValueError("gamma must be a rank-1 tensor")

        K = len(gamma)
        self.device = gamma.device

        super().__init__(M, K, device=self.device, debug_config=debug_config)

        self.t = transformation

        if gamma_param:
            self._inv_gamma = torch.nn.Parameter(
                (self.t.inverse(gamma)).to(self.device)
            )
        else:
            self._inv_gamma = self.t.inverse(gamma)

        self.rank = rank
        self.init_scale = torch.tensor([init_scale])

        # Initialize low-rank parameters
        self._inv_mu = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(M)).to(self.device)
        )
        self._inv_alpha_diag = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(K, M)).to(self.device)
        )  # Diagonal of alpha
        self._inv_U = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(K, M, rank)).to(self.device)
        )
        self._inv_V = torch.nn.Parameter(
            (self.t.inverse(self.init_scale) * torch.ones(K, rank, M)).to(self.device)
        )

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
        # Construct alpha as UV
        alpha_diag = torch.diag_embed(
            self.t.forward(self._inv_alpha_diag), dim1=-2, dim2=-1
        )  # Shape: (K, M, M)

        alpha_off_diag = torch.matmul(
            self.t.forward(self._inv_U), self.t.forward(self._inv_V)
        )

        # This makes the diagonal elements sensitive to both U/V and alpha_diag
        alpha_diag = alpha_diag * alpha_off_diag

        mask = torch.eye(self.M, device=self._inv_alpha_diag.device).bool()
        alpha = torch.where(
            mask,
            alpha_diag,
            alpha_off_diag,
        )

        return alpha

    @alpha.setter
    def alpha(self, value: torch.Tensor | float):
        if isinstance(value, float):
            # For scalar initialization, initialize U and V with sqrt(value)
            sqrt_val = math.sqrt(value)
            self._inv_U.data = self.t.inverse(
                sqrt_val * torch.ones_like(self._inv_U).data
            )
            self._inv_V.data = self.t.inverse(
                sqrt_val * torch.ones_like(self._inv_V).data
            )
        else:
            # For tensor initialization, use SVD for each kernel
            for k in range(self.K):
                U, S, V = torch.linalg.svd(value[k])
                # Take only top 'rank' singular values/vectors
                U_k = U[:, : self.rank] * torch.sqrt(S[: self.rank])
                V_k = V[: self.rank, :] * torch.sqrt(S[: self.rank, None])
                # Set log parameters
                self._inv_U.data[k] = self.t.inverse(
                    torch.abs(U_k) + 1e-8
                )  # small constant for numerical stability
                self._inv_V.data[k] = self.t.inverse(torch.abs(V_k) + 1e-8)

    @property
    def gamma(self) -> torch.Tensor:
        return self.t.forward(self._inv_gamma)

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        self._inv_gamma.data = self.t.inverse(value)
