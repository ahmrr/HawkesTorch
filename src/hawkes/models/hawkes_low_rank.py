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
        transformation: typing.Literal["exp", "softplus"] | None = "exp",
        debug_config=config.HawkesDebugConfig(),
    ):
        assert len(gamma.shape) == 1, "gamma must be a rank-1 tensor"
        self.K = len(gamma)
        self.device = gamma.device

        super().__init__(M, self.K, device=self.device, debug_config=debug_config)

        match transformation:
            case "exp":
                self.trans = torch.exp
                self.inv_trans = torch.log
            case "softplus":
                self.trans = torch.nn.functional.softplus
                self.inv_trans = lambda x: x + torch.log(-torch.expm1(-x))
            case _:
                print(f'Error: unsupported transformation "{transformation}"')

        if gamma_param:
            self._inv_gamma = torch.nn.Parameter(
                (self.inv_trans(gamma)).to(self.device)
            )
            # assert self.K == 1
        else:
            self._inv_gamma = gamma

        self.rank = rank
        self.init_scale = init_scale

        # Initialize low-rank parameters
        self._inv_mu = torch.nn.Parameter(
            (self.inv_trans(torch.tensor([init_scale])) * torch.ones(self.M)).to(
                self.device
            )
        )
        self._inv_alpha_diag = torch.nn.Parameter(
            (
                self.inv_trans(torch.tensor([init_scale])) * torch.ones(self.K, self.M)
            ).to(self.device)
        )  # Diagonal of alpha
        self._inv_U = torch.nn.Parameter(
            (
                self.inv_trans(torch.tensor([init_scale]))
                * torch.ones(self.K, self.M, rank)
            ).to(self.device)
        )
        self._inv_V = torch.nn.Parameter(
            (
                self.inv_trans(torch.tensor([init_scale]))
                * torch.ones(self.K, rank, self.M)
            ).to(self.device)
        )

    @property
    def mu(self) -> torch.Tensor:
        return self.trans(self._inv_mu)

    @mu.setter
    def mu(self, value: torch.Tensor | float):
        self._inv_mu.data = self.inv_trans(
            value * torch.ones_like(self._inv_mu).data.clone()
        )

    @property
    def alpha(self) -> torch.Tensor:
        # Construct alpha as UV
        alpha_diag = torch.diag_embed(
            self.trans(self._inv_alpha_diag), dim1=-2, dim2=-1
        )  # Shape: (K, M, M)

        alpha_off_diag = torch.matmul(self.trans(self._inv_U), self.trans(self._inv_V))

        # this makes the diagonal elements sensitive to both U/V and alpha_diag
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
            self._inv_U.data = self.inv_trans(
                sqrt_val * torch.ones_like(self._inv_U).data
            )
            self._inv_V.data = self.inv_trans(
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
                self._inv_U.data[k] = self.inv_trans(
                    torch.abs(U_k) + 1e-8
                )  # small constant for numerical stability
                self._inv_V.data[k] = self.inv_trans(torch.abs(V_k) + 1e-8)

    @property
    def gamma(self) -> torch.Tensor:
        if isinstance(self._inv_gamma, torch.nn.Parameter):
            return self.trans(self._inv_gamma)
        else:
            return self._inv_gamma

    @gamma.setter
    def gamma(self, value: torch.Tensor):
        if isinstance(self._inv_gamma, torch.nn.Parameter):
            self._inv_gamma.data = self.inv_trans(value)
        else:
            self._inv_gamma = value
