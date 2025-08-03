import math
import torch
import typing
import torch.nn.functional as F

from . import HawkesBase, HawkesNLL
from ..utils import config, _torch_scan


# TODO: Accomodate parametrized gamma in gradient
class HawkesFullRankNLL(HawkesNLL):
    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor):
        # TODO: can use existing intensity exp_decay

        intensity, M, K, T, ti, mi = ctx.saved_tensors
        M, K, T = M.item(), K.item(), T.item()

        dti = ti - torch.cat([ti[:, 0:1, :], ti[:, 0:-1, :]], dim=1)  # Shape: [1, N, 1]
        exp_decay = torch.exp(-gamma * dti)  # Shape: [K, N, 1]
        one_hot_vectors = torch.cat(
            [torch.zeros(M).unsqueeze(0), F.one_hot(mi[:-1], num_classes=M)], dim=0
        )  # Shape: [N, M]

        # Compute K sequence (of shape [N, M, K]) where K[:, q, k] corresponds to alpha[p, q, k]

        transition_matrices = torch.cat(
            [exp_decay, exp_decay * one_hot_vectors.unsqueeze(0)], dim=2
        )  # Shape: [K, N, M+1]

        # Compute prefix matrices P
        P = _torch_scan.prefix_scan(
            transition_matrices, prefix_func=_torch_scan.state_mult, dim=1
        )  # Shape: [K, N, M+1]

        # TODO: optimize this using sorting and binning
        # split_intensity = []  # Length: M
        # for p in range(1, M + 1):
        #     split_intensity.append(intensity[mi == p])
        # Obtain sorting indices of event nodes and the length of each node type in the sorted tensor
        mi_sorted, idx = torch.sort(mi)
        mi_counts = torch.bincount(mi_sorted, minlength=M + 1)[1:].tolist()  # Ignore 0

        # Sort intensity and exp_decay according to their node type, and split based on which node they correspond to
        intensity_split = torch.split(
            intensity[idx].squeeze(-1), mi_counts
        )  # Shape: [M, Np]
        # exp_decay_split = torch.split(
        #     exp_decay[:, idx].squeeze(-1), counts.tolist(), dim=1
        # )  # Shape: [M, K, Np]

        P_split = torch.split(P[:, idx, :], mi_counts, dim=1)  # Shape: [M, K, Np, M+1]

        # Compute mu gradient; only depends on each node type's intensity
        mu_grad = (
            torch.tensor(
                [intensity_p.reciprocal().sum() for intensity_p in intensity_split]
            )
            - T
        )  # Shape: [Np]

        # Compute alpha gradient components using prefix scanned intensity

        return (None, alpha_grad, mu_grad, None) + (None,) * 6


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

        super().__init__(M, K, device=self.device, debug_config=debug_config)

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
