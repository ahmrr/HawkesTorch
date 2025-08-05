import math
import torch
import typing
import torch.nn.functional as F

from . import HawkesBase, HawkesNLL
from ..utils import config, _torch_scan


class HawkesFullRankNLL(HawkesNLL):
    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor):
        # TODO: can use existing intensity exp_decay, maybe that would save some time or mem
        # TODO: replace this with argument to forward pass and accomodate parametrized gamma in gradient
        GAMMA_PARAM = False

        intensity, mu, alpha, gamma, ti, mi = ctx.saved_tensors
        M, K, T = ctx.M, ctx.K, ctx.T

        dti = ti - torch.cat([ti[:, 0:1, :], ti[:, 0:-1, :]], dim=1)  # Shape: [1, N, 1]
        exp_decay = torch.exp(-gamma * dti).permute(2, 1, 0)  # Shape: [K, N, 1]
        one_hot_vectors = torch.cat(
            [
                torch.zeros(M, device=ti.device).unsqueeze(0),
                F.one_hot(mi[:-1], num_classes=M),
            ],
            dim=0,
        )  # Shape: [N, M]

        # Compute K sequence (of shape [N, M, K]) where K[:, q, k] corresponds to alpha[p, q, k]

        transition_matrices = torch.cat(
            [exp_decay, exp_decay * one_hot_vectors.unsqueeze(0)], dim=2
        )  # Shape: [K, N, M+1]

        # Perform prefix scan on transition matrices
        prefix_matrices = _torch_scan.prefix_scan(
            transition_matrices, prefix_func=_torch_scan.state_mult, dim=1
        )  # Shape: [K, N, M+1]

        # Obtain sorting indices of event nodes and the length of each node type in the sorted tensor
        mi_sorted, idx = torch.sort(mi, stable=True)
        mi_counts = torch.bincount(mi_sorted, minlength=M).tolist()  # Ignore 0

        # Sort ti, intensity, and states according to their node type (preserving temporal order as well)
        # Split based on which node they correspond to, with each split subsequence being sorted already
        # The first tuple index is the node type; i.e., mi_split[p] contains events of node type p + 1
        # The docs say that split returns a view of the original tensor, so this should be efficient
        ti_split = torch.split(ti[:, idx, :], mi_counts, dim=1)  # Shape: [M, 1, Np, 1]
        intensity_split = torch.split(
            intensity[idx, None, None], mi_counts, dim=0
        )  # Shape: [M, Np, 1, 1]
        states = prefix_matrices[:, :, 1:].permute(1, 2, 0)  # Shape: [N, M, K]
        states_split = torch.split(states, mi_counts, dim=0)  # Shape: [M, Np, M, K]

        # print(len(states_split), states_split[0].shape)
        # print(len(intensity_split), intensity_split[0].shape)
        # print(gamma.shape)

        # Compute mu gradient; only depends on each node type's intensity
        mu_grad = (
            (
                torch.tensor(
                    [λp.reciprocal().sum() for λp in intensity_split], device=ti.device
                )
                - T
            )
            * ctx.trans.derivative(ctx.trans.inverse(mu))
            * grad_output
        )

        # Compute alpha gradient components using intensity and prefix matrices
        intensity_state_sum = torch.stack(
            [
                (gamma * Kp / λp).sum(dim=0)
                for (Kp, λp) in zip(states_split, intensity_split)
            ],
            dim=0,
        )  # Shape: [M, M, K]
        exp_decay_sum = torch.stack(
            [-torch.expm1(-gamma * (T - tiq)).sum(dim=1) for tiq in ti_split],
            dim=1,
        )  # Shape: [1, M, K]

        alpha_grad = (
            (intensity_state_sum - exp_decay_sum).permute(2, 0, 1)
            * ctx.trans.derivative(ctx.trans.inverse(alpha))
            * grad_output
        )
        gamma_grad = None  # * grad_output

        return (None, alpha_grad, mu_grad, gamma_grad) + (None,) * 7


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
            nll_function=HawkesFullRankNLL,
            transformation=transformation,
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
