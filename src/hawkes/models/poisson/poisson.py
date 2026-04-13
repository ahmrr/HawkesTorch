import math
import time
import torch
import typing
import logging

from . import PoissonBase, PoissonPenalty
from ..penalty import Penalty
from ... import utils
from ...utils import config


class Poisson(PoissonBase):
    """
    Homogeneous (constant) Poisson process implementation.
    """

    def __init__(
        self,
        M: int,
        mu_init: torch.Tensor | float | None = None,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
        penalization: Penalty = PoissonPenalty(),
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        """
        Initialize a homogeneous Poisson process with constant intensity rates.

        This class models a multivariate Poisson process where each variate has a constant
        intensity rate over time. The intensity for variate m is: μ_m = activation(raw_mu_m),
        where raw_mu_m is the learnable parameter.

        Args:
            M: Number of variates (event types) in the process
            mu_init: Initial intensity values for each variate. If None, random initialization in [0.1, 2.1] is used.
            t_start: Start times for each variate's activity window. If None, defaults to 0 for all variates.
            t_end: End times for each variate's activity window. If None, defaults to infinity for all variates.
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            runtime_config: Runtime configuration for debugging and profiling
            device: PyTorch device for computation ("cpu" or "cuda")

        Raises:
            ValueError: If mu_init tensor has wrong shape

        Example:
            >>> # Simple case: 3 variates, same initial intensity
            >>> model = HomogeneousPoisson(M=3, mu_init=2.0)

            >>> # Different intensities per variate
            >>> model = HomogeneousPoisson(
            ...     M=3,
            ...     mu_init=torch.tensor([1.0, 2.0, 1.5]),
            ...     device="cuda"
            ... )

            >>> # With time windows
            >>> model = HomogeneousPoisson(
            ...     M=2,
            ...     t_start=[0.0, 10.0],  # Variate 1 starts at t=10
            ...     t_end=[100.0, 80.0]   # Variate 1 ends earlier
            ... )
        """

        super().__init__(M, t_start, t_end, penalization, runtime_config, device)

        self.t = transformation

        if mu_init is None:
            # Create random initial mu values (positive values)
            mu_init_tensor = torch.rand(M, device=device) * 0.1
        elif isinstance(mu_init, (int, float)):
            mu_init_tensor = torch.full((M,), float(mu_init), device=device)
        else:
            mu_init_tensor = mu_init.to(device)

        if mu_init_tensor.shape != (M,):
            raise ValueError(
                f"mu_init must be scalar, None (for random), or tensor of shape ({M},), got shape {mu_init_tensor.shape}"
            )

        # Store raw parameters (will be transformed via activation)
        self.raw_mu = torch.nn.Parameter(self.t.inverse(mu_init_tensor))

    @property
    def mu_values(self) -> torch.Tensor:
        """Get the current intensity values (always positive via activation)"""

        return self.t.forward(self.raw_mu)

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute constant intensity at given times, respecting variate activity windows.
        """

        N = t.shape[0]
        mu_vals = self.mu_values  # Shape: (M,)
        base_intensity = mu_vals.unsqueeze(0).expand(N, -1)  # Shape: (N, M)

        # Apply active mask to zero out inactive variates
        active_mask = self._active_mask(t)  # Shape: (N, M)
        return base_intensity * active_mask.float()

    def integral_mu(
        self, t_start: torch.Tensor | float, t_end: torch.Tensor | float
    ) -> torch.Tensor:
        """
        Compute integral of constant intensity from t_start to t_end, respecting variate activity windows.
        """

        mu_vals = self.mu_values  # Shape: (M,)

        # Convert to tensors for easier computation
        if isinstance(t_start, (int, float)):
            t_start = torch.full((self.M,), float(t_start), device=self.device)
        else:
            t_start = t_start.to(self.device)

        if isinstance(t_end, (int, float)):
            t_end = torch.full((self.M,), float(t_end), device=self.device)
        else:
            t_end = t_end.to(self.device)

        # For each variate, compute overlap with its active window
        # Overlap is [max(t_start, self.t_start[m]), min(t_end, self.t_end[m]))
        overlap_start = torch.maximum(t_start, self.t_start)
        overlap_end = torch.minimum(t_end, self.t_end)

        # Duration is positive only if there's actual overlap
        duration = torch.maximum(
            overlap_end - overlap_start, torch.zeros_like(overlap_end)
        )

        return mu_vals * duration

    def upper_bound_in_interval(
        self, t_start: torch.Tensor | float, t_end: torch.Tensor | float
    ) -> float:
        """
        Compute upper bound of total intensity over time interval, respecting variate activity windows.
        """

        mu_vals = self.mu_values  # Shape: (M,)

        # Convert to scalars for interval overlap computation
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.min().item()  # Most conservative bound
        if isinstance(t_end, torch.Tensor):
            t_end = t_end.max().item()  # Most conservative bound

        # Check which variates have any overlap with [t_start, t_end]
        # Variate m overlaps if: t_start < self.t_end[m] AND t_end > self.t_start[m]
        overlaps = (t_start < self.t_end) & (t_end > self.t_start)

        # Sum intensities of overlapping variates
        return (mu_vals * overlaps.float()).sum().item()

    def report_parameters(self) -> str:
        return f"μ_mean={self.mu_values.detach().mean().item():.4f}"

    def get_save_data(self) -> dict:
        return {
            "mu_values": self.mu_values.detach().cpu().numpy(),
        }
