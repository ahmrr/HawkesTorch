import math
import time
import torch
import typing
import logging
import torch.nn as nn
from dataclasses import dataclass

from . import PoissonBase
from ..penalty import Penalty
from ... import utils
from ...utils import config


@dataclass
class PoissonPiecewisePenalty:
    """
    Penalization hyperparameters for the PoissonPiecewise model.

    Both penalties act on the transformed (positive) rates, so their weights
    are interpretable in intensity units.

    Attributes:
        rate: Penalty applied to the bin rates μ_{m,j} themselves
        smoothness: Penalty applied to the differences between adjacent bins,
            μ_{m,j+1} - μ_{m,j}, which discourages large jumps between bins
            (e.g. L2Penalty for a smooth profile, L1Penalty for a sparse,
            "few change points" profile)
    """

    rate: Penalty | None = None
    smoothness: Penalty | None = None


class PoissonPiecewise(PoissonBase):
    """
    Poisson process with a piecewise-constant intensity for each variate.
    """

    def __init__(
        self,
        M: int,
        breakpoints: torch.Tensor | None = None,
        num_bins: int | None = None,
        mu_init: torch.Tensor | float | None = None,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
        penalization=PoissonPiecewisePenalty(),
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        """
        Initialize a PoissonPiecewise process with a piecewise-constant intensity.

        This class models a multivariate Poisson process whose intensity is constant
        on each of B bins defined by a set of breakpoints e_0 < e_1 < ... < e_B, which
        are shared across variates. The intensity of variate m at time t is:
        μ_m(t) = activation(raw_μ_{m,j(t)}), where j(t) is the index of the bin
        containing t. This is the natural base rate for data whose activity level
        drifts over long timescales (e.g. months of social media activity), since it
        absorbs slow trend that would otherwise be pushed into the excitation matrix.

        Bins are half-open, [e_j, e_{j+1}), so a time falling exactly on a breakpoint
        belongs to the bin starting there. The first and last bins extend to -inf and
        +inf respectively, meaning the intensity is defined (by constant extrapolation)
        outside [e_0, e_B); the activity windows t_start/t_end remain the mechanism
        that turns a variate off. Both mu and integral_mu follow this convention, so
        they are always consistent with one another.

        Args:
            M: Number of variates (event types) in the process
            breakpoints: Bin edges of shape (B + 1,), strictly increasing and shared by
                all variates. Exactly one of breakpoints or num_bins must be given.
            num_bins: Convenience alternative to breakpoints, producing a uniform grid
                of num_bins bins spanning [min(t_start), max(t_end)], which must
                therefore both be finite. Exactly one of breakpoints or num_bins must
                be given.
            mu_init: Initial intensity values.
                - If float: Same rate for every variate and every bin
                - If torch.Tensor: Must have shape (M, B), per-variate, per-bin rates
                - If None: Random initialization in [0, 0.1)
                Default: None
            t_start: Start times for each variate's activity window. If None, defaults to 0 for all variates.
            t_end: End times for each variate's activity window. If None, defaults to infinity for all variates.
            penalization: Penalization hyperparameters, see PoissonPiecewisePenalty
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            runtime_config: Runtime configuration for debugging and profiling
            device: PyTorch device for computation ("cpu" or "cuda")

        Raises:
            ValueError: If the breakpoints are malformed, if neither or both of
                breakpoints and num_bins are given, or if mu_init has the wrong shape.

        Notes:
            The intensity is, with j(t) the index of the bin containing t,

            .. math::
                \\mu_m(t) = \\text{activation}\\left(\\text{raw}\\mu_{m, j(t)}\\right),
                \\quad j(t) = \\max\\{j : e_j \\le t\\}

            and the integral over an interval is available in closed form as the sum of
            each bin's rate times its overlap with the interval,

            .. math::
                \\int_a^b \\mu_m(s)\\,ds = \\sum_j \\mu_{m,j}
                \\left|[e_j, e_{j+1}) \\cap [a, b]\\right|

            which integral_mu implements (differentiably) in place of the numerical
            fallback in PoissonBase.

        Examples:
            >>> # Uniform monthly bins over three years of daily-scale data
            >>> model = PoissonPiecewise(M=4, num_bins=36, t_end=1080.0)

            >>> # Explicit breakpoints and a known step profile
            >>> model = PoissonPiecewise(
            ...     M=2,
            ...     breakpoints=torch.tensor([0.0, 10.0, 20.0, 30.0]),
            ...     mu_init=torch.tensor([[0.5, 3.0, 1.0], [2.0, 0.5, 2.5]]),
            ...     t_end=30.0,
            ... )

            >>> # Discourage large jumps between adjacent bins
            >>> model = PoissonPiecewise(
            ...     M=2,
            ...     num_bins=50,
            ...     t_end=100.0,
            ...     penalization=PoissonPiecewisePenalty(smoothness=L2Penalty(1e-2)),
            ... )

        Note:
            - More bins allow more detailed trends, but each bin is estimated from only
              the events it contains; a smoothness penalty helps when bins are narrow
            - Bins need not be uniform; pass breakpoints directly for e.g. finer
              resolution in a period of interest
        """

        super().__init__(M, t_start, t_end, penalization, runtime_config, device)

        self.t = transformation

        self.breakpoints = self._resolve_breakpoints(breakpoints, num_bins)

        # Effective bin edges, with the outer bins extended to +/- infinity so that
        # mu(t) is defined for every t and agrees with integral_mu
        self._bin_start = self.breakpoints[:-1].clone()
        self._bin_end = self.breakpoints[1:].clone()
        self._bin_start[0] = float("-inf")
        self._bin_end[-1] = float("inf")

        if mu_init is None:
            mu_init_tensor = (
                torch.rand(M, self.num_bins, device=self.device) * 0.1
            )  # Random values in [0, 0.1)
        elif isinstance(mu_init, (int, float)):
            mu_init_tensor = torch.full(
                (M, self.num_bins), float(mu_init), device=self.device
            )
        else:
            mu_init_tensor = mu_init.to(self.device)

        if mu_init_tensor.shape != (M, self.num_bins):
            raise ValueError(
                f"mu_init must be scalar, None (for random), or tensor of shape ({M}, {self.num_bins}), got shape {tuple(mu_init_tensor.shape)}"
            )

        # Store raw parameters (will be transformed via activation)
        self._mu = nn.Parameter(self.t.inverse(mu_init_tensor))

    def _resolve_breakpoints(
        self, breakpoints: torch.Tensor | None, num_bins: int | None
    ) -> torch.Tensor:
        """
        Validate user-supplied breakpoints, or build a uniform grid of num_bins bins
        spanning the union of the variate activity windows.

        Args:
            breakpoints: Bin edges of shape (B + 1,), or None
            num_bins: Number of uniform bins to create, or None

        Returns:
            Tensor of shape (B + 1,) with the strictly increasing bin edges
        """

        if (breakpoints is None) == (num_bins is None):
            raise ValueError("Exactly one of breakpoints or num_bins must be given")

        if breakpoints is None:
            if num_bins < 1:
                raise ValueError(f"num_bins must be positive, got {num_bins}")

            grid_start, grid_end = self.simulation_bounds
            if math.isinf(grid_start) or math.isinf(grid_end):
                raise ValueError(
                    "num_bins requires finite t_start and t_end; pass breakpoints explicitly instead"
                )

            return torch.linspace(
                grid_start, grid_end, num_bins + 1, device=self.device
            )

        breakpoints = breakpoints.to(self.device)

        if breakpoints.ndim != 1 or breakpoints.numel() < 2:
            raise ValueError(
                f"breakpoints must be a rank-1 tensor with at least 2 entries, got shape {tuple(breakpoints.shape)}"
            )
        if not torch.all(torch.diff(breakpoints) > 0):
            raise ValueError("breakpoints must be strictly increasing")

        return breakpoints

    @property
    def num_bins(self) -> int:
        """Number of piecewise-constant bins."""

        return self.breakpoints.numel() - 1

    @property
    def mu_values(self) -> torch.Tensor:
        """Get the current per-bin intensity values, of shape (M, B) (always positive via activation)"""

        return self.t.forward(self._mu)

    def bin_index(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the index of the bin containing each time in t.

        Bins are half-open, [e_j, e_{j+1}), with the outer bins extended to
        +/- infinity, so the returned indices always lie in [0, B - 1].

        Args:
            t: Tensor of shape (N,) with times

        Returns:
            Long tensor of shape (N,) with bin indices
        """

        # right=True gives boundaries[i - 1] <= t < boundaries[i], i.e. half-open bins.
        # Only the interior breakpoints are boundaries, since the outer bins are
        # unbounded, which also keeps the indices in range.
        return torch.bucketize(
            t.to(self.breakpoints.dtype), self.breakpoints[1:-1], right=True
        )

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute piecewise-constant intensity at given times, respecting variate activity windows.
        """

        indices = self.bin_index(t)  # Shape: (N,)

        # mu_values is (M, B); transposing and indexing by bin gives (N, M)
        base_intensity = self.mu_values.transpose(0, 1)[indices]  # Shape: (N, M)

        # Apply active mask to zero out inactive variates
        active_mask = self._active_mask(t)  # Shape: (N, M)
        return base_intensity * active_mask.float()

    def integral_mu(
        self, t_start: torch.Tensor | float, t_end: torch.Tensor | float
    ) -> torch.Tensor:
        """
        Compute the exact integral of the piecewise-constant intensity from t_start to
        t_end, respecting variate activity windows.

        The integral is the sum over bins of each bin's rate times the length of the
        overlap between that bin and the requested interval (further intersected with
        the variate's activity window), which is differentiable with respect to the
        bin rates.
        """

        mu_vals = self.mu_values  # Shape: (M, B)

        # Convert to tensors for easier computation
        if isinstance(t_start, (int, float)):
            t_start = torch.full((self.M,), float(t_start), device=self.device)
        else:
            t_start = t_start.to(self.device)

        if isinstance(t_end, (int, float)):
            t_end = torch.full((self.M,), float(t_end), device=self.device)
        else:
            t_end = t_end.to(self.device)

        # For each variate, intersect the requested interval with its active window
        overlap_start = torch.maximum(t_start, self.t_start).unsqueeze(-1)  # (M, 1)
        overlap_end = torch.minimum(t_end, self.t_end).unsqueeze(-1)  # (M, 1)

        # Then intersect with each bin, [max(a, e_j), min(b, e_{j+1}))
        bin_start = torch.maximum(overlap_start, self._bin_start)  # (M, B)
        bin_end = torch.minimum(overlap_end, self._bin_end)  # (M, B)

        # Duration is positive only if there's actual overlap
        durations = torch.clamp(bin_end - bin_start, min=0.0)  # (M, B)

        return (mu_vals * durations).sum(dim=-1)  # Shape: (M,)

    def upper_bound_in_interval(
        self, t_start: torch.Tensor | float, t_end: torch.Tensor | float
    ) -> float:
        """
        Compute an exact upper bound of total intensity over time interval, respecting
        variate activity windows.

        The total intensity is piecewise constant, so the bound is the largest bin
        total among the bins overlapping [t_start, t_end].
        """

        mu_vals = self.mu_values  # Shape: (M, B)

        # Convert to scalars for interval overlap computation
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.min().item()  # Most conservative bound
        if isinstance(t_end, torch.Tensor):
            t_end = t_end.max().item()  # Most conservative bound

        if t_end <= t_start:
            return 0.0

        # Check which variates have any overlap with [t_start, t_end]
        overlaps = (t_start < self.t_end) & (t_end > self.t_start)  # Shape: (M,)

        # Check which bins have any overlap with [t_start, t_end]
        bin_overlaps = (self._bin_end > t_start) & (self._bin_start < t_end)  # (B,)

        with torch.no_grad():
            # Total intensity in each bin, summed over the overlapping variates
            bin_totals = (mu_vals * overlaps.float().unsqueeze(-1)).sum(dim=0)  # (B,)

            return bin_totals[bin_overlaps].max().item()

    def penalty(self) -> torch.Tensor:
        """Compute the regularization penalty for the model parameters."""

        penalty = 0.0

        mu_vals = self.mu_values  # Shape: (M, B)

        if self.penalization.rate is not None:
            penalty += self.penalization.rate(mu_vals)
        if self.penalization.smoothness is not None:
            penalty += self.penalization.smoothness(torch.diff(mu_vals, dim=-1))

        return penalty

    def report_parameters(self) -> str:
        """
        Report the current parameter state of the model for logging purposes.
        """

        mu_vals = self.mu_values.detach().cpu().numpy()

        return (
            f"μ_mean={mu_vals.mean():.4f}, "
            f"μ_min={mu_vals.min():.4f}, "
            f"μ_max={mu_vals.max():.4f}, "
            f"num_bins={self.num_bins}"
        )

    def get_save_data(self) -> dict:
        """
        Get PoissonPiecewise-specific data for saving to file.
        """

        return {
            "mu_values": self.mu_values.detach().cpu().numpy(),
            "breakpoints": self.breakpoints.detach().cpu().numpy(),
            "num_bins": self.num_bins,
        }
