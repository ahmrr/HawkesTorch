import math
import time
import torch
import typing
import logging
from abc import ABC, abstractmethod

from .. import utils
from ..utils import config


class PoissonBase(torch.nn.Module, ABC):
    """Base class for Poisson process implementations."""

    def __init__(
        self,
        M: int,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        super().__init__()
        self.M = M
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

        self.t = transformation

        # Set up time bounds for each variate
        if t_start is None:
            self.t_start = torch.zeros(M, device=self.device).float()
        elif isinstance(t_start, (int, float)):
            self.t_start = torch.full((M,), float(t_start), device=self.device)
        else:
            self.t_start = t_start.to(self.device)

        if t_end is None:
            self.t_end = torch.full((M,), float("inf"), device=self.device)
        elif isinstance(t_end, (int, float)):
            self.t_end = torch.full((M,), float(t_end), device=self.device)
        else:
            self.t_end = t_end.to(self.device)

        # Validate shapes and bounds
        if self.t_start.shape != (M,):
            raise ValueError(f"t_start must be of shape (M,), got {self.t_start.shape}")
        if self.t_end.shape != (M,):
            raise ValueError(f"t_end must be of shape (M,), got {self.t_end.shape}")
        if not torch.all(self.t_end > self.t_start):
            raise ValueError(
                "All t_start values must be less than corresponding t_end values"
            )

    def _active_mask(self, t: torch.Tensor | float) -> torch.Tensor:
        """
        Utility to get a mask of which variates are active at time t.

        Variate m is active at time t if t_start[m] <= t < t_end[m].

        Args:
            t: Scalar time or tensor of times

        Returns:
            Boolean tensor of shape (M,) if t is scalar, or (N, M) if t has shape (N,)
        """

        if isinstance(t, torch.Tensor):
            t = t.unsqueeze(-1)

        return (self.t_start <= t) & (t < self.t_end)  # Shape (N, M)

    @property
    def simulation_bounds(self) -> tuple[float, float]:
        """
        Get overall simulation time bounds across all variates.

        Returns:
            Tuple of (sim_start, sim_end)
        """

        sim_start = self.t_start.min().item()
        sim_end = self.t_end.max().item()
        if math.isinf(sim_end):
            sim_end = float("inf")

        return sim_start, sim_end

    @abstractmethod
    def mu(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-varying base intensity at given times.

        Args:
            t: Tensor of shape (N,) with times, or scalar time

        Returns:
            Tensor of shape (N, M) with base intensities for each variate at each time
        """

        pass

    def integral_mu(
        self,
        t_start: torch.Tensor | float,
        t_end: torch.Tensor | float,
        n_points: int = 100,
    ):
        """
        Compute integral of base intensity between t_start and t_end for each variate.

        Uses numerical integration with the trapezoidal rule.

        Args:
            t_start: Start times. Can be scalar (same for all variates) or tensor of shape (M,)
            t_end: End times. Can be scalar (same for all variates) or tensor of shape (M,)
            n_points: Number of points to use for numerical integration

        Returns:
            Tensor of shape (M,) with integral of base intensity for each variate
        """

        # Use same interval for all variates since mu(t) handles masking
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.min().item()
        if isinstance(t_end, torch.Tensor):
            t_end = t_end.max().item()

        if t_end <= t_start:
            return torch.zeros(self.M, device=self.device)

        # Create uniform time grid
        t_points = torch.linspace(t_start, t_end, n_points, device=self.device)
        dt = (t_end - t_start) / (n_points - 1)

        # Compute mu(t) at time points
        mu_values = self.mu(t_points)  # Shape (n_points, M)

        # Trapezoidal integration
        weights = torch.ones(n_points, device=self.device) * dt
        weights[0] *= 0.5
        weights[-1] *= 0.5

        integrals = dt * (mu_values * weights.unsqueeze(-1)).sum(dim=0)  # Shape (M,)

        return integrals

    def upper_bound_in_interval(
        self,
        t_start: torch.Tensor | float,
        t_end: torch.Tensor | float,
        n_samples: int = 200,
        safety_multiplier: float = 1.2,
    ) -> float:
        """
        Compute upper bound of total intensity over time interval by sampling.
        Used for modified thinning algorithm during simulation when mu(t) is not monotonic.

        Args:
            t_start: Start of time interval. Can be scalar (same for all variates) or tensor of shape (M,)
            t_end: End of time interval. Can be scalar (same for all variates) or tensor of shape (M,)
            n_samples: Number of time samples to use for estimating maximum intensity
            safety_multiplier: Multiplier to apply to sampled maximum to ensure upper bound

        Returns:
            Scalar upper bound on max_{t ∈ [t_start, t_end]} λ(t) where λ(t) = sum over all variates.
        """

        # Convert to scalars
        if isinstance(t_start, torch.Tensor):
            t_start = t_start.min().item()
        if isinstance(t_end, torch.Tensor):
            t_end = t_end.max().item()

        if t_end <= t_start:
            return 0.0

        # Sample densely and find maximum - mu(t) handles variate masking
        t_samples = torch.linspace(t_start, t_end, n_samples, device=self.device)

        with torch.no_grad():
            intensities = self.mu(t_samples)  # Shape: (n_samples, M)
            total_intensities = intensities.sum(dim=1)  # Sum over variates
            max_intensity = total_intensities.max().item()

        # Add 2% safety margin to handle sampling-based underestimation
        return max_intensity * safety_multiplier

    @abstractmethod
    def report_parameters(self) -> str:
        """
        Report the current parameter state of the model for logging purposes.

        Returns:
            String representation of parameters suitable for logging/printing.
        """
        pass

    @abstractmethod
    def get_save_data(self) -> dict:
        """
        Get model-specific data for saving to file.

        Returns:
            Dictionary containing model-specific data for serialization
        """
        pass

    def simulate(self, max_events: int = 100) -> utils.EventSequence:
        """
        Simulate events from the Poisson process using modified thinning algorithm.

        Uses the modified thinning algorithm to generate event times and types. The algorithm
        employs upper_bound_in_interval to handle non-monotonic intensity functions mu(t)
        by finding an appropriate upper bound on the total intensity over time intervals.

        The simulation runs from the earliest variate start time (min(t_start)) to the
        latest variate end time (max(t_end)). Only active variates participate in
        event generation at each time point.

        Args:
            max_events: Maximum number of events to generate (default: 100)

        Returns:
            ti: Event times tensor of shape (N,) where N is number of generated events
            mi: Event types tensor of shape (N,) with integer type indices in [0, M-1]

        Note:
            The algorithm uses rejection sampling with exponential inter-arrival times
            based on the upper bound, then accepts/rejects based on the actual intensity ratio.
            If no events are generated, returns empty tensors with appropriate shapes.
        """

        sim_start, sim_end = self._simulation_bounds
        t = sim_start
        events_list = []
        types_list = []

        with torch.no_grad():
            while t < sim_end and len(events_list) < max_events:
                # Use upper bound function to get upper bound on intensity
                λ_star = self.upper_bound_in_interval(t, sim_end)
                if λ_star <= 0:
                    break  # No active variates, simulation ends

                τ = torch.distributions.Exponential(λ_star).sample().to(self.device)
                t = t + τ.item()

                if t > sim_end:
                    break

                # Compute actual intensity at time t
                λ_t_new = self.mu(torch.tensor([t], device=self.device))
                λ_t_sum = λ_t_new.sum()

                r = λ_t_sum / λ_star
                if r > 1.0 + 1e-4:  # Increased tolerance for numerical errors
                    raise ValueError(
                        f"Intensity ratio {r.item():0.6f} exceeds theoretical maximum. Check upper_bound_in_interval implementation."
                    )

                if torch.rand(1).item() <= r.item():
                    p = λ_t_new.squeeze() / λ_t_sum
                    # Use probabilities directly for better numerical stability
                    event_type = torch.distributions.Categorical(probs=p).sample()
                    events_list.append(t)
                    types_list.append(event_type.item())

        # Convert lists to tensors at the end for better performance
        if events_list:
            ti = torch.tensor(events_list, device=self.device)
            mi = torch.tensor(types_list, dtype=torch.long, device=self.device)
        else:
            self.logger.warning("No events were generated.")
            ti = torch.zeros(0, device=self.device)
            mi = torch.zeros(0, dtype=torch.long, device=self.device)

        return utils.EventSequence(ti, mi, T=sim_end, M=self.M)

    def fit(
        self,
        seq: utils.EventSequence,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
        fit_config=config.FitConfig(),
    ) -> list[float]:
        """
        Fit the Poisson process parameters using Maximum Likelihood Estimation.

        Args:
            seq: EventSequence containing event times and types
            t_start: Start time for integration. If None, uses minimum time in seq.ti
            t_end: End time for integration. If None, uses maximum time in seq.ti
            fit_config: Configuration object containing fit parameters

        Returns:
            List of training losses at each optimization step
        """

        # Log training configuration
        self.logger.info(
            f"Starting Poisson process training: {self.__class__.__name__}"
        )
        self.logger.info(
            f"Configuration: M={self.M}, N={seq.N:,}, steps={fit_config.num_steps}"
        )
        self.logger.info(f"Device: {self.device}, model params: {self.num_params:,}")

        # Log learning rate
        self.logger.info(f"Parameters: lr={fit_config.learning_rate}")
        optimizer = optim.Adam(self.parameters(), lr=fit_config.learning_rate)
        losses = []

        self.logger.info("Starting training loop...")

        for epoch in range(fit_config.num_steps):
            optimizer.zero_grad()

            # Use simple batching for Poisson process
            loss = self.nll(seq, t_start, t_end)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Progress logging at monitor intervals
            if (epoch + 1) % fit_config.monitor_interval == 0:
                log_message = (
                    f"Epoch {epoch + 1}/{fit_config.num_steps}: Loss={loss.item():.4f}"
                )
                self.logger.info(log_message)

        self.logger.info("Training completed successfully!")

        return losses

    def nll(
        self,
        seq: utils.EventSequence,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood for Poisson process.

        Args:
            seq: EventSequence containing event times and types
            t_start: Start time for integration. If None, uses minimum time in seq.ti
            t_end: End time for integration. If None, uses maximum time in seq.ti

        Returns:
            Negative log-likelihood normalized by number of events
        """

        # Set default values for t_start and t_end if None
        if t_start is None:
            t_start = seq.ti.min().item()
        if t_end is None:
            t_end = seq.ti.max().item()

        # Validate time bounds
        if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
            if t_start >= t_end:
                raise ValueError(
                    f"t_start ({t_start}) must be less than t_end ({t_end})"
                )

        # Compute intensities at event times
        intensities = self.mu(seq.ti)  # Shape: [N, M]

        # Select intensity for the specific event type at each time
        batch_indices = torch.arange(len(seq.ti), device=seq.ti.device)
        event_intensities = intensities[batch_indices, seq.mi]

        # Add numerical stability
        epsilon = 1e-10
        if (event_intensities <= 0).any():
            self.logger.warning(
                "Zero or negative intensities detected, adding numerical stability term"
            )

        nll_events = -torch.sum(torch.log(event_intensities + epsilon))
        nll_int = self.integral_mu(t_start, t_end)  # Shape: (M,)

        # Correct mathematical form: add integral term and sum over variates
        return (nll_events + nll_int.sum()) / seq.N

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
