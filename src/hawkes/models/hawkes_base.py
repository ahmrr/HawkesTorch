import math
import time
import torch
import typing
import logging
from abc import ABC, abstractmethod

from .. import utils
from ..utils import config, _torch_scan
from .hawkes_nll_function import HawkesLogSumIntensity, HawkesIntegratedExcitation
from .poisson_base import PoissonBase


class HawkesBase(torch.nn.Module, ABC):
    """
    Abstract base class defining the API and shared utilities for a
    multivariate Hawkes process implementation.

    Concrete subclasses must provide properties for mu, alpha, and gamma
    (with both getters and setters) and can reuse the provided simulation,
    fitting, and intensity utility methods.
    """

    def __init__(
        self,
        K: int,
        gamma_param: bool,
        base_process: PoissonBase,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        """
        Args:
            K: Number of exponential kernels per pair (memory components)
            gamma_param: Whether the decay rates are learnable parameters
            base_process: Base Poisson process model
            device: Torch device (e.g., "cpu" or "cuda")
            runtime_config: Runtime configuration for debugging and profiling
        """

        super().__init__()
        self.K = K
        self.M = base_process.M
        self.gamma_param = gamma_param
        self.base_process = base_process

        self.device = torch.device(device or base_process.device)
        self.runtime_config = runtime_config
        self.logger = logging.getLogger(__name__)

        if runtime_config.deterministic_sim:
            # Seed RNGs for reproducible simulations when requested
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        if runtime_config.prefix_scan_implementation is not None:
            _torch_scan.PREFIX_SCAN_IMPLEMENTATION = (
                runtime_config.prefix_scan_implementation
            )

    def state_dict(self, *args, **kwargs):
        """Include the current computed alpha in the saved state dictionary."""

        # TODO: update

        state_dict = super().state_dict(*args, **kwargs)
        # alpha may be a derived tensor in some subclasses; store its current value
        with torch.no_grad():
            state_dict["mu"] = self.mu.detach()
            state_dict["alpha"] = self.alpha.detach()
            state_dict["gamma"] = self.gamma.detach()
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load model parameters while ignoring stored 'alpha' (derived)."""

        # TODO: update

        # Copy to avoid mutating the caller's dict
        state_dict = state_dict.copy()
        # alpha is derived from parameters — drop it before loading
        state_dict.pop("alpha", None)
        super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def simulation_bounds(self):
        """Get the simulation start and end times from the base process."""

        return self.base_process.simulation_bounds

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        """Time-varying base intensity vector of shape [N, M]."""

        return self.base_process.mu(t)

    def mu_at_events(self, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Base intensity values at event times, shape [N]."""

        mu = self.base_process.mu(t)  # Shape [N, M]
        return mu.gather(index=m.unsqueeze(1), dim=1).squeeze(1)

    @property
    @abstractmethod
    def alpha(self) -> torch.Tensor:
        """Excitation tensor of shape [K, M, M]."""
        pass

    @alpha.setter
    @abstractmethod
    def alpha(self, value: torch.Tensor | float):
        pass

    @property
    @abstractmethod
    def gamma(self) -> torch.Tensor:
        """Decay memory vector of length K (shape [K])."""
        pass

    @gamma.setter
    @abstractmethod
    def gamma(self, value: torch.Tensor):
        pass

    def simulate(self, max_events: int = 1000):
        """
        Simulate events from the Hawkes process using modified thinning algorithm. Supports time-varying base rates.
        """

        sim_start, sim_end = self.simulation_bounds
        t = sim_start
        if isinstance(t, torch.Tensor):
            t = t.item()

        ti = torch.zeros(0, device=self.device)
        mi = torch.zeros(0, dtype=torch.long, device=self.device)

        # R tracks the excitation state at the current time
        R = torch.zeros(self.K, self.M, device=self.device)
        event_count = 0

        self.logger.info(f"Starting simulation: t_start={sim_start}, t_end={sim_end}")

        with torch.no_grad():
            while t < sim_end and ti.shape[0] < max_events:
                # Get current excitation intensity from state R
                current_excitation = torch.sum(self.gamma.unsqueeze(-1) * R).item()

                # Use a small lookahead interval for upper bound calculation
                lookahead = min(1.0, sim_end - t)
                base_upper_bound = self.base_process.upper_bound_in_interval(
                    t, t + lookahead
                )
                λ_star = base_upper_bound + current_excitation

                τ = torch.distributions.Exponential(λ_star).sample()
                t_new = t + τ.item()

                if t_new > sim_end:
                    break

                # Decay the state from t to t_new
                decay_factor = torch.exp(-self.gamma * (t_new - t)).unsqueeze(-1)
                R = R * decay_factor

                # Calculate intensity at the new time
                base_intensity = self.base_process.mu(
                    torch.tensor([t_new], device=self.device)
                ).squeeze(0)

                # Calculate excitation and apply active mask
                excitation = torch.sum(self.gamma.unsqueeze(-1) * R, dim=0)
                active_mask = self.base_process._active_mask(t_new)
                excitation = active_mask * excitation

                λ_t = base_intensity + excitation
                λ_star_new = λ_t.sum()

                r = λ_star_new / λ_star
                if r > 1.01:
                    t = t_new  # Update time even if rejected
                    continue

                if torch.rand(1, device=self.device) <= r:
                    # Accept the event
                    p = λ_t / λ_star_new
                    event_type = torch.distributions.Categorical(
                        logits=torch.log(p)
                    ).sample()

                    # Add event to history
                    mi = torch.cat([mi, event_type.reshape(-1)])
                    ti = torch.cat([ti, torch.tensor([t_new], device=self.device)])

                    # Add the excitation jump from this event to the state
                    R = (
                        R + self.alpha[:, event_type, :]
                    )  # Add excitation from new event

                    event_count += 1

                # Update time
                t = t_new

        return utils.EventSequence(
            ti, mi, T=sim_end if sim_end != float("inf") else None, M=self.M
        )

    def fit(
        self,
        seq: utils.EventSequence,
        fit_config=config.FitConfig(),
    ) -> list[float]:
        """
        Fit model parameters by maximizing likelihood (minimizing NLL).

        Returns:
            List of scalar loss values (training loss per epoch).
        """

        assert seq.M <= self.M, f"sequence M is {seq.M} but model M is {self.M}"

        if self.runtime_config.detect_anomalies:
            torch.autograd.set_detect_anomaly(True)

        # Logging basic training configuration and model size
        self.logger.info(
            f"Starting Hawkes process training: {self.__class__.__name__} with base process {self.base_process.__class__.__name__}"
        )
        self.logger.info(
            f"Configuration: M={self.M}, K={self.K}, N={seq.N:,}, T={seq.T:,.2f}, steps={fit_config.num_steps}"
            + (f", batch_size={fit_config.batch_size}" if fit_config.batch_size else "")
        )
        self.logger.info(
            f"Parameters: lr={fit_config.learning_rate}, l1={fit_config.l1_penalty}, nuc={fit_config.nuc_penalty}"
        )
        self.logger.info(f"Device: {self.device}, model params: {self.num_params:,}")

        optimizer = torch.optim.Adam(self.parameters(), lr=fit_config.learning_rate)
        losses = []
        epoch_times = []

        self.logger.info("Starting training loop...")

        if self.device.type != "cpu":
            torch.cuda.reset_peak_memory_stats()

        total_fit_time = time.perf_counter()

        for epoch in range(fit_config.num_steps):
            epoch_time_real = time.perf_counter()

            optimizer.zero_grad()

            # Compute negative log-likelihood
            nll = self.nll(seq, batch_size=fit_config.batch_size)

            # Optional L1 hinge penalty applied to small parameters
            if fit_config.l1_penalty > 0:
                if fit_config.l1_alpha_diag:
                    l1_alpha = torch.where(
                        self.alpha < fit_config.l1_hinge, self.alpha, 0
                    ).sum()
                else:
                    # Penalize off-diagonal alpha entries only
                    off_diagonal_mask = ~torch.eye(
                        self.M, dtype=torch.bool, device=self.device
                    )[None, ...]
                    l1_alpha = torch.where(
                        (self.alpha < fit_config.l1_hinge) & off_diagonal_mask,
                        self.alpha,
                        0,
                    ).sum()

                l1 = fit_config.l1_penalty * l1_alpha
            else:
                l1 = 0

            # Optional nuclear-norm penalty on alpha
            if fit_config.nuc_penalty > 0:
                nuclear_norm = (
                    fit_config.nuc_penalty
                    * torch.linalg.matrix_norm(self.alpha, ord="nuc", dim=(1, 2)).sum()
                )
            else:
                nuclear_norm = 0

            loss = nll + l1 + nuclear_norm
            loss.backward()

            # Ensure gradients are finite
            for n, p in self.named_parameters():
                if torch.isnan(p.grad).any():
                    self.logger.error(f"Gradient of {n} is nan at epoch {epoch + 1}")
                    self.logger.info(p.grad)
                    raise ValueError(f"Gradient of {n} is nan")

            optimizer.step()

            epoch_time_real = time.perf_counter() - epoch_time_real

            losses.append(loss.item())
            epoch_times.append(epoch_time_real)

            # Periodic logging with more diagnostics
            if (epoch + 1) % fit_config.monitor_interval == 0:
                with torch.no_grad():
                    full_nll = self.nll(seq, batch_size=fit_config.batch_size)
                    sparsity_factor = (
                        self.alpha.isclose(
                            torch.zeros_like(self.alpha), atol=0.03
                        ).sum()
                    ).sum() / self.alpha.numel()

                    self.logger.info(
                        f"Epoch {epoch + 1}/{fit_config.num_steps}: "
                        f"Loss={full_nll.item():.4f}, "
                        f"Sparsity={sparsity_factor:.3f}, "
                        f"{self.base_process.report_parameters()}, "
                        f"α_mean={self.alpha.mean().item():.4f}, "
                        f"γ={'[' + ', '.join(f'{g:.2g}' for g in self.gamma.tolist()) + ']'}, "
                    )

        total_fit_time = time.perf_counter() - total_fit_time

        if self.device.type != "cpu":
            peak_memory_usage = torch.cuda.max_memory_allocated()

        self.logger.info("Training completed successfully!")

        # Final summary statistics
        with torch.no_grad():
            final_sparsity = (
                self.alpha.isclose(torch.zeros_like(self.alpha), atol=0.03).sum()
            ).sum() / self.alpha.numel()
            loss_reduction = (losses[0] - losses[-1]) / losses[0]

        hours = int(total_fit_time // 3600)
        minutes = int((total_fit_time % 3600) // 60)
        seconds = int(total_fit_time % 60)

        self.logger.info(
            f"Training summary: Final loss={losses[-1]:.4f}, "
            f"Loss reduction={loss_reduction:.1%}, "
            f"Final sparsity={final_sparsity:.3f}"
        )
        self.logger.info(
            f"Performance summary: Fitting time={hours:02d}:{minutes:02d}:{seconds:02d}, "
            f"Average epoch time={sum(epoch_times) / len(epoch_times) * 1000:.3f}ms"
            + (
                f", Peak memory={round(peak_memory_usage / 2**20)}MiB"
                if self.device.type != "cpu"
                else ""
            )
        )

        return {
            "losses": losses,
            "epoch_times": epoch_times,
            "total_fit_time": total_fit_time,
            "peak_memory_usage": (
                peak_memory_usage if self.device.type != "cpu" else None
            ),
        }

    def nll(
        self,
        seq: utils.EventSequence,
        batch_size: int | None = None,
    ):
        """
        Computes the negative log-likelihood of the given event sequence.
        """

        batch_size = min(batch_size or seq.N, seq.N)

        mu_at_events = self.mu_at_events(seq.ti, seq.mi)

        # Compute log-sum intensity term

        nll_log_sum = HawkesLogSumIntensity.apply(
            seq,
            mu_at_events,
            self.alpha,
            self.gamma,
            self.gamma_param,
            self.intensity_states,
            self.intensity_at_events,
            batch_size,
        )

        # Compute integral of intensity in [0, T]

        base_rate_integral = self.base_process.integral_mu(0, seq.T).sum()
        excitation_integral = HawkesIntegratedExcitation.apply(
            seq,
            self.alpha,
            self.gamma,
            self.gamma_param,
            batch_size,
        )
        nll_integral = base_rate_integral + excitation_integral

        return -(nll_log_sum - nll_integral) / seq.N

    def intensity(
        self,
        t: torch.Tensor,
        seq: utils.EventSequence,
        states: torch.Tensor | None = None,
    ):
        """
        General function for computing the intensity.

        Args:
            seq: Event sequence
            t: Times at which to compute the intensity; if None, use event times
            states: Precomputed intensity states R; if None, compute them via intensity_states(...)

        Returns:
            Tensor shaped (len(t), M) giving intensities for all nodes at each t.
        """

        if states is None:
            states = self.intensity_states(seq)

        # Find index of the most recent event before each t
        idx = torch.searchsorted(seq.ti, t).to(self.device) - 1
        after_first_event = idx >= 0
        idx_ = idx[after_first_event]
        t_ = t[after_first_event]  # times that are after at least one event

        # Exponential decay for each time relative to its last event
        exp_decay = torch.exp(
            -self.gamma * (t_ - seq.ti[idx_]).unsqueeze(-1)
        )  # Shape: [t_.shape, K]

        # Contribution from stored R states (decayed) and the immediate alpha impulse
        state_sum = torch.sum(
            self.gamma * exp_decay * states[idx_].permute(1, 0, 2), dim=-1
        )  # Shape: [M, t_.shape]
        alpha_sum = torch.sum(
            self.alpha[:, seq.mi[idx_], :].permute(2, 1, 0) * self.gamma * exp_decay,
            dim=-1,
        )  # Shape: [M, t_.shape]

        t_intensity = (
            self.mu(t_).permute(1, 0) + state_sum + alpha_sum
        )  # Shape: [M, t_.shape]

        # Handle times before the first event
        baseline_intensity = self.mu(t[~after_first_event])  # Shape: [t.shape, M]

        # Prepend baseline intensities for times before the first event
        return torch.cat(
            [
                baseline_intensity,
                t_intensity.permute(1, 0),
            ],
            dim=0,
        )  # Shape: [t.shape, M]

    def intensity_states(
        self,
        seq: utils.EventSequence,
        bounds: tuple[int, int] | None = None,
        prev_state: torch.Tensor | None = None,
        next_state: bool = False,
        full_states: bool = True,
    ):
        """
        Compute the intensity states R at event times in seq between start and end indices.

        Args:
            seq: Event sequence
            bounds: Tuple of (start, end) indices of events to compute
            prev_state: Previous augmented state tensor of shape (K, M+1)
            next_state: If True, return the next state after the end index
            full_states: If True, return full states; otherwise return only states
                         corresponding to the event nodes

        Returns:
            States of shape (n, M, K) where n is the number of events in bounds,
            and optionally the augmented next state tensor of shape (K, M+1).
        """

        alpha = self.alpha
        start = bounds[0] if bounds else 0
        end = bounds[1] if bounds else seq.N

        if start != 0:
            # Inter-event times Δt_i = t_i - t_{i-1} with Δt_1 = 0
            dti = seq.ti[start:end] - seq.ti[(start - 1) : (end - 1)]  # Shape: [n]
            # Influences α_{:, m_{i-1}, k}
            alphas = alpha[:, seq.mi[(start - 1) : (end - 1)], :]  # Shape: [K, n, M]
        else:
            dti = seq.ti[0:end] - torch.cat([seq.ti[0:1], seq.ti[0 : (end - 1)]])
            alphas = torch.cat(
                [
                    torch.zeros_like(alpha[:, 0:1, :]),
                    alpha[:, seq.mi[0 : (end - 1)], :],
                ],
                dim=1,
            )

        dti = dti[None, :, None]  # Shape: [1, n, 1]
        exp_dti = torch.exp(-self.gamma * dti).permute(2, 1, 0)  # Shape: [K, n, 1]

        # Build transition matrices
        M = torch.cat([exp_dti, exp_dti * alphas], dim=2)  # Shape: [K, n, M+1]
        if prev_state is not None:
            M[:, 0:1, :] = _torch_scan.state_left_mult(
                prev_state[:, None, :], M[:, 0:1, :]
            )

        # Compute prefix products P via scan
        P = _torch_scan.prefix_scan(M, _torch_scan.state_left_mult, dim=1)

        # Extract non-augmented states
        R = P[:, :, 1:].permute(1, 2, 0)  # Shape: [n, M, K]

        if not full_states:
            R = torch.gather(
                R, dim=1, index=seq.mi[start:end, None, None].expand(-1, 1, self.K)
            )

        return (R, P[:, -1, :]) if next_state else R

    def intensity_at_events(
        self,
        seq: utils.EventSequence,
        states: torch.Tensor,
        full_intensity: bool = True,
    ):
        """
        Compute the intensity at event times in seq using precomputed states R.

        Args:
            seq: Event sequence
            states: Intensity states R computed via intensity_states(...)
            full_intensity: If True, return full M-dimensional intensity at each event;
                            otherwise return only the intensity of the node that
                            experienced the event.

        Returns:
            Tensor of shape (N, M) or (N,) giving intensities at each event time.
        """

        mu = self.mu(seq.ti)  # Shape: [N]
        excitation = torch.sum(self.gamma * states, dim=2)  # Shape: [N, M]

        intensity = mu + excitation

        if not full_intensity:
            intensity = intensity.gather(dim=1, index=seq.mi.unsqueeze(1)).squeeze(1)

        return intensity

    def rescaled_times(
        self, seq: utils.EventSequence, states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the rescaled inter-event times using the time-rescaling theorem.

        Args:
            seq: Event sequence
            states: Intensity states R computed via intensity_states(...)

        Returns:
            Tensor of shape (N-1, M) containing the rescaled inter-event times.
        """

        # R_{i}^k
        states = states.permute(2, 0, 1)  # Shape: [K, N, M]

        # Δt_{i+1} = t_{i+1} - t_{i}
        dti = seq.ti[1:] - seq.ti[:-1]  # Shape: [N-1]

        dti = dti[None, :, None]  # Shape: [1, N-1, 1]
        exp_decay = 1 - torch.exp(-self.gamma * dti)
        exp_decay = exp_decay.permute(2, 1, 0)  # Shape: [K, N-1, 1]

        # α_{:, m_i, k}
        alpha_mi = self.alpha[:, seq.mi[:-1], :]  # Shape: [K, N-1, M]

        excitation_integrals = torch.sum(
            exp_decay * (states[:, :-1, :] + alpha_mi), dim=0
        )  # Shape: [N-1, M]

        # TODO: Implement batched base rate integrals
        base_integrals = []
        for i in range(seq.N - 1):
            base_integral = self.base_process.integral_mu(
                seq.ti[i].item(), seq.ti[i + 1].item()
            )  # Shape: [M]
            base_integrals.append(base_integral)
        base_integrals = torch.stack(base_integrals, dim=0)  # Shape: [N-1, M]

        return base_integrals + excitation_integrals

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
