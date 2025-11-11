import math
import time
import torch
import typing
import logging
from abc import ABC, abstractmethod

from .. import utils
from ..utils import config, _torch_scan
from .hawkes_nll_function import HawkesLogSumIntensity, HawkesIntegratedExcitation


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
        M: int,
        K: int,
        device="cpu",
        debug_config=config.HawkesDebugConfig(),
    ):
        """
        Args:
            M: Number of nodes (processes)
            K: Number of exponential kernels per pair (memory components)
            device: Torch device (e.g., "cpu" or "cuda")
            debug_config: Debugging and profiling configuration
        """

        super().__init__()
        self.M = M
        self.K = K

        self.device = device
        self.debug_config = debug_config
        self.logger = logging.getLogger(__name__)

        if debug_config.deterministic_sim:
            # Seed RNGs for reproducible simulations when requested
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        # Print parameters and their device for quick sanity check
        for param in self.parameters():
            print(param, param.device)

    def state_dict(self, *args, **kwargs):
        """Include the current computed alpha in the saved state dictionary."""

        state_dict = super().state_dict(*args, **kwargs)
        # alpha may be a derived tensor in some subclasses; store its current value
        with torch.no_grad():
            state_dict["mu"] = self.mu.detach()
            state_dict["alpha"] = self.alpha.detach()
            state_dict["gamma"] = self.gamma.detach()
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load model parameters while ignoring stored 'alpha' (derived)."""

        # Copy to avoid mutating the caller's dict
        state_dict = state_dict.copy()
        # alpha is derived from parameters — drop it before loading
        state_dict.pop("alpha", None)
        super().load_state_dict(state_dict, *args, **kwargs)

    @property
    @abstractmethod
    def mu(self) -> torch.Tensor:
        """Base intensity vector of shape [M]."""
        pass

    @mu.setter
    @abstractmethod
    def mu(self, value: torch.Tensor | float):
        pass

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

    def simulate(
        self,
        T: float,
        max_events=1000,
    ) -> utils.EventSequence:
        """
        Simulate an event sequence using Ogata's thinning algorithm.

        Terminates when an event time exceeds T or when max_events is reached.

        Returns:
            ti: Tensor of event times with shape (N,)
            mi: Tensor of event types with shape (N,)
        """

        t = 0.0
        ti = torch.zeros(0, device=self.device)
        mi = torch.zeros(0, dtype=torch.long, device=self.device)

        R_λ = None  # right-limit state (R) after the most recent accepted event

        # self.logger.info("Simulating data")

        with torch.no_grad():
            while t < T and ti.shape[0] < max_events:
                # Upper-bound intensity (sum across nodes) at current state
                λ_star = self.intensity_at_next_event(t, None, ti, mi, R_λ).sum()

                # Sample candidate inter-arrival time
                τ = torch.distributions.Exponential(λ_star).sample()
                t = t + τ.item()

                if t > T:
                    break

                # Compute intensity at candidate time and proposed new state
                λ_t, R = self.intensity_at_next_event(
                    t, None, ti, mi, R_λ, return_next_state=True
                )
                λ_star_new = λ_t.sum()

                # Acceptance probability r = λ(t) / λ*
                r = λ_star_new / λ_star
                assert (
                    r <= 1
                ), f"λ_t.sum() / λ_star must be less than or equal to 1. Got {r.item():0.3f}"

                if torch.rand(1, device=self.device) <= r:
                    # Choose event type proportional to λ_t
                    p = λ_t / λ_star_new
                    # use logits to avoid floating point rounding issues
                    event_type = (
                        torch.distributions.Categorical(logits=torch.log(p))
                        .sample()
                        .unsqueeze(0)
                    )
                    mi = torch.cat([mi, event_type])
                    ti = torch.cat([ti, torch.tensor([t], device=self.device)])
                    R_λ = R  # update right-limit state

        return utils.EventSequence(ti, mi, T, self.M)

    def fit(
        self,
        seq: utils.EventSequence,
        fit_config=config.HawkesFitConfig(),
    ) -> list[float]:
        """
        Fit model parameters by maximizing likelihood (minimizing NLL).

        Returns:
            List of scalar loss values (training loss per epoch).
        """

        assert seq.M <= self.M, f"sequence M is {seq.M} but model M is {self.M}"

        if self.debug_config.detect_anomalies:
            torch.autograd.set_detect_anomaly(True)

        # Logging basic training configuration and model size
        self.logger.info(f"Starting Hawkes process training: {self.__class__.__name__}")
        self.logger.info(
            f"Configuration: M={self.M}, K={self.K}, N={seq.N:,}, T={seq.T}, steps={fit_config.num_steps}"
            + (f", batch_size={fit_config.batch_size}" if fit_config.batch_size else "")
        )
        self.logger.info(
            f"Parameters: lr={fit_config.learning_rate}, l1={fit_config.l1_penalty}, nuc={fit_config.nuc_penalty}"
        )
        self.logger.info(f"Device: {self.device}, model params: {self.num_params:,}")

        optimizer = torch.optim.Adam(self.parameters(), lr=fit_config.learning_rate)
        losses = []
        epoch_times = []

        if self.debug_config.profile_mem_iters:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self.logger.info(
                f"Profiling {self.debug_config.profile_mem_iters} training iterations (up to {self.debug_config.profile_mem_entries} entries)"
            )

        self.logger.info("Starting training loop...")

        if self.device != "cpu":
            torch.cuda.reset_peak_memory_stats()

        fit_time_real = time.perf_counter()

        for epoch in range(fit_config.num_steps):
            epoch_time_real = time.perf_counter()

            optimizer.zero_grad()

            # Compute NLL (forward + backward handled by custom Function)
            nll = (
                self.nll(seq, batch_size=fit_config.batch_size)
                if not self.debug_config.use_autograd_gradients
                else self._compute_nll(seq, compute_backward=True)
            )

            # Optional L1 hinge penalty applied to small parameters
            if fit_config.l1_penalty > 0:
                l1_mu = torch.where(self.mu < fit_config.l1_hinge, self.mu, 0).sum()
                l1_alpha = torch.where(
                    self.alpha < fit_config.l1_hinge, self.alpha, 0
                ).sum()
                l1 = fit_config.l1_penalty * (l1_mu + l1_alpha)
            else:
                l1 = 0

            # Optional nuclear-norm penalty over alpha (sum across appropriate dims)
            if fit_config.nuc_penalty > 0:
                nuclear_norm = (
                    fit_config.nuc_penalty
                    * torch.linalg.matrix_norm(self.alpha, ord="nuc", dim=(1, 2)).sum()
                )
            else:
                nuclear_norm = 0

            loss = nll + l1 + nuclear_norm
            if self.debug_config.use_autograd_gradients:
                (l1 + nuclear_norm).backward()
            else:
                # Backpropagate (retain_graph when debugging gradient comparisons)
                loss.backward(retain_graph=self.debug_config.check_grad_epsilon)

            # Ensure gradients are finite
            for n, p in self.named_parameters():
                if torch.isnan(p.grad).any():
                    self.logger.error(f"Gradient of {n} is nan at epoch {epoch + 1}")
                    self.logger.info(p.grad)
                    raise ValueError(f"Gradient of {n} is nan")

            # Optionally compute autograd gradients for cross-checking
            if self.debug_config.check_grad_epsilon:
                param_grads = torch.autograd.grad(
                    (self._compute_nll(seq) + l1 + nuclear_norm),
                    self.parameters(),
                )

            optimizer.step()

            epoch_time_real = time.perf_counter() - epoch_time_real

            losses.append(loss.item())
            epoch_times.append(epoch_time_real)

            if epoch >= self.debug_config.profile_mem_iters and self.device != "cpu":
                torch.cuda.memory._record_memory_history(enabled=None)

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
                        f"μ_mean={self.mu.mean().item():.4f}, "
                        f"α_mean={self.alpha.mean().item():.4f}, "
                        f"γ_mean={self.gamma.mean().item():.4f}"
                    )

                # Checking manual gradients against autograd
                if self.debug_config.check_grad_epsilon:
                    for (n, p), grad_auto in zip(self.named_parameters(), param_grads):
                        abs_diff = torch.abs(p.grad - grad_auto)
                        rel_diff = abs_diff / ((p.grad + grad_auto) / 2)
                        self.logger.info(
                            f" >- Absolute gradient diff for {n}: "
                            f"min={float(abs_diff.min()):.2e}, "
                            f"max={float(abs_diff.max()):.2e}, "
                            f"avg={float(abs_diff.mean()):.2e}"
                        )
                        self.logger.info(
                            f" >- Relative gradient diff for {n}: "
                            f"min={float(rel_diff.abs().min()):.2e}, "
                            f"max={float(rel_diff.abs().max()):.2e}, "
                            f"avg={float(rel_diff.abs().mean()):.2e}"
                        )

        peak_mem_gpu = torch.cuda.max_memory_allocated()
        fit_time_real = time.perf_counter() - fit_time_real

        if self.debug_config.profile_mem_iters and self.device != "cpu":
            mem_file = f"outputs/mem/mem_snapshot_n{seq.N}_m{self.M}_b{fit_config.batch_size}_e{self.debug_config.profile_mem_iters}.pkl"
            torch.cuda.memory._dump_snapshot(mem_file)
            self.logger.info(f"Saved memory profiling snapshot at {mem_file}")

        self.logger.info("Training completed successfully!")

        # Final summary statistics
        with torch.no_grad():
            final_sparsity = (
                self.alpha.isclose(torch.zeros_like(self.alpha), atol=0.03).sum()
            ).sum() / self.alpha.numel()
            loss_reduction = (
                ((losses[0] - losses[-1]) / losses[0] * 100) if len(losses) > 1 else 0
            )

        self.logger.info(
            f"Training summary: Final loss={losses[-1]:.4f}, "
            f"Loss reduction={loss_reduction:.1f}%, "
            f"Final sparsity={final_sparsity:.3f}"
        )
        self.logger.info(
            f"Performance summary: Fitting time={time.strftime('%H:%M:%S', time.gmtime(fit_time_real))}, "
            f"Average epoch time={sum(epoch_times) / len(epoch_times) * 1000:.3f}ms, "
            f"Peak memory={round(peak_mem_gpu / 2**20)}MiB"
        )

        return {
            "losses": losses,
            "epoch_times": epoch_times,
            "peak_mem_gpu": peak_mem_gpu,
            "fit_time_real": fit_time_real,
        }

    def nll(
        self,
        seq: utils.EventSequence,
        batch_size: int | None = None,
    ):
        gamma_param = any([("gamma" in n) for n, _ in self.named_parameters()])
        batch_size = min(batch_size or seq.N, seq.N)

        # TODO: Support arbitrary mu parameterization by passing in base rate at each event
        mu_at_events = self.mu[seq.mi]

        # TODO: Incorporate and support batching
        nll_logsum_term = HawkesLogSumIntensity.apply(
            seq,
            mu_at_events,  # \mu_{m_i}(t_i)
            self.alpha,  # \alpha_{p,q,k}
            self.gamma,  # \gamma_k
            gamma_param,
            self.intensity_at_events,
            batch_size,
        )

        nll_integral_term = self.integrated_intensity(seq, batch_size=batch_size)

        return -(nll_logsum_term - nll_integral_term) / seq.N

    def _compute_nll(
        self,
        seq: utils.EventSequence,
        batch_size=None,
        compute_backward=False,
    ) -> float:
        """
        Compute the negative log-likelihood across all events, optionally
        performing the backward pass batch-wise.

        Args:
            T: End time of observation period
            ti: Tensor of event times with shape (N,)
            mi: Tensor of event types with shape (N,)
            batch_size: If provided, compute terms in batches of this many events
            compute_backward: If True, call backward() on each batch's contribution
                              to compute gradients in a batched way

        Returns:
            Scalar NLL (sum of -log-intensity at events and integrated intensity).
        """

        batch_size = min(batch_size or seq.N, seq.N)

        nll_neg_logsum = 0.0
        nll_integral = 0.0

        prev_state = None
        for bs in range(0, seq.N, batch_size):
            be = min(bs + batch_size, seq.N)
            Nb = be - bs

            batch_integrated_intensity = (
                self.integrated_intensity(seq, batch_size=Nb) / seq.N
            )
            if compute_backward:
                batch_integrated_intensity.backward()

            # Compute batch intensities; prev_state ensures continuity across batches
            batch_intensity, prev_state = self.intensity_at_events(
                seq,
                return_full_intensity=False,
                return_last_state=True,
                batch_prev_state=prev_state,
                batch_start=bs,
                batch_end=be,
            )
            batch_neg_logsum = -torch.sum(torch.log(batch_intensity)) / seq.N
            if compute_backward:
                batch_neg_logsum.backward()

            nll_neg_logsum += batch_neg_logsum
            nll_integral += batch_integrated_intensity

        return nll_neg_logsum + nll_integral

    def integrated_base_rate(self, seq: utils.EventSequence) -> torch.Tensor:
        """
        Compute the integrated base rate over the observation interval [0, T].
        """

        # TODO: generalize to arbitrary base rate parametrization
        return seq.T * torch.sum(self.mu)

    def integrated_excitation(
        self,
        seq: utils.EventSequence,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Wrapper for computing the integrated excitation over the observation interval [0, T].
        """

        gamma_param = any([("gamma" in n) for n, _ in self.named_parameters()])
        batch_size = min(batch_size or seq.N, seq.N)

        return HawkesIntegratedExcitation.apply(
            seq,
            self.alpha,
            self.gamma,
            gamma_param,
            batch_size,
        )

    def integrated_intensity(
        self,
        seq: utils.EventSequence,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Compute the total integrated intensity over the observation interval [0, T].
        """

        batch_size = min(batch_size or seq.N, seq.N)

        base_rate_integral = self.integrated_base_rate(seq)
        excitation_integral = self.integrated_excitation(seq, batch_size=batch_size)

        return base_rate_integral + excitation_integral

    def intensity_at_next_event(
        self,
        t: float,
        m: int | None,
        ti: torch.Tensor,
        mi: torch.Tensor,
        prev_state: torch.Tensor = None,
        return_next_state=False,
    ) -> torch.Tensor:
        """
        Compute intensity at a candidate time t (and optionally the next state R)
        using a recurrence relation that updates the KxM state matrix.

        This function is intended for sequential simulation or CPU-friendly
        sequential computations; for batched GPU computations use intensity_at_events.

        Args:
            t: Candidate event time
            m: Candidate event type (unused for intensity computation here but kept
               for API symmetry)
            ti: Tensor of previous event times (1, N, 1)
            mi: Tensor of previous event types (N,)
            prev_state: Right-limit state R from the previous accepted event
            return_next_state: If True, return both (λ, R) where R is the new right-limit

        Returns:
            If return_next_state is False: λ (1, M) vector of intensities at time t
            If return_next_state is True: tuple (λ (1, M), R (K, M)) where R is next state
        """

        # TODO: Fix calling convention

        N = ti.shape[0]

        if prev_state is None:
            prev_state = torch.zeros(self.K, self.M, device=self.device)

        # time since last event (0 if no prior events)
        dti = t - ti[-1] if N != 0 else 0

        # alphas: contributions from the most recent event types, or zeros if none
        alphas = (
            self.alpha[:, mi[-1], :]
            if N != 0
            else torch.zeros(self.K, self.M, device=self.device)
        )  # Shape: [K, M]

        exp_decay = torch.exp(-self.gamma[:, None] * dti)  # Shape: [K, 1]

        # R = γ-decayed previous state + γ-decayed new impulse from last event
        R = exp_decay * prev_state + exp_decay * alphas  # Shape: [K, M]

        λ = self.mu + torch.sum(self.gamma[:, None] * R, dim=0)

        if return_next_state:
            return λ, R  # λ shape: [1, M], R shape: [K, M]
        else:
            return λ

    def intensity_at_events(
        self,
        seq: utils.EventSequence | tuple[float, torch.Tensor, torch.Tensor],
        batch_prev_state: torch.Tensor | None = None,
        batch_start: int | None = None,
        batch_end: int | None = None,
        return_full_intensity=True,
        return_last_state=False,
        return_all_states=False,
    ) -> torch.Tensor:
        """
        Compute intensities at the sequence of event times ti using a state-augmented
        matrix recurrence and prefix scan. This is the preferred fast implementation
        for GPU/batched evaluation of the intensity terms used in NLL.

        Args:
            seq: Either an EventSequence object or tuple of (T, ti, mi)
            full_intensity: If True, return the full M-dimensional intensity at
                            each event; otherwise return only the intensity of the
                            node that actually experienced the event at that time.
            batch_size: Number of events to use in each batch, or None to disable batchin

        Returns:
            Depending on flags, returns λ (Nb x M) or (Nb x 1) plus optionally
            the last prefix matrix (for batching) and the R states.
        """

        if not isinstance(seq, utils.EventSequence):
            seq = utils.EventSequence(seq[1], seq[2], T=seq[0], M=self.M)

        # To prevent multiple computations of alpha matrix
        alpha = self.alpha

        bs = batch_start or 0
        be = batch_end or seq.N
        Nb = be - bs

        # Δt_i of shape [1, Nb, 1]
        dti = seq.ti[bs:be] - (
            torch.cat([seq.ti[0:1], seq.ti[0 : (be - 1)]])
            if bs == 0
            else seq.ti[(bs - 1) : (be - 1)]
        )
        dti = dti[None, :, None]

        # α_{m_{i-1}} of shape [K, Nb, M]
        alpha_mi = (
            torch.cat(
                [
                    torch.zeros_like(alpha[:, 0:1, :]),
                    alpha[:, seq.mi[0 : (be - 1)], :],
                ],
                dim=1,
            )
            if bs == 0
            else alpha[:, seq.mi[(bs - 1) : (be - 1)], :]
        )

        # e^{-γ_k Δt_i}
        exp_dti = torch.exp(-self.gamma * dti).permute(2, 1, 0)  # Shape: [K, Nb, 1]

        # Build transition matrices for each time step: [exp_decay, exp_decay * alphas]
        M = torch.cat([exp_dti, exp_dti * alpha_mi], dim=2)  # Shape: [K, Nb, M+1]
        # if batch_prev_state is not None:
        #     M = torch.cat([batch_prev_state[:, None, :], M], dim=1)
        if batch_prev_state is not None:
            M[:, 0:1, :] = _torch_scan.state_left_mult(
                batch_prev_state[:, None, :], M[:, 0:1, :]
            )  # Shape: [K, Nb+1, M+1]

        # Compute prefix products P via scan; P contains augmented states
        # Shape: [K, Nb, M+1] (first batch) or [K, Nb+1, M+1] (regular batch)
        P = _torch_scan.prefix_scan(M, _torch_scan.state_left_mult, dim=1)
        # if batch_prev_state is not None:
        #     P = _torch_scan.state_left_mult(
        #         batch_prev_state[:, None, :], P
        #     )  # Shape: [K, Nb, M+1]

        # Extract non-augmented right-limit states R for the last Nb prefixes
        states = P[:, -Nb:, 1:].permute(1, 2, 0)  # Shape: [Nb, M, K]

        # Convert states to intensities: μ + sum_k γ_k * R[..., k]
        λ = self.mu[None, :] + torch.sum(self.gamma * states, dim=2)  # Shape: [Nb, M]

        # If only the intensity at the actual event node is required, pick it out
        if not return_full_intensity:
            λ = torch.gather(λ, dim=1, index=seq.mi[bs:be, None])  # Shape: [Nb, 1]
            λ = λ.squeeze(1)

        # Return assembled results according to requested flags
        res = (λ,)
        if return_all_states:
            res += (states,)
        if return_last_state:
            res += (P[:, -1, :],)  # last prefix for next batch continuity

        return res if len(res) > 1 else res[0]

    def intensity_at_t(
        self,
        t: torch.Tensor,
        seq: utils.EventSequence,
        intensity_states: torch.Tensor = None,
    ):
        """
        Compute the right-limit intensity at arbitrary times t (not necessarily events).

        If R (per-event states) is not provided, it is computed internally by
        calling intensity_at_events(..., return_states=True).

        Returns:
            Tensor shaped (len(t), M) giving intensities for all nodes at each t.
        """

        if intensity_states is None:
            with torch.no_grad():
                _, intensity_states = self.intensity_at_events(
                    seq, return_all_states=True
                )

        # Find index of the most recent event before each t
        idx = torch.searchsorted(seq.ti, t).to(seq.ti.device) - 1  # Shape: t.shape
        after_first_event = idx >= 0
        idx_ = idx[after_first_event]
        t_ = t[after_first_event]  # times that are after at least one event

        # exp_decay for each kernel at each queried time relative to its last event
        exp_decay = torch.exp(
            -self.gamma * (t_ - seq.ti[idx_]).unsqueeze(-1)
        )  # Shape: [t_.shape, K]

        # Contribution from stored R states (decayed) and the immediate alpha impulse
        state_sum = torch.sum(
            self.gamma * exp_decay * intensity_states[idx_].permute(1, 0, 2), dim=-1
        )  # Shape: [M, t_.shape]
        alpha_sum = torch.sum(
            self.alpha[:, seq.mi[idx_], :].permute(2, 1, 0) * self.gamma * exp_decay,
            dim=-1,
        )  # Shape: [M, t_.shape]

        t_intensity = (
            self.mu.unsqueeze(-1) + state_sum + alpha_sum
        )  # Shape: [M, t_.shape]

        # Prepend baseline intensities for times before the first event
        return torch.cat(
            [
                self.mu.unsqueeze(0).repeat(t.shape[0] - t_.shape[0], 1),
                t_intensity.permute(1, 0),
            ],
            dim=0,
        )  # Shape: [t.shape, M]

    def _intensity_reference_implementation(
        self,
        t: torch.Tensor | float,
        m: torch.Tensor | int | None,
        ti: torch.Tensor,
        mi: torch.Tensor,
        right_limit=False,
    ) -> torch.Tensor:
        """
        Straightforward (inefficient) reference implementation of the intensity formula.

        Args:
            t: scalar or 1-D tensor of query times
            m: If provided, select intensities only for these node indices per t
            ti: Event times tensor (1, N, 1)
            mi: Event types tensor (N,)
            right_limit: If True, include contributions from events that occur at t
        """

        if not isinstance(t, torch.Tensor):
            if not isinstance(t, float):
                raise ValueError(f"t must be a float or torch.Tensor but got {type(t)}")
            t = torch.tensor([t]).reshape(1, 1, 1).to(ti.device)

        if m is not None:
            if not isinstance(m, torch.Tensor):
                assert isinstance(m, int), "m must be an integer"
                m = torch.tensor([m]).repeat(t.shape[0]).to(ti.device)
            else:
                assert len(m.shape) == 1, "m must be a rank-1 tensor"
                assert (
                    m.shape[0] == t.shape[0]
                ), "m must have same number of elements as t"

        B = t.shape[0]
        N = ti.shape[0]

        # Compute pairwise time differences dt = t - ti with broadcasting
        dt = t - ti

        # For right-limit vs left-limit handling: remove contributions from
        # events at or after the query time as appropriate
        if right_limit:
            dt[dt < 0] = torch.inf
        else:
            dt[dt <= 0] = torch.inf

        # Add dimension for kernels
        dt = dt.unsqueeze(-1)  # Shape: [B, N, 1, 1]

        # Base intensity handling: choose mu elements per batch if m provided
        if m is not None:
            mus = self.mu[m].view(B, 1)  # Shape: [B, 1]
        else:
            mus = self.mu.expand(B, self.M)  # Shape: [B, M]

        # Prepare alpha contributions: select columns for each event type
        alpha_mi = self.alpha[:, mi]  # Shape: [K, N, M]
        alpha_mi = alpha_mi.permute(1, 2, 0)  # Shape: [N, M, K]

        if m is not None:
            # Compute excitation only for specified node indices per time
            batch_indices = torch.arange(B)
            alpha_batch = alpha_mi[:, m, :]  # Shape: [N, B, K]
            alpha_batch = alpha_batch.permute(1, 0, 2)  # Shape: [B, N, K]

            excitation = torch.sum(
                alpha_batch * self.gamma * torch.exp(-self.gamma * dt.squeeze(-2)),
                dim=(1, 2),
            )

            return mus + excitation.unsqueeze(1)
        else:
            # Compute excitation for all nodes using broadcasting
            alpha_mi = alpha_mi.unsqueeze(0)  # Shape: [1, N, M, K]

            # gamma_term shape: [B, N, 1, K]
            gamma_term = self.gamma * torch.exp(-self.gamma * dt)

            # Sum contributions over events and kernels
            excitation = torch.sum(
                alpha_mi * gamma_term, dim=(1, 3)
            )  # Sum over events and kernels

            return mus + excitation

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
