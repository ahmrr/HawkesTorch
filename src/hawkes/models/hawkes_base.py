import math
import time
import torch
import typing
import logging
from abc import ABC, abstractmethod

from ..utils import config, _torch_scan


class HawkesNLL(torch.autograd.Function):
    """
    Custom autograd Function that computes the negative log-likelihood (NLL)
    for a multivariate Hawkes process and provides a custom backward pass
    that yields gradients for the model parameters.

    This Function expects helper callables to compute the intensity at event
    times and the integrated intensity over the observation window. It is
    optimized to work with the prefix-scan state representation used by the
    Hawkes model implementation.
    """

    @staticmethod
    def forward(
        ctx: typing.Any,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        gamma_param: bool,
        intensity_at_events_fun: typing.Callable,
        integrated_intensity_fun: typing.Callable,
        M: int,
        K: int,
        T: float,
        ti: torch.Tensor,
        mi: torch.Tensor,
        batching=False,
        batch_start=0,
        batch_size=None,
    ):
        N = ti.shape[1]
        batch_size = batch_size or N

        # Accumulate the two terms of the NLL:
        # 1) negative log of intensities at events
        # 2) integrated intensity over [0, T]
        nll_neg_logsum = 0.0
        nll_int = 0.0

        # We'll compute intensity and (optionally) states in batches,
        # keeping the previous state's right-limit for continuity across batches.
        prev_state = None

        # Preallocate buffers for intensities and states (device same as ti)
        intensity = torch.empty(N, device=ti.device)
        states = torch.empty(N, M, K, device=ti.device)

        # TODO: batching
        # the same behavior while iterating in chunks.
        for batch_start in range(0, N, batch_size):
            Nb = min(batch_size, N - batch_start)

            # Add the integrated intensity for this batch
            nll_int += integrated_intensity_fun(
                T, ti, mi, batching=True, batch_start=batch_start, batch_size=Nb
            )

            # Compute intensities at event times for this batch, plus states
            (
                batch_intensity,
                prev_state,
                batch_states,
            ) = intensity_at_events_fun(
                ti,
                mi,
                full_intensity=False,
                batching=True,
                batch_start=batch_start,
                batch_size=Nb,
                batch_prev_state=prev_state,
                return_states=True,
            )

            # Store results in preallocated buffers
            intensity[batch_start : (batch_start + Nb)] = batch_intensity.squeeze()
            states[batch_start : (batch_start + Nb)] = batch_states

            # Sum negative log intensities (the -log lambda term in NLL)
            nll_neg_logsum -= torch.sum(torch.log(batch_intensity))

        # Save context for backward
        ctx.M, ctx.K, ctx.T = M, K, T
        ctx.gamma_param = gamma_param
        ctx.save_for_backward(
            intensity,
            states if gamma_param else None,
            alpha,
            gamma,
            ti,
            mi,
        )

        # Return average NLL per event
        return (nll_neg_logsum + nll_int) / N

    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor):
        (
            intensity,
            intensity_states,
            alpha,
            gamma,
            ti,
            mi,
        ) = ctx.saved_tensors
        M, K, T = ctx.M, ctx.K, ctx.T
        N = ti.shape[1]

        # Scale applied to all computed parameter gradients (derivative of mean)
        grad_scale = -grad_output / N

        # dti: vector of inter-event times Δt_i = t_i - t_{i-1}; first Δt is 0
        dti = ti - torch.cat([ti[:, 0:1, :], ti[:, 0:-1, :]], dim=1)  # Shape: [1, N, 1]

        # exp_decay is e^{-γ_k Δt_i} with shape [K, N, 1]
        exp_decay = torch.exp(-gamma * dti).permute(2, 1, 0)  # Shape: [K, N, 1]

        # one_hot_vectors encodes which node produced the previous event for each i.
        # First column corresponds to a "no-previous-event" zero vector.
        one_hot_vectors = torch.cat(
            [
                torch.zeros(M, device=ti.device).unsqueeze(0),
                torch.nn.functional.one_hot(mi[:-1], num_classes=M),
            ],
            dim=0,
        )  # Shape: [N, M]

        # Build transition matrices for the alpha-state recurrence:
        # Each transition row contains [exp_decay, exp_decay * one_hot_vector].
        # Concatenate to shape [K, N, M+1]
        alpha_transition_matrices = torch.cat(
            [exp_decay, exp_decay * one_hot_vectors], dim=2
        )  # Shape: [K, N, M+1]

        # Prefix-scan across time to get prefix transition products (prefix matrices)
        alpha_prefix_matrices = _torch_scan.prefix_scan(
            alpha_transition_matrices, prefix_func=_torch_scan.state_mult, dim=1
        )  # Shape: [K, N, M+1]

        # Extract per-event non-augmented states R from the prefix matrices
        alpha_states = alpha_prefix_matrices[:, :, 1:].permute(
            1, 2, 0
        )  # Shape: [N, M, K]

        if ctx.gamma_param:
            # Build analogous transition matrices for d/dγ terms.
            gamma_transition_matrices = torch.cat(
                [exp_decay, -dti * intensity_states.permute(2, 0, 1)], dim=2
            )  # Shape: [K, N, M+1]
            gamma_prefix_matrices = _torch_scan.prefix_scan(
                gamma_transition_matrices, prefix_func=_torch_scan.state_mult, dim=1
            )  # Shape: [K, N, M+1]
            gamma_states = gamma_prefix_matrices[:, :, 1:].permute(
                1, 2, 0
            )  # Shape: [N, M, K]

            # Gather the per-event gamma-state and intensity-state for the event's node
            gamma_states_at_events = torch.gather(
                gamma_states, dim=1, index=mi[:, None, None].expand(-1, 1, K)
            )  # Shape: [N, 1, K]
            intensity_states_at_events = torch.gather(
                intensity_states, dim=1, index=mi[:, None, None].expand(-1, 1, K)
            )  # Shape: [N, 1, K]

            # Compose the term required for gamma gradient from log-sum contribution
            gamma_state_sum = torch.sum(
                (
                    intensity_states_at_events.squeeze(1)
                    + gamma * gamma_states_at_events.squeeze(1)
                )
                / intensity.unsqueeze(-1),
                dim=0,
            )  # Shape: [K]

            # Compose the term required for gamma gradient from integrated-intensity contribution
            gamma_exp_decay_sum = torch.sum(
                alpha[:, mi, :]
                * (T - ti)
                * torch.exp(-gamma[:, None, None] * (T - ti)),
                dim=(1, 2),
            )  # Shape: [K]

        # --- Prepare data for alpha gradient computation per source node ---
        # Sort events by their node index while preserving temporal order
        mi_sorted, idx = torch.sort(mi, stable=True)
        mi_counts = torch.bincount(mi_sorted, minlength=M).tolist()

        # Reorder ti, intensity, and alpha_states according to the sorted node indices
        # Splitting yields per-node sequences (each subsequence remains time-ordered)
        ti_split = torch.split(ti[:, idx, :], mi_counts, dim=1)  # Shape: [M, 1, Np, 1]
        intensity_split = torch.split(
            intensity[idx, None, None], mi_counts, dim=0
        )  # Shape: [M, Np, 1, 1]
        alpha_states_split = torch.split(
            alpha_states[idx], mi_counts, dim=0
        )  # Shape: [M, Np, M, K]

        # Compute components for alpha gradient:
        #  - from the -log(lambda) term: sum over (gamma * state / lambda)
        alpha_intensity_state_sum = torch.stack(
            [
                (gamma * Kp / λp).sum(dim=0)
                for (Kp, λp) in zip(alpha_states_split, intensity_split)
            ],
            dim=0,
        )  # Shape: [M, M, K]

        #  - from the integrated-intensity term: sum over (1 - e^{-γ (T - t_i)})
        alpha_exp_decay_sum = torch.stack(
            [-torch.expm1(-gamma * (T - tiq)).sum(dim=1) for tiq in ti_split],
            dim=1,
        )  # Shape: [M, K]

        # --- Final gradient composition ---

        # Gradient for mu: depends only on counts of reciprocals of intensities and T
        mu_grad = grad_scale * torch.tensor(
            [λp.reciprocal().sum() - T for λp in intensity_split], device=ti.device
        )  # Shape: [M]

        # Gradient for alpha: combine intensity term and integrated-intensity term,
        # then permute to match alpha's shape [M, M, K]
        alpha_grad = grad_scale * (
            alpha_intensity_state_sum - alpha_exp_decay_sum
        ).permute(2, 1, 0)

        # Gradient for gamma if gamma is a parameter
        if ctx.gamma_param:
            gamma_grad = grad_scale * (gamma_state_sum - gamma_exp_decay_sum)
        else:
            gamma_grad = None

        # Return gradients for (mu, alpha, gamma) and None for the remaining forwarded args
        return (
            mu_grad,
            alpha_grad,
            gamma_grad,
        ) + (None,) * 8


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
            state_dict["alpha"] = self.alpha.detach()
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

    def simulate(self, T: float, max_events=100):
        """
        Simulate an event sequence using Ogata's thinning algorithm.

        Terminates when an event time exceeds T or when max_events is reached.

        Returns:
            ti: Tensor of event times shaped (1, N, 1)
            mi: Tensor of event types shaped (N,)
        """

        t = 0.0
        ti = torch.zeros(1, 0, 1, device=self.device)
        mi = torch.zeros(0, dtype=torch.long, device=self.device)

        R_λ = None  # right-limit state (R) after the most recent accepted event

        with torch.no_grad():
            while t < T and ti.shape[1] < max_events:
                # Upper-bound intensity (sum across nodes) at current state
                λ_star = self.intensity_at_next_event(t, None, ti, mi, R_λ).sum()

                # Sample candidate inter-arrival time
                τ = torch.distributions.Exponential(λ_star).sample()
                t = t + τ.item()

                if t > T:
                    break

                # Compute intensity at candidate time and proposed new state
                λ_t, R = self.intensity_at_next_event(
                    t, None, ti, mi, R_λ, next_state=True
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
                    event_type = torch.distributions.Categorical(
                        logits=torch.log(p)
                    ).sample()
                    mi = torch.cat([mi, event_type])
                    ti = torch.cat(
                        [ti, torch.tensor([t], device=self.device).reshape(1, 1, 1)],
                        dim=1,
                    )
                    R_λ = R  # update right-limit state

        return ti, mi

    def fit(self, T, ti, mi, fit_config=config.HawkesFitConfig()) -> list:
        """
        Fit model parameters by maximizing likelihood (minimizing NLL).

        Returns:
            List of scalar loss values (training loss per epoch).
        """

        N = ti.shape[1]

        if self.debug_config.detect_anomalies:
            torch.autograd.set_detect_anomaly(True)

        # Logging basic training configuration and model size
        self.logger.info(f"Starting Hawkes process training: {self.__class__.__name__}")
        self.logger.info(
            f"Configuration: M={self.M}, K={self.K}, N={N:,}, steps={fit_config.num_steps}"
        )
        self.logger.info(
            f"Parameters: lr={fit_config.learning_rate}, l1={fit_config.l1_penalty}, nuc={fit_config.nuc_penalty}"
        )
        self.logger.info(f"Device: {self.device}, model params: {self.num_params:,}")

        optimizer = torch.optim.Adam(self.parameters(), lr=fit_config.learning_rate)
        losses = []

        if self.debug_config.profile_mem_iters:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self.logger.info(
                f"Profiling {self.debug_config.profile_mem_iters} training iterations (up to {self.debug_config.profile_mem_entries} entries)"
            )

        self.logger.info("Starting training loop...")

        fit_time_real = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        for epoch in range(fit_config.num_steps):
            optimizer.zero_grad()

            # Compute NLL (forward + backward handled by custom Function)
            nll = self.nll(T, ti, mi)

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
            # Backpropagate (retain_graph when debugging gradient comparisons)
            loss.backward(retain_graph=self.debug_config.check_grad_epsilon)

            # Validate gradients are finite
            for n, p in self.named_parameters():
                if torch.isnan(p.grad).any():
                    self.logger.error(f"Gradient of {n} is nan at epoch {epoch + 1}")
                    self.logger.info(p.grad)
                    raise ValueError(f"Gradient of {n} is nan")

            # Optionally compute autograd gradients for cross-checking
            if self.debug_config.check_grad_epsilon:
                param_grads = torch.autograd.grad(
                    (self._compute_nll(T, ti, mi) + l1 + nuclear_norm),
                    self.parameters(),
                )

            optimizer.step()
            losses.append(loss.item())

            if epoch >= self.debug_config.profile_mem_iters:
                torch.cuda.memory._record_memory_history(enabled=None)

            # Periodic logging with fuller diagnostics
            if (epoch + 1) % fit_config.monitor_interval == 0:
                with torch.no_grad():
                    full_nll = self._compute_nll(
                        T, ti, mi, batch_size=fit_config.batch_size
                    )
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
                        f"α_mean={self.alpha.mean().item():.4f}"
                    )

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

        if self.debug_config.profile_mem_iters:
            mem_file = f"outputs/mem/mem_snapshot_n{N}_m{self.M}_b{batch_size}_e{self.debug_config.profile_mem_iters}.pkl"
            torch.cuda.memory._dump_snapshot(mem_file)
            self.logger.info(f"Saved memory profiling snapshot at {mem_file}")

        self.logger.info("Training completed successfully!")

        # Final summary statistics computed without gradient tracking
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
            f"Performance summary: Fitting took {time.strftime('%H:%M:%S', time.gmtime(fit_time_real))}, "
            f"Peak GPU memory usage was {round(peak_mem_gpu / 2**20)}MiB"
        )

        return losses

    def nll(
        self,
        T: float,
        ti: torch.Tensor,
        mi: torch.Tensor,
    ):
        # TODO: Incorporate and support batching
        return HawkesNLL.apply(
            self.mu,
            self.alpha,
            self.gamma,
            any([("gamma" in n) for n, _ in self.named_parameters()]),
            self.intensity_at_events,
            self.integrated_intensity,
            self.M,
            self.K,
            T,
            ti,
            mi,
        )

    def _compute_nll(
        self,
        T: float,
        ti: torch.Tensor,
        mi: torch.Tensor,
        batch_size=None,
        compute_backward=False,
    ) -> float:
        """
        Compute the negative log-likelihood across all events, optionally
        performing the backward pass batch-wise.

        Args:
            T: End time of observation period
            ti: Tensor of event times shaped (1, N, 1)
            mi: Tensor of event types shaped (N,)
            batch_size: If provided, compute terms in batches of this many events
            compute_backward: If True, call backward() on each batch's contribution
                              to compute gradients in a batched way

        Returns:
            Scalar NLL (sum of -log-intensity at events and integrated intensity).
        """

        N = ti.shape[1]
        batch_size = batch_size or N

        nll_neg_logsum = 0.0
        nll_int = 0.0

        prev_state = None
        for batch_start in range(0, N, batch_size):
            Nb = min(batch_size, N - batch_start)

            batch_integrated_intensity = (
                self.integrated_intensity(
                    T,
                    ti,
                    mi,
                    batching=True,
                    batch_start=batch_start,
                    batch_size=Nb,
                )
                / N
            )
            if compute_backward:
                batch_integrated_intensity.backward()

            # Compute batch intensities; prev_state ensures continuity across batches
            batch_intensity, prev_state = self.intensity_at_events(
                ti,
                mi,
                full_intensity=False,
                batching=True,
                batch_start=batch_start,
                batch_size=Nb,
                batch_prev_state=prev_state,
            )
            batch_logsum = -torch.sum(torch.log(batch_intensity)) / N
            if compute_backward:
                batch_logsum.backward()

            nll_int += batch_integrated_intensity
            nll_neg_logsum += batch_logsum

        return nll_neg_logsum + nll_int

    def integrated_intensity(
        self,
        T: float,
        ti: torch.Tensor,
        mi: torch.Tensor,
        batching=False,
        batch_start=0,
        batch_size=None,
    ) -> torch.Tensor:
        """
        Compute the integrated intensity over the observation interval [0, T].

        When batching is enabled, this returns either the full integral for the
        first batch (including mu * T) or only the excitation contribution for
        subsequent batches to avoid double-counting mu * T.
        """

        if batching and batch_size and batch_size < ti.shape[1]:
            ti = ti[:, batch_start : (batch_start + batch_size), :]
            mi = mi[batch_start : (batch_start + batch_size)]

        # Consider only events that occurred before T
        mask = ti < T
        ti = ti[mask].reshape(1, -1, 1)
        mi = mi[mask.squeeze(0, 2)]

        # Select alpha entries corresponding to the events and align dims
        alphas = self.alpha[:, mi].permute(1, 2, 0)[None]
        ti = ti[..., None]

        # Excitation contribution: sum_k sum_events α * (1 - e^{-γ_k (T - t_i)})
        excitation = torch.sum(alphas * (1 - torch.exp(-self.gamma * (T - ti))))

        # Include baseline mu * T only for the first batch or when batching disabled
        if batching and batch_start == 0 or not batching:
            return torch.sum(self.mu * T) + excitation
        else:
            return excitation

    def intensity_at_next_event(
        self,
        t: float,
        m: int | None,
        ti: torch.Tensor,
        mi: torch.Tensor,
        prev_state: torch.Tensor = None,
        next_state=False,
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
            next_state: If True, return both (λ, R) where R is the new right-limit

        Returns:
            If next_state is False: λ (1, M) vector of intensities at time t
            If next_state is True: tuple (λ (1, M), R (K, M)) where R is next state
        """

        if prev_state is None:
            prev_state = torch.zeros(self.K, self.M, device=self.device)

        # time since last event (0 if no prior events)
        dti = t - ti[0, -1, 0] if ti.shape[1] != 0 else 0

        # alphas: contributions from the most recent event types, or zeros if none
        alphas = (
            self.alpha[:, mi[-1], :]
            if mi.shape[0] != 0
            else torch.zeros(self.K, self.M, device=self.device)
        )  # Shape: [K, M]

        exp_decay = torch.exp(-self.gamma.unsqueeze(-1) * dti)  # Shape: [K, 1]

        # R = γ-decayed previous state + γ-decayed new impulse from last event
        R = exp_decay * prev_state + exp_decay * alphas  # Shape: [K, M]

        λ = self.mu + torch.sum(self.gamma * R, dim=0).unsqueeze(0)

        if next_state:
            return λ, R  # λ shape: [1, M], R shape: [K, M]
        else:
            return λ

    def intensity_at_events(
        self,
        ti: torch.Tensor,
        mi: torch.Tensor,
        full_intensity=True,
        batching=False,
        batch_start=0,
        batch_size=None,
        batch_prev_state: torch.Tensor = None,
        return_states=False,
    ) -> torch.Tensor:
        """
        Compute intensities at the sequence of event times ti using a state-augmented
        matrix recurrence and prefix scan. This is the preferred fast implementation
        for GPU/batched evaluation of the intensity terms used in NLL.

        Args:
            ti: Event times tensor (1, N, 1)
            mi: Event types tensor (N,)
            full_intensity: If True, return the full M-dimensional intensity at
                            each event; otherwise return only the intensity of the
                            node that actually experienced the event at that time.
            batching: If True, operate on a subrange [batch_start : batch_start+batch_size]
            batch_start: Index where this batch starts
            batch_size: Number of events in this batch
            batch_prev_state: Optional previous prefix matrix to maintain continuity
            return_states: If True, also return the per-event state R required for
                           intensity_at_t computations.

        Returns:
            Depending on flags, returns λ (Nb x M) or (Nb x 1) plus optionally
            the last prefix matrix (for batching) and the R states.
        """

        if not batching and (batch_start or batch_size or batch_prev_state):
            raise ValueError("batch arguments were given when batching is disabled")

        N = ti.shape[1]

        # Batch indices shorthand
        bs = batch_start
        be = batch_start + (batch_size or N)
        Nb = be - bs

        # Compute inter-event times Δt and the corresponding alpha impulses
        # If this is not the first batch, Δt is computed relative to ti[bs-1]
        if bs != 0:
            dti = ti[:, bs:be, :] - ti[:, (bs - 1) : (be - 1), :]  # Shape: [1, Nb, 1]
            alphas = self.alpha[:, mi[(bs - 1) : (be - 1)], :]  # Shape: [K, Nb, M]
        else:
            # For the first element in the batch, previous event is treated as zero
            dti = ti[:, bs:be, :] - torch.cat(
                [ti[:, bs : (bs + 1), :], ti[:, bs : (be - 1), :]], dim=1
            )  # Shape: [1, Nb, 1]
            alphas = torch.cat(
                [
                    torch.zeros(self.K, 1, self.M, device=self.device),
                    self.alpha[:, mi[bs : (be - 1)], :],
                ],
                dim=1,
            )  # Shape: [K, Nb, M]

        # Precompute decay factors e^{-γ_k Δt_i} with shape [K, Nb, 1]
        exp_decay = torch.exp(-self.gamma * dti).permute(2, 1, 0)  # Shape: [K, Nb, 1]

        # Build transition matrices for each time step: [exp_decay, exp_decay * alphas]
        transition_matrices = torch.cat(
            [exp_decay, exp_decay * alphas], dim=2
        )  # Shape: [K, Nb, M+1]
        if batch_prev_state is not None:
            # Prepend previous prefix/state for continuity across batches
            transition_matrices = torch.cat(
                [batch_prev_state.unsqueeze(1), transition_matrices], dim=1
            )  # Shape: [K, Nb+1, M+1]

        # Compute prefix products P via scan; P contains augmented states
        P = _torch_scan.prefix_scan(
            transition_matrices, prefix_func=_torch_scan.state_mult, dim=1
        )  # Shape: [K, Nb, M+1] or [K, Nb+1, M+1]

        # Extract non-augmented right-limit states R for the last Nb prefixes
        R = P[:, -Nb:, 1:].permute(1, 2, 0)  # Shape: [Nb, M, K]

        # Convert states to intensities: μ + sum_k γ_k * R[..., k]
        λ = self.mu.unsqueeze(0) + torch.sum(self.gamma * R, dim=2)  # Shape: [Nb, M]

        # If only the intensity at the actual event node is required, pick it out
        if not full_intensity:
            λ = torch.gather(λ, dim=1, index=mi[bs:be].unsqueeze(-1))  # Shape: [Nb, 1]

        # Return assembled results according to requested flags
        res = (λ,)
        if batching:
            res += (P[:, -1, :],)  # last prefix for next batch continuity
        if return_states:
            res += (R,)

        return res if len(res) > 1 else res[0]

    def intensity_at_t(self, t, ti, mi, R: torch.Tensor = None):
        """
        Compute the right-limit intensity at arbitrary times t (not necessarily events).

        If R (per-event states) is not provided, it is computed internally by
        calling intensity_at_events(..., return_states=True).

        Returns:
            Tensor shaped (len(t), M) giving intensities for all nodes at each t.
        """

        if R is None:
            with torch.no_grad():
                _, R = self.intensity_at_events(ti, mi, return_states=True)

        # Find index of the most recent event before each t
        idx = torch.searchsorted(ti, t).to(ti.device) - 1  # Shape: t.shape
        after_first_event = idx >= 0
        idx_ = idx[after_first_event]
        t_ = t[after_first_event]  # times that are after at least one event

        # exp_decay for each kernel at each queried time relative to its last event
        exp_decay = torch.exp(
            -self.gamma * (t_ - ti[idx_]).unsqueeze(-1)
        )  # Shape: [t_.shape, K]

        # Contribution from stored R states (decayed) and the immediate alpha impulse
        state_sum = torch.sum(
            self.gamma * exp_decay * R[idx_].permute(1, 0, 2), dim=-1
        )  # Shape: [M, t_.shape]
        alpha_sum = torch.sum(
            self.alpha[:, mi[idx_], :].permute(2, 1, 0) * self.gamma * exp_decay, dim=-1
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

        This is useful for debugging and testing; it is intentionally simple and
        not optimized for speed. Prefer intensity_at_events / intensity_at_t
        for production computations.

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
            t = torch.tensor([t]).reshape(1, 1, 1)

        if m is not None:
            if not isinstance(m, torch.Tensor):
                assert isinstance(m, int), "m must be an integer"
                m = torch.tensor([m]).repeat(t.shape[0])
            else:
                assert len(m.shape) == 1, "m must be a rank-1 tensor"
                assert (
                    m.shape[0] == t.shape[0]
                ), "m must have same number of elements as t"

        B = t.shape[0]
        N = ti.shape[1]

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
            print(alpha_mi.shape, gamma_term.shape)
            excitation = torch.sum(
                alpha_mi * gamma_term, dim=(1, 3)
            )  # Sum over events and kernels

            return mus + excitation

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
