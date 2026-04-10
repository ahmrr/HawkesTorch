import math
import time
import torch
from abc import ABC, abstractmethod

from ... import utils
from ...utils import config, _torch_scan
from .hawkes_nll_function import HawkesLogSumIntensity
from .hawkes_base import HawkesBase


class HawkesBaseReference(HawkesBase, ABC):
    """
    Reference implementation of Hawkes base class.
    """

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
        self.logger.info(f"Starting Hawkes process training: {self.__class__.__name__}")
        self.logger.info(
            f"Configuration: M={self.M}, K={self.K}, N={seq.N:,}, T={seq.T:,.2f}, steps={fit_config.num_steps}"
            + (f", batch_size={fit_config.batch_size}" if fit_config.batch_size else "")
        )
        self.logger.info(
            f"Parameters: lr={fit_config.learning_rate}, l1={fit_config.l1_penalty}, nuc={fit_config.nuc_penalty}"
        )
        self.logger.info(f"Device: {self.device}, model params: {self.num_params:,}")
        self.logger.info(
            f"Using {self.runtime_config.intensity_implementation} intensity implementation"
            + (
                " with autograd gradients"
                if self.runtime_config.use_autograd_gradients
                else " with custom gradients"
            )
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=fit_config.learning_rate)
        losses = []
        epoch_times = []

        if self.runtime_config.profile_mem_iters:
            torch.cuda.memory._record_memory_history(
                max_entries=self.runtime_config.profile_mem_entries
            )
            self.logger.info(
                f"Profiling {self.runtime_config.profile_mem_iters} training iterations (up to {self.runtime_config.profile_mem_entries} entries)"
            )

        self.logger.info("Starting training loop...")

        if torch.device(self.device).type != "cpu":
            print(f'Resetting CUDA peak memory stats, device was "{self.device}"')
            torch.cuda.reset_peak_memory_stats()

        fit_time_real = time.perf_counter()

        for epoch in range(fit_config.num_steps):
            epoch_time_real = time.perf_counter()

            optimizer.zero_grad()

            # Compute NLL (forward + backward handled by custom Function)
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

            # Optional nuclear-norm penalty over alpha
            if fit_config.nuc_penalty > 0:
                nuclear_norm = (
                    fit_config.nuc_penalty
                    * torch.linalg.matrix_norm(self.alpha, ord="nuc", dim=(1, 2)).sum()
                )
            else:
                nuclear_norm = 0

            loss = nll + l1 + nuclear_norm
            loss.backward(retain_graph=self.runtime_config.check_grad_epsilon)

            # Ensure gradients are finite
            for n, p in self.named_parameters():
                if torch.isnan(p.grad).any():
                    self.logger.error(f"Gradient of {n} is nan at epoch {epoch + 1}")
                    self.logger.info(p.grad)
                    raise ValueError(f"Gradient of {n} is nan")

            # Optionally compute autograd gradients for cross-checking
            if self.runtime_config.check_grad_epsilon:
                param_grads = torch.autograd.grad(
                    (self._nll_autograd_safe(seq) + l1 + nuclear_norm),
                    self.parameters(),
                )

            optimizer.step()

            epoch_time_real = time.perf_counter() - epoch_time_real

            losses.append(loss.item())
            epoch_times.append(epoch_time_real)

            if (
                epoch >= self.runtime_config.profile_mem_iters
                and torch.device(self.device).type != "cpu"
            ):
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
                        f"{self.base_process.report_parameters()}, "
                        f"α_mean={self.alpha.mean().item():.4f}, "
                        f"γ={'[' + ', '.join(f'{g:.2g}' for g in self.gamma.tolist()) + ']'}, "
                    )

                # Checking manual gradients against autograd
                if self.runtime_config.check_grad_epsilon:
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

        if (
            self.runtime_config.profile_mem_iters
            and torch.device(self.device).type != "cpu"
        ):
            mem_file = f"outputs/mem/mem_snapshot_n{seq.N}_m{self.M}_b{fit_config.batch_size}_e{self.runtime_config.profile_mem_iters}.pkl"
            torch.cuda.memory._dump_snapshot(mem_file)
            self.logger.info(f"Saved memory profiling snapshot at {mem_file}")

        self.logger.info("Training completed successfully!")

        # Final summary statistics
        with torch.no_grad():
            final_sparsity = (
                self.alpha.isclose(torch.zeros_like(self.alpha), atol=0.03).sum()
            ).sum() / self.alpha.numel()
            loss_reduction = (losses[0] - losses[-1]) / losses[0]

        hours = int(fit_time_real // 3600)
        minutes = int((fit_time_real % 3600) // 60)
        seconds = int(fit_time_real % 60)

        self.logger.info(
            f"Training summary: Final loss={losses[-1]:.4f}, "
            f"Loss reduction={loss_reduction:.1%}, "
            f"Final sparsity={final_sparsity:.3f}"
        )
        self.logger.info(
            f"Performance summary: Fitting time={hours:02d}:{minutes:02d}:{seconds:02d}, "
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
        """
        Computes the negative log-likelihood of the given event sequence.
        """

        batch_size = min(batch_size or seq.N, seq.N)

        mu_at_events = self.mu_at_events(seq.ti, seq.mi)

        if self.runtime_config.use_autograd_gradients:
            nll = self._nll_parallel_intensity_autograd_safe(seq)
        else:
            match self.runtime_config.intensity_implementation:
                case "general":
                    nll = self._nll_general_intensity(seq)
                case "sequential":
                    nll = self._nll_sequential_intensity(seq)
                case "parallel":
                    nll = self._nll_parallel_intensity(seq, batch_size=batch_size)
                case _:
                    raise ValueError(
                        f"Unknown intensity implementation: {self.runtime_config.intensity_implementation}"
                    )

        return -nll / seq.N

    def _nll_general_intensity(
        self,
        seq: utils.EventSequence,
        **kwargs,
    ) -> float:
        """
        Uses general intensity computation (reference implementation).
        """

        nll_log_sum = self._log_sum_intensity_general_implementation(seq)
        nll_integral = self.integrated_intensity(seq)

        return nll_log_sum - nll_integral

    def _nll_sequential_intensity(
        self,
        seq: utils.EventSequence,
        **kwargs,
    ) -> float:
        """
        Uses sequential intensity computation (reference implementation).
        """

        nll_log_sum_excitation = self._log_sum_intensity_sequential_implementation(
            seq, integrated_excitation=True
        )

        return nll_log_sum_excitation - self.integrated_base_rate(seq)

    def _nll_parallel_intensity_autograd_safe(
        self,
        seq: utils.EventSequence,
        **kwargs,
    ) -> float:
        """
        Uses prefix scan intensity computation compatible with autograd (reference implementation).
        """

        intensity_at_events = self.intensity_at_events(seq, return_full_intensity=False)

        nll_logsum = intensity_at_events.log().sum()
        nll_integral = self.integrated_intensity(seq)

        return nll_logsum - nll_integral

    def _nll_parallel_intensity(
        self,
        seq: utils.EventSequence,
        batch_size: int,
        **kwargs,
    ) -> float:
        """
        Uses parallel prefix scan intensity computation.
        """

        mu_at_events = self.mu_at_events(seq.ti, seq.mi)

        nll_log_sum = HawkesLogSumIntensity.apply(
            seq,
            mu_at_events,  # mu_{m_i}(t_i)
            self.alpha,  # alpha_{p,q,k}
            self.gamma,  # gamma_k
            self.gamma_param,
            self.intensity_at_events,
            batch_size,
        )
        nll_integral = self.integrated_intensity(seq, batch_size=batch_size)

        return nll_log_sum - nll_integral

    def _log_sum_intensity_sequential_implementation(
        self,
        seq: utils.EventSequence,
        integrated_excitation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sequential implementation of intensity computation at each event time using the recurrence relation.
        """

        gamma_param = any([("gamma" in n) for n, _ in self.named_parameters()])
        mu_at_events = self.mu_at_events(seq.ti, seq.mi)

        return HawkesLogSumIntensitySequential.apply(
            seq,
            mu_at_events,
            self.alpha,
            self.gamma,
            gamma_param,
            integrated_excitation,
        )

    def _log_sum_intensity_general_implementation(
        self,
        seq: utils.EventSequence,
        **kwargs,
    ) -> torch.Tensor:
        """
        General (inefficient) reference implementation of the log-sum intensity formula.
        """

        t = seq.ti.view(-1, 1, 1)
        ti = seq.ti.view(1, -1, 1)
        mi = seq.mi

        N = ti.shape[1]

        # Compute inter event times dt = t_i - t_j
        dt = t - ti  # [N, N, 1]
        valid = dt > 0  # mask of shape [N, N, 1]

        dt_pos = dt.masked_fill(~valid, 0.0)

        exp_dt = torch.exp(-self.gamma * dt_pos)  # Safe: no positive exponent ever
        exp_dt = exp_dt * valid  # Zero out invalid entries (no gradient leak)

        mus = self.mu_at_events(seq.ti, seq.mi).view(N, 1)

        alpha_mi = self.alpha[:, mi].permute(1, 2, 0)  # [N, M, K]
        alpha_batch = alpha_mi[:, mi, :].permute(1, 0, 2)  # [N, N, K]

        excitation = torch.sum(alpha_batch * self.gamma * exp_dt, dim=(1, 2))

        intensity = mus + excitation.unsqueeze(1)

        return intensity.log().sum()

    def intensity_at_next_event(
        self,
        t: float,
        m: int | None,
        ti: torch.Tensor,
        mi: torch.Tensor,
        prev_state: torch.Tensor = None,
        return_full_intensity=True,
        return_next_state=False,
    ) -> torch.Tensor:
        """
        Compute intensity at a candidate time t (and optionally the next state R)
        using a recurrence relation that updates the KxM state matrix.

        This function is intended for sequential simulation or CPU-friendly
        sequential computations; for batched GPU computations use intensity_at_events.

        Args:
            t: Candidate event time
            m: Candidate event type
            ti: Tensor of previous event times (1, N, 1)
            mi: Tensor of previous event types (N,)
            prev_state: Right-limit state R from the previous accepted event
            return_full_intensity: If True, return full M-dimensional intensity; else only for event type m
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

        exp_dti = torch.exp(-self.gamma[:, None] * dti)  # Shape: [K, 1]

        # R = γ-decayed previous state + γ-decayed new impulse from last event
        R = exp_dti * prev_state + exp_dti * alphas  # Shape: [K, M]

        λ = self.mu + torch.sum(self.gamma[:, None] * R, dim=0)

        if not return_full_intensity and m is not None:
            λ = λ[m].unsqueeze(0)

        if return_next_state:
            return λ, R
        else:
            return λ


class HawkesLogSumIntensitySequential(torch.autograd.Function):
    """
    Computes log-sum intensity term in NLL and its derivatives using a sequential intensity recurrence.
    """

    @staticmethod
    def forward(
        ctx: Any,
        seq: utils.EventSequence,
        mu_at_events: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        gamma_param: bool,
        integrated_excitation: bool = False,
    ):
        K = gamma.shape[0]

        nll_logsum_term = 0.0
        nll_integral_term = 0.0

        state = torch.zeros(K, seq.M, device=seq.ti.device)

        for i in range(seq.N):
            t, m = seq.ti[i], seq.mi[i]
            dt = (t - seq.ti[i - 1]) if i > 0 else 0.0

            exp_dt = torch.exp(-gamma * dt).unsqueeze(-1)
            alphas = (
                alpha[:, seq.mi[i - 1], :] if i > 0 else torch.zeros_like(alpha[:, 0])
            )

            state = exp_dt * state + exp_dt * alphas

            excitation = torch.sum(gamma[:, None] * state, dim=0)
            intensity = mu_at_events[i] + excitation[m]

            nll_logsum_term += torch.log(intensity)

            if integrated_excitation:
                # 1 - e^{-γ_k (T - t_i)}
                exp_term = -torch.expm1(-gamma[:, None] * (seq.T - t))
                nll_integral_term += torch.sum(alpha[:, :, m] * exp_term)

        # Save context for backward
        ctx.M, ctx.T = seq.M, seq.T
        ctx.gamma_param = gamma_param
        ctx.integrated_excitation = integrated_excitation
        ctx.save_for_backward(
            seq.ti,
            seq.mi,
            mu_at_events,
            alpha,
            gamma,
        )

        return nll_logsum_term - nll_integral_term

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (
            _ti_saved,
            _mi_saved,
            mu_at_events,
            alpha,
            gamma,
        ) = ctx.saved_tensors
        K = gamma.shape[0]

        seq = utils.EventSequence(_ti_saved, _mi_saved, T=ctx.T, M=ctx.M)

        mu_grad = torch.empty_like(mu_at_events)
        alpha_grad = torch.zeros_like(alpha)
        gamma_grad = torch.zeros_like(gamma) if ctx.gamma_param else None

        intensity_state = torch.zeros(K, seq.M, device=seq.ti.device)
        alpha_grad_state = torch.zeros(K, seq.M, device=seq.ti.device)
        if ctx.gamma_param:
            gamma_grad_state = torch.zeros(K, seq.M, device=seq.ti.device)

        for i in range(seq.N):
            t, m = seq.ti[i], seq.mi[i]
            dt = (t - seq.ti[i - 1]) if i > 0 else 0.0

            exp_dt = torch.exp(-gamma * dt).unsqueeze(-1)
            alphas = (
                alpha[:, seq.mi[i - 1], :] if i > 0 else torch.zeros_like(alpha[:, 0])
            )

            # R_i = e^{-γ Δt_i} R_{i-1} + e^{-γ Δt_i} α_{m_{i-1}}
            intensity_state = exp_dt * intensity_state + exp_dt * alphas

            intensity_excitation = torch.sum(gamma[:, None] * intensity_state, dim=0)
            intensity = mu_at_events[i] + intensity_excitation[m]

            # K_i = e^{-γ Δt_i} K_{i-1} + e^{-γ Δt_i} e_{m_{i-1}}
            alpha_grad_state = exp_dt * alpha_grad_state
            if i > 0:
                alpha_grad_state[:, seq.mi[i - 1]] += exp_dt.squeeze(-1)

            # L_i = e^{-γ Δt_i} L_{i-1} + t_{i-1} e^{-γ Δt_i} α_{m_{i-1}}
            if ctx.gamma_param:
                gamma_grad_state = (
                    exp_dt * gamma_grad_state + seq.ti[i - 1] * exp_dt * alphas
                )

            alpha_grad_term = (
                gamma[:, None] * alpha_grad_state / intensity
            )  # Shape: [K, M]
            alpha_grad[:, :, m] += alpha_grad_term

            if ctx.gamma_param:
                gamma_grad_term = (
                    gamma * gamma_grad_state[:, m]
                    + (1 - gamma * t) * intensity_state[:, m]
                ) / intensity
                gamma_grad += gamma_grad_term

            # Gradient for mu depends only on intensities
            mu_grad[i] = 1 / intensity

            # Gradients for integrated excitation term
            if ctx.integrated_excitation:
                alpha_grad[:, m, :] -= 1 - torch.exp(-gamma[:, None] * (ctx.T - t))
                gamma_grad -= torch.sum(
                    (ctx.T - t)
                    * alpha[:, m, :]
                    * torch.exp(-gamma[:, None] * (ctx.T - t)),
                    dim=1,
                )

        mu_grad *= grad_output
        alpha_grad *= grad_output
        if ctx.gamma_param:
            gamma_grad *= grad_output

        return (None,) + (mu_grad, alpha_grad, gamma_grad) + (None,) * 8
