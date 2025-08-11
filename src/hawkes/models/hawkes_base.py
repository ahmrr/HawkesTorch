import math
import time
import torch
import typing
import logging
from abc import ABC, abstractmethod
import torchviz

from ..utils import config, _torch_scan


class HawkesNLL(torch.autograd.Function):
    """
    Custom autograd Function to compute the NLL of a multivariate Hawkes process, as well as its parameter gradients.
    """

    @staticmethod
    def forward(
        ctx: typing.Any,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
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

        # Negative log-sum and integral terms in negative likelihood
        nll_neg_logsum = 0.0
        nll_int = 0.0

        # Calculate intensity and integrated intensity in batches
        prev_state = None

        # TODO: Make batching work properly
        intensity = torch.empty(N, device=ti.device)
        for batch_start in range(0, N, batch_size):
            Nb = min(batch_size, N - batch_start)

            nll_int += integrated_intensity_fun(
                T, ti, mi, batching=True, batch_start=batch_start, batch_size=Nb
            )

            batch_intensity, prev_state = intensity_at_events_fun(
                ti,
                mi,
                full_intensity=False,
                batching=True,
                batch_start=batch_start,
                batch_size=Nb,
                batch_prev_state=prev_state,
            )
            nll_neg_logsum -= torch.sum(torch.log(batch_intensity))
            intensity[batch_start : (batch_start + Nb)] = batch_intensity.squeeze()

        ctx.M, ctx.K, ctx.T = M, K, T
        ctx.save_for_backward(intensity, gamma, ti, mi)

        return (nll_neg_logsum + nll_int) / N

    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor):
        # TODO: can use existing intensity exp_decay, maybe that would save some time or mem

        intensity, gamma, ti, mi = ctx.saved_tensors
        M, K, T = ctx.M, ctx.K, ctx.T
        N = ti.shape[1]

        dti = ti - torch.cat([ti[:, 0:1, :], ti[:, 0:-1, :]], dim=1)  # Shape: [1, N, 1]
        exp_decay = torch.exp(-gamma * dti).permute(2, 1, 0)  # Shape: [K, N, 1]
        one_hot_vectors = torch.cat(
            [
                torch.zeros(M, device=ti.device).unsqueeze(0),
                torch.nn.functional.one_hot(mi[:-1], num_classes=M),
            ],
            dim=0,
        )  # Shape: [N, M]

        # Compute K sequence (of shape [N, M, K]) where K[:, q, k] corresponds to alpha[p, q, k]

        transition_matrices = torch.cat(
            [exp_decay, exp_decay * one_hot_vectors], dim=2
        )  # Shape: [K, N, M+1]

        # Perform prefix scan on transition matrices
        prefix_matrices = _torch_scan.prefix_scan(
            transition_matrices, prefix_func=_torch_scan.state_mult, dim=1
        )  # Shape: [K, N, M+1]

        # Extract states from prefix matrices
        states = prefix_matrices[:, :, 1:].permute(1, 2, 0)  # Shape: [N, M, K]

        # Obtain sorting indices of event nodes and the length of each node type in the sorted tensor
        mi_sorted, idx = torch.sort(mi, stable=True)
        mi_counts = torch.bincount(mi_sorted, minlength=M).tolist()

        # Sort ti, intensity, and states according to their node type (preserving temporal order as well)
        # Split based on which node they correspond to, with each split subsequence being sorted already
        # The first tuple index is the node type; i.e., mi_split[p] contains events of node type p + 1
        # The docs say that split returns a view of the original tensor, so this should be efficient
        ti_split = torch.split(ti[:, idx, :], mi_counts, dim=1)  # Shape: [M, 1, Np, 1]
        intensity_split = torch.split(
            intensity[idx, None, None], mi_counts, dim=0
        )  # Shape: [M, Np, 1, 1]
        states_split = torch.split(
            states[idx], mi_counts, dim=0
        )  # Shape: [M, Np, M, K]

        # Compute mu gradient; only depends on each node type's intensity
        mu_grad = torch.tensor(
            [λp.reciprocal().sum() - T for λp in intensity_split], device=ti.device
        )  # Shape: [M]

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
        )  # Shape: [M, K]

        alpha_grad = (intensity_state_sum - exp_decay_sum).permute(2, 1, 0)

        # TODO: Handle parametrized gamma case
        gamma_grad = None

        # Return negative of log-likelihood gradient scaled by 1/N
        return (
            -mu_grad * grad_output / N,
            -alpha_grad * grad_output / N,
            gamma_grad,
        ) + (None,) * 7


class HawkesBase(torch.nn.Module, ABC):
    """
    Abstract base class for multivariate Hawkes process implementations.
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
            M: Number of nodes
            K: Number of exponential decay kernels in intensity function
            device: Which device the model should live on
            debug_config: Debug configuration settings
        """

        super().__init__()
        self.M = M
        self.K = K

        self.device = device
        self.debug_config = debug_config
        self.logger = logging.getLogger(__name__)

        if debug_config.deterministic_sim:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        for param in self.parameters():
            print(param, param.device)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to include current alpha value"""
        state_dict = super().state_dict(*args, **kwargs)
        # Compute and store current alpha
        with torch.no_grad():
            state_dict["alpha"] = self.alpha.detach()
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Override load_state_dict to handle alpha"""
        # Remove alpha from state_dict before loading since it's derived
        state_dict = state_dict.copy()  # avoid modifying original
        state_dict.pop("alpha", None)
        super().load_state_dict(state_dict, *args, **kwargs)

    @property
    @abstractmethod
    def mu(self) -> torch.Tensor:
        """Base intensity vector of shape [M]"""
        pass

    @mu.setter
    @abstractmethod
    def mu(self, value: torch.Tensor | float):
        pass

    @property
    @abstractmethod
    def alpha(self) -> torch.Tensor:
        """Excitation matrix of shape [M, M, K]"""
        pass

    @alpha.setter
    @abstractmethod
    def alpha(self, value: torch.Tensor | float):
        pass

    @property
    @abstractmethod
    def gamma(self) -> torch.Tensor:
        """Exponential decay kernel memory values of shape [K]"""
        pass

    @gamma.setter
    @abstractmethod
    def gamma(self, value: torch.Tensor):
        pass

    def simulate(self, T: float, max_events=100):
        """
        Simulate events from the Hawkes process using Ogata's thinning algorithm. Terminates either when event time reaches T or when the number of events reaches max_events.

        Args:
            T: End time of observation period; i.e., the maximum simulated event time
            max_events: Maximum number of events to simulate

        Returns:
            Event sequence (ti, mi) consisting of event-node pairs
        """

        t = 0.0
        ti = torch.zeros(1, 0, 1, device=self.device)
        mi = torch.zeros(0, dtype=torch.long, device=self.device)

        R_λ = None  # Left-limit state for the last known event

        with torch.no_grad():
            while t < T and ti.shape[1] < max_events:
                λ_star = self.intensity_at_next_event(t, None, ti, mi, R_λ).sum()

                τ = torch.distributions.Exponential(λ_star).sample()
                t = t + τ.item()

                if t > T:
                    break

                λ_t, R = self.intensity_at_next_event(
                    t, None, ti, mi, R_λ, next_state=True
                )
                λ_star_new = λ_t.sum()

                r = λ_star_new / λ_star
                assert (
                    r <= 1
                ), f"λ_t.sum() / λ_star must be less than or equal to 1. Got {r.item():0.3f}"

                if torch.rand(1, device=self.device) <= r:
                    p = λ_t / λ_star_new
                    # Use logits to avoid floating point issues with probabilities summing to 1
                    event_type = torch.distributions.Categorical(
                        logits=torch.log(p)
                    ).sample()
                    mi = torch.cat([mi, event_type])
                    ti = torch.cat(
                        [ti, torch.tensor([t], device=self.device).reshape(1, 1, 1)],
                        dim=1,
                    )
                    R_λ = R  # Save new event state for use in next iteration

        return ti, mi

    def fit(self, T, ti, mi, fit_config=config.HawkesFitConfig()) -> list:
        """
        Fit the Hawkes process parameters using Maximum Likelihood Estimation.

        Args:
            T: End time of observation period
            ti: Tensor of event times with shape (1, N, 1)
            mi: Tensor of event types with shape (N,)
            fit_config: Contains model hyperparameters and other configuration values

        Returns:
            List of training losses at each optimization step
        """

        N = ti.shape[1]

        if self.debug_config.detect_anomalies:
            torch.autograd.set_detect_anomaly(True)

        # Log training configuration
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

            # Calculate NLL across entire data and perform the backward pass
            nll = self.nll(T, ti, mi)
            # nll = self._compute_nll(
            #     T, ti, mi, batch_size=fit_config.batch_size, compute_backward=True
            # )

            if fit_config.l1_penalty > 0:
                l1_mu = torch.where(self.mu < fit_config.l1_hinge, self.mu, 0).sum()
                l1_alpha = torch.where(
                    self.alpha < fit_config.l1_hinge, self.alpha, 0
                ).sum()
                l1 = fit_config.l1_penalty * (l1_mu + l1_alpha)
            else:
                l1 = 0

            if fit_config.nuc_penalty > 0:
                nuclear_norm = (
                    fit_config.nuc_penalty
                    * torch.linalg.matrix_norm(self.alpha, ord="nuc", dim=(1, 2)).sum()
                )
            else:
                nuclear_norm = 0

            loss = nll + l1 + nuclear_norm
            loss.backward(
                retain_graph=(self.debug_config.check_grad_epsilon < torch.inf)
            )

            # Check for NaN gradients
            for n, p in self.named_parameters():
                if torch.isnan(p.grad).any():
                    self.logger.error(f"Gradient of {n} is nan at epoch {epoch + 1}")
                    self.logger.info(p.grad)
                    raise ValueError(f"Gradient of {n} is nan")

            if self.debug_config.check_grad_epsilon < torch.inf:
                param_grads = torch.autograd.grad(
                    (self._compute_nll(T, ti, mi) + l1 + nuclear_norm),
                    self.parameters(),
                )

            optimizer.step()
            losses.append(loss.item())

            if epoch >= self.debug_config.profile_mem_iters:
                torch.cuda.memory._record_memory_history(enabled=None)

            # Progress logging at monitor intervals
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

                if self.debug_config.check_grad_epsilon < torch.inf:
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

        # Final summary
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
        Compute negative log-likelihood across the entire data using batching.

        Args:
            T: End time of observation period
            ti: Tensor of event times with shape (1, N, 1)
            mi: Tensor of event types with shape (N,)
            batch_size: Number of events to use in each batch. If None, use all events
            perform_backward: Perform backward pass, computing gradients in a batched manner

        Returns:
            NLL of the model's parameters given the data
        """

        N = ti.shape[1]
        batch_size = batch_size or N

        # Integral and negative log-sum terms in NLL
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

            # TODO: Detaching prev_state causes incorrect gradients because of dependency
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
        Compute integral of intensity function across the observation period.

        Args:
            T: End time of observation period
            ti: Tensor of event times with shape (1, N, 1)
            mi: Tensor of event types with shape (N,)
            batching: Whether to split computaiton of the integral into batches
            batch_start: If batching, the starting index of the current batch
            batch_size: If batching, the size of the current batch

        Returns:
            Intensity function integrated from 0 to T.
        """

        if batching and batch_size and batch_size < ti.shape[1]:
            ti = ti[:, batch_start : (batch_start + batch_size), :]
            mi = mi[batch_start : (batch_start + batch_size)]

        mask = ti < T
        ti = ti[mask].reshape(1, -1, 1)
        mi = mi[mask.squeeze(0, 2)]

        # alphas = self.alpha[:, :, mi].permute(2, 1, 0)[None]
        alphas = self.alpha[:, mi].permute(1, 2, 0)[None]
        ti = ti[..., None]

        excitation = torch.sum(alphas * (1 - torch.exp(-self.gamma * (T - ti))))

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
        Compute the intensity at and including a new next event in a Hawkes process using a recurrence relation. This should only be used for simulating data, or for work-efficient non-parallel computation on the CPU.

        Args:
            t: New event time
            m: New event type
            ti: Tensor of all previous event times, of shape (1, N, 1)
            mi: Tensor of all previous event types, of shape (N,)
            prev_state: Previous intensity state of shape (K, M), and None if first event
            next_state: Whether to return the next state, for use in the next recursive iteration

        Returns:
            Intensity of shape (M)
        """

        if prev_state is None:
            prev_state = torch.zeros(self.K, self.M, device=self.device)

        dti = t - ti[0, -1, 0] if ti.shape[1] != 0 else 0
        alphas = (
            self.alpha[:, mi[-1], :]
            if mi.shape[0] != 0
            else torch.zeros(self.K, self.M, device=self.device)
        )  # Shape: [K, M]

        exp_decay = torch.exp(-self.gamma.unsqueeze(-1) * dti)  # Shape: [K, 1]

        R = exp_decay * prev_state + exp_decay * alphas  # Shape: [K, M]

        λ = self.mu + torch.sum(self.gamma * R, dim=0).unsqueeze(0)

        if next_state:
            return λ, R  # Shape: [1, M]
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
        Compute intensity of the Hawkes process at events ti using an optimized state-augmented matrix multiplication.
        This should only be used for NLL calculations; to simulate data, use the recursive version.

        Args:
            ti: Tensor of all event times, of shape (1, N, 1)
            mi: Tensor of all event types, of shape (N,)
            full_intensity: Whether to return intensity for all accounts or for the single corresponding account at each event
            batching: If True, allows batching params and returns the last state, used to calculate the next batch intensity
            batch_start: Start of current intensity batch
            batch_size: Size of current intensity batch
            batch_prev_state: Previous intensity state, if intensity has been batched, of shape (K, M+1)
            return_states: Whether to return the full state vector R, required for calculating intensity at a time.

        Returns:
            Intensities at each time ti, of shape (N, M) if full_intensity is True or (N, 1) otherwise.

        Raises:
            ValueError: If batch_* arguments are given when batching is disabled, or if both return_states and batching are enabled
        """

        if not batching and (batch_start or batch_size or batch_prev_state):
            raise ValueError("batch arguments were given when batching is disabled")

        N = ti.shape[1]

        # Shorthand batch start and end indices for convenience
        bs = batch_start
        be = batch_start + (batch_size or N)
        Nb = be - bs

        # Calculate dti and alphas used in recursive formula
        # Each element of dti is ∆t_i = t_i - t_{i-1}, and ∆t_1 is 0 for the first batch
        # Each element of alphas is α^k_{:,m_{i-1}}, and α^k_{:,m_0} is [0 ... 0] for the first batch
        if bs != 0:
            dti = ti[:, bs:be, :] - ti[:, (bs - 1) : (be - 1), :]  # Shape: [1, Nb, 1]
            alphas = self.alpha[:, mi[(bs - 1) : (be - 1)], :]  # Shape: [K, Nb, M]
        else:
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

        # Each element is e^(-γ^k * ∆t_i)
        exp_decay = torch.exp(-self.gamma * dti).permute(2, 1, 0)  # Shape: [K, Nb, 1]

        # Construct transition matrices M
        # Each element is M_i, and M_1 = prev_state if given
        transition_matrices = torch.cat(
            [exp_decay, exp_decay * alphas], dim=2
        )  # Shape: [K, Nb, M+1]
        if batch_prev_state is not None:
            transition_matrices = torch.cat(
                [batch_prev_state.unsqueeze(1), transition_matrices], dim=1
            )  # Shape: [K, Nb+1, M+1]

        # Compute prefix matrices P
        P = _torch_scan.prefix_scan(
            transition_matrices, prefix_func=_torch_scan.state_mult, dim=1
        )  # Shape: [K, Nb, M+1] or [K, Nb+1, M+1]

        # Obtain non-augmented states R for each of last N prefixes
        R = P[:, -Nb:, 1:].permute(1, 2, 0)  # Shape: [Nb, M, K]

        # Scale R by gamma, sum over kernels, and add mu to obtain intensity
        λ = self.mu.unsqueeze(0) + torch.sum(  # Shape: [1, M]
            self.gamma * R, dim=2
        )  # Shape: [Nb, M]

        # Provide intensity only for the single account at each event
        if not full_intensity:
            λ = torch.gather(λ, dim=1, index=mi[bs:be].unsqueeze(-1))  # Shape: [Nb, 1]

        # Return last intensity state for use in the next batch, if needed
        res = (λ,)
        if batching:
            res += (P[:, -1, :],)
        if return_states:
            res += (R,)

        return res if len(res) > 1 else res[0]

    def intensity_at_t(self, t, ti, mi, R: torch.Tensor = None):
        """
        Computes the intensity at (and including) times that are not necessarily events. This is useful for plotting the intensity function of an event sequence.

        Args:
            t: Tensor of times to compute right-limit intensities at.
            ti: Tensor of event times.
            mi: Tensor of event types.
            R: Tensor of intensity states saved during intensity calculation at each ti and mi. If None, this is computed internally.

        Returns:
            Intensities across all nodes at each time t, of shape (t.shape[0], M).
        """

        if R is None:
            with torch.no_grad():
                _, R = self.intensity_at_events(ti, mi, return_states=True)

        # ti[idx] is the event that happens right before the corresponding t
        idx = torch.searchsorted(ti, t).to(ti.device) - 1  # Shape: t.shape
        after_first_event = idx >= 0
        idx_ = idx[after_first_event]
        t_ = t[after_first_event]  # Base intensity for values before the first event

        exp_decay = torch.exp(
            -self.gamma * (t_ - ti[idx_]).unsqueeze(-1)
        )  # Shape: [t_.shape, K]

        state_sum = torch.sum(
            self.gamma * exp_decay * R[idx_].permute(1, 0, 2), dim=-1
        )  # Shape: [M, t_.shape]
        alpha_sum = torch.sum(
            self.alpha[:, mi[idx_], :].permute(2, 1, 0) * self.gamma * exp_decay, dim=-1
        )  # Shape: [M, t_.shape]

        t_intensity = (
            self.mu.unsqueeze(-1) + state_sum + alpha_sum
        )  # Shape: [M, t_.shape]

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
        Reference implementation: calculates the intensity in a manner akin to the intensity formula. Very inefficient; use a combination of intensity_at_events and intensity_at_t for fast parallel GPU computation or looped intensity_at_next_event calls for work-efficient sequential CPU computation.

        Args:
            t: Times to calculate intensity at.
            m: The node to return the intensity for at each time t. If None, return intensity for all nodes.
            ti: Tensor of event times.
            mi: Tensor of event nodes.
            right_limit: If True, calculates the intensity at each event including the event itself.
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

        # Calculate time differences - use broadcasting instead of repeat
        # t has shape [B, 1, 1], ti has shape [1, N, 1]
        # The result dt will have shape [B, N, 1]
        dt = t - ti

        if right_limit:
            dt[dt < 0] = torch.inf
        else:
            dt[dt <= 0] = torch.inf

        # Add dimension for gamma
        dt = dt.unsqueeze(-1)  # Shape: [B, N, 1, 1]

        # Handle base intensity (mu)
        if m is not None:
            # Select the specific mu values for each batch element
            mus = self.mu[m].view(B, 1)  # Shape: [B, 1]
        else:
            # Use all mu values
            mus = self.mu.expand(B, self.M)  # Shape: [B, M]

        # Handle excitation intensity (alpha)
        # Get relevant alpha values based on event types
        # alpha has shape [K, M, M], mi has shape [N]
        alpha_mi = self.alpha[:, mi]  # Shape: [K, N, M]
        alpha_mi = alpha_mi.permute(1, 2, 0)  # Shape: [N, M, K]

        if m is not None:
            # For specific event types m
            # Select the relevant column from alpha_mi for each batch element
            # We need alpha_mi[:, m[batch_idx], :] for each batch_idx
            batch_indices = torch.arange(B)
            alpha_batch = alpha_mi[:, m, :]  # Shape: [N, B, K]
            alpha_batch = alpha_batch.permute(1, 0, 2)  # Shape: [B, N, K]

            # Calculate the excitation term
            excitation = torch.sum(
                alpha_batch * self.gamma * torch.exp(-self.gamma * dt.squeeze(-2)),
                dim=(1, 2),
            )

            return mus + excitation.unsqueeze(1)
        else:
            # For all event types
            # Using broadcasting for the batch dimension
            alpha_mi = alpha_mi.unsqueeze(0)  # Shape: [1, N, M, K]

            # Calculate the excitation term
            # gamma has shape [K]
            gamma_term = self.gamma * torch.exp(-self.gamma * dt)  # Shape: [B, N, 1, K]

            # For each event type, sum the contributions
            print(alpha_mi.shape, gamma_term.shape)
            excitation = torch.sum(
                alpha_mi * gamma_term, dim=(1, 3)
            )  # Sum over events and kernels

            return mus + excitation

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
