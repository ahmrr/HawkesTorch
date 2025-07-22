import math
import torch
from abc import ABC, abstractmethod


class HawkesBaseOld(torch.nn.Module, ABC):
    """
    Abstract base class for Hawkes process implementations.
    """

    def __init__(self, M: int, gamma: torch.Tensor):
        super().__init__()
        self.M = M
        self.gamma = gamma

        # assert that gamma is a rank-1 tensor
        assert len(gamma.shape) == 1, "gamma must be a rank-1 tensor"
        self.K = len(gamma)

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
        """Base intensity vector R_+^M"""
        pass

    @mu.setter
    @abstractmethod
    def mu(self, value: torch.Tensor | float):
        pass

    @property
    @abstractmethod
    def alpha(self) -> torch.Tensor:
        """Excitation matrix R_+^{KxMxM}"""
        pass

    @alpha.setter
    @abstractmethod
    def alpha(self, value: torch.Tensor | float):
        pass

    def simulate(self, T, max_events=100):
        """
        Simulate events from the Hawkes process.
        """
        t = 0.0
        ti = torch.zeros(1, 0, 1)
        mi = torch.zeros(0, dtype=torch.long)

        with torch.no_grad():
            while t < T and ti.shape[1] < max_events:
                λ_star = self.intensity(t, None, ti, mi, right_limit=True).sum()
                τ = torch.distributions.Exponential(λ_star).sample()
                t = t + τ.item()

                if t > T:
                    break

                λ_t = self.intensity(t, None, ti, mi)
                λ_star_new = λ_t.sum()
                r = λ_star_new / λ_star
                assert (
                    r <= 1
                ), f"λ_t.sum() / λ_star must be less than or equal to 1. Got {r.item():0.3f}"

                if torch.rand(1) <= r:
                    p = λ_t / λ_star_new
                    event_type = torch.distributions.Categorical(p).sample()
                    mi = torch.cat([mi, event_type])
                    ti = torch.cat([ti, torch.tensor([t]).reshape(1, 1, 1)], dim=1)

        return ti, mi

    def fit(
        self,
        T,
        ti,
        mi,
        num_steps=1000,
        monitor_interval=100,
        learning_rate=0.01,
        l1_penalty=0.01,
        nuc_penalty=0.01,
        batch_size=None,
        l1_hinge=1,
    ):
        """
        Fit the Hawkes process parameters using Maximum Likelihood Estimation.

        Args:
            T: End time of observation period
            ti: Tensor of event times with shape (1, N, 1)
            mi: Tensor of event types with shape (N,)
            num_steps: Number of optimization steps
            monitor_interval: Print progress every monitor_interval steps
            learning_rate: Learning rate for Adam optimizer
            l1_penalty: L1 regularization strength for sparsity
            nuc_penalty: Nuclear norm regularization strength for low-rank structure
            batch_size: Number of events to use in each batch. If None, use all events
            li_hinge: Hinge point for l1 regularization

        Returns:
            list: Training losses at each optimization step
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        N = ti.shape[1]
        losses = []

        for epoch in range(num_steps):
            optimizer.zero_grad()
            batch_ix = (
                torch.randint(0, N, (batch_size,))
                if batch_size is not None
                else torch.arange(N)
            )

            nll = self.nll(T, ti, mi, batch_ix)

            if l1_penalty > 0:
                l1_mu = torch.where(self.mu < l1_hinge, self.mu, 0).sum()
                l1_alpha = torch.where(self.alpha < l1_hinge, self.alpha, 0).sum()
                l1 = l1_penalty * (l1_mu + l1_alpha)
            else:
                l1 = 0

            if nuc_penalty > 0:
                nuclear_norm = (
                    nuc_penalty * torch.norm(self.alpha, p="nuc", dim=(1, 2)).sum()
                )
            else:
                nuclear_norm = 0

            loss = nll + l1 + nuclear_norm
            loss.backward()

            for n, p in self.named_parameters():
                if torch.isnan(p.grad).any():
                    raise ValueError(f"Gradient of {n} is nan")

            optimizer.step()
            losses.append(loss.item())

            if (epoch + 1) % monitor_interval == 0:
                with torch.no_grad():
                    full_nll = self.nll(T, ti, mi, torch.arange(N))
                    sparsity_factor = (
                        self.alpha.isclose(
                            torch.zeros_like(self.alpha), atol=0.03
                        ).sum()
                    ).sum() / self.alpha.numel()
                    print(
                        f"Epoch {epoch + 1}/{num_steps}, Loss: {full_nll.item():.6f}, "
                        f"Sparsity: {sparsity_factor:.2f}"
                    )

        return losses

    def nll(self, T, ti, mi, batch_ix, debug=False):
        """Compute negative log-likelihood."""
        B = batch_ix.shape[0]
        N = ti.shape[1]
        weight = N / B

        t = ti[:, batch_ix].permute(1, 0, 2)
        m = mi[batch_ix]

        intensity = self.intensity(t, m, ti, mi)
        nll_events = -weight * torch.sum(torch.log(intensity))
        nll_int = self.integrated_intensity(T, ti, mi)

        if debug:
            print("\tEvent time loss:", nll_events)
            print("\tIntegral loss:", nll_int)

        return (nll_events + nll_int) / N

    def integrated_intensity(
        self, t: float, ti: torch.Tensor, mi: torch.Tensor
    ) -> torch.Tensor:
        """Compute integral of intensity function."""
        mask = ti < t
        ti = ti[mask].reshape(1, -1, 1)
        mi = mi[mask.squeeze()]

        alphas = self.alpha[:, mi].permute(1, 2, 0)[None]
        ti = ti[..., None]

        baseline = torch.sum(self.mu * t)
        excitation = torch.sum(alphas * (1 - torch.exp(-self.gamma * (t - ti))))

        return baseline + excitation

    def intensity(
        self,
        t: torch.Tensor | float,
        m: torch.Tensor | int | None,
        ti: torch.Tensor,
        mi: torch.Tensor,
        right_limit=False,
    ) -> torch.Tensor:
        """Compute intensity of the Hawkes process."""
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
            excitation = torch.sum(
                alpha_mi * gamma_term, dim=(1, 3)
            )  # Sum over events and kernels

            return mus + excitation

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
