import math
import time
import torch
import typing
import logging

from . import PoissonBase
from .. import utils
from ..utils import config


class PoissonFourierSeries(PoissonBase):
    """
    Poisson process with intensity modeled as activation(r0 + fourier series).
    The intensity for each variate m is: activation(r0_m + sum_k(a_k_m * cos(2*pi*k*t/T) + b_k_m * sin(2*pi*k*t/T)))
    """

    def __init__(
        self,
        M: int,
        T: float,
        num_modes: int,
        r0_init: torch.Tensor | float | None = None,
        fourier_init: torch.Tensor | None = None,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str = "cpu",
    ):
        """
        Initialize a FourierSeriesPoisson process with periodic intensity modeled by Fourier series.

        This class models a multivariate Poisson process where each variate has time-varying
        intensity following a periodic pattern. The intensity for variate m at time t is:
        μ_m(t) = activation(r0_m + Σ_k [a_k_m * cos(2πkt/T) + b_k_m * sin(2πkt/T)])
        where k ranges from 1 to num_modes, providing rich periodic behavior.

        Args:
            M (int): Number of variates (event types) in the process.
                Must be positive integer.
            T: Period of the Fourier series in time units.
                The intensity pattern repeats every T time units.
                Must be positive.
            num_modes: Number of Fourier modes to include in the series.
                Higher values allow more complex periodic patterns but increase parameters.
                Must be positive integer.
            device: PyTorch device for computation ("cpu" or "cuda").
                Default: "cpu"
            transformation: Activation function to ensure positive intensities.
                Options: "softplus" (smooth, recommended) or "exp" (sharp).
                Default: "softplus"
            t_start: Start times for each variate.
                - If float: Same start time for all variates
                - If torch.Tensor: Must have shape (M,), per-variate start times
                - If None: All variates start at time 0
                Default: None
            t_end: End times for each variate.
                - If float: Same end time for all variates
                - If torch.Tensor: Must have shape (M,), per-variate end times
                - If None: All variates have infinite end time
                Default: None
            r0_init: Initial baseline values.
                - If float: Same baseline for all variates
                - If torch.Tensor: Must have shape (M,), per-variate baselines
                - If None: Small random initialization from N(0, 0.01)
                Default: None
            fourier_init: Initial Fourier coefficients.
                - If torch.Tensor: Must have shape (M, 2, num_modes) where [:, 0, :] are
                  cosine coefficients and [:, 1, :] are sine coefficients
                - If None: Small random initialization from N(0, 0.01)
                Default: None

        Raises:
            ValueError: If parameters have incompatible shapes or invalid values.

        Mathematical Model:
            The Fourier series representation allows modeling complex periodic patterns:
            - r0_m: Baseline intensity for variate m (DC component)
            - a_k_m: Cosine coefficient for mode k, variate m
            - b_k_m: Sine coefficient for mode k, variate m
            - T: Period (intensity repeats every T time units)

            The full intensity is: μ_m(t) = activation(r0_m + fourier_series_m(t))
            where fourier_series_m(t) = Σ_{k=1}^{num_modes} [a_k_m * cos(2πkt/T) + b_k_m * sin(2πkt/T)]

        Examples:
            >>> # Simple periodic process with 3 variates, 2 modes, period 24 hours
            >>> model = FourierSeriesPoisson(
            ...     M=3,
            ...     num_modes=2,
            ...     T=24.0,
            ...     r0_init=1.0  # Same baseline for all
            ... )

            >>> # Complex model with different baselines and custom Fourier coefficients
            >>> fourier_coeffs = torch.zeros(4, 2, 3)  # 4 variates, 3 modes
            >>> fourier_coeffs[:, 0, 0] = torch.tensor([1.0, 0.5, -0.3, 0.8])  # First cosine mode
            >>> model = FourierSeriesPoisson(
            ...     M=4,
            ...     num_modes=3,
            ...     T=12.0,
            ...     r0_init=torch.tensor([2.0, 1.5, 1.0, 2.5]),
            ...     fourier_init=fourier_coeffs,
            ...     device="cuda"
            ... )

            >>> # Daily pattern with different activity windows
            >>> model = FourierSeriesPoisson(
            ...     M=2,
            ...     num_modes=4,
            ...     T=24.0,  # 24-hour period
            ...     t_start=[0.0, 6.0],   # Second variate starts at 6 AM
            ...     t_end=[18.0, 22.0]    # Different end times
            ... )

        Note:
            - Higher num_modes allow more complex patterns but increase computational cost
            - The period T should match the expected periodicity in your data
            - Fourier coefficients are learned parameters that will be optimized during fitting
            - Use activation="softplus" for smooth intensities, "exp" for sharper transitions
        """
        super().__init__(M, device, activation, t_start, t_end)

        self.num_modes = num_modes
        self.T = T

        # Initialize r0 parameters (baseline for each variate)
        if r0_init is None:
            r0_init_tensor = torch.randn(M, device=device) * 0.1
        elif isinstance(r0_init, (int, float)):
            r0_init_tensor = torch.full((M,), float(r0_init), device=device)
        else:
            r0_init_tensor = r0_init.to(device)

        if r0_init_tensor.shape != (M,):
            raise ValueError(
                f"r0_init must be scalar, None, or tensor of shape ({M},), got shape {r0_init_tensor.shape}"
            )

        self.r0 = nn.Parameter(r0_init_tensor)

        # Initialize Fourier coefficients: a_k and b_k for each mode k and variate m
        # Shape: (M, num_modes) for both cosine (a_k) and sine (b_k) coefficients
        if fourier_init is None:
            # Small random initialization for Fourier coefficients
            a_init = torch.randn(M, num_modes, device=device) * 0.1
            b_init = torch.randn(M, num_modes, device=device) * 0.1
        else:
            if fourier_init.shape != (M, 2, num_modes):
                raise ValueError(
                    f"fourier_init must have shape ({M}, 2, {num_modes}), got {fourier_init.shape}"
                )
            a_init = fourier_init[:, 0, :].to(device)
            b_init = fourier_init[:, 1, :].to(device)

        self.a_coeffs = nn.Parameter(a_init)  # Cosine coefficients
        self.b_coeffs = nn.Parameter(b_init)  # Sine coefficients

    def _fourier_series(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier series at given times for all variates.

        Args:
            t: Time points of shape (N,)

        Returns:
            Fourier series values of shape (N, M)
        """

        # Create frequency terms: 2*pi*k*t/T for k=1,2,...,num_modes
        # t shape: (N,), we need (N, num_modes)
        t_expanded = t.unsqueeze(1)  # (N, 1)
        k_values = torch.arange(
            1, self.num_modes + 1, device=self.device, dtype=torch.float32
        )  # (num_modes,)
        freq_terms = 2 * torch.pi * k_values * t_expanded / self.T  # (N, num_modes)

        cos_terms = torch.cos(freq_terms)  # (N, num_modes)
        sin_terms = torch.sin(freq_terms)  # (N, num_modes)

        # Vectorized computation for all variates
        # a_coeffs and b_coeffs have shape (M, num_modes)
        # cos_terms and sin_terms have shape (N, num_modes)
        # We want: sum_k(a_k_m * cos_k + b_k_m * sin_k) for each m

        # Use matrix multiplication: (N, num_modes) @ (num_modes, M) -> (N, M)
        cos_contribution = cos_terms @ self.a_coeffs.T  # (N, M)
        sin_contribution = sin_terms @ self.b_coeffs.T  # (N, M)

        return cos_contribution + sin_contribution

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-varying intensity mu(t) at given times.
        Intensity is activation(r0 + fourier_series(t)) for each variate.
        """
        N = t.shape[0]

        # Get baseline values (M,) and expand to (N, M)
        r0_expanded = self.r0.unsqueeze(0).expand(N, -1)

        # Get Fourier series values (N, M)
        fourier_vals = self._fourier_series(t)

        # Combine and apply activation
        combined = r0_expanded + fourier_vals
        base_intensity = self.activation(combined)

        # Apply active mask to zero out inactive variates
        active_mask = self._active_mask(t)  # Shape: (N, M)
        return base_intensity * active_mask.float()

    def report_parameters(self) -> str:
        """
        Report the current parameter state of the model for logging purposes.
        """
        r0_vals = self.r0.detach().cpu().numpy()
        a_vals = self.a_coeffs.detach().cpu().numpy()
        b_vals = self.b_coeffs.detach().cpu().numpy()

        report = f"FourierSeriesPoisson parameters:\n"
        report += f"  Period T: {self.T}\n"
        report += f"  Modes: {self.num_modes}\n"
        report += f"  r0 values: {r0_vals}\n"
        report += f"  Cosine coeffs: {a_vals}\n"
        report += f"  Sine coeffs: {b_vals}\n"

        return report

    def get_save_data(self) -> dict:
        """
        Get FourierSeriesPoisson-specific data for saving to file.
        """
        return {
            "r0": self.r0.detach().cpu().numpy(),
            "a": self.a_coeffs.detach().cpu().numpy(),
            "b": self.b_coeffs.detach().cpu().numpy(),
            "num_modes": self.num_modes,
            "T": self.T,
        }
