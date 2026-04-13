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
class PoissonGlobalFourierPenalty:
    baseline: Penalty | None = None
    weight: Penalty | None = None
    cosine: Penalty | None = None
    sine: Penalty | None = None


class PoissonGlobalFourier(PoissonBase):

    def __init__(
        self,
        M: int,
        T: float,
        num_modes: int,
        r0_init: torch.Tensor | float | None = None,
        w_init: torch.Tensor | float | None = None,
        fourier_init: torch.Tensor | None = None,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
        penalization: Penalty = PoissonGlobalFourierPenalty(),
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        super().__init__(M, t_start, t_end, penalization, runtime_config, device)

        self.t = transformation

        self.num_modes = num_modes
        self.T = T

        if r0_init is None:
            r0_tensor = torch.randn(M, device=device) * 0.1
        elif isinstance(r0_init, (int, float)):
            r0_tensor = torch.full((M,), float(r0_init), device=device)
        else:
            r0_tensor = r0_init.to(device)

        if r0_tensor.shape != (M,):
            raise ValueError(
                f"r0_init must be scalar, None, or tensor of shape ({M},), got {r0_tensor.shape}"
            )
        self.r0 = nn.Parameter(r0_tensor)

        if w_init is None:
            w_tensor = torch.ones(M, device=device)
        elif isinstance(w_init, (int, float)):
            w_tensor = torch.full((M,), float(w_init), device=device)
        else:
            w_tensor = w_init.to(device)

        if w_tensor.shape != (M,):
            raise ValueError(
                f"w_init must be scalar, None, or tensor of shape ({M},), got {w_tensor.shape}"
            )
        self.w = nn.Parameter(w_tensor)

        if fourier_init is None:
            a_init = torch.randn(num_modes, device=device) * 0.1
            b_init = torch.randn(num_modes, device=device) * 0.1
        else:
            if fourier_init.shape != (2, num_modes):
                raise ValueError(
                    f"fourier_init must have shape (2, {num_modes}), got {fourier_init.shape}"
                )
            a_init = fourier_init[0].to(device)
            b_init = fourier_init[1].to(device)

        self.a_coeffs = nn.Parameter(a_init)
        self.b_coeffs = nn.Parameter(b_init)

    def _fourier_series(self, t: torch.Tensor) -> torch.Tensor:
        t_expanded = t.unsqueeze(1)
        k_values = torch.arange(
            1, self.num_modes + 1, device=self.device, dtype=torch.float32
        )
        freq_terms = 2 * torch.pi * k_values * t_expanded / self.T

        cos_contribution = torch.cos(freq_terms) @ self.a_coeffs
        sin_contribution = torch.sin(freq_terms) @ self.b_coeffs

        return cos_contribution + sin_contribution

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        N = t.shape[0]

        fourier_vals = self._fourier_series(t)
        weighted = fourier_vals.unsqueeze(1) * self.w.unsqueeze(0)
        r0_expanded = self.r0.unsqueeze(0).expand(N, -1)

        combined = r0_expanded + weighted
        base_intensity = self.t.forward(combined)

        active_mask = self._active_mask(t)
        return base_intensity * active_mask.float()

    def penalty(self) -> torch.Tensor:
        penalty = 0.0

        if self.penalization.baseline is not None:
            penalty += self.penalization.baseline(self.r0)
        if self.penalization.weight is not None:
            penalty += self.penalization.weight(self.w)
        if self.penalization.cosine is not None:
            penalty += self.penalization.cosine(self.a_coeffs)
        if self.penalization.sine is not None:
            penalty += self.penalization.sine(self.b_coeffs)

        return penalty

    def report_parameters(self) -> str:
        r0_vals = self.r0.detach().cpu().numpy()
        w_vals = self.w.detach().cpu().numpy()
        a_vals = self.a_coeffs.detach().cpu().numpy()
        b_vals = self.b_coeffs.detach().cpu().numpy()

        return f"[r0_mean={r0_vals.mean():.4f}, weight_mean={w_vals.mean():.4f}, cosine_mean={a_vals.mean():.4f}, sine_mean={b_vals.mean():.4f}]"

    def get_save_data(self) -> dict:
        return {
            "r0": self.r0.detach().cpu().numpy(),
            "w": self.w.detach().cpu().numpy(),
            "a": self.a_coeffs.detach().cpu().numpy(),
            "b": self.b_coeffs.detach().cpu().numpy(),
            "num_modes": self.num_modes,
            "T": self.T,
        }
