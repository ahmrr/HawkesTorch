import math
import time
import torch
import typing
import logging
import torch.nn as nn
from dataclasses import dataclass

from . import PoissonBase
from .. import utils
from ..utils import config


@dataclass
class PoissonGlobalFourierPenalty:
    l1_baseline: float = 0.0
    l1_baseline_hinge: float = float("inf")
    l2_baseline: float = 0.0
    l2_baseline_hinge: float = float("inf")

    l1_weight: float = 0.0
    l1_weight_hinge: float = float("inf")
    l2_weight: float = 0.0
    l2_weight_hinge: float = float("inf")

    l1_cosine: float = 0.0
    l1_cosine_hinge: float = float("inf")
    l2_cosine: float = 0.0
    l2_cosine_hinge: float = float("inf")

    l1_sine: float = 0.0
    l1_sine_hinge: float = float("inf")
    l2_sine: float = 0.0
    l2_sine_hinge: float = float("inf")


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
        penalization=PoissonGlobalFourierPenalty(),
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str = "cpu",
    ):
        super().__init__(M, t_start, t_end, transformation, runtime_config, device)

        self.penalization = penalization
        self.num_modes = num_modes
        self.T = T

        if r0_init is None:
            r0_tensor = torch.randn(M, device=device) * 0.01
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
            a_init = torch.randn(num_modes, device=device) * 0.01
            b_init = torch.randn(num_modes, device=device) * 0.01
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

        penalty += utils.norm_penalty(
            self.r0,
            1.0,
            self.penalization.l1_baseline,
            self.penalization.l1_baseline_hinge,
        )
        penalty += utils.norm_penalty(
            self.r0,
            2.0,
            self.penalization.l2_baseline,
            self.penalization.l2_baseline_hinge,
        )

        penalty += utils.norm_penalty(
            self.w, 1.0, self.penalization.l1_weight, self.penalization.l1_weight_hinge
        )
        penalty += utils.norm_penalty(
            self.w, 2.0, self.penalization.l2_weight, self.penalization.l2_weight_hinge
        )

        penalty += utils.norm_penalty(
            self.a_coeffs,
            1.0,
            self.penalization.l1_cosine,
            self.penalization.l1_cosine_hinge,
        )
        penalty += utils.norm_penalty(
            self.a_coeffs,
            2.0,
            self.penalization.l2_cosine,
            self.penalization.l2_cosine_hinge,
        )

        penalty += utils.norm_penalty(
            self.b_coeffs,
            1.0,
            self.penalization.l1_sine,
            self.penalization.l1_sine_hinge,
        )
        penalty += utils.norm_penalty(
            self.b_coeffs,
            2.0,
            self.penalization.l2_sine,
            self.penalization.l2_sine_hinge,
        )

        return penalty

    def report_parameters(self) -> str:
        r0_vals = self.r0.detach().cpu().numpy()
        w_vals = self.w.detach().cpu().numpy()
        a_vals = self.a_coeffs.detach().cpu().numpy()
        b_vals = self.b_coeffs.detach().cpu().numpy()

        report = "PoissonGlobalFourier parameters:\n"
        report += f"  Period T:            {self.T}\n"
        report += f"  Modes:               {self.num_modes}\n"
        report += f"  r0 mean:             {r0_vals.mean():.4f}\n"
        report += f"  w mean:              {w_vals.mean():.4f}\n"
        report += f"  Cosine coeffs mean:  {a_vals.mean():.4f}\n"
        report += f"  Sine coeffs mean:    {b_vals.mean():.4f}\n"

        return report

    def get_save_data(self) -> dict:
        return {
            "r0": self.r0.detach().cpu().numpy(),
            "w": self.w.detach().cpu().numpy(),
            "a": self.a_coeffs.detach().cpu().numpy(),
            "b": self.b_coeffs.detach().cpu().numpy(),
            "num_modes": self.num_modes,
            "T": self.T,
        }
