from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Penalty(ABC):
    """
    Abstract base class for parameter penalties.

    Subclasses implement `compute(param)`, which returns the unweighted
    penalty value. The base class handles weighting, early-exit on w=0,
    and addition of same-type penalties.

    The intended use is:
        penalty = SomePenalty(w=1e-3)
        loss = loss + penalty(param)
    """

    weight: float = 0.0

    @abstractmethod
    def compute(self, param: torch.Tensor) -> torch.Tensor:
        """Unweighted penalty value."""
        ...

    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        if self.weight == 0:
            return torch.tensor(0.0, device=param.device)
        return self.weight * self.compute(param)


@dataclass
class SumPenalty(Penalty):
    """
    A sum of penalties.

    Example:
        penalty = SumPenalty([NormPenalty(1, weight=1e-3), NormPenalty("nuc", weight=1e-4)])
    """

    penalties: list[Penalty] = field(default_factory=list)
    weight: float = 1.0  # global rescaling; usually left at 1.0

    def compute(self, param: torch.Tensor) -> torch.Tensor:
        return sum(p(param) for p in self.penalties)

    def __add__(self, other):
        if isinstance(other, SumPenalty):
            return SumPenalty(penalties=self.penalties + other.penalties)
        if isinstance(other, Penalty):
            return SumPenalty(penalties=self.penalties + [other])
        return NotImplemented


@dataclass
class NormPenalty(Penalty):
    """
    General L-norm / nuclear norm / max-norm penalty implementation.

    Args:
        order:       "inf", "nuc", or a numeric exponent (e.g., 1 for L1, 2 for L2).
        weight:      Penalty weight.
        hinge:       Threshold — penalty is applied to elements with |x| <= hinge.
        kwargs:      Forwarded to the underlying torch call (e.g. dim for "nuc").

    Examples:
        NormPenalty(1, weight=1e-3)               # L1
        NormPenalty(2, weight=1e-3)               # L2 (weight decay)
        NormPenalty("inf", weight=0.1, hinge=0.5)     # thresholded max-norm
        NormPenalty("nuc", weight=1e-4, dim=(0,1))
    """

    order: int | float | str = 2
    weight: float = 0.0
    hinge: float = float("inf")
    kwargs: dict = field(default_factory=dict)

    def compute(self, param: torch.Tensor) -> torch.Tensor:
        param_abs = param.abs()
        clipped = torch.where(
            param_abs < self.hinge, param_abs, torch.zeros_like(param)
        )
        match self.order:
            case "inf":
                return clipped.max()
            case "nuc":
                dim = self.kwargs.get("dim", self._infer_matrix_dim(param))
                return torch.linalg.matrix_norm(param, ord="nuc", dim=dim).sum()
            case _ if isinstance(self.order, (int, float)):
                return clipped.pow(self.order).sum()
            case _:
                raise ValueError(f"Unsupported norm type: {self.order!r}")

    @staticmethod
    def _infer_matrix_dim(param: torch.Tensor) -> tuple[int, int]:
        if param.ndim < 2:
            raise ValueError("Nuclear norm requires at least a 2D tensor.")
        return (-2, -1)


def L1Penalty(weight: float = 0.0, hinge: float = float("inf")) -> NormPenalty:
    """L1 norm penalty."""
    return NormPenalty(1, weight, hinge)


def L2Penalty(weight: float = 0.0, hinge: float = float("inf")) -> NormPenalty:
    """L2 norm penalty."""
    return NormPenalty(2, weight, hinge)


def MaxPenalty(weight: float = 0.0, hinge: float = float("inf")) -> NormPenalty:
    """Max-norm penalty."""
    return NormPenalty("inf", weight, hinge)


def NuclearPenalty(
    weight: float = 0.0, dim: tuple[int, int] | None = None
) -> NormPenalty:
    """Nuclear norm penalty."""
    kwargs = {"dim": dim} if dim is not None else {}
    return NormPenalty("nuc", weight, float("inf"), kwargs=kwargs)
