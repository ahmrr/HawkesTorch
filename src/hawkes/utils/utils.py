import torch

from dataclasses import dataclass


@dataclass
class EventSequence:
    ti: torch.Tensor  # Event times with shape [N]
    mi: torch.Tensor  # Event nodes/dimensions with shape [N]

    T: float | None = None  # Observation horizon
    M: int | None = None  # Number of node types

    validate: bool = True

    def __post_init__(self):
        assert self.ti.ndim == 1 and self.mi.ndim == 1, "ti and mi must be 1D tensors"
        assert self.ti.shape == self.mi.shape, "ti and mi must have same shape"

        # Compute defaults
        if self.M is None:
            self.M = int(self.mi.max().item()) + 1
        if self.T is None or self.T == float("inf"):
            self.T = float(self.ti.max().item())

        self.N = self.ti.shape[0]

        if self.validate:
            assert torch.all(self.ti >= 0) and torch.all(
                self.ti <= self.T
            ), f"Event times must be in [0, T={self.T}]"
            assert torch.all(self.mi >= 0) and torch.all(
                self.mi < self.M
            ), f"Event nodes must be in [0, M={self.M - 1}]"
            assert torch.all(
                self.ti[:-1] <= self.ti[1:]
            ), "Event times must be sorted non-decreasingly"

    def __getitem__(self, idx):
        """Return a sliced EventSequence with same T, M, and updated N."""
        if isinstance(idx, slice) or torch.is_tensor(idx) or isinstance(idx, list):
            return EventSequence(
                ti=self.ti[idx],
                mi=self.mi[idx],
                T=self.T,
                M=self.M,
                validate=self.validate,
            )
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def to(self, device):
        """Move tensors to a specified device."""
        return EventSequence(
            ti=self.ti.to(device),
            mi=self.mi.to(device),
            T=self.T,
            M=self.M,
            validate=self.validate,
        )

    def cpu(self):
        return self.to("cpu")

    def __len__(self):
        return self.N

    def __repr__(self):
        return (
            f"EventSequence(N={self.N}, M={self.M}, T={self.T:.3f}, "
            f"device={self.ti.device}, dtype={self.ti.dtype})"
        )
