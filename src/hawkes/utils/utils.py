import torch

from typing import Self
from dataclasses import dataclass, field


@dataclass
class EventSequence:
    """
    A sorted sequence of marked temporal events over [0, T].

    Each event is a (time, node) pair. Times must be non-decreasing;
    nodes are integer labels in [0, M).

    Args:
        ti: Event times, shape [N].
        mi: Event nodes/marks, shape [N], integer dtype.
        T:  Observation horizon. Defaults to the last event time.
        M:  Number of node types. Defaults to max(mi) + 1.

    Example:
        >>> seq = EventSequence(
        ...     ti=torch.tensor([0.1, 0.5, 0.9]),
        ...     mi=torch.tensor([0, 1, 0]),
        ...     T=1.0,
        ... )
        >>> seq[seq.mi == 0]          # filter to node 0
        >>> seq.split_by_node()[0]    # same, for all nodes at once
    """

    ti: torch.Tensor  # Event times,  shape [N]
    mi: torch.Tensor  # Event marks,  shape [N], integer
    T: float | None = None  # Observation horizon
    M: int | None = None  # Number of node types

    _validate: bool = field(default=True, repr=False)

    def __post_init__(self):
        assert self.ti.ndim == 1 and self.mi.ndim == 1, "ti and mi must be 1D tensors"
        assert self.ti.shape == self.mi.shape, "ti and mi must have the same shape"

        self.N = self.ti.shape[0]

        if self.M is None:
            self.M = int(self.mi.max().item()) + 1 if self.N > 0 else 0
        if self.T is None or self.T == float("inf"):
            self.T = float(self.ti.max().item()) if self.N > 0 else 0.0

        if self._validate:
            assert torch.all(self.ti >= 0) and torch.all(
                self.ti <= self.T
            ), f"Event times must be in [0, T={self.T}]"
            assert torch.all(self.mi >= 0) and torch.all(
                self.mi < self.M
            ), f"Event nodes must be in [0, M={self.M - 1}]"
            assert torch.all(
                torch.diff(self.ti) >= 0
            ), "Event times must be sorted non-decreasingly"

    def node(self, m: int) -> Self:
        """Return a new sequence containing only events on node m."""
        return self[self.mi == m]

    def split_by_node(self) -> list[Self]:
        """Return a list of M sequences, one per node, preserving T and M."""
        return [self.node(m) for m in range(self.M)]

    def window(self, t_start: float, t_end: float) -> Self:
        """
        Restrict to events in the half-open interval [t_start, t_end).

        T is updated to t_end so the returned sequence has a well-defined horizon.
        """

        mask = (self.ti >= t_start) & (self.ti < t_end)
        return EventSequence(ti=self.ti[mask], mi=self.mi[mask], T=t_end, M=self.M)

    def __getitem__(self, idx):
        """
        Slice or filter the sequence.

        Integer index returns (ti, mi) scalars; everything else
        (slice, bool/index tensor, list) returns a new EventSequence.
        T and M are preserved so downstream models see a consistent horizon.
        """

        if isinstance(idx, int):
            return self.ti[idx], self.mi[idx]
        if isinstance(idx, (slice, list)) or torch.is_tensor(idx):
            return EventSequence(ti=self.ti[idx], mi=self.mi[idx], T=self.T, M=self.M)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __add__(self, other: Self) -> Self:
        """
        Merge two sequences, re-sorting by time.

        T and M are taken as the elementwise max of both sequences,
        so the merged sequence covers the full observed span.
        """

        if not isinstance(other, EventSequence):
            return NotImplemented
        ti_cat = torch.cat([self.ti, other.ti])
        mi_cat = torch.cat([self.mi, other.mi])
        order = torch.argsort(ti_cat, stable=True)
        return EventSequence(
            ti=ti_cat[order],
            mi=mi_cat[order],
            T=max(self.T, other.T),
            M=max(self.M, other.M),
        )

    def __len__(self) -> int:
        return self.N

    def __repr__(self) -> str:
        return (
            f"EventSequence(N={self.N}, M={self.M}, T={self.T:.3f}, "
            f"device={self.ti.device}, dtype={self.ti.dtype})"
        )

    def to(self, device) -> Self:
        """Move tensors to device; T and M are scalars and stay on CPU."""
        return EventSequence(
            ti=self.ti.to(device), mi=self.mi.to(device), T=self.T, M=self.M
        )

    def cpu(self) -> Self:
        return self.to("cpu")
