import math
import torch

from typing import Callable


PREFIX_SCAN_IMPLEMENTATION = "BL"

# TODO: Pad outside of scan function instead of repeatedly inside


def state_left_mult(B: torch.Tensor, A: torch.Tensor):
    """Left-multiply operation for compressed transition matrices"""

    result = torch.empty_like(A)
    result[..., 0:1] = A[..., 0:1] * B[..., 0:1]  # Sa * Sb
    result[..., 1:] = A[..., 0:1] * B[..., 1:] + A[..., 1:]  # Sa * Vb + Va
    return result


def _prefix_scan_sequential(
    x: torch.Tensor,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dim: int = 0,
):
    """
    Naive sequential inclusive prefix scan (reference implementation).

    Args:
        x: Sequence tensor of shape [*batch_dims, n, *op_dims]
        op: Broadcastable associative operation on two tensors of shape [..., *op_dims]
        dim: Dimension over which to compute the parallel scan

    Returns:
        torch.Tensor: Tensor of the same shape as x, containing the inclusive prefix scan
    """

    x = x.movedim(dim, -1)
    out = x.clone()

    for i in range(1, out.shape[-1]):
        out[..., i] = op(out[..., i - 1], out[..., i])

    return out.movedim(-1, dim)


def _prefix_scan_hillis_steele(
    x: torch.Tensor,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dim: int = 0,
    pad_value: torch.Tensor | float = 0,
):
    """
    Adapted from https://github.com/glassroom/torch_parallel_scan/blob/main/torch_parallel_scan/torch_parallel_scan.py

    Span-efficient inclusive prefix scan using Hillis-Steele algorithm.
    This is better for smaller sequences, particularly when

        # of available parallel processors <= length of sequence

    Args:
        x: Tensor of shape [*batch_dims, seq_len, *op_dims]
        op: Broadcastable binary associative function
        dim: Dimension over which to compute the parallel scan
        pad_value: For padding sequences to a power of two

    Output:
        torch.Tensor: Tensor of the same shape as x, containing the inclusive prefix scan
    """

    x = x.movedim(dim, -1)
    other_dims, seq_len = (x.shape[:-1], x.size(-1))
    n_powers_of_2 = int(math.ceil(math.log2(seq_len)))
    n_pads = 2**n_powers_of_2 - seq_len
    x = torch.nn.functional.pad(x, (0, n_pads), value=pad_value)

    for n in (2 ** torch.arange(n_powers_of_2)).tolist():
        x = x.view(*other_dims, -1, n * 2)

        last_on_L = x[..., (n - 1) : n].movedim((-2, -1), (dim - 1, dim))
        all_on_R = x[..., n:].movedim((-2, -1), (dim - 1, dim))

        updated_on_R = op(last_on_L, all_on_R).movedim((dim - 1, dim), (-2, -1))

        x = torch.cat([x[..., :n], updated_on_R], dim=-1)

    x = x.view(*other_dims, -1)
    x = x[..., :seq_len]

    return x.movedim(-1, dim)


def _prefix_scan_blelloch(
    x: torch.Tensor,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dim: int = 0,
    pad_value: torch.Tensor | float = 0,
    identity: torch.Tensor | float = 0,
):
    """
    Work-efficient inclusive parallel prefix scan using the Blelloch algorithm.

    Args:
        x: Sequence tensor of shape [*batch_dims, n, *op_dims]
        op: Broadcastable associative operation on two tensors of shape [..., *op_dims]
        dim: Dimension over which to compute the parallel scan
        pad_value: Used for padding sequences to a power of two,
            either a float or a tensor of shape [*batch_dims, *op_dims]
        identity: Identity element under op used for the the downsweep

    Returns:
        torch.Tensor: Tensor of the same shape as x, containing the inclusive prefix scan
    """

    align_dims = lambda t: t.movedim((-2, -1), (dim - 1, dim))  # Move op dims to last
    reset_dims = lambda t: t.movedim((dim - 1, dim), (-2, -1))  # Undo moving op dims
    apply_op = lambda a, b: reset_dims(
        op(
            align_dims(a),
            align_dims(b),
        )
    )

    x = x.movedim(dim, -1)  # Move scan dim to back for convenience
    other_dims, n = x.shape[:-1], x.shape[-1]

    # pad to next power of 2
    L = 2 ** math.ceil(math.log2(n))
    if L - n > 0:
        if isinstance(pad_value, torch.Tensor):
            pad_seq = pad_value.unsqueeze(-1).expand(*pad_value.shape, N)
            x = torch.cat([x, pad_seq], dim=-1)
        else:
            x = torch.nn.functional.pad(x, (0, L - n), value=pad_value)

    # Upsweep algorithm (reduce)

    d = 1  # = 2 ** (tree depth)
    while d < L:
        y = x.view(*other_dims, -1, 2 * d)

        left = y[..., d - 1 : d]
        right = y[..., 2 * d - 1 : 2 * d]

        joined = apply_op(left, right)

        y[..., 2 * d - 1 : 2 * d] = joined
        x = y.reshape(*other_dims, L)
        d *= 2

    # For inclusive scan, save the total and zero the root
    total = x[..., L - 1 : L]
    x[..., L - 1] = identity

    # Downsweep algorithm (construct prefixes)

    d = L // 2
    while d > 0:
        y = x.view(*other_dims, -1, 2 * d)

        left = y[..., d - 1 : d]
        right = y[..., 2 * d - 1 : 2 * d]

        new_left = right
        new_right = apply_op(right, left)

        y[..., d - 1 : d] = new_left
        y[..., 2 * d - 1 : 2 * d] = new_right
        x = y.reshape(*other_dims, L)
        d //= 2

    # Convert exclusive scan to inclusive by appending total
    x = torch.cat([x[..., 1:L], total], dim=-1)

    # Drop padding and move dimension back
    x = x[..., :n]
    return x.movedim(-1, dim)


match PREFIX_SCAN_IMPLEMENTATION:
    case "HS":
        prefix_scan = _prefix_scan_hillis_steele
    case "BL":
        prefix_scan = _prefix_scan_blelloch
    case "PY":
        prefix_scan = lambda x, op, dim=0: torch._higher_order_ops.associative_scan(
            op, x, dim=dim, combine_mode="generic"
        )
    case _:
        prefix_scan = _prefix_scan_sequential
