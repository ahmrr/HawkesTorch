"""Load the installation-built CUB extension for the Hawkes affine scan."""

from functools import lru_cache
from importlib.util import find_spec

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def _load_extension():
    # Find the installed library without importing it as a Python module
    # Cache the operation so later scans skip library lookup and loading
    binary = find_spec(f"{__package__}._cub_scan_cuda")
    if binary is None:
        raise RuntimeError(
            "The CUB extension is not installed. From the repository root, run "
            "HAWKES_BUILD_CUB=1 pip install --no-build-isolation . "
            "(add -e for an editable installation), or use "
            "prefix_scan_implementation='blelloch'."
        )
    try:
        # Loading the library registers torch.ops.hawkes_cub.scan
        torch.ops.load_library(binary.origin)
    except OSError as exc:
        raise RuntimeError(
            "Could not load the installed CUB extension. Reinstall with "
            "HAWKES_BUILD_CUB=1 and pip --no-build-isolation against your "
            "current PyTorch and CUDA toolkit."
        ) from exc
    return torch.ops.hawkes_cub.scan


class _AffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = _load_extension()(x)
        # The backward pass needs the input scales and the states reached at each event
        ctx.save_for_backward(x, result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, result = ctx.saved_tensors
        if x.shape[1] == 0:
            return grad.clone()

        # Each state gets its own gradient plus a contribution from the next event
        # Recurrence: lambda_i = grad_i + s_{i+1} * lambda_{i+1}
        decay = torch.cat((x[:, 1:, :1], torch.zeros_like(x[:, :1, :1])), dim=1)
        # Reverse the events so the same forward scan can accumulate gradients backward
        # The final zero scale above stops the recurrence at the end of the sequence
        transitions = torch.cat((decay, grad), dim=-1).flip(1)
        adjoint = _load_extension()(transitions)[:, :, 1:].flip(1)
        # Before the first event, the accumulated transform is the identity (1, 0, ...)
        initial = torch.zeros_like(result[:, :1])
        initial[:, :, 0] = 1
        previous = torch.cat((initial, result[:, :-1]), dim=1)
        # The scale multiplies every previous state component, so its gradient is a dot product
        scale_grad = (adjoint * previous).sum(dim=-1, keepdim=True)
        # Each additive value affects only its matching state component
        return torch.cat((scale_grad, adjoint[:, :, 1:]), dim=-1)


def affine_scan(x: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of [K, N, M+1] affine states; supports first derivatives."""
    if not x.is_cuda:
        raise ValueError("The CUB scan requires a CUDA tensor; use 'blelloch' on CPU.")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("The CUB scan supports float32 and float64 tensors.")
    if x.ndim != 3 or x.shape[-1] < 2:
        raise ValueError(
            "The CUB scan expects affine states of shape [K, N, M+1], M >= 1."
        )
    # CUB reads contiguous storage, so copy only when the input layout requires it
    x = x.contiguous()
    # Save backward intermediates only when autograd needs input gradients
    if torch.is_grad_enabled() and x.requires_grad:
        return _AffineScan.apply(x)
    return _load_extension()(x)
