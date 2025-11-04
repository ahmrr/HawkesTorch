import math
import torch

from typing import Any, Callable

from .. import utils
from ..utils import config, _torch_scan


class HawkesLogSumIntensity(torch.autograd.Function):
    """
    Custom autograd Function that computes the log-sum intensity term and derivatives for computing the Hawkes NLL.
    """

    @staticmethod
    def forward(
        ctx: Any,
        seq: utils.EventSequence,
        mu_at_events: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        gamma_param: bool,
        intensity_at_events_fun: Callable,
        batch_size: int,
    ):
        K = gamma.shape[0]

        nll_logsum_term = 0.0

        intensity_at_events = torch.empty(seq.N, device=seq.ti.device)
        intensity_states = torch.empty(seq.N, seq.M, K, device=seq.ti.device)

        prev_state = None
        for bs in range(0, seq.N, batch_size):
            be = min(bs + batch_size, seq.N)

            # TODO: Saving all states for backward is memory intensive
            (
                batch_intensity,
                batch_prev_state,
                batch_states,
            ) = intensity_at_events_fun(
                seq,
                batch_prev_state=prev_state,
                batch_start=bs,
                batch_end=be,
                return_full_intensity=False,
                return_last_state=True,
                return_all_states=True,
            )

            intensity_at_events[bs:be] = batch_intensity
            intensity_states[bs:be] = batch_states

            # Sum log intensities (the log lambda term in NLL)
            nll_logsum_term += torch.sum(torch.log(batch_intensity))

        # Save context for backward
        ctx.M, ctx.T = seq.M, seq.T
        ctx.gamma_param = gamma_param
        ctx.batch_size = batch_size
        ctx.save_for_backward(
            seq.ti,
            seq.mi,
            mu_at_events,
            alpha,
            gamma,
            intensity_at_events,
            intensity_states if gamma_param else None,
        )

        return nll_logsum_term

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (
            _ti_saved,
            _mi_saved,
            mu_at_events,
            alpha,
            gamma,
            intensity_at_events,
            intensity_states,
        ) = ctx.saved_tensors
        K = gamma.shape[0]

        seq = utils.EventSequence(_ti_saved, _mi_saved, T=ctx.T, M=ctx.M)

        alpha_prev_state = None
        gamma_prev_state = None

        # mu_grad = torch.zeros_like(mu_at_events)
        alpha_grad = torch.zeros_like(alpha)
        gamma_grad = torch.zeros_like(gamma) if ctx.gamma_param else None

        for bs in range(0, seq.N, ctx.batch_size):
            be = min(bs + ctx.batch_size, seq.N)

            alpha_batch_grad, alpha_prev_state = _compute_alpha_grad(
                seq, alpha, gamma, intensity_at_events, alpha_prev_state, bs, be
            )
            alpha_grad += alpha_batch_grad

            if ctx.gamma_param:
                gamma_batch_grad, gamma_prev_state = _compute_gamma_grad(
                    seq,
                    alpha,
                    gamma,
                    intensity_states,
                    intensity_at_events,
                    gamma_prev_state,
                    bs,
                    be,
                )
                gamma_grad += gamma_batch_grad

            # Gradient for mu with a constant base rate parameterization (unused)
            # mu_grad += torch.stack(
            #     [(1 / intensity_p).sum() for intensity_p in intensity_split],
            # )

        # Gradient for mu depends only on counts of reciprocals of intensities and T
        mu_grad = intensity_at_events.reciprocal()

        mu_grad *= grad_output
        alpha_grad *= grad_output
        if ctx.gamma_param:
            gamma_grad *= grad_output

        return (None,) + (mu_grad, alpha_grad, gamma_grad) + (None,) * 8


def _compute_alpha_grad(
    seq: utils.EventSequence,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    intensity_at_events: torch.Tensor,
    alpha_prev_state: torch.Tensor,
    batch_start: int,
    batch_end: int,
):
    K = gamma.shape[0]

    # Convenient shorthand
    bs, be = batch_start, batch_end
    Nb = be - bs

    ti_b, mi_b = seq.ti[bs:be], seq.mi[bs:be]  # Shape: [Nb]
    intensity_b = intensity_at_events[bs:be]  # Shape: [Nb]

    # e_{m_{i-1}} of shape [Nb, M]
    e_mi = torch.zeros(Nb, seq.M, dtype=seq.mi.dtype, device=seq.mi.device)
    if bs == 0:
        e_mi[1:].scatter_(1, seq.mi[0 : (be - 1), None], 1.0)
    else:
        e_mi.scatter_(1, seq.mi[(bs - 1) : (be - 1), None], 1.0)

    # Δt_i of shape [1, Nb, 1]
    dti = seq.ti[bs:be] - (
        torch.cat([seq.ti[0:1], seq.ti[0 : (be - 1)]])
        if bs == 0
        else seq.ti[(bs - 1) : (be - 1)]
    )
    dti = dti[None, :, None]

    # e^{-γ_k Δt_i}
    exp_dti = torch.exp(-gamma * dti).permute(2, 1, 0)  # Shape: [K, Nb, 1]

    # Transition matrices for the alpha-state recurrence
    M_alpha = torch.cat([exp_dti, exp_dti * e_mi], dim=2)  # Shape: [K, Nb, M+1]
    if alpha_prev_state is not None:
        M_alpha = torch.cat([alpha_prev_state[:, None, :], M_alpha], dim=1)

    # Prefix scan to get cumulative products of transition matrices
    P_alpha = _torch_scan.prefix_scan(M_alpha, _torch_scan.state_left_mult, dim=1)

    alpha_states = P_alpha[:, -Nb:, 1:].permute(1, 2, 0)
    alpha_prev_state = P_alpha[:, -1, :]

    # Sort events by their node index while preserving temporal order
    mi_sorted, idx = torch.sort(mi_b, stable=True)
    mi_counts = torch.bincount(mi_sorted, minlength=seq.M).tolist()

    # Reorder ti, intensity, and alpha_states according to the sorted node indices
    # Splitting yields per-node sequences (each subsequence remains time-ordered)
    # ti_split = torch.split(
    #     ti_b[:, idx, :], mi_counts, dim=1
    # )  # Shape: [M, 1, Nq, 1]
    intensity_split = torch.split(
        intensity_b[idx, None, None], mi_counts, dim=0
    )  # Shape: [M, Np, 1, 1]
    alpha_states_split = torch.split(
        alpha_states[idx], mi_counts, dim=0
    )  # Shape: [M, Np, M, K]

    # From the -log(lambda) term: sum over (gamma * state / lambda)
    alpha_grad_terms = [
        (gamma * state_p / intensity_p).sum(dim=0)
        for (state_p, intensity_p) in zip(alpha_states_split, intensity_split)
    ]
    alpha_grad = torch.stack(alpha_grad_terms, dim=0).permute(2, 1, 0)

    # Integral term alpha derivative (unused)
    # alpha_exp_decay_sum = torch.stack(
    #     [-torch.expm1(-gamma * (T - tiq)).sum(dim=1) for tiq in ti_split],
    #     dim=1,
    # )  # Shape: [M, K]

    return alpha_grad, alpha_prev_state


def _compute_gamma_grad(
    seq: utils.EventSequence,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    intensity_states: torch.Tensor,
    intensity_at_events: torch.Tensor,
    gamma_prev_state: torch.Tensor,
    batch_start: int,
    batch_end: int,
):
    K = gamma.shape[0]

    # Convenient shorthand
    bs, be = batch_start, batch_end
    Nb = be - bs

    ti_b, mi_b = seq.ti[bs:be], seq.mi[bs:be]  # Shape: [Nb]
    intensity_b = intensity_at_events[bs:be]  # Shape: [Nb]
    intensity_states_b = intensity_states[bs:be]  # Shape: [Nb, M, K]

    intensity_states_at_events = torch.gather(
        intensity_states[bs:be], dim=1, index=seq.mi[bs:be, None, None].expand(-1, 1, K)
    )  # Shape: [Nb, 1, K]

    # Δt_i of shape [1, Nb, 1]
    dti = seq.ti[bs:be] - (
        torch.cat([seq.ti[0:1], seq.ti[0 : (be - 1)]])
        if bs == 0
        else seq.ti[(bs - 1) : (be - 1)]
    )
    dti = dti[None, :, None]

    # e^{-γ_k Δt_i}
    exp_dti = torch.exp(-gamma * dti).permute(2, 1, 0)  # Shape: [K, Nb, 1]

    # Build analogous transition matrices for d/dγ terms.
    M_gamma = torch.cat(
        [exp_dti, -dti * intensity_states_b.permute(2, 0, 1)], dim=2
    )  # Shape: [K, Nb, M+1]
    if gamma_prev_state is not None:
        M_gamma = torch.cat([gamma_prev_state[:, None, :], M_gamma], dim=1)

    P_gamma = _torch_scan.prefix_scan(M_gamma, _torch_scan.state_left_mult, dim=1)

    gamma_states = P_gamma[:, -Nb:, 1:].permute(1, 2, 0)
    gamma_prev_state = P_gamma[:, -1, :]

    # Gather the per-event gamma-state and intensity-state for the event's node
    gamma_states_at_events = torch.gather(
        gamma_states, dim=1, index=mi_b[:, None, None].expand(-1, 1, K)
    )  # Shape: [Nb, 1, K]

    # Compose the term required for gamma gradient from log-sum contribution
    gamma_grad_terms = (
        intensity_states_at_events + gamma * gamma_states_at_events
    )  # Shape: [Nb, 1, K]
    gamma_grad = torch.sum(
        gamma_grad_terms.squeeze(1) / intensity_b.unsqueeze(-1),
        dim=0,
    )  # Shape: [K]

    # Integral term gamma derivative (unused)
    # gamma_exp_decay_sum = torch.sum(
    #     alpha[:, mi, :]
    #     * (T - ti)
    #     * torch.exp(-gamma[:, None, None] * (T - ti)),
    #     dim=(1, 2),
    # )  # Shape: [K]

    return gamma_grad, gamma_prev_state
