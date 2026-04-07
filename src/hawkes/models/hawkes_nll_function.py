import math
import torch

from typing import Any, Callable

from .. import utils
from ..utils import config, _torch_scan

# Need this for deterministic behavior in index_add_
# torch.use_deterministic_algorithms(True)


class HawkesLogSumIntensity(torch.autograd.Function):
    """
    Computes the log-sum intensity term in the NLL and its derivatives.
    """

    @staticmethod
    def forward(
        ctx: Any,
        seq: utils.EventSequence,
        mu_at_events: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        gamma_param: bool,
        intensity_states_fn: Callable,
        intensity_at_events_fn: Callable,
        batch_size: int,
    ):
        K = gamma.shape[0]

        nll_logsum_term = 0.0

        intensity_at_events = torch.empty(seq.N, device=seq.ti.device)
        intensity_states_at_events = torch.empty(seq.N, 1, K, device=seq.ti.device)

        prev_state = None
        for batch_start in range(0, seq.N, batch_size):
            batch_end = min(batch_start + batch_size, seq.N)

            batch_states, prev_state = intensity_states_fn(
                seq,
                bounds=(batch_start, batch_end),
                prev_state=prev_state,
                next_state=True,
                full_states=False,
            )

            batch_intensity = intensity_at_events_fn(
                seq,
                states=batch_states,
                full_intensity=False,
            )

            intensity_at_events[batch_start:batch_end] = batch_intensity
            intensity_states_at_events[batch_start:batch_end] = batch_states

            # Accumulate log intensities
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
            intensity_states_at_events if gamma_param else None,
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
            intensity_states_at_events,
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
                seq,
                alpha,
                gamma,
                intensity_at_events,
                alpha_prev_state,
                bs,
                be,
            )
            alpha_grad += alpha_batch_grad

            if ctx.gamma_param:
                gamma_batch_grad, gamma_prev_state = _compute_gamma_grad(
                    seq,
                    alpha,
                    gamma,
                    intensity_at_events,
                    intensity_states_at_events,
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

        return (None,) + (mu_grad, alpha_grad, gamma_grad) + (None,) * 9


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
        M_alpha[:, 0:1, :] = _torch_scan.state_left_mult(
            alpha_prev_state[:, None, :], M_alpha[:, 0:1, :]
        )  # Shape: [K, Nb+1, M+1]

    # Prefix scan to get cumulative products of transition matrices
    P_alpha = _torch_scan.prefix_scan(M_alpha, _torch_scan.state_left_mult, dim=1)

    alpha_states = P_alpha[:, -Nb:, 1:].permute(1, 2, 0)
    alpha_prev_state = P_alpha[:, -1, :]

    # Sum over events of each type
    alpha_term = gamma * alpha_states / intensity_b[:, None, None]  # Shape: [Nb, M, K]
    alpha_grad = torch.zeros(seq.M, seq.M, K, device=seq.ti.device)
    alpha_grad.index_add_(0, mi_b, alpha_term)
    alpha_grad = alpha_grad.permute(2, 1, 0)  # Shape: [K, M, M]

    return alpha_grad, alpha_prev_state


def _compute_gamma_grad(
    seq: utils.EventSequence,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    intensity_at_events: torch.Tensor,
    intensity_states_at_events: torch.Tensor,
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
    intensity_states_b = intensity_states_at_events[bs:be]  # Shape: [Nb, 1, K]

    # t_{i - 1} of shape [1, Nb, 1]
    ti1 = (
        torch.cat([seq.ti[0:1], seq.ti[0 : (be - 1)]])
        if bs == 0
        else seq.ti[(bs - 1) : (be - 1)]
    )

    # Δt_i of shape [1, Nb, 1]
    dti = ti_b - ti1

    ti1 = ti1[None, :, None]
    dti = dti[None, :, None]

    # α_{m_{i-1}} of shape [K, Nb, M]
    alpha_mi = (
        torch.cat(
            [
                torch.zeros_like(alpha[:, 0:1, :]),
                alpha[:, seq.mi[0 : (be - 1)], :],
            ],
            dim=1,
        )
        if bs == 0
        else alpha[:, seq.mi[(bs - 1) : (be - 1)], :]
    )

    # e^{-γ_k Δt_i}
    exp_dti = torch.exp(-gamma * dti).permute(2, 1, 0)  # Shape: [K, Nb, 1]

    # Build analogous transition matrices for d/dγ terms.
    M_gamma = torch.cat(
        [exp_dti, exp_dti * ti1 * alpha_mi], dim=2
    )  # Shape: [K, Nb, M+1]
    if gamma_prev_state is not None:
        M_gamma[:, 0:1, :] = _torch_scan.state_left_mult(
            gamma_prev_state[:, None, :], M_gamma[:, 0:1, :]
        )  # Shape: [K, Nb+1, M+1]

    P_gamma = _torch_scan.prefix_scan(M_gamma, _torch_scan.state_left_mult, dim=1)

    gamma_states = P_gamma[:, -Nb:, 1:].permute(1, 2, 0)
    gamma_prev_state = P_gamma[:, -1, :]

    # Gather the per-event gamma-state and intensity-state for the event's node
    gamma_states_at_events = torch.gather(
        gamma_states, dim=1, index=mi_b[:, None, None].expand(-1, 1, K)
    )  # Shape: [Nb, 1, K]

    # Compose the term required for gamma gradient from log-sum contribution
    gamma_grad_terms = (
        gamma * gamma_states_at_events
        + (1 - gamma * ti_b[:, None, None]) * intensity_states_b
    )  # Shape: [Nb, 1, K]
    gamma_grad = torch.sum(
        gamma_grad_terms.squeeze(1) / intensity_b.unsqueeze(-1), dim=0
    )  # Shape: [K]

    return gamma_grad, gamma_prev_state


class HawkesIntegratedExcitation(torch.autograd.Function):
    """
    Computes the integral of the intensity excitation term over [0, T].
    """

    @staticmethod
    def forward(
        ctx: Any,
        seq: utils.EventSequence,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        gamma_param: bool,
        batch_size: int,
    ):
        integrated_excitation = 0.0

        for bs in range(0, seq.N, batch_size):
            be = min(bs + batch_size, seq.N)

            ti_b, mi_b = seq.ti[bs:be], seq.mi[bs:be]

            alphas = alpha[:, mi_b, :].permute(1, 2, 0)[None]
            ti_b = ti_b[None, :, None, None]

            # 1 - e^{-γ_k (T - t_i)}
            exp_term = -torch.expm1(-gamma * (seq.T - ti_b))

            integrated_excitation += torch.sum(alphas * exp_term)

        ctx.M, ctx.T = seq.M, seq.T
        ctx.gamma_param = gamma_param
        ctx.batch_size = batch_size
        ctx.save_for_backward(seq.ti, seq.mi, alpha, gamma)

        return integrated_excitation

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (
            _ti_saved,
            _mi_saved,
            alpha,
            gamma,
        ) = ctx.saved_tensors
        K = gamma.shape[0]

        seq = utils.EventSequence(_ti_saved, _mi_saved, T=ctx.T, M=ctx.M)

        alpha_grad = torch.zeros_like(alpha)
        gamma_grad = torch.zeros_like(gamma) if ctx.gamma_param else None

        for bs in range(0, seq.N, ctx.batch_size):
            be = min(bs + ctx.batch_size, seq.N)

            ti_b, mi_b = seq.ti[bs:be], seq.mi[bs:be]

            # Sum over events of each type
            alpha_term = -torch.expm1(-gamma * (seq.T - ti_b[:, None]))
            alpha_exp_decay_sum = torch.zeros(seq.M, K, device=seq.ti.device)
            alpha_exp_decay_sum.index_add_(0, mi_b, alpha_term)  # Shape: [M, K]

            alpha_grad += alpha_exp_decay_sum.t()[..., None]

            if ctx.gamma_param:
                # Integral term gamma derivative
                gamma_exp_decay_sum = torch.sum(
                    alpha[:, mi_b, :]
                    * (seq.T - ti_b[None, :, None])
                    * torch.exp(-gamma[:, None, None] * (seq.T - ti_b[None, :, None])),
                    dim=(1, 2),
                )  # Shape: [K]

                gamma_grad += gamma_exp_decay_sum

        alpha_grad *= grad_output
        if ctx.gamma_param:
            gamma_grad *= grad_output

        return (None,) + (alpha_grad, gamma_grad) + (None,) * 2
