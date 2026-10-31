import torch

from typing import Any, Callable

from ... import utils
from ...utils import _torch_scan

# Need this for deterministic behavior in index_add_
torch.use_deterministic_algorithms(False)


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
        batch_size: int,
        scan_implementation: str,
    ):
        K = gamma.shape[0]

        nll_logsum_term = 0.0

        intensity_at_events = mu_at_events.new_empty(seq.N)
        intensity_states_at_events = (
            gamma.new_empty(seq.N, 1, K) if gamma_param else None
        )

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

            batch_intensity = mu_at_events[batch_start:batch_end] + (
                gamma * batch_states.squeeze(1)
            ).sum(dim=-1)

            intensity_at_events[batch_start:batch_end] = batch_intensity
            if gamma_param:
                intensity_states_at_events[batch_start:batch_end] = batch_states

            # Accumulate log intensities
            nll_logsum_term += torch.sum(torch.log(batch_intensity))

        # Save context for backward
        ctx.M, ctx.T = seq.M, seq.T
        ctx.gamma_param = gamma_param
        ctx.batch_size = batch_size
        ctx.scan_implementation = scan_implementation
        ctx.save_for_backward(
            seq.ti,
            seq.mi,
            alpha,
            gamma,
            intensity_at_events,
            intensity_states_at_events,
        )

        return nll_logsum_term

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (
            _ti_saved,
            _mi_saved,
            alpha,
            gamma,
            intensity_at_events,
            intensity_states_at_events,
        ) = ctx.saved_tensors
        seq = utils.EventSequence(
            _ti_saved, _mi_saved, T=ctx.T, M=ctx.M, _validate=False
        )

        alpha_prev_state = None
        gamma_prev_state = None

        alpha_grad = torch.zeros_like(alpha)
        gamma_grad = torch.zeros_like(gamma) if ctx.gamma_param else None

        for bs in range(0, seq.N, ctx.batch_size):
            be = min(bs + ctx.batch_size, seq.N)
            previous_times = (
                torch.cat((seq.ti[:1], seq.ti[: be - 1]))
                if bs == 0
                else seq.ti[bs - 1 : be - 1]
            )
            # Both derivative recurrences use the same decay factors.
            exp_dti = torch.exp(
                -gamma[:, None, None] * (seq.ti[bs:be] - previous_times)[None, :, None]
            )

            alpha_batch_grad, alpha_prev_state = _compute_alpha_grad(
                seq,
                alpha,
                gamma,
                intensity_at_events,
                alpha_prev_state,
                bs,
                be,
                exp_dti,
                ctx.scan_implementation,
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
                    exp_dti,
                    previous_times,
                    ctx.scan_implementation,
                )
                gamma_grad += gamma_batch_grad

        mu_grad = intensity_at_events.reciprocal()

        mu_grad *= grad_output
        alpha_grad *= grad_output
        if ctx.gamma_param:
            gamma_grad *= grad_output

        return (None, mu_grad, alpha_grad, gamma_grad) + (None,) * 4


def _compute_alpha_grad(
    seq: utils.EventSequence,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    intensity_at_events: torch.Tensor,
    alpha_prev_state: torch.Tensor,
    batch_start: int,
    batch_end: int,
    exp_dti: torch.Tensor,
    scan_implementation: str = "blelloch",
):
    K = gamma.shape[0]

    # Convenient shorthand
    bs, be = batch_start, batch_end
    Nb = be - bs

    mi_b = seq.mi[bs:be]
    intensity_b = intensity_at_events[bs:be]  # Shape: [Nb]

    # Write decay * one_hot(previous_mark) directly into the transitions.
    M_alpha = exp_dti.new_zeros(K, Nb, alpha.shape[1] + 1)
    M_alpha[..., :1] = exp_dti
    first = 1 if bs == 0 else 0
    columns = (seq.mi[max(bs - 1, 0) : be - 1] + 1)[None, :, None].expand(K, -1, -1)
    M_alpha[:, first:].scatter_(2, columns, exp_dti[:, first:])
    if alpha_prev_state is not None:
        M_alpha[:, 0:1, :] = _torch_scan.state_left_mult(
            alpha_prev_state[:, None, :], M_alpha[:, 0:1, :]
        )

    # Prefix scan to get cumulative products of transition matrices
    P_alpha = _torch_scan.prefix_scan(
        M_alpha,
        _torch_scan.state_left_mult,
        dim=1,
        implementation=scan_implementation,
    )

    alpha_states = P_alpha[:, -Nb:, 1:].permute(1, 2, 0)
    alpha_prev_state = P_alpha[:, -1, :].clone()

    # Sum over events of each type
    alpha_term = gamma * alpha_states / intensity_b[:, None, None]  # Shape: [Nb, M, K]
    alpha_grad = alpha.new_zeros(alpha.shape[2], alpha.shape[1], K)
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
    exp_dti: torch.Tensor,
    previous_times: torch.Tensor,
    scan_implementation: str = "blelloch",
):
    K = gamma.shape[0]

    # Convenient shorthand
    bs, be = batch_start, batch_end
    Nb = be - bs

    ti_b, mi_b = seq.ti[bs:be], seq.mi[bs:be]  # Shape: [Nb]
    intensity_b = intensity_at_events[bs:be]  # Shape: [Nb]
    intensity_states_b = intensity_states_at_events[bs:be]  # Shape: [Nb, 1, K]

    ti1 = previous_times[None, :, None]

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

    # Build analogous transition matrices for d/dγ terms.
    M_gamma = torch.cat(
        [exp_dti, exp_dti * ti1 * alpha_mi], dim=2
    )  # Shape: [K, Nb, M+1]
    if gamma_prev_state is not None:
        M_gamma[:, 0:1, :] = _torch_scan.state_left_mult(
            gamma_prev_state[:, None, :], M_gamma[:, 0:1, :]
        )

    P_gamma = _torch_scan.prefix_scan(
        M_gamma,
        _torch_scan.state_left_mult,
        dim=1,
        implementation=scan_implementation,
    )

    gamma_states = P_gamma[:, -Nb:, 1:].permute(1, 2, 0)
    gamma_prev_state = P_gamma[:, -1, :].clone()

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
        alpha_totals = alpha.sum(dim=-1)  # Sum destinations before indexing events.

        for bs in range(0, seq.N, batch_size):
            be = min(bs + batch_size, seq.N)

            ti_b, mi_b = seq.ti[bs:be], seq.mi[bs:be]

            # 1 - e^{-γ_k (T - t_i)}
            exp_term = -torch.expm1(-gamma[:, None] * (seq.T - ti_b))
            integrated_excitation += torch.sum(alpha_totals[:, mi_b] * exp_term)

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

        seq = utils.EventSequence(
            _ti_saved, _mi_saved, T=ctx.T, M=ctx.M, _validate=False
        )

        alpha_grad = torch.zeros_like(alpha)
        gamma_grad = torch.zeros_like(gamma) if ctx.gamma_param else None
        alpha_totals = alpha.sum(dim=-1) if ctx.gamma_param else None

        for bs in range(0, seq.N, ctx.batch_size):
            be = min(bs + ctx.batch_size, seq.N)

            ti_b, mi_b = seq.ti[bs:be], seq.mi[bs:be]

            # Sum over events of each type
            alpha_term = -torch.expm1(-gamma * (seq.T - ti_b[:, None]))
            alpha_exp_decay_sum = alpha.new_zeros(alpha.shape[1], K)
            alpha_exp_decay_sum.index_add_(0, mi_b, alpha_term)  # Shape: [M, K]

            alpha_grad += alpha_exp_decay_sum.t()[..., None]

            if ctx.gamma_param:
                # Integral term gamma derivative
                gamma_exp_decay_sum = torch.sum(
                    alpha_totals[:, mi_b]
                    * (seq.T - ti_b)
                    * torch.exp(-gamma[:, None] * (seq.T - ti_b)),
                    dim=1,
                )  # Shape: [K]

                gamma_grad += gamma_exp_decay_sum

        alpha_grad *= grad_output
        if ctx.gamma_param:
            gamma_grad *= grad_output

        return (None,) + (alpha_grad, gamma_grad) + (None,) * 2
