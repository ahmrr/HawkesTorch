import torch
import math
import time
from torch import cuda


def state_mult(B: torch.Tensor, A: torch.Tensor):
    result = torch.empty_like(A)
    result[..., 0:1] = A[..., 0:1] * B[..., 0:1]  # Sa * Sb
    result[..., 1:] = A[..., 0:1] * B[..., 1:] + A[..., 1:]  # Sa * Vb + Va
    return result


def prefix_scan(x, prefix_func, dim=0, pad_value=0):
    x = x.movedim(dim, -1)
    other_dims, seq_len = x.shape[:-1], x.size(-1)
    n_powers_of_2 = int(math.ceil(math.log2(seq_len)))
    n_pads = 2**n_powers_of_2 - seq_len
    x = torch.nn.functional.pad(x, (0, n_pads), value=pad_value)
    for n in (2 ** torch.arange(n_powers_of_2)).tolist():
        x = x.view(*other_dims, -1, n * 2)
        last_on_L = x[..., (n - 1) : n]
        last_on_L = last_on_L.movedim((-2, -1), (dim - 1, dim))
        all_on_R = x[..., n:]
        all_on_R = all_on_R.movedim((-2, -1), (dim - 1, dim))
        updated_on_R = prefix_func(last_on_L, all_on_R)
        updated_on_R = updated_on_R.movedim((dim - 1, dim), (-2, -1))
        x = torch.cat([x[..., :n], updated_on_R], dim=-1)
    x = x.view(*other_dims, -1)
    x = x[..., :seq_len]
    y = x.movedim(-1, dim)
    return y


def cumsum_gradient(delta_t, alpha, gamma_k):
    """Compute gradients for all R_m^k using torch.cumsum."""
    N, d = alpha.shape
    device = delta_t.device

    # Compute R_m^k using prefix products
    S = torch.exp(-gamma_k * delta_t)  # Shape [N]
    cumprod_S = torch.cumprod(S, dim=0)  # Shape [N]
    cumprod_S = torch.cat(
        [torch.ones(1, device=device), cumprod_S[:-1]], dim=0
    )  # Shift: [1, S_1, S_1*S_2, ...]
    R = torch.zeros(N, d, device=device)
    R[0] = alpha[0]
    for m in range(1, N):
        R[m] = S[m] * (R[m - 1] + alpha[m - 1])

    # Compute gradients: dR_m^k/dγ^k = -∑_{j=2}^m Δt_j e^{-γ^k (t_m - t_j)} R_j^k
    cumsum_dt = torch.cumsum(delta_t, dim=0)  # Shape [N]
    cumsum_dt = torch.cat(
        [torch.zeros(1, device=device), cumsum_dt[:-1]], dim=0
    )  # Shift: [0, Δt_2, Δt_2+Δt_3, ...]
    t_m = cumsum_dt[-1]  # t_N
    t_j = cumsum_dt  # Shape [N]
    exp_terms = torch.exp(-gamma_k * (t_m - t_j))  # Shape [N]
    exp_terms = exp_terms.view(-1, 1)  # Shape [N, 1]
    weights = delta_t.view(-1, 1) * exp_terms  # Shape [N, 1]

    # Create lower triangular matrix for summation
    indices = torch.arange(N, device=device)
    mask = indices.view(-1, 1) >= indices.view(1, -1)  # Lower triangular mask [N, N]
    mask[0, :] = False  # Exclude j=1
    weights = weights.view(N, 1) * mask  # Shape [N, N]
    grad = -torch.matmul(weights, R)  # Shape [N, d]

    return grad


def prefix_scan_gradient(delta_t, alpha, gamma_k):
    """Compute gradients for all R_m^k using prefix scan."""
    N, d = alpha.shape
    S = torch.exp(-gamma_k * delta_t).unsqueeze(-1)  # Shape [N, 1]
    state = torch.cat([S, alpha], dim=-1)  # Shape [N, d+1]
    R_all = prefix_scan(state, state_mult, dim=0)  # Shape [N, d+1]
    R = R_all[..., 1:]  # Shape [N, d]

    # Compute gradients using prefix scan for the recurrence
    V_grad = -delta_t.unsqueeze(-1) * R  # Shape [N, d]
    state_grad = torch.cat([S, V_grad], dim=-1)  # Shape [N, d+1]
    grad_all = prefix_scan(state_grad, state_mult, dim=0)  # Shape [N, d+1]
    grad = grad_all[..., 1:]  # Shape [N, d]

    return grad


def test_performance(N_values, d=64, device="cuda"):
    """Test performance of cumsum vs. prefix scan for different N."""
    print(f"{'N':<10} {'Cumsum (ms)':<15} {'Prefix Scan (ms)':<15}")
    print("-" * 40)

    for N in N_values:
        # Generate inputs
        delta_t = torch.rand(N, device=device)
        delta_t[0] = 0.0  # Δt_1 = 0
        alpha = torch.rand(N, d, device=device)
        gamma_k = torch.tensor(0.5, device=device)

        # Time cumsum
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(100):
            result_cumsum = cumsum_gradient(delta_t, alpha, gamma_k)
        end_event.record()
        cuda.synchronize()
        cumsum_time = start_event.elapsed_time(end_event) / 100

        # Time prefix scan
        start_event.record()
        for _ in range(100):
            result_prefix = prefix_scan_gradient(delta_t, alpha, gamma_k)
        end_event.record()
        cuda.synchronize()
        prefix_time = start_event.elapsed_time(end_event) / 100

        print(f"{N:<10} {cumsum_time:<15.3f} {prefix_time:<15.3f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    N_values = [1000, 10000, 100000, 1000000]
    test_performance(N_values, d=64, device=device)
