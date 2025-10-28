import math
import torch
import statistics
import matplotlib.pyplot as plt

from hawkes import models, utils


def plot_intensity(
    seq: utils.EventSequence,
    model_sim: models.HawkesBase,
    grid: int = 1000,
    output: str = "intensity_plot.png",
    plot_events: bool = True,
):
    """
    Plot the intensity of a Hawkes process given an event sequence.

    Args:
        model_sim: The model used to simulate the data. Use an model with chosen parameter values or a learned model for real data.
        M: Number of nodes
        T: End time of event sequence observation period
        ti: Tensor of event times with shape (1, N, 1)
        mi: Tensor of event types with shape (N,)
        grid: Resolution of plot; i.e., how many times to plot the intensity for
        output: Output location for the plot
        plot_events: Whether to mark the location of events on the plot
    """

    assert (
        seq.M == model_sim.M
    ), "error: event sequence and sim model have different number of nodes"

    t = torch.linspace(0, seq.T, grid, device=seq.ti.device)
    with torch.no_grad():
        intensity, intensity_states = model_sim.intensity_at_events(
            seq, return_all_states=True
        )
        t_intensity = model_sim.intensity_at_t(t, seq, intensity_states).cpu()

    t = t.cpu()
    seq = seq.to("cpu")
    intensity = intensity.cpu()

    fig, ax = plt.subplots(seq.M, 1, figsize=(9, seq.M), sharex=True, sharey=False)

    for g in range(seq.M):
        mask = seq.mi.squeeze() == g
        times_g = seq.ti.squeeze()[mask]

        if plot_events:
            for eventt in times_g:
                ax[g].axvline(eventt.item(), color="r", linestyle="-", linewidth=0.5)

        ax[g].plot(t.squeeze(), t_intensity[:, g], label=f"Account {g}", linewidth=1)
        # ax[g].plot(times_g, intensity[mask, g], ".")  # Experimental: plot event points
        ax[g].set_xlim(0, seq.T)
        ax[g].set_ylim(0, torch.ceil(torch.max(intensity)))

        if g > 0:
            ax[g].set_yticklabels([])
            ax[g].set_yticks([])

        if g == seq.M - 1:
            ax[g].set_xlabel("Time (arbitrary)")

        label = f"Account {g}" if g == 0 else g
        ax[g].text(
            0.01,
            0.88,
            label,
            transform=ax[g].transAxes,
            ha="left",
            va="top",
            bbox=dict(
                facecolor="lightgray", edgecolor="black", boxstyle="round,pad=0.2"
            ),
        )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.text(0.0, 0.5, "Intensity", va="center", rotation="vertical")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic(
    true_alpha: torch.Tensor,
    estimated_alpha: torch.Tensor,
    output: str = "diagnostic_plot.png",
):
    """
    A diagnostic plot comparing an arbitrarily chosen or known-to-be-true interaction matrix with a learned one.

    Args:
        true_alpha: The baseline interaction matrix
        estimated_alpha: The comparison interaction matrix
        output: Output location for the plot
    """

    # List of matplotlib's diverging colormaps
    diverging_cmaps = {
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",
    }

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    matrices = [
        (true_alpha, "True Interaction Matrix", "Reds"),
        (estimated_alpha, "Estimated Interaction Matrix", "Reds"),
        (estimated_alpha - true_alpha, "Error", "seismic"),
    ]

    for ax, (matrix, title, cmap) in zip(axes, matrices):
        cmap_ = plt.get_cmap(cmap).copy()
        cmap_.set_bad("white")
        if cmap in diverging_cmaps:
            # For diverging colormaps, center at zero with symmetric limits
            abs_max = max(abs(matrix.max()), abs(matrix.min()))
            vmin, vmax = -abs_max, abs_max
            heatmap = ax.imshow(matrix, cmap=cmap_, aspect="auto", vmin=vmin, vmax=vmax)
        else:
            heatmap = ax.imshow(matrix, cmap=cmap_, aspect="auto")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    model,
    model_sim,
    T,
    max_events,
    num_simulations=10,
    output: str = "residual_plot.png",
):
    """
    Simulate data from a simulation model and plot the residuals of a fitted model, and compare the residuals to that of a Poisson process.

    Args:
        model: The learned model
        model_sim: The model to use for simulating data
        T: End time of event sequence observation period
        max_events: The maximum number of events to consider in the residual plot
        num_simulations: number of simulation rounds to perform; i.e., number of calculated residual curves

    [DOES NOT WORK]
    """

    # TODO: too much memory is used for plotting residuals

    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect("equal", adjustable="box")

    hawkes_rmse_list = []
    poisson_rmse_list = []

    for i in range(num_simulations):
        seq_new = model_sim.simulate(T, max_events=max_events)
        ts = torch.linspace(2, T, 100)
        EN, N = [], []

        for t in ts:
            with torch.no_grad():
                EN_t = model.integrated_intensity(t, seq_new.ti, seq_new.mi)
                EN.append(EN_t)
            N.append((seq_new.ti < t).sum().item())

        EN = torch.stack(EN).cpu()
        N = torch.tensor(N)

        seq_new.cpu()

        # Hawkes model residuals
        resid_hawkes = EN - N
        hawkes_rmse = (resid_hawkes**2).mean().sqrt().item()
        hawkes_rmse_list.append(hawkes_rmse)
        plt.plot(N, resid_hawkes, "r-", alpha=0.2)

        # Poisson model residuals
        lam0 = seq_new.N / T
        N_poisson = lam0 * ts
        resid_poisson = N_poisson - N
        poisson_rmse = (resid_poisson**2).mean().sqrt().item()
        poisson_rmse_list.append(poisson_rmse)
        plt.plot(N, resid_poisson, "b-", alpha=0.2)

    # Calculate average RMSE for both models
    avg_hawkes_rmse = statistics.fmean(hawkes_rmse_list)
    avg_poisson_rmse = statistics.fmean(poisson_rmse_list)

    # Add legend with average RMSE
    plt.plot([], [], "r-", label=f"Hawkes (Avg RMSE={avg_hawkes_rmse:.2f})")
    plt.plot([], [], "b-", label=f"Poisson (Avg RMSE={avg_poisson_rmse:.2f})")

    plt.xlabel("Observed " + r"$N(t)$")
    plt.ylabel(r"$E[N(t)] - N(t)$")
    plt.ylim(-500, 500)
    plt.xlim(0, max(N))  # Set x-axis limit based on maximum observed events
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left")
    plt.title(f"Residuals for {num_simulations} Simulations")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
