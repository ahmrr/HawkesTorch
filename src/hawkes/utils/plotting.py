import math
import torch
import statistics
import matplotlib.pyplot as plt

from hawkes import models, utils


def plot_intensity(
    seq: utils.EventSequence,
    model_sim: models.HawkesBase,
    plot_events: bool = True,
    grid: int = 1000,
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
        plot_events: Whether to mark the location of events on the plot
    """

    assert (
        seq.M == model_sim.M
    ), "Event sequence and sim model must have the same number of nodes"

    t = torch.linspace(0, seq.ti.max(), grid, device=seq.ti.device)
    with torch.no_grad():
        intensity_states = model_sim.intensity_states(seq)
        intensity_at_events = model_sim.intensity_at_events(
            seq, states=intensity_states, full_intensity=False
        )
        t_intensity = model_sim.intensity(t, seq, intensity_states).cpu()

    t = t.cpu()
    seq = seq.to("cpu")
    intensity_at_events = intensity_at_events.cpu()

    with plt.style.context("seaborn-v0_8-white"):
        fig, axes = plt.subplots(
            seq.M, 1, figsize=(9, seq.M), sharex=True, sharey=False
        )

        for g in range(seq.M):
            mask = seq.mi.squeeze() == g
            times_g = seq.ti.squeeze()[mask]

            if plot_events:
                for eventt in times_g:
                    axes[g].axvline(
                        eventt.item(),
                        color="r",
                        linestyle="-",
                        linewidth=0.5,
                        alpha=0.25,
                    )

            axes[g].plot(t.squeeze(), t_intensity[:, g], label=f"{g}", linewidth=1)
            axes[g].set_xlim(0, seq.ti.max())
            axes[g].set_ylim(0, torch.ceil(torch.max(intensity_at_events)))

            if g > 0:
                axes[g].set_yticklabels([])
                axes[g].set_yticks([])

            if g == seq.M - 1:
                axes[g].set_xlabel("Time")

            axes[g].text(
                0.02,
                0.88,
                g,
                transform=axes[g].transAxes,
                ha="left",
                va="top",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.4",
                ),
            )

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.text(0.0, 0.5, "Intensity", va="center", rotation="vertical")

        return fig, axes


def plot_alpha(
    alpha: torch.Tensor,
):
    K = alpha.shape[0]

    with plt.style.context("seaborn-v0_8-white"):
        fig, axes = plt.subplots(1, K, figsize=(8 * K, 6), squeeze=False)

        for k in range(K):
            vmin, vmax = 0, alpha[k].max().item()
            im = axes[0, k].imshow(
                alpha[k], aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax
            )
            axes[0, k].set_title(f"Kernel {k+1}")
            plt.colorbar(im, ax=axes[0, k])

        return fig, axes


def plot_alpha_comparison(
    true_alpha: torch.Tensor,
    estimated_alpha: torch.Tensor,
):
    # List of diverging colormaps
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

    with plt.style.context("seaborn-v0_8-white"):
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
                heatmap = ax.imshow(
                    matrix, cmap=cmap_, aspect="auto", vmin=vmin, vmax=vmax
                )
            else:
                heatmap = ax.imshow(matrix, cmap=cmap_, aspect="auto")

            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

        return fig, axes
