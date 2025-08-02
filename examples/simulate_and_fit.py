import torch
import logging
import argparse
import numpy as np

from hawkes import models
from hawkes.utils import config, plotting

# Setup logging and argument parsing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Run Hawkes process simulation and estimation"
)
parser.add_argument(
    "--T",
    type=int,
    default=500,
    help="Set maximum simulated event time (default: T = 500)",
)
parser.add_argument(
    "--N",
    type=int,
    default=2000,
    help="Set number of simulated events (default: N = 2000)",
)
parser.add_argument(
    "--M",
    type=int,
    default=10,
    help="Set number of event types (default: M = 10)",
)
parser.add_argument(
    "--Nb",
    type=int,
    default=None,
    help="Set batch size (default: batching is disabled)",
)
parser.add_argument(
    "--device",
    type=str,
    default=("cuda" if torch.cuda.is_available() else "cpu"),
    help="Set PyTorch device to use (default: cuda if available, else cpu)",
)
parser.add_argument(
    "--deterministic-sim",
    action="store_true",
    help="Set to enable deterministic simulation (default: disabled)",
)
parser.add_argument(
    "--model-type",
    type=str,
    default="low-rank",
    choices=["full-rank", "low-rank", "upper-triangular"],
    help="What type of Hawkes model parametrization to use (default: low-rank)",
)
parser.add_argument(
    "--intensity-plot",
    type=str,
    default="outputs/intensity_plot.png",
    help="Location of intensity plot output (default: outputs/intensity_plot.png)",
)
parser.add_argument(
    "--diagnostic-plot",
    type=str,
    default="outputs/diagnostic_plot.png",
    help="Location of diagnostic plot output (default: outputs/diagnostic_plot.png)",
)
parser.add_argument(
    "--residual-plot",
    type=str,
    default="outputs/residual_plot.png",
    help="Location of residual plot output (default: outputs/residual_plot.png)",
)
args = parser.parse_args()

# Simulate Hawkes process data

# Fixed simulation hyperparameters to imitate "hub-and-spoke" interaction behavior
S = 0.1  # Spontaneous rate
I = 0.9  # Influence rate
sim_K = 1
sim_gamma = [1.5] * sim_K
sim_init_scale = 0.1
sim_mu = [S] + [0] * (args.M - 1)  # Only hub has spontaneous activity
sim_alpha = [[0] * args.M for _ in range(args.M)]
sim_alpha[0] = [I] * args.M  # Hub influences all nodes including self

# Initialize model used for simulation
model_sim = models.HawkesFullRank(
    M=args.M,
    gamma=torch.tensor(sim_gamma).to(args.device),
    init_scale=sim_init_scale,
    debug_config=config.HawkesDebugConfig(deterministic_sim=args.deterministic_sim),
).to(args.device)


# Set base excitation and interaction rates according to fixed sim params
model_sim.mu = torch.tensor(sim_mu).to(args.device)
model_sim.alpha = (
    torch.tensor(sim_alpha).to(args.device).unsqueeze(0).repeat(1, 1, sim_K)
)

# Simulate the event sequence and save intensity plot
ti, mi = model_sim.simulate(T=args.T, max_events=args.N)
logger.info(f"Simulated event sequence of length {ti.shape[1]}")
plotting.plot_intensity(
    model_sim, ti, mi, T=args.T, M=args.M, output=args.intensity_plot
)
logger.info(f"Saved intensity plot to {args.intensity_plot}")

# Fit Hawkes model to simulated data

# Fitting hyperparameters
fit_config = config.HawkesFitConfig(
    num_steps=4000,
    monitor_interval=400,
    learning_rate=0.1,
    l1_penalty=0.01,
    l1_hinge=0.05,
    nuc_penalty=0,
)
est_gamma = np.linspace(0.1, 5, 5).tolist()
est_init_scale = 0.01
est_rank = 3

# Fit estimation model
match args.model_type:
    case "full-rank":
        model_est = models.HawkesFullRank(
            M=args.M,
            gamma=torch.tensor(est_gamma).to(args.device),
            init_scale=est_init_scale,
            gamma_param=True,
        ).to(args.device)
    case "low-rank":
        model_est = models.HawkesLowRank(
            M=args.M,
            gamma=torch.tensor(est_gamma).to(args.device),
            rank=est_rank,
            init_scale=est_init_scale,
            gamma_param=True,
        ).to(args.device)
    case "upper-triangular":
        model_est = models.HawkesUpperTriangular(
            M=args.M,
            gamma=torch.tensor(est_gamma).to(args.device),
            rank=est_rank,
            init_scale=est_init_scale,
            gamma_param=True,
        ).to(args.device)

_ = model_est.fit(
    1_000_000, ti, mi, fit_config
)  # Use a large number to capture all events

# Save diagnostic and residual plots
plotting.plot_diagnostic(
    true_alpha=model_sim.alpha.detach().cpu().sum(axis=0),
    estimated_alpha=model_est.alpha.detach().cpu().sum(axis=0),
    output=args.diagnostic_plot,
)
logger.info(f"Saved diagnostic plot to {args.diagnostic_plot}")
plotting.plot_residuals(
    model=model_est,
    model_sim=model_sim,
    T=args.T,
    max_events=args.N,
    output=args.residual_plot,
)
logger.info(f"Saved residual plot to {args.residual_plot}")
