import torch
import logging
import argparse
import warnings
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
    "--model-sim-file",
    type=str,
    default="outputs/data/model_sim.pt",
    help="Location of saved event simulation model (default: outputs/data/model_sim.pt)",
)
parser.add_argument(
    "--model-est-file",
    type=str,
    default="outputs/data/model_est.pt",
    help="Location of saved estimation model (default: outputs/data/model_est.pt)",
)
parser.add_argument(
    "--diagnostic-plot",
    type=str,
    default="outputs/plots/diagnostic_plot.png",
    help="Location of diagnostic plot output (default: outputs/diagnostic_plot.png)",
)
parser.add_argument(
    "--residual-plot",
    type=str,
    default="outputs/plots/residual_plot.png",
    help="Location of residual plot output (default: outputs/residual_plot.png)",
)
args = parser.parse_args()

# Suppress message warning against loading without weights_only=True
warnings.simplefilter(action="ignore", category=FutureWarning)

# Obtain event sequence and reference interaction matrix
model_sim = torch.load(args.model_sim_file)
model_est = torch.load(args.model_est_file)

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
