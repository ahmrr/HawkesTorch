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
    description="Simulate an event sequence from a Hawkes process"
)
parser.add_argument(
    "--T",
    type=int,
    default=500,
    help="Set maximum simulated event time (default: T = 500)",
)
parser.add_argument(
    "--M",
    type=int,
    default=10,
    help="Set number of event types (default: M = 10)",
)
parser.add_argument(
    "--N",
    type=int,
    default=2000,
    help="Set number of simulated events (default: N = 2000)",
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
    "--events-file",
    type=str,
    default="outputs/data/events.pt",
    help="Location to save event sequence to (default: outputs/data/events.pt)",
)
parser.add_argument(
    "--model-sim-file",
    type=str,
    default="outputs/data/model_sim.pt",
    help="Location to save event simulation model to (default: outputs/data/model_sim.pt)",
)
parser.add_argument(
    "--intensity-plot",
    type=str,
    default="outputs/plots/intensity_plot.png",
    help="Location of intensity plot output (default: outputs/intensity_plot.png)",
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

# Alternative: "three bridged communities" interaction behavior
# S = 0.1
# I = 0.5
# M = 10
# sim_K = 1
# sim_gamma = [1.5] * sim_K
# sim_init_scale = 0.1
# sim_mu = [S, 0, S, 0, 0, S, S, S, S, S]
# sim_alpha = [
#     [I, I, 0, 0, 0, 0, 0, 0, 0, 0],
#     [S, I, 0, 0, 0, 0, 0, 0, 0, S],
#     [0, 0, I, I, I, 0, 0, 0, 0, 0],
#     [0, 0, S, I, S, 0, 0, 0, 0, 0],
#     [0, 0, S, S, I, 0, 0, 0, 0, S],
#     [0, 0, 0, 0, 0, I, I, I, I, 0],
#     [0, 0, 0, 0, 0, 0, S, S, S, 0],
#     [0, 0, 0, 0, 0, 0, 0, S, S, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, S, S],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, I],
# ]

# Initialize model used for simulation
model_sim = models.HawkesFullRank(
    M=args.M,
    gamma=torch.tensor(sim_gamma).to(args.device),
    init_scale=sim_init_scale,
    runtime_config=config.HawkesRuntimeConfig(deterministic_sim=args.deterministic_sim),
).to(args.device)


# Set base excitation and interaction rates according to fixed sim params
model_sim.mu = torch.tensor(sim_mu).to(args.device)
model_sim.alpha = (
    torch.tensor(sim_alpha).to(args.device).unsqueeze(0).repeat(sim_K, 1, 1)
)

# Simulate the event sequence and save intensity plot
seq = model_sim.simulate(T=args.T, max_events=args.N)
logger.info(f"Simulated event sequence of length {seq.N}")

if args.intensity_plot != "false":
    plotting.plot_intensity(seq, model_sim, output=args.intensity_plot)
    logger.info(f"Saved intensity plot to {args.intensity_plot}")

# Save simulated event sequence and simulation model
torch.save(seq, args.events_file)
torch.save(model_sim, args.model_sim_file)
logger.info(
    f"Saved event sequence to {args.events_file} "
    f"and simulation model to {args.model_sim_file}"
)
