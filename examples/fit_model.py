# import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import logging
import argparse
import numpy as np

from hawkes import models, utils
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
# parser.add_argument(
#     "--T",
#     type=int,
#     default=500,
#     help="Set maximum simulated event time (default: T = 500)",
# )
# parser.add_argument(
#     "--N",
#     type=int,
#     default=2000,
#     help="Set number of simulated events (default: N = 2000)",
# )
# parser.add_argument(
#     "--M",
#     type=int,
#     default=10,
#     help="Set number of event types (default: M = 10)",
# )
parser.add_argument(
    "--batch-size",
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
    "--model-type",
    type=str,
    default="full-rank",
    choices=["full-rank", "low-rank", "upper-triangular"],
    help="What type of Hawkes model parametrization to use (default: full-rank)",
)
parser.add_argument(
    "--events-file",
    type=str,
    default="outputs/data/events.pt",
    help="Location of saved event sequence (default: outputs/data/events.pt)",
)
parser.add_argument(
    "--model-est-file",
    type=str,
    default="outputs/data/model_est.pt",
    help="Location to save estimated model to (default: outputs/data/model_est.pt)",
)
args = parser.parse_args()

# Obtain event sequence and reference interaction matrix
seq = torch.load(args.events_file, weights_only=False).to(args.device)

# Fitting hyperparameters
fit_config = config.FitConfig(
    num_steps=4000,
    batch_size=args.batch_size,
    monitor_interval=400,
    learning_rate=0.1,
    l1_penalty=0.01,
    l1_hinge=0.05,
    l1_alpha_diag=True,
    nuc_penalty=0,
)
est_gamma = [0.1] * 1
est_init_scale = 0.01
est_rank = 3

base_process = models.PoissonHomogeneous(M=seq.M, mu_init=0.01, device=args.device)
# base_process = models.PoissonFourierSeries(
#     M=seq.M,
#     T=0.1,
#     num_modes=1,
#     fourier_init=torch.tensor([2.0, 2.0], device=args.device)[None, :, None].repeat(
#         seq.M, 1, 1
#     ),
#     device=args.device,
# )

# Fit estimation model
match args.model_type:
    case "full-rank":
        model_est = models.HawkesFullRank(
            gamma=torch.tensor(est_gamma).to(args.device),
            gamma_param=True,
            base_process=base_process,
            alpha_init=est_init_scale,
        ).to(args.device)
    case "low-rank":
        model_est = models.HawkesLowRank(
            rank=est_rank,
            gamma=torch.tensor(est_gamma).to(args.device),
            gamma_param=True,
            base_process=base_process,
            alpha_init=est_init_scale,
        ).to(args.device)
    case "upper-triangular":
        model_est = models.HawkesUpperTriangular(
            rank=est_rank,
            gamma=torch.tensor(est_gamma).to(args.device),
            gamma_param=True,
            base_process=base_process,
            alpha_init=est_init_scale,
        ).to(args.device)

_ = model_est.fit(seq, fit_config)

# Save estimated model
torch.save(model_est, args.model_est_file)
logger.info(f"Saved estimated model to {args.model_est_file}")
