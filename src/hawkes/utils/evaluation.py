import math
import time
import torch
import typing
import logging

from . import utils
from .. import models


def kolmogorov_smirnov_test(
    seq: utils.EventSequence,
    model: models.HawkesBase,
):
    # TODO: implement K-S test on transformed inter-event times
    pass
