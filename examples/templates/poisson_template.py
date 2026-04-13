"""
A starting point for implementing custom Poisson models, primarily for use as
the base process (base rate) of Hawkes models. See Poisson and PoissonFourier
for complete reference implementations.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from hawkes.models import PoissonBase, Penalty
from hawkes.utils import config


@dataclass
class PoissonExamplePenalty:
    """
    Optionally define penalties for each learnable nn.Parameter in your model.
    You can just use PoissonPenalty if penalizing mu (instead of the parameters
    it is constructed from) is sufficient.
    """

    custom_parameter_1: Penalty | None = None
    custom_parameter_2: Penalty | None = None


class PoissonExample(PoissonBase):
    """
    Custom Poisson process implementation.
    """

    def __init__(
        self,
        # Add or remove constructor arguments as needed (e.g. default values)
        M: int,
        mu_init: torch.Tensor | float | None = None,
        t_start: torch.Tensor | float | None = None,
        t_end: torch.Tensor | float | None = None,
        penalization: Penalty | None = PoissonExamplePenalty(),
        transformation=config.SOFTPLUS,
        runtime_config=config.RuntimeConfig(),
        device: str | None = None,
    ):
        """
        Args:
            M: Number of variates (event types) in the process
            mu_init: Initial intensity values for each variate. If None, random initialization is used.
            t_start: Start times for each variate's activity window. If None, defaults to 0 for all variates.
            t_end: End times for each variate's activity window. If None, defaults to infinity for all variates.
            transformation: A mapping for the learned parameters, meaning the model learns the inverse values of the params
            runtime_config: Runtime configuration for debugging and profiling
            device: PyTorch device for computation ("cpu" or "cuda")
        """

        super().__init__(M, t_start, t_end, penalization, runtime_config, device)

        self.t = transformation

        # Define any custom values and nn.Parameter variables here
        _custom_parameter_1 = self.t.inverse(torch.tensor(0.0))
        _custom_parameter_2 = self.t.inverse(torch.tensor(0.0))

        self._custom_parameter_1 = nn.Parameter(_custom_parameter_1.to(self.device))
        self._custom_parameter_2 = nn.Parameter(_custom_parameter_2.to(self.device))

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute intensity at given times, respecting variate activity windows.
        """

        return ...

    def penalty(self) -> float:
        """
        Optionally override the default penalty method, if your parameters
        require custom penalization that is not covered by the base class.
        """

        return ...

    def integral_mu(
        self, t_start: torch.Tensor | float, t_end: torch.Tensor | float
    ) -> torch.Tensor:
        """
        Compute integral of constant intensity from t_start to t_end,
        respecting the variate activity windows (self.t_start, self.t_end).

        NOTE: It is more efficient to implement analytical integrals, if your
        custom parameterization allows for it. Otherwise, the base class will
        compute numerical integrals using the trapezoid rule.

        NOTE: Even if your model does not allow for analytical integrals, it is
        advisable to change the default number of points used for numerical
        integration (n_points) in the base class.
        """

        return ...

    def upper_bound_in_interval(
        self, t_start: torch.Tensor | float, t_end: torch.Tensor | float
    ) -> float:
        """
        Compute an upper bound of total intensity over time interval,
        respecting the variate activity windows (self.t_start, self.t_end).

        NOTE: This is used for thinning-based simulation. If you do not need to
        simulate Poisson/Hawkes processes, no need to implement this method.

        NOTE: If your model allows for it, it is best implement an analytical
        upper bound, and have it it should be as tight as possible. Otherwise,
        the base class has a conservative implementation that is sufficient.
        """

        return ...

    def report_parameters(self) -> str:
        """
        Return a string representation of the current parameter values. Used
        for logging in the training loop.
        """

        return ...

    def get_save_data(self) -> dict:
        """
        Return a dictionary of parameter values to be saved. Used for saving
        model parameters after training.
        """

        return {}
