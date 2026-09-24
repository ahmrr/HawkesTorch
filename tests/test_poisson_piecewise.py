"""Tests for the PoissonPiecewise base process."""

import math
import torch
import pytest

from hawkes import models
from hawkes.models import (
    Hawkes,
    PoissonBase,
    PoissonPiecewise,
    PoissonPiecewisePenalty,
    L1Penalty,
    L2Penalty,
)
from hawkes.utils import config


BREAKPOINTS = torch.tensor([0.0, 10.0, 20.0, 30.0])
RATES = torch.tensor([[0.5, 3.0, 1.0], [2.0, 0.5, 2.5]])


def make_model(**kwargs) -> PoissonPiecewise:
    """A two-variate model with a known step profile on [0, 30)."""

    defaults = dict(
        M=2,
        breakpoints=BREAKPOINTS,
        mu_init=RATES,
        t_end=30.0,
    )
    defaults.update(kwargs)

    return PoissonPiecewise(**defaults)


def test_mu_follows_step_profile():
    model = make_model()

    assert model.num_bins == 3
    assert model.num_params == 6

    # One time in each bin, plus the breakpoints themselves (bins are half-open)
    t = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    expected = RATES.T[[0, 0, 1, 1, 2, 2]]

    assert torch.allclose(model.mu(t), expected, atol=1e-6)


def test_mu_respects_activity_windows():
    model = make_model(
        t_start=torch.tensor([0.0, 12.0]), t_end=torch.tensor([18.0, 30.0])
    )

    t = torch.tensor([5.0, 15.0, 25.0])
    mu = model.mu(t)

    # Variate 0 is active on [0, 18), variate 1 on [12, 30)
    assert torch.allclose(mu[:, 0], torch.tensor([0.5, 3.0, 0.0]), atol=1e-6)
    assert torch.allclose(mu[:, 1], torch.tensor([0.0, 0.5, 2.5]), atol=1e-6)


def test_mu_extrapolates_outside_breakpoints():
    # Outer bins extend to +/- infinity, so mu is defined wherever a variate is active
    model = make_model(t_start=-10.0, t_end=float("inf"))

    t = torch.tensor([-5.0, 100.0])
    expected = RATES.T[[0, 2]]

    assert torch.allclose(model.mu(t), expected, atol=1e-6)


def test_single_bin_matches_constant_rate():
    model = PoissonPiecewise(
        M=2,
        breakpoints=torch.tensor([0.0, 30.0]),
        mu_init=torch.tensor([[0.5], [2.0]]),
        t_end=30.0,
    )

    t = torch.tensor([0.0, 15.0, 29.0])

    assert torch.allclose(
        model.mu(t), torch.tensor([0.5, 2.0]).expand(3, -1), atol=1e-6
    )
    assert torch.allclose(
        model.integral_mu(0.0, 30.0), torch.tensor([15.0, 60.0]), atol=1e-5
    )


def test_uniform_grid_from_num_bins():
    model = PoissonPiecewise(M=3, num_bins=4, t_start=2.0, t_end=10.0)

    assert model.num_bins == 4
    assert torch.allclose(model.breakpoints, torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0]))
    assert model.mu_values.shape == (3, 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),  # neither breakpoints nor num_bins
        dict(breakpoints=BREAKPOINTS, num_bins=3),  # both
        dict(num_bins=4),  # infinite t_end
        dict(breakpoints=torch.tensor([0.0, 20.0, 10.0])),  # not increasing
        dict(breakpoints=torch.tensor([0.0])),  # too few edges
        dict(breakpoints=BREAKPOINTS.unsqueeze(0)),  # not rank 1
        dict(breakpoints=BREAKPOINTS, mu_init=torch.ones(2, 5)),  # wrong mu_init shape
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        PoissonPiecewise(M=2, **kwargs)


def test_integral_matches_numeric_fallback():
    torch.manual_seed(0)
    model = make_model()

    # Random intervals, including ones that start/end inside a bin
    bounds = torch.rand(20, 2) * 30.0
    bounds, _ = bounds.sort(dim=1)

    for t_start, t_end in bounds.tolist():
        exact = model.integral_mu(t_start, t_end)
        numeric = PoissonBase.integral_mu(model, t_start, t_end, n_points=200001)

        assert torch.allclose(exact, numeric, rtol=1e-3, atol=1e-3)


def test_integral_respects_activity_windows():
    model = make_model(
        t_start=torch.tensor([0.0, 10.0]), t_end=torch.tensor([20.0, 30.0])
    )

    integral = model.integral_mu(0.0, 30.0)

    # Variate 0 only accumulates on [0, 20), variate 1 only on [10, 30)
    assert torch.allclose(
        integral, torch.tensor([0.5 * 10 + 3.0 * 10, 0.5 * 10 + 2.5 * 10]), atol=1e-5
    )


def test_integral_is_differentiable():
    """The closed-form integral must carry gradients matching finite differences."""

    model = make_model(mu_init=RATES.double())

    model.integral_mu(2.5, 27.5).sum().backward()
    analytic = model._mu.grad.clone()

    eps = 1e-6
    numeric = torch.zeros_like(analytic)
    with torch.no_grad():
        for index in range(model._mu.numel()):
            i, j = divmod(index, model.num_bins)

            model._mu.data[i, j] += eps
            plus = model.integral_mu(2.5, 27.5).sum()
            model._mu.data[i, j] -= 2 * eps
            minus = model.integral_mu(2.5, 27.5).sum()
            model._mu.data[i, j] += eps

            numeric[i, j] = (plus - minus) / (2 * eps)

    assert torch.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_integral_is_additive():
    model = make_model()

    whole = model.integral_mu(3.0, 27.0)
    split = model.integral_mu(3.0, 11.5) + model.integral_mu(11.5, 27.0)

    assert torch.allclose(whole, split, atol=1e-5)


def test_upper_bound_is_tight_and_valid():
    model = make_model()

    # Largest bin total is bin 1 on variate 0 plus bin 1 on variate 1 => 3.5
    assert model.upper_bound_in_interval(0.0, 30.0) == pytest.approx(3.5, abs=1e-5)
    assert model.upper_bound_in_interval(0.0, 5.0) == pytest.approx(2.5, abs=1e-5)
    assert model.upper_bound_in_interval(30.0, 0.0) == 0.0

    bound = model.upper_bound_in_interval(0.0, 30.0)
    t = torch.linspace(0.0, 29.9, 500)
    assert model.mu(t).sum(dim=1).max().item() <= bound


def test_penalty_terms():
    model = make_model()

    assert model.penalty() == 0.0

    rate_only = make_model(penalization=PoissonPiecewisePenalty(rate=L1Penalty(1.0)))
    assert rate_only.penalty().item() == pytest.approx(RATES.sum().item(), abs=1e-4)

    smooth_only = make_model(
        penalization=PoissonPiecewisePenalty(smoothness=L2Penalty(1.0))
    )
    expected = torch.diff(RATES, dim=-1).pow(2).sum().item()
    assert smooth_only.penalty().item() == pytest.approx(expected, abs=1e-4)


def test_get_save_data_and_report():
    model = make_model()
    data = model.get_save_data()

    assert data["num_bins"] == 3
    assert data["mu_values"].shape == (2, 3)
    assert data["breakpoints"].shape == (4,)
    assert "num_bins=3" in model.report_parameters()


def test_recovers_known_step_profile():
    torch.manual_seed(0)

    breakpoints = torch.tensor([0.0, 200.0, 400.0, 600.0])
    true_rates = torch.tensor([[0.4, 2.0, 0.8], [1.5, 0.3, 1.2]])

    sim_model = PoissonPiecewise(
        M=2, breakpoints=breakpoints, mu_init=true_rates, t_end=600.0
    )
    seq = sim_model.simulate(max_events=100000)

    est_model = PoissonPiecewise(M=2, breakpoints=breakpoints, mu_init=0.5, t_end=600.0)
    losses = est_model.fit(
        seq,
        t_start=0.0,
        t_end=600.0,
        fit_config=config.FitConfig(num_steps=1500, learning_rate=0.05),
    )

    assert losses[-1] < losses[0]

    recovered = est_model.mu_values.detach()
    assert torch.allclose(recovered, true_rates, rtol=0.3, atol=0.1)


def test_hawkes_with_piecewise_base_simulates_and_fits():
    torch.manual_seed(0)

    breakpoints = torch.tensor([0.0, 100.0, 200.0, 300.0])
    sim_base = PoissonPiecewise(
        M=2,
        breakpoints=breakpoints,
        mu_init=torch.tensor([[0.3, 1.2, 0.6], [1.0, 0.4, 0.9]]),
        t_end=300.0,
    )
    sim_model = Hawkes(
        gamma=torch.tensor([1.5]),
        gamma_param=False,
        base_process=sim_base,
        alpha_init=torch.full((1, 2, 2), 0.2),
    )

    seq = sim_model.simulate(max_events=2000)
    assert seq.N > 100

    est_base = PoissonPiecewise(M=2, breakpoints=breakpoints, mu_init=0.5, t_end=300.0)
    est_model = Hawkes(
        gamma=torch.tensor([1.0]),
        gamma_param=True,
        base_process=est_base,
        alpha_init=0.1,
    )

    stats = est_model.fit(
        seq, fit_config=config.FitConfig(num_steps=100, learning_rate=0.05)
    )
    losses = stats["losses"] if isinstance(stats, dict) else stats

    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0]
    assert torch.isfinite(est_base.mu_values).all()
