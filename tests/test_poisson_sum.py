import torch
import pytest

from hawkes.models.hawkes.hawkes import Hawkes
from hawkes.models.poisson.poisson import Poisson
from hawkes.models.poisson.poisson_base import PoissonBase, PoissonSum
from hawkes.models.poisson.poisson_global_fourier import PoissonGlobalFourier
from hawkes.utils.utils import EventSequence

M = 3


@pytest.fixture
def windowed_constants():
    """Two constant-rate components with staggered per-variate activity windows."""
    a = Poisson(
        M,
        mu_init=torch.tensor([0.5, 1.0, 2.0]),
        t_start=torch.tensor([0.0, 3.3, 7.1]),
        t_end=torch.tensor([10.0, 8.9, 9.4]),
    )
    b = Poisson(
        M,
        mu_init=torch.tensor([0.25, 0.75, 1.5]),
        t_start=torch.tensor([1.7, 0.0, 2.2]),
        t_end=torch.tensor([6.3, 10.0, 5.05]),
    )
    return a, b


def test_integral_matches_sum_of_components(windowed_constants):
    a, b = windowed_constants
    s = PoissonSum([a, b])
    for lo, hi in [(0.0, 10.0), (2.5, 7.75), (4.0, 4.0)]:
        expected = a.integral_mu(lo, hi) + b.integral_mu(lo, hi)
        torch.testing.assert_close(s.integral_mu(lo, hi), expected)


def test_integral_is_exact_for_closed_form_components(windowed_constants):
    """The numerical fallback is off at activity-window edges; delegation is not."""
    a, b = windowed_constants
    s = PoissonSum([a, b])
    exact = a.integral_mu(0.0, 10.0) + b.integral_mu(0.0, 10.0)
    torch.testing.assert_close(s.integral_mu(0.0, 10.0), exact, rtol=0, atol=1e-6)
    numerical = PoissonBase.integral_mu(s, 0.0, 10.0, n_points=101)
    assert (numerical - exact).abs().max() > 1e-3


def test_per_variate_intervals_are_respected(windowed_constants):
    a, b = windowed_constants
    s = PoissonSum([a, b])
    lo = torch.tensor([0.0, 2.0, 4.0])
    hi = torch.tensor([3.0, 6.0, 10.0])
    expected = a.integral_mu(lo, hi) + b.integral_mu(lo, hi)
    torch.testing.assert_close(s.integral_mu(lo, hi), expected)


def test_numerical_components_still_supported(windowed_constants):
    a, _ = windowed_constants
    f = PoissonGlobalFourier(M, T=1.0, num_modes=2)
    s = PoissonSum([a, f])
    expected = a.integral_mu(0.0, 10.0) + f.integral_mu(0.0, 10.0)
    torch.testing.assert_close(s.integral_mu(0.0, 10.0), expected)


def test_nested_sums(windowed_constants):
    a, b = windowed_constants
    c = Poisson(M, mu_init=0.3)
    s = PoissonSum([a, b]) + c
    expected = a.integral_mu(0.0, 10.0) + b.integral_mu(0.0, 10.0) + c.integral_mu(0.0, 10.0)
    torch.testing.assert_close(s.integral_mu(0.0, 10.0), expected)


def test_gradients_reach_every_component(windowed_constants):
    a, b = windowed_constants
    s = PoissonSum([a, b])
    s.integral_mu(0.0, 10.0).sum().backward()
    assert a._mu.grad is not None and a._mu.grad.abs().sum() > 0
    assert b._mu.grad is not None and b._mu.grad.abs().sum() > 0


def test_hawkes_nll_with_summed_base(windowed_constants):
    a, b = windowed_constants
    model = Hawkes(
        gamma=torch.tensor([1.0]), gamma_param=False, base_process=PoissonSum([a, b])
    )
    seq = EventSequence(
        ti=torch.tensor([0.5, 1.2, 2.8, 4.1, 6.0, 7.5, 8.2, 9.3]),
        mi=torch.tensor([0, 1, 0, 2, 1, 2, 0, 2]),
        T=10.0,
    )
    nll = model.nll(seq)
    assert torch.isfinite(nll)
    nll.backward()
    assert a._mu.grad is not None and b._mu.grad is not None


def test_constructor_and_add():
    a = Poisson(M, mu_init=0.5)
    b = Poisson(M, mu_init=0.25)
    c = Poisson(M, mu_init=0.1)
    s = PoissonSum([a, b]) + c
    assert isinstance(s, PoissonSum) and len(s.components) == 3
    t = torch.linspace(0.0, 1.0, 5)
    torch.testing.assert_close(s.mu(t), a.mu(t) + b.mu(t) + c.mu(t))


def test_penalty_sums_component_penalties():
    from hawkes.models.penalty import L1Penalty
    from hawkes.models.poisson.poisson_base import PoissonPenalty

    a = Poisson(M, mu_init=0.5, penalization=PoissonPenalty(mu=L1Penalty(weight=1.0)))
    b = Poisson(M, mu_init=0.25, penalization=PoissonPenalty(mu=L1Penalty(weight=2.0)))
    s = PoissonSum([a, b])
    expected = a.penalty() + b.penalty()
    assert expected > 0
    torch.testing.assert_close(torch.as_tensor(s.penalty()), torch.as_tensor(expected))
