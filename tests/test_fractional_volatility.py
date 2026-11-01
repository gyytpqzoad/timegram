"""Fractional integration, ARFIMA and GARCH (ported from the grammar_v2_20261028 prototype).
The independent `arch` comparisons are skipped when the package is not installed."""
import math
import warnings

import numpy as np
import pytest
import torch

import timegram as tg
from timegram import library
from timegram.model import Source
from timegram.simulate import Batch, Series


class Given(Source):
    name = "given"

    def __init__(self, values):
        self.values = np.asarray(values, float)

    def run(self, s):
        return self.values

    def run_batch(self, b):
        return torch.as_tensor(self.values, dtype=torch.float32).expand(b.n, -1)


def run(model, width, pre=0, seed=1):
    s = Series(np.random.default_rng(seed))
    s.t = np.arange(1 - pre, width + 1)
    return s.value(model), s


def batch(model, width, pre=0, n=1):
    b = Batch(n, torch.Generator().manual_seed(1), "cpu")
    b.t = torch.arange(1 - pre, width + 1, dtype=torch.float32).expand(n, -1)
    return b.value(model).numpy(), b


def contexts(width, n=3):
    s = Series(np.random.default_rng(3))
    s.t = np.arange(1, width + 1)
    b = Batch(n, torch.Generator().manual_seed(3), "cpu")
    b.t = torch.arange(1, width + 1).float().expand(n, -1)
    return s, b


# ---------------------------------------------------------------------------------------
# fractional integration and ARFIMA
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("order", [0.0, 0.2, 0.4, -0.3, 1.0, 2.0])
def test_fractional_coefficients_match_generalized_binomial(order):
    expected = [1.0] + [math.prod(order + k for k in range(lag)) / math.factorial(lag) for lag in range(1, 18)]
    np.testing.assert_allclose(tg.fractional_weights(order, 18), expected, atol=1e-13)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("period", [1, 4])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_fractional_integer_limits_match_ordinary_integration(period, order):
    x = np.random.default_rng(4).normal(size=47)
    left = Given(x) >> tg.fractional_integrate(order, period=period)
    right = Given(x) >> tg.integrate(order, period=period)
    np.testing.assert_allclose(run(left, len(x))[0], run(right, len(x))[0], atol=1e-11)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("order", [0, 1, 2])
def test_fractional_difference_integer_limits(order):
    x = np.random.default_rng(5).normal(size=47)
    left = Given(x) >> tg.fractional_difference(order, period=4)
    right = Given(x) >> tg.difference(order, period=4)
    np.testing.assert_allclose(run(left, len(x))[0], run(right, len(x))[0], atol=1e-12)


def test_fractional_integration_and_differencing_invert_each_other():
    x = np.random.default_rng(6).normal(size=80)
    model = Given(x) >> tg.fractional_integrate(0.35) >> tg.fractional_difference(0.35)
    np.testing.assert_allclose(run(model, 80)[0], x, atol=1e-12)


@pytest.mark.parametrize("period,memory,pre", [(1, None, 0), (4, 9, 0), (1, 30, 19)])
def test_fractional_fast_matches_reference(period, memory, pre):
    width = 75
    x = np.random.default_rng(7).normal(size=width + pre)
    model = Given(x) >> tg.fractional_integrate(0.32, period=period, memory=memory, presample=pre)
    np.testing.assert_allclose(batch(model, width, pre)[0][0], run(model, width, pre)[0], atol=2e-6)


def test_fractional_zero_start_ignores_other_blocks_prehistory():
    y, _ = run(Given(np.ones(30)) >> tg.fractional_integrate(0.3), 20, pre=10)
    assert np.all(y[:10] == 0) and y[10] == 1


def test_noninteger_fractional_order_has_no_boundary_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tg.fractional_integrate(0.3)
    assert not captured


def test_named_arfima_is_its_atomic_construction():
    named = tg.arfima(1, 0.3, 1, phi=[0.5], theta=[0.2], presample=200)
    atomic = (tg.noise() >> tg.ma(1, theta=[0.2]) >> tg.ar(1, phi=[0.5])
              >> tg.fractional_integrate(0.3, presample=200))
    np.testing.assert_allclose(named.simulate(n=1, length=80, seed=5).context,
                               atomic.simulate(n=1, length=80, seed=5).context, atol=1e-12)
    assert any(n["kind"] == "fractional_integrate" for n in named.graph(expanded=True).nodes)


def test_arfima_default_ma_is_invertible():
    rng = np.random.default_rng(8)
    for _ in range(500):
        theta = np.atleast_1d(tg.arfima(0, 0.3, 2).theta.sample(rng, None))
        assert np.all(np.abs(np.roots(np.r_[theta[::-1], 1.0])) > 1)


# ---------------------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("p,q,alpha,beta,gamma,power", [
    (0, 1, [.3], [], None, 2.), (1, 1, [.18], [.72], None, 2.), (2, 1, [.2], [.48, .2], None, 2.),
    (1, 2, [.15, .1], [.63], None, 2.), (1, 1, [.1], [.72], [.16], 2.), (1, 1, [.15], [.65], [.12], 1.),
])
def test_paths_equal_independent_arch_simulator(p, q, alpha, beta, gamma, power):
    external = pytest.importorskip("arch.univariate")
    shocks = np.random.default_rng(17).normal(size=400)
    filt = tg.garch(p, q, omega=.2, alpha=alpha, beta=beta, gamma=gamma, power=power, burn_in=0)
    s, b = contexts(len(shocks))
    actual, variance = filt._path(s, shocks)
    initial = filt._parameters(s)[-1]
    simulator = external.GARCH(p=q, o=q if gamma is not None else 0, q=p, power=power)
    expected, expected_variance = simulator.simulate([.2] + alpha + (gamma or []) + beta, len(shocks),
                                                     rng=lambda size: shocks.copy(), burn=0, initial_value=initial)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(variance, expected_variance, rtol=3e-13, atol=3e-13)


def test_arch1_matches_textbook_updates_and_fast_path():
    shocks = np.array([1., -2., .5, 1.2, -.7])
    model = tg.arch(1, omega=.7, alpha=[.3], initial=1., burn_in=0)
    s, b = contexts(len(shocks))
    expected, variances = [], []
    for t, z in enumerate(shocks):
        v = 1. if t == 0 else .7 + .3 * expected[-1] ** 2
        variances.append(v)
        expected.append(np.sqrt(v) * z)
    actual, variance = model._path(s, shocks)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(variance, variances, atol=1e-14)
    fast, _ = model._path_batch(b, torch.tensor(shocks, dtype=torch.float32).expand(b.n, -1))
    np.testing.assert_allclose(fast.numpy(), np.tile(actual, (b.n, 1)), atol=1e-5)


def test_ar_garch_composition_matches_two_recursions():
    shocks = np.random.default_rng(19).normal(size=300)
    filt = tg.garch(1, 1, omega=.1, alpha=[.18], beta=[.72], initial=1, burn_in=0)
    model = tg.constant(1.5) + (Given(shocks) >> filt >> tg.ar(2, phi=[.45, -.2], init="zero"))
    s, _ = contexts(len(shocks))
    eps, variance, y = np.empty(300), np.empty(300), np.empty(300)
    for t, z in enumerate(shocks):
        variance[t] = 1 if t == 0 else .1 + .18 * eps[t - 1] ** 2 + .72 * variance[t - 1]
        eps[t] = np.sqrt(variance[t]) * z
        y[t] = eps[t] + (.45 * y[t - 1] if t else 0) - (.2 * y[t - 2] if t > 1 else 0)
    np.testing.assert_allclose(s.value(model), 1.5 + y, atol=2e-13)


@pytest.mark.parametrize("kwargs", [
    {"omega": 0}, {"omega": np.nan}, {"alpha": [-.1]}, {"beta": [1.]}, {"alpha": [.4], "beta": [.6]},
    {"gamma": [-.1]}, {"initial": 0}, {"alpha": [.1, .2]}, {"power": 0}, {"burn_in": -1},
])
def test_invalid_garch_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        tg.garch(**kwargs)


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_per_series_garch_parameters_are_sampled(backend):
    model = tg.noise() >> tg.garch(1, 1, omega=tg.uniform(.05, .15), alpha=[.15], beta=[.7], burn_in=20)
    omega = next(v for k, v in model.simulate(n=4, length=30, backend=backend).params.items()
                 if k.endswith("omega"))
    assert len(np.unique(omega)) == 4


def test_garch_scale_parameters_are_refused_as_estimands():
    model = tg.noise() >> tg.garch(1, 1, omega=tg.uniform(.05, .15), alpha=[.15], beta=[.7])
    with pytest.raises(NotImplementedError, match="level or scale"):
        model.estimate(tg.params("omega")).layout()


def test_library_distinguishes_volatility_laws():
    def key(**kw):
        return library.canonical(tg.garch(**kw).estimate(tg.forecast(h=2)))
    assert len({key(), key(power=1), key(gamma=[.1]), key(burn_in=2000), key(initial=2.)}) == 5
