"""Blocks against independent implementations, and the fast backend against the
reference backend, given the same noise and parameters."""
import numpy as np
import pytest
import torch
from scipy.signal import lfilter

import timegram as tg
from timegram.model import Source
from timegram.simulate import Batch, Series


class Given(Source):
    """A source with fixed values on the simulated grid (for exact comparisons)."""

    name = "given"

    def __init__(self, values):
        self.values = np.asarray(values, float)

    def run(self, s):
        return self.values

    def run_batch(self, b):
        return torch.as_tensor(self.values, dtype=torch.float64, device=b.device).unsqueeze(0).float()


def reference(model, T, pre=0, seed=0):
    s = Series(np.random.default_rng(seed))
    s.t = np.arange(1 - pre, T + 1)
    return s.value(model), s


def fast(model, T, pre=0, draws=None):
    b = Batch(1, torch.Generator().manual_seed(0), "cpu")
    b.t = torch.arange(1 - pre, T + 1, dtype=torch.float32).unsqueeze(0)
    for dist, v in (draws or {}).items():
        b._draws[id(dist)] = torch.as_tensor(np.asarray(v, float), dtype=torch.float32).reshape(1, -1).squeeze(-1) \
            if np.ndim(v) == 0 else torch.as_tensor(np.asarray(v, float), dtype=torch.float32).reshape(1, -1)
        b._keep.append(dist)
    return b.value(model)[0].double().numpy()


# ---------------------------------------------------------------------------------------
def run_holt_winters_additive(alpha, beta, gamma, level0, trend0, season0, eps):
    """An independent hand-written additive Holt-Winters recursion."""
    season = np.asarray(season0, float).copy()
    m, y = len(season), np.empty(len(eps))
    level, trend = float(level0), float(trend0)
    for t in range(len(eps)):
        k = t % m
        y[t] = level + trend + season[k] + eps[t]
        level, trend, season[k] = level + trend + alpha * eps[t], trend + beta * eps[t], season[k] + gamma * eps[t]
    return y


def test_ets_aaa_equals_holt_winters_recursion():
    rng = np.random.default_rng(1)
    eps = rng.standard_normal(40)
    s0 = np.array([1.5, -0.5, 2.0, -3.0])
    model = Given(eps) >> tg.ets("AAA", period=4, alpha=0.3, beta=0.1, gamma=0.2, level0=2.0, slope0=0.5, season0=s0)
    y, _ = reference(model, 40)
    np.testing.assert_allclose(y, run_holt_winters_additive(0.3, 0.1, 0.2, 2.0, 0.5, s0, eps), atol=1e-12)
    np.testing.assert_allclose(fast(model, 40), y, atol=1e-4)


def test_ses_equals_arima_011():
    """SES = ARIMA(0,1,1) with theta = alpha - 1, same noise, zero start."""
    eps = np.random.default_rng(2).standard_normal(50)
    alpha = tg.uniform(0.1, 0.9)
    ses = Given(eps) >> tg.ets("ANN", alpha=alpha, level0=0.0)
    arima = Given(eps) >> tg.ma(1, theta=alpha - 1) >> tg.integrate()
    s = Series(np.random.default_rng(0))
    s.t = np.arange(1, 51)
    np.testing.assert_allclose(s.value(ses), s.value(arima), atol=1e-12)


def test_arma_and_seasonal_arima_equal_lfilter():
    eps = np.random.default_rng(3).standard_normal(120)
    phi, theta, Phi, Theta = [0.5, -0.2], [0.4], [0.6], [-0.3]
    model = (Given(eps) >> tg.ma(1, theta=theta) >> tg.ma(1, period=4, theta=Theta)
             >> tg.ar(2, phi=phi) >> tg.ar(1, period=4, phi=Phi))
    ar_poly = np.convolve([1, -phi[0], -phi[1]], [1, 0, 0, 0, -Phi[0]])
    ma_poly = np.convolve([1, theta[0]], [1, 0, 0, 0, Theta[0]])
    expected = lfilter(ma_poly, ar_poly, eps)
    y, _ = reference(model, 120)
    np.testing.assert_allclose(y, expected, atol=1e-10)
    np.testing.assert_allclose(fast(model, 120), expected, atol=1e-4)


@pytest.mark.parametrize("make", [
    lambda: tg.integrate(2),
    lambda: tg.integrate(1, period=4),
    lambda: tg.difference(1, period=4),
    lambda: tg.lag(3),
    lambda: tg.ma(2, theta=[0.3, -0.4]),
    lambda: tg.ar(1, period=12, phi=[0.9]),
])
def test_fast_matches_reference(make):
    pre, T = 10, 60
    x = np.random.default_rng(4).standard_normal(pre + T)
    model = Given(x) >> make()
    y, _ = reference(model, T, pre)
    np.testing.assert_allclose(fast(model, T, pre), y, atol=1e-4 * max(1, np.abs(y).max()))


def test_sources_fast_matches_reference():
    pre, T = 5, 30
    model = tg.constant(2.0) + tg.trend(0.5) + tg.pattern(4, values=[1.0, -1.0, 2.0, -2.0])
    y, _ = reference(model, T, pre)
    np.testing.assert_allclose(fast(model, T, pre), y, atol=1e-5)
    assert np.all(y[:pre] == 0)                     # deterministic sources start at t = 1


def test_filters_commute_up_to_initial_conditions():
    """ma(1) >> ar(1) equals ar(1) >> ma(1), here exactly: both start at zero."""
    x = np.random.default_rng(5).standard_normal(80)
    a = Given(x) >> tg.ma(1, theta=[0.4]) >> tg.ar(1, phi=[0.7])
    b = Given(x) >> tg.ar(1, phi=[0.7]) >> tg.ma(1, theta=[0.4])
    np.testing.assert_allclose(reference(a, 80)[0], reference(b, 80)[0], atol=1e-12)


def test_shared_noise_and_independent_copies():
    e = tg.noise()
    shared = e + e
    sim = shared.simulate(n=1, length=20, seed=0)
    single = (2 * tg.noise()).simulate(n=1, length=20, seed=0)
    np.testing.assert_allclose(sim.context, single.context)          # a + a is 2 * a
    indep = (e + e.copy()).simulate(n=2000, length=1, seed=0)
    assert abs(np.nanvar(indep.context) - 2.0) < 0.2                 # variance 2, not 4


def test_selection_uses_one_operand_and_flattens():
    sel = tg.constant(1.0) | tg.constant(2.0) | tg.constant(3.0)
    assert len(sel.operands) == 3
    sim = sel.simulate(n=3000, length=1, seed=0)
    vals, counts = np.unique(sim.context[:, -1], return_counts=True)
    assert list(vals) == [1.0, 2.0, 3.0] and np.all(np.abs(counts / 3000 - 1 / 3) < 0.03)


def test_fast_admissibility_equals_eigenvalue_definition():
    from timegram.compounds import _hw_second_modulus, hw_admissible_batch
    rng = np.random.default_rng(6)
    for m in (4, 12):
        a = rng.uniform(0.05, 0.95, 5000)
        b = rng.uniform(0.02, 0.98, 5000) * a * rng.uniform(0.5, 4, 5000)
        g = rng.uniform(0.02, 0.98, 5000) * (1 - a) * rng.uniform(0.5, 3, 5000)
        expected = _hw_second_modulus(a, b, g, m) < 1 - 1e-9
        got = hw_admissible_batch(*(torch.as_tensor(v) for v in (a, b, g)), m).numpy()
        assert (got == expected).mean() > 0.999


def ets_recursion(alpha, beta, gamma, phi, l0, b0, s0, eps, trend, seasonal):
    """The textbook additive ETS recursion with damping (reference)."""
    m = len(s0) if seasonal else 1
    season = np.asarray(s0 if seasonal else [0.0], float).copy()
    level, slope, y = l0, (b0 if trend else 0.0), np.empty(len(eps))
    for t in range(len(eps)):
        k = t % m
        y[t] = level + phi * slope + season[k] + eps[t]
        level, slope = level + phi * slope + alpha * eps[t], phi * slope + beta * eps[t]
        if seasonal:
            season[k] += gamma * eps[t]
    return y


@pytest.mark.parametrize("spec", ["ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA"])
def test_ets_family_equals_recursion(spec):
    eps = np.random.default_rng(7).standard_normal(50)
    s0 = np.array([1.0, -2.0, 0.5, 0.5])
    trend, damped, seasonal = spec[1] == "A", "d" in spec, spec[-1] == "A"
    phi = 0.9 if damped else 1.0
    model = Given(eps) >> tg.ets(spec, period=4 if seasonal else None, alpha=0.4, beta=0.1, gamma=0.2,
                                 phi=phi, level0=3.0, slope0=0.5, season0=s0)
    expected = ets_recursion(0.4, 0.1 if trend else 0.0, 0.2, phi, 3.0, 0.5, s0, eps, trend, seasonal)
    y, _ = reference(model, 50)
    np.testing.assert_allclose(y, expected, atol=1e-10)
    np.testing.assert_allclose(fast(model, 50), expected, atol=1e-3)


def test_fast_admissibility_all_variants():
    from timegram.compounds import ets_admissible_batch, ets_second_modulus
    rng = np.random.default_rng(8)
    n = 3000
    for trend, seasonal, m in [(False, False, None), (True, False, None), (False, True, 4), (True, True, 12)]:
        a = rng.uniform(0.05, 1.9, n)
        b = rng.uniform(0, 1.5, n) * a
        g = rng.uniform(0, 1.5, n) * (1 - a / 2)
        ph = rng.uniform(0.7, 1.0, n)
        expected = ets_second_modulus(a, b, g, ph, trend, seasonal, m) < 1 - 1e-9
        got = ets_admissible_batch(*(torch.as_tensor(v) for v in (a, b, g, ph)), trend, seasonal, m).numpy()
        assert (got == expected).mean() > 0.998, (trend, seasonal, m)


@pytest.mark.parametrize("kind", ["minmax", "std", "robust"])
def test_normalize_uses_only_the_observed_window(kind):
    pre, T, h = 3, 30, 6
    x = np.random.default_rng(9).standard_normal(pre + T) * 5 + 20
    model = Given(x) >> tg.normalize(kind)
    s = Series(np.random.default_rng(0))
    s.t, s.horizon = np.arange(1 - pre, T + 1), h
    y = s.value(model)
    obs = y[pre:pre + T - h]
    if kind == "minmax":
        assert abs(obs.min()) < 1e-12 and abs(obs.max() - 1) < 1e-12
    elif kind == "std":
        assert abs(obs.mean()) < 1e-12 and abs(obs.std() - 1) < 1e-12
    b = Batch(1, torch.Generator().manual_seed(0), "cpu")
    b.t, b.horizon = torch.arange(1 - pre, T + 1, dtype=torch.float32).unsqueeze(0), h
    np.testing.assert_allclose(b.value(model)[0].double().numpy(), y, atol=1e-4)


def test_convex_mix_weights_sum_to_one():
    m = tg.mix(tg.constant(1.0), tg.constant(2.0), tg.constant(4.0))
    sim = m.simulate(n=200, length=1, seed=0)
    w = sim.params["convex.weight"]
    np.testing.assert_allclose(w.sum(1), 1.0)
    np.testing.assert_allclose(sim.context[:, -1], w @ np.array([1.0, 2.0, 4.0]))


def _invertible(theta):
    """All roots of 1 + sum theta_j z^j outside the unit circle."""
    theta = np.atleast_1d(theta)
    return np.all(np.abs(np.roots(np.r_[theta[::-1], 1.0])) > 1 - 1e-9)


def test_default_ma_prior_is_invertible():
    rng = np.random.default_rng(10)
    draws = [tg.ma(3).theta.sample(rng, None) for _ in range(3000)]
    assert all(_invertible(t) for t in draws)
    batch = tg.ma(3).theta.sample_batch(torch.Generator().manual_seed(0), 3000, "cpu", None).double().numpy()
    assert all(_invertible(t) for t in batch)


def test_sarima_tree_ma_matches_the_original_convention():
    """The curated prior's MA polynomial was 1 - theta_old(B): invertible by construction."""
    from timegram.priors import SarimaTree
    prior, rng = SarimaTree(period=4), np.random.default_rng(11)
    for _ in range(3000):
        d = prior.sample(rng, None)
        assert _invertible(d["theta"]) and _invertible(np.r_[np.zeros(3), d["Theta"][0], np.zeros(3), d["Theta"][1]])
    b = prior.sample_batch(torch.Generator().manual_seed(1), 3000, "cpu", None)
    assert all(_invertible(t) for t in b["theta"].double().numpy())


def test_fast_selection_leaves_no_reference_cycles():
    """Subset batches link back to their parent weakly, so a finished batch is freed by
    reference counting alone (a cycle would hold its GPU tensors until a full gc)."""
    import gc
    import weakref
    from timegram.simulate import simulate_batch
    gc.disable()
    try:
        model = tg.ets("ANN") | tg.ets("AAN") | tg.ar(1)
        _, _, b = simulate_batch(model, 64, (10, 20), 4, torch.Generator().manual_seed(0), "cpu")
        children = [v for k, v in b._memo.items() if k[0] == "subset"]
        assert children
        refs = [weakref.ref(b)] + [weakref.ref(c) for c in children]
        del b, children
        assert all(r() is None for r in refs)
    finally:
        gc.enable()


def _from_pacf(pacf):
    from timegram.dists import pacf_to_coefs
    return pacf_to_coefs(np.array(pacf))


def _repeated_root(root, k):
    """AR coefficients with a k-fold root, rounded to float32 as the fast backend stores them."""
    return (-np.poly([root] * k)[1:]).astype(np.float32).astype(float)


@pytest.mark.parametrize("phi,period", [
    (_from_pacf([0.5, -0.3, 0.2, 0.1, 0.99998]), 1), (_from_pacf([0.9, 0.99999]), 4),
    (_from_pacf([-0.99999]), 1), (_from_pacf([0.3, 0.2]), 12),
    (_repeated_root(0.95, 5), 1), (_repeated_root(0.9, 5), 1), (_repeated_root(-0.97, 3), 1)])
def test_fast_ar_is_exact_near_unit_and_repeated_roots(phi, period):
    """The fast AR must not assume a quickly decaying impulse response (near-unit roots once
    gave wrapped-around or NaN series), nor evaluate a badly conditioned polynomial in
    float32 (five roots at 0.95 once gave all-NaN output)."""
    from timegram.simulate import Batch, Series
    m = tg.ar(len(phi), period=period, phi=list(phi))
    x = np.random.default_rng(0).normal(size=700)
    s = Series(np.random.default_rng(0))
    s.t = np.arange(1, 701)
    ref = m.apply(s, x)
    b = Batch(1, torch.Generator().manual_seed(0), "cpu")
    b.t = torch.arange(1, 701).float().unsqueeze(0)
    fast = m.apply_batch(b, torch.tensor(x, dtype=torch.float32).unsqueeze(0))[0].numpy()
    assert np.isfinite(fast).all() and np.abs(fast - ref).max() <= 1e-5 * np.abs(ref).max()


def test_exp_makes_an_additive_world_positive_and_multiplicative():
    world = tg.ets("AAA", period=4) >> tg.exp(rate=0.05)
    for backend in ("reference", "fast"):
        sim = world.simulate(n=6, length=40, seed=0, backend=backend)
        assert (sim.context > 0).all() and np.isfinite(sim.context).all()
    x = np.random.default_rng(0).normal(size=50).cumsum()
    s = Series(np.random.default_rng(0))
    s.t = np.arange(1, 51)
    ref = tg.exp(rate=0.05).apply(s, x)
    b = Batch(1, torch.Generator().manual_seed(0), "cpu")
    b.t = torch.arange(1, 51).float().unsqueeze(0)
    fast = tg.exp(rate=0.05).apply_batch(b, torch.tensor(x, dtype=torch.float32).unsqueeze(0))[0].numpy()
    np.testing.assert_allclose(fast, np.exp(0.05 * x), rtol=1e-5)
    np.testing.assert_allclose(ref, np.exp(0.05 * x), rtol=1e-12)
