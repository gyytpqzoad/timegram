"""Definitions are checked where they are made: invalid parameters fail at construction,
drawn values with the wrong shape or kind fail when used, and both backends agree."""
import numpy as np
import pytest

import timegram as tg
from timegram.dists import Joint


@pytest.mark.parametrize("make", [
    lambda: tg.normal(0, -1), lambda: tg.normal(float("nan"), 1), lambda: tg.uniform(2, 1),
    lambda: tg.student_t(0), lambda: tg.laplace(0, -1), lambda: tg.dists.Choice([1, 2], probs=[1, -1]),
    lambda: tg.dists.Stationary(2, pacf_max=1.5), lambda: tg.dists.Dirichlet(0),
    lambda: tg.dists.Integers(1.5, 3),
])
def test_invalid_distribution_parameters_fail_at_construction(make):
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize("make", [
    lambda: tg.integrate(1.5), lambda: tg.integrate(-1), lambda: tg.integrate(tg.uniform(0, 2)),
    lambda: tg.integrate(tg.dists.Choice([0, 0.5])), lambda: tg.difference(1.5),
    lambda: tg.ar(2, phi=[.5]), lambda: tg.ma(1, theta=[.2, .1]), lambda: tg.ar(1, phi=[float("nan")]),
    lambda: tg.ar(1.5), lambda: tg.ma(1, period=0), lambda: tg.pattern(0),
])
def test_invalid_block_definitions_fail_at_construction(make):
    with pytest.raises(ValueError):
        make()


def test_valid_integer_orders_and_scalar_coefficients_still_work():
    for m in (tg.integrate(2), tg.integrate(np.int64(1)), tg.integrate(tg.integers(0, 2)),
              tg.integrate(tg.dists.Choice([0, 1])), tg.ar(1, phi=0.5), tg.ma(1, theta=-0.3)):
        for backend in ("reference", "fast"):
            assert np.isfinite(m.simulate(n=3, length=16, seed=0, backend=backend).context).all()


class _Bad(Joint):
    """A joint prior whose draws have the wrong kind: a fractional order and 3 coefficients."""
    keys = ("d", "phi")

    def sample(self, rng, draw):
        return {"d": 1.5, "phi": np.array([.2, .3, .4])}

    def sample_batch(self, gen, n, device, draw):
        import torch
        return {"d": torch.full((n,), 1.5), "phi": torch.full((n, 3), .1)}


@pytest.mark.parametrize("backend", ["reference", "fast"])
@pytest.mark.parametrize("which", ["d", "phi"])
def test_bad_draws_fail_in_both_backends(backend, which):
    bad = _Bad()
    m = tg.integrate(bad["d"]) if which == "d" else tg.ar(2, phi=bad["phi"])
    with pytest.raises(ValueError, match="drawn"):
        m.simulate(n=2, length=16, seed=0, backend=backend)


def test_simulation_can_skip_the_structural_check():
    m = tg.ets("AAA", period=4)
    a = tg.simulate(m, n=2, lengths=(16, 16), seed=0)
    b = tg.simulate(m, n=2, lengths=(16, 16), seed=0, validate=False)
    np.testing.assert_array_equal(a.context, b.context)


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_garch11_prior_stays_inside_the_stationary_region(backend):
    g = tg.priors.Garch11()
    m = tg.noise() >> tg.garch(omega=g["omega"], alpha=g["alpha"], beta=g["beta"], burn_in=200)
    sim = m.simulate(n=64, length=64, seed=0, backend=backend)
    assert np.isfinite(sim.context).all()
    a, b, w = (np.asarray(sim.params[k], float).reshape(-1) for k in
               ("garch.alpha", "garch.beta", "garch.omega"))
    assert (w > 0).all() and (a >= 0).all() and (b >= 0).all() and (a + b < 1).all()


def test_garch11_rejects_ranges_outside_the_region():
    with pytest.raises(ValueError):
        tg.priors.Garch11(persistence=(0.5, 1.0))
    with pytest.raises(ValueError):
        tg.priors.Garch11(variance=(0.0, 1.0))


def test_a_random_vector_times_a_random_scalar_in_both_backends():
    from timegram.blocks import Centered
    season0 = Centered(tg.uniform(-1, 1), 4) * tg.uniform(0, 6)
    for backend in ("reference", "fast"):
        sim = tg.simulate(tg.pattern(4, season0), n=8, lengths=(8, 8), seed=0, backend=backend)
        first = sim.context[:, :4]
        np.testing.assert_allclose(first.sum(1), 0, atol=1e-5)                 # still zero-sum
        np.testing.assert_allclose(sim.context[:, 4:], first, atol=1e-5)        # and periodic


# ---------------------------------------------------------------------------------------
# defects found by review of the typed grammar
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("make", [lambda: tg.lag(1.5), lambda: tg.lag(-1), lambda: tg.integrate(float("inf")),
                                  lambda: tg.priors.Garch11(variance=(1e-46, 1e-46)),
                                  lambda: tg.priors.Garch11(variance=(1e40, 1e40))])
def test_review_definitions_fail_at_construction(make):
    with pytest.raises(ValueError):
        make()


def test_a_degenerate_continuous_law_at_an_integer_is_an_integer_order():
    for backend in ("reference", "fast"):
        sim = tg.simulate(tg.noise() >> tg.integrate(tg.uniform(1, 1)), n=3, lengths=(8, 8), seed=0, backend=backend)
        drawn = next(v for k, v in sim.params.items() if k.endswith(".d"))
        np.testing.assert_array_equal(drawn, 1.0)
    with pytest.raises(ValueError):
        tg.integrate(tg.uniform(1.5, 1.5))


class _ScalarBatch(Joint):
    """A joint prior whose batched draw is a single scalar instead of one row per series."""
    keys = ("phi",)

    def sample(self, rng, draw):
        return {"phi": np.array([.1, .2, .3])}

    def sample_batch(self, gen, n, device, draw):
        import torch
        return {"phi": torch.tensor(.1, device=device)}


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_a_malformed_scalar_draw_is_a_validation_error(backend):
    with pytest.raises(ValueError, match="drawn"):
        tg.ar(2, phi=_ScalarBatch()["phi"]).simulate(n=2, length=16, seed=0, backend=backend)


def test_check_batches_catches_an_extension_that_rewires_itself():
    """Validation runs once per training run; a custom block that changes the model while
    it is simulated breaks that contract. check_batches=True catches it as a cycle."""
    from timegram.model import Source
    from timegram.train import TrainConfig, training_batch

    class Rewiring(Source):
        name = "rewiring"

        def __init__(self):
            self.child = tg.constant(1)

        def inputs(self):
            return [("child", self.child)]

        def run(self, s):
            result = s.value(self.child)
            self.child = self                                   # a cycle, after the first use
            return result

    problem = Rewiring().estimate(tg.forecast(1))
    cfg = TrainConfig(lengths=(8, 8), device="cpu", check_batches=True)
    training_batch(problem, cfg, 1, {}, np.random.default_rng(0), None, "cpu", 1)
    with pytest.raises(ValueError, match="cycle"):
        training_batch(problem, cfg, 1, {}, np.random.default_rng(0), None, "cpu", 1)
    assert "check_batches" not in TrainConfig(check_batches=True).key_dict()


# ---------------------------------------------------------------------------------------
# further regressions
# ---------------------------------------------------------------------------------------
def test_training_rejects_non_finite_observed_values_not_only_targets():
    """NaN is legitimate only as padding before a series' first observation."""
    import torch
    from timegram.train import _check_finite
    context = torch.tensor([[float("nan"), 1.0, 2.0], [float("nan"), float("nan"), 3.0]])
    _check_finite(context, torch.tensor([2, 1]), torch.zeros(2, 0))            # padding only: fine
    with pytest.raises(FloatingPointError):
        _check_finite(context, torch.tensor([3, 1]), torch.zeros(2, 0))        # a NaN observation


def test_cancelling_or_degenerate_levels_do_not_silence_the_normalization_warning():
    from timegram.equivariance import EquivarianceWarning, level_status, warn_if_restricted
    level = tg.constant(tg.uniform(-10, 10))
    assert level_status(level - level + tg.noise()) == "unverified"           # could cancel
    assert level_status(0 * tg.constant(tg.uniform(-1, 1)) + tg.noise()) == "fixed"
    assert level_status(tg.constant(tg.uniform(0, 0)) + tg.noise()) == "fixed"
    assert level_status(tg.constant(tg.uniform(-1, 1)) + tg.ar(1)) == "free"
    for world in (tg.ets("AAA", period=4), tg.arima(1, 1, 1), tg.worlds.m1_ets()):
        assert level_status(world) == "free"                                   # unchanged
    with pytest.warns(EquivarianceWarning):
        warn_if_restricted(level - level + tg.noise())
