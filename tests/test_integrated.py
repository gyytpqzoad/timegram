"""Regressions for the integrated implementation."""
import warnings

import numpy as np
import pytest

import timegram as tg
from timegram import library
from timegram.equivariance import EquivarianceWarning, level_status


# F2: associative compositions share parameter paths and head layout ------------------------
def test_associative_compositions_have_the_same_parameter_layout():
    x = tg.noise()
    ma, ar = tg.ma(1, theta=[0.4]), tg.ar(1, phi=[0.6])
    left = ((x >> ma) >> ar).estimate(tg.forecast(h=2), tg.params())
    right = (x >> (ma >> ar)).estimate(tg.forecast(h=2), tg.params())
    assert left.layout() == right.layout()
    assert list(left.layout()[1]) == ["ar.phi", "ar.ma.theta"]
    assert library.canonical(left) == library.canonical(right)


def test_a_standalone_composite_is_named_as_its_chain():
    ma, ar = tg.ma(1), tg.ar(1)
    assert list(tg.model.named_params(ma >> ar)) == list(tg.model.named_params(tg.model.Feed(ma, ar)))


# F3: feeding innovations does not change the default horizon ------------------------------
@pytest.mark.parametrize("make", [
    lambda: tg.ets("AAA", period=12),
    lambda: tg.noise() >> tg.ets("AAA", period=12),
    lambda: tg.ma(1, period=12),
    lambda: tg.noise() >> tg.ma(1, period=12),
    lambda: tg.ma(1, period=12) >> tg.ar(1),
    lambda: tg.noise(tg.student_t(5)) >> (tg.ma(1, period=12) >> tg.ar(1)),
])
def test_default_horizon_sees_fed_and_composed_blocks(make):
    assert make().estimate(tg.forecast()).layout()[0] == 24


# F4: convex weights are validated, identically in both backends ----------------------------
@pytest.mark.parametrize("values,weight", [
    ([0., 1.], 1.2), ([0., 1.], -0.1), ([1., 2., 3.], [.2, .2]), ([1., 1., 1.], [.5, .5, .5]),
    ([1., 2., 3.], .5), ([1., 2., 3.], [.5, -.1, .6]),
])
def test_invalid_fixed_convex_weights_are_rejected(values, weight):
    with pytest.raises(ValueError):
        tg.convex(*(tg.constant(v) for v in values), weight=weight)


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_invalid_drawn_convex_weights_are_rejected(backend):
    model = tg.convex(tg.constant(0.), tg.constant(1.), weight=tg.uniform(0.5, 1.5))
    with pytest.raises(ValueError, match="convex weights"):
        model.simulate(n=50, length=3, seed=0, backend=backend)


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_valid_convex_weights_reach_every_operand(backend):
    values = np.array([1., 2., 4.])
    model = tg.convex(*(tg.constant(v) for v in values), weight=[.2, .3, .5])
    out = model.simulate(n=2, length=3, seed=0, backend=backend).context
    np.testing.assert_allclose(out[:, -1], values @ [.2, .3, .5], rtol=1e-6)


# F1: equivariance warning: once, non-blocking, only when the level is not free ------------
@pytest.mark.parametrize("model,status", [
    (lambda: tg.ets("AAA", period=4), "free"),
    (lambda: tg.worlds.m1_ets(), "free"),
    (lambda: tg.noise() >> tg.integrate(), "free"),
    (lambda: tg.ar(1), "fixed"),
    (lambda: tg.noise() >> tg.garch(1, 1), "fixed"),
    (lambda: tg.constant(tg.uniform(-5, 5)) + tg.ar(1), "free"),
    (lambda: tg.worlds.low_informative("quarterly"), "partly"),
    (lambda: tg.ets("ANN") | tg.ar(1), "partly"),
])
def test_level_status(model, status):
    assert level_status(model()) == status


def test_pretraining_warns_for_a_fixed_mean_world_and_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    problem = tg.ar(1, phi=[0.5]).estimate(tg.forecast(h=1)).loss(tg.absolute(scale="relative"))
    with pytest.warns(EquivarianceWarning, match="equivariant"):
        est = problem.pretrain(lengths=(8, 16), budget=256, batch_size=128, verbose=False, device="cpu")
    assert est.forecast([np.arange(10.0)]).shape == (1, 1)


def test_pretraining_is_silent_for_a_free_level_world(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    problem = tg.ets("ANN").estimate(tg.forecast(h=1)).loss(tg.absolute(scale="relative"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", EquivarianceWarning)
        problem.pretrain(lengths=(9, 16), budget=256, batch_size=128, verbose=False, device="cpu")
