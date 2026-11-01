"""Grammar semantics: operators, parameters, canonical forms, priors."""
import numpy as np
import pytest

import timegram as tg
from timegram import library
from timegram.model import named_params


def test_precedence_gotcha_is_an_error_not_a_silent_model():
    with pytest.raises(TypeError, match="precedence"):
        tg.noise() >> tg.ar(1) + tg.pattern(4)


def test_parameter_paths_and_shared_parameters():
    w = tg.uniform(0, 1)
    blend = w * tg.ets("ANN") + (1 - w) * tg.ar(1)
    paths = named_params(blend)
    assert "add.0.scale.c" in paths and "add.0.scale.ets.alpha" in paths
    sim = blend.simulate(n=50, length=10, seed=0)
    w_draws = sim.params["add.0.scale.c"]
    one_minus = sim.params["add.1.scale.c"]
    np.testing.assert_allclose(w_draws + one_minus, 1.0)          # one w per series, shared


def test_named_compound_is_its_tree():
    m = tg.ets("AAA", period=4)
    assert "integrate(1, period=4)" in m.expanded()
    assert "lag(4)" in m.expanded()


def test_canonical_form_equivalences():
    def key(model):
        return library.canonical(model.estimate(tg.forecast(h=4)))
    a, b = tg.ar(1, phi=tg.uniform(-0.5, 0.5)), tg.ma(1, theta=tg.uniform(-0.5, 0.5))
    assert key(a + b) == key(b + a)                                # addition commutes
    assert key((a + b) + tg.constant(1.0)) == key(a + (b + tg.constant(1.0)))   # and flattens
    assert key(tg.ets("AAA", period=4)) == key(tg.ets("AAA", period=4))      # fresh objects, same spec
    assert key(tg.ets("AAA", period=4)) != key(tg.ets("AAA", period=12))
    e = tg.noise()
    assert key(e + e) != key(e + e.copy())                         # sharing is part of the model


def test_reference_and_fast_priors_agree_in_distribution():
    model = tg.priors.low_informative("quarterly")
    ref = tg.simulate(model, n=1500, lengths=(10, 64), horizon=8, seed=0)
    fast = tg.simulate(model, n=6000, lengths=(10, 64), horizon=8, seed=0, backend="fast")
    for key in ("arima.d", "arima.D", "arima.p", "arima.P"):
        assert abs(ref.params[key].mean() - fast.params[key].mean()) < 0.06, key
    assert abs(ref.lengths.mean() - fast.lengths.mean()) < 1.5


def test_structure_first_lengths_respect_identifiability():
    model = tg.priors.low_informative("quarterly")
    sim = tg.simulate(model, n=500, lengths=(10, 64), horizon=0, seed=1)
    from timegram.compounds import arima_min_length
    p = sim.params
    floor = arima_min_length(p["arima.p"], p["arima.q"], p["arima.d"], p["arima.P"], p["arima.Q"],
                             p["arima.D"], 4)
    assert np.all(sim.lengths >= np.maximum(floor, 10))


def test_batch_reuse_is_opt_in():
    from timegram.train import TrainConfig
    assert TrainConfig().reuse == 1 and TrainConfig().backend == "reference"


def test_only_the_pipe_operator_flattens_selection():
    a, b, c = tg.constant(1.0), tg.constant(2.0), tg.constant(3.0)
    assert len(((a | b) | c).operands) == 3
    assert len(tg.select(tg.select(a, b), c).operands) == 2


def test_canonical_form_sees_the_configuration_of_fed_blocks():
    def key(model):
        return library.canonical(model.estimate(tg.forecast(h=4)))
    e = tg.noise()
    assert key(e >> tg.integrate(1)) != key(e >> tg.integrate(2))
    assert key(e >> tg.lag(4)) != key(e >> tg.lag(12))
    assert key(tg.ets("ANN") >> tg.normalize("std")) != key(tg.ets("ANN") >> tg.normalize("minmax"))
    assert key(tg.mix(tg.ets("ANN"), tg.ets("AAN"))) != key(tg.select(tg.ets("ANN"), tg.ets("AAN")))
    assert "0x" not in key(tg.mix(tg.ets("ANN"), tg.ets("AAN")))          # no memory addresses


def test_ets_letter_codes_and_keywords_build_the_same_model():
    pairs = [("ANN", {}), ("AAN", {"trend": "additive"}), ("AAdN", {"trend": "damped"}),
             ("ANA", {"season": 4}), ("AAA", {"trend": "additive", "season": 12}),
             ("AAdA", {"trend": "damped", "season": 4})]
    for code, kw in pairs:
        period = kw.get("season")
        a, b = tg.ets(code, period=period), tg.ets(**kw)
        assert a.label() == b.label().replace(b.spec, code)
        assert (a.has_trend, a.damped, a.has_season, a.period) == (b.has_trend, b.damped, b.has_season, b.period)
        assert library.canonical(a.estimate(tg.forecast(h=2))) == library.canonical(b.estimate(tg.forecast(h=2)))
    with pytest.raises(ValueError):
        tg.ets("AAA", trend="damped")


def test_convex_is_the_combination_name_and_mix_an_alias():
    assert tg.mix is tg.convex
    assert tg.convex(tg.ets("ANN"), tg.ets("AAN")).label().startswith("convex(")


def test_composition_is_associative_and_reusable():
    x = tg.noise(tg.student_t(5))
    ma, ar = tg.ma(1, theta=[0.4]), tg.ar(1, phi=[0.6])
    left = (x >> ma) >> ar
    right = x >> (ma >> ar)
    for backend in ("reference", "fast"):
        a = left.simulate(n=3, length=30, seed=2, backend=backend).context
        b = right.simulate(n=3, length=30, seed=2, backend=backend).context
        np.testing.assert_allclose(a, b, atol=1e-5)
    filt = tg.ma(1) >> tg.ar(1)                                    # an open transform
    assert isinstance(filt, tg.Block) and filt.label().startswith("compose(")
    assert set(tg.model.named_params(tg.noise() >> filt)) >= {"ar.ma.theta", "ar.phi"}   # named as the chain
    assert tg.compose(x, ma, ar).label() == left.label()
    assert tg.feed is tg.compose


def test_a_standalone_composite_simulates_as_before():
    """Before compose, `ma >> ar` fed ma's default noise into ar; it still does."""
    ma, ar = tg.ma(1, theta=[0.4]), tg.ar(1, phi=[0.6])
    composite = ma >> ar
    explicit = tg.model.Feed(ma, ar)
    np.testing.assert_allclose(composite.simulate(n=2, length=25, seed=4).context,
                               explicit.simulate(n=2, length=25, seed=4).context, atol=1e-12)


def test_equivalent_compositions_share_a_library_key():
    def key(model):
        return library.canonical(model.estimate(tg.forecast(h=2)))
    x = tg.noise()
    ma, ar = tg.ma(1, theta=[0.4]), tg.ar(1, phi=[0.6])
    assert key((x >> ma) >> ar) == key(x >> (ma >> ar)) == key(tg.compose(x, ma, ar))


def test_arma_sd_reference_and_fast_agree_and_match_theory():
    import torch
    from timegram.priors import arma_sd, arma_sd_batch
    assert abs(arma_sd([0.5], [], [], [], 1) - 1 / np.sqrt(1 - 0.25)) < 1e-9          # AR(1)
    assert abs(arma_sd([], [0.4], [], [], 1) - np.sqrt(1 + 0.16)) < 1e-9              # MA(1)
    rng = np.random.default_rng(3)
    phi, theta = tg.stationary(3, 0.9).sample(rng, None), tg.invertible(2).sample(rng, None)
    Phi, Theta = np.array([0.6, 0.2]), np.array([-0.3, 0.0])
    ref = arma_sd(phi, theta, Phi, Theta, 4)
    t = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32).unsqueeze(0)   # noqa: E731
    fast = float(arma_sd_batch(t(phi), t(theta), t(Phi), t(Theta), 4)[0])
    assert abs(fast - ref) / ref < 1e-3


def test_curated_prior_starting_states_are_explicit():
    model = tg.priors.low_informative("quarterly")
    assert "start" not in model.params() and {"level0", "season0"} <= set(model.params())
    sim = tg.simulate(model, n=400, lengths=(10, 64), horizon=0, seed=0)
    D, d = sim.params["arima.D"], sim.params["arima.d"]
    s0, l0 = sim.params["arima.season0"], sim.params["arima.level0"]
    assert np.all(s0[D == 0] == 0) and np.any(s0[D == 1] != 0)
    np.testing.assert_allclose(s0.sum(1), 0, atol=1e-9)                 # zero-sum pattern
    assert np.all(l0[(d == 0) | (D == 0)] == 0)


def test_spectral_starting_states_match_simulated_warm_up_sums():
    """White noise: per-phase sums over m seasons are i.i.d. N(0, m) before centering."""
    from timegram.priors import starting_states
    rng, s = np.random.default_rng(5), 4
    pats = np.array([starting_states(rng, [], [], [], [], s, 0, 1, 40)[1] for _ in range(20000)])
    k_mean = (40 + 5 * s) / 2                                   # E[k], k ~ U(0, 40 + 5 s)
    expected_var = k_mean / s * (1 - 1 / s)                     # centred i.i.d. N(0, m), m = k / s
    assert abs(pats.var(0).mean() - expected_var) / expected_var < 0.05
    np.testing.assert_allclose(pats.sum(1), 0, atol=1e-9)


def test_worlds_namespace():
    assert tg.worlds.low_informative("quarterly").label().startswith("arima(")
    assert len(tg.worlds.backbone().operands) == 7
    assert len(tg.worlds.ets_family().operands) == 9
    m1 = tg.worlds.m1_ets()
    assert len(m1.operands) == 2 and m1.operands[1].label().startswith("convex(")
    m1.simulate(n=20, length=40, horizon=18, seed=0, backend="fast")
    assert tg.priors.low_informative("monthly").period == 12          # earlier name still works
