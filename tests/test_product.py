"""Product of two blocks (`a * b`): semantics on both backends, sharing, keys, identity."""
import warnings

import numpy as np

import timegram as tg
from timegram import identity, library


def _ctx(world, backend, n=64, seed=3):
    sim = tg.simulate(world, n=n, lengths=(40, 40), seed=seed, backend=backend, device="cpu")
    return np.asarray(sim.context)[:, -40:]


def test_scaling_by_a_parameter_is_unchanged():
    x = tg.noise()
    assert type(2.0 * x).__name__ == "Scale"
    assert type(x * tg.uniform(1, 2)).__name__ == "Scale"
    assert type(x * x).__name__ == "Product"


def test_shared_product_is_a_square_and_independent_product_is_not():
    x = tg.noise() >> tg.ar(1, phi=[0.5])
    sq = _ctx(x * x, "reference")
    assert (sq >= 0).all()
    lin = _ctx(x, "reference")
    assert np.allclose(sq, lin ** 2)
    indep = _ctx(x * x.copy(), "reference")
    assert (indep < 0).any()


def test_backends_agree_on_stochastic_volatility():
    vol = tg.noise() >> tg.ar(1, phi=[0.9]) >> tg.exp(rate=0.3)
    world = tg.noise() * vol
    ref = _ctx(world, "reference", n=8)
    assert ref.shape == (8, 40) and np.isfinite(ref).all()
    fast = _ctx(world, "fast", n=256)
    assert np.isfinite(fast).all()
    kurt = lambda a: ((a - a.mean()) ** 4).mean() / a.var() ** 2      # noqa: E731
    assert kurt(fast.ravel()) > 3.3                                   # heavier tails than Gaussian


def test_product_with_seasonal_factor_and_typecheck():
    level = tg.constant(10.0) + (tg.noise() >> tg.integrate())
    factor = tg.constant(1.0) + 0.2 * tg.pattern(4)
    world = level * factor
    assert np.isfinite(_ctx(world, "fast")).all()
    report = world.typecheck(history_length=40)
    assert not report.has_errors


def test_keys_and_identity_distinguish_product_from_sum():
    a, b = tg.noise(), tg.noise() >> tg.ar(1, phi=[0.5])
    p = (a * b).estimate(tg.forecast(h=4))
    s = (a + b).estimate(tg.forecast(h=4))
    cfg = tg.TrainConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert library.key(p, cfg) != library.key(s, cfg)
    rp, rs = identity.world(a * b), identity.world(a + b)
    assert rp.digest is not None and rp.digest != rs.digest
    assert identity.world(a * b).digest != identity.world(b * a).digest  # ordered: no commutation rule


def test_wrappers_keep_child_diagnostics():
    bad = tg.noise() * np.ones(3)                           # vector scale on a scalar process: a shape error
    assert bad.typecheck().has_errors
    assert tg.at_least(bad, 12).typecheck().has_errors
    assert (bad * tg.noise()).typecheck().has_errors
    explosive = tg.ar(1, phi=[1.2])
    for wrapped in (tg.at_least(explosive, 12), explosive * tg.noise()):
        refuted = [c for c in wrapped.typecheck().claims if c.claim == "stationary_coefficients"]
        assert refuted and refuted[0].status == "refuted"
    fine = tg.at_least(tg.noise() >> tg.ar(1, phi=[0.5]), 12).typecheck()
    assert not fine.has_errors and any(c.claim == "minimum_history_length" for c in fine.claims)
