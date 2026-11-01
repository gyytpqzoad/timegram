"""tg.at_least: a minimum history length for any world."""
import warnings

import numpy as np
import pytest

import timegram as tg
from timegram import identity, library


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_floor_is_enforced_and_values_are_unchanged(backend):
    world = tg.noise() >> tg.ar(1, phi=[0.5])
    sim = tg.simulate(tg.at_least(world, 30), n=64, lengths=(7, 40), seed=4, backend=backend, device="cpu")
    assert np.asarray(sim.lengths).min() >= 30
    same = tg.simulate(tg.at_least(world, 1), n=8, lengths=(35, 35), seed=4, backend=backend, device="cpu")
    plain = tg.simulate(world, n=8, lengths=(35, 35), seed=4, backend=backend, device="cpu")
    assert np.allclose(np.asarray(same.context), np.asarray(plain.context), equal_nan=True)


def test_inner_floor_still_counts_and_bad_lengths_are_rejected():
    inner = tg.ets("AAA", period=12)                     # carries its own floor
    sim = tg.simulate(tg.at_least(inner, 9), n=32, lengths=(7, 60), seed=1, backend="fast", device="cpu")
    assert np.asarray(sim.lengths).min() >= 9
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            tg.at_least(tg.noise(), bad)


def test_key_and_identity_record_the_floor():
    w = tg.noise() >> tg.integrate()
    cfg = tg.TrainConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k1 = library.key(tg.at_least(w, 10).estimate(tg.forecast(h=2)), cfg)
        k2 = library.key(tg.at_least(w, 20).estimate(tg.forecast(h=2)), cfg)
    assert k1 != k2
    d1, d2 = identity.world(tg.at_least(w, 10)).digest, identity.world(tg.at_least(w, 20)).digest
    assert d1 is not None and d1 != d2
    assert not tg.at_least(w, 10).typecheck(history_length=12).has_errors
