import numpy as np

import timegram as tg


def _sim(world, backend):
    sim = tg.simulate(world, n=300, lengths=(80, 80), seed=3, backend=backend, device="cpu")
    return np.asarray(sim.context)[:, -80:]


def test_indicator_turns_a_latent_process_into_persistent_occurrence():
    latent = tg.constant(-0.5) + (tg.noise() >> tg.ar(1, phi=0.9))
    world = latent >> tg.indicator()
    for backend in ("reference", "fast"):
        x = _sim(world, backend)
        assert set(np.unique(x)) <= {0.0, 1.0}
        same = (x[:, 1:] == x[:, :-1]).mean()           # persistence: runs of demand and of none
        assert same > 0.7
    world.typecheck(history_length=80)


def test_counts_rounds_to_non_negative_integers():
    world = (tg.uniform(0.5, 3) * tg.noise()) >> tg.counts()
    for backend in ("reference", "fast"):
        x = _sim(world, backend)
        assert (x >= 0).all() and np.allclose(x, np.round(x))
    world.typecheck(history_length=80)


def test_filters_have_distinct_identities():
    a, b = tg.noise() >> tg.indicator(), tg.noise() >> tg.counts()
    assert tg.identity.world(a).digest != tg.identity.world(b).digest
