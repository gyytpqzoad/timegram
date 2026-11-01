import numpy as np
import pytest
import torch

import timegram as tg


def test_bernoulli_draws_zeros_and_ones_with_rate_p_on_both_backends():
    d = tg.bernoulli(0.3)
    x = d.sample_iid(np.random.default_rng(0), 20000)
    y = d.sample_iid_batch(torch.Generator().manual_seed(0), (20000,), "cpu").numpy()
    for z in (x, y):
        assert set(np.unique(z)) <= {0.0, 1.0}
        assert abs(z.mean() - 0.3) < 0.02


def test_bernoulli_validates_and_has_an_identity():
    with pytest.raises(ValueError):
        tg.bernoulli(1.5)
    a, b = tg.noise(tg.bernoulli(0.2)), tg.noise(tg.bernoulli(0.3))
    assert tg.identity.world(a).digest != tg.identity.world(b).digest
    assert tg.identity.world(a).digest == tg.identity.world(tg.noise(tg.bernoulli(0.2))).digest


def test_intermittent_demand_is_a_product_of_occurrence_and_size():
    size = (tg.uniform(0.2, 0.8) * tg.noise()) >> tg.exp()
    world = tg.noise(tg.bernoulli(0.25)) * size
    for backend in ("reference", "fast"):
        sim = tg.simulate(world, n=400, lengths=(60, 60), seed=1, backend=backend, device="cpu")
        x = np.asarray(sim.context)[:, -60:]
        assert (x >= 0).all()
        assert abs((x == 0).mean() - 0.75) < 0.03
    world.typecheck(history_length=60)
