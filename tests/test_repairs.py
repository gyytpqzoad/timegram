"""Result-preserving repairs: these tests assert correct behavior."""
import hashlib
import importlib

import numpy as np
import pytest
import torch

import timegram as tg
from timegram import library
from timegram.estimator import Estimator
from timegram.train import TrainConfig, training_batch


@pytest.mark.parametrize("settings,field", [
    ({"batch_size": 4, "chunk": 2}, "chunk"),
    ({"replicates": 0}, "replicates"),
    ({"accumulate": 0}, "accumulate"),
    ({"batch_size": 0}, "batch_size"),
    ({"chunk": 0}, "chunk"),
    ({"budget": 0}, "budget"),
    ({"budget": -1}, "budget"),
    ({"batch_size": 4.5}, "batch_size"),
])
def test_invalid_batch_configuration_fails_at_construction(settings, field):
    with pytest.raises(ValueError, match=field):
        TrainConfig(**settings)


def test_existing_batch_configuration_contract_is_preserved():
    for batch, reps in [(2048, 1), (4096, 32), (4096, 128)]:
        assert TrainConfig(batch_size=batch, chunk=16384, replicates=reps).chunk == 16384
    assert TrainConfig(batch_size=4).chunk is None
    # Preserve the existing treatment of trailing rows; changing it would alter draws.
    assert TrainConfig(batch_size=4, chunk=6).chunk == 6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.0, 2.0])
def test_laplace_endpoint_is_finite_and_other_values_are_unchanged(monkeypatch, dtype, scale):
    draws = torch.tensor([0., 2**-24, .125, .25, .5, .75, 1 - 2**-24], dtype=dtype)
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: draws.clone())
    actual = tg.laplace(loc=3., scale=scale).sample_iid_batch(torch.Generator(), draws.shape, "cpu")
    u = draws - .5
    legacy = 3. - scale * torch.sign(u) * torch.log1p(-2 * u.abs())
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[1:], legacy[1:], rtol=0, atol=0)
    if scale == 0:
        assert torch.equal(actual, torch.full_like(actual, 3.))
    else:
        assert actual[0] < actual[1]  # preserve the extreme left tail's ordering


def test_laplace_keeps_finite_seeded_draws_and_rng_consumption():
    a, b = torch.Generator().manual_seed(13), torch.Generator().manual_seed(13)
    actual = tg.laplace().sample_iid_batch(a, (4096,), "cpu")
    u = torch.rand((4096,), generator=b) - .5
    legacy = -torch.sign(u) * torch.log1p(-2 * u.abs())
    assert torch.isfinite(legacy).all()  # this fixture contains no singular draw
    assert torch.equal(actual, legacy)
    assert torch.equal(a.get_state(), b.get_state())


def test_joint_key_shapes_follow_prior_configuration_without_new_serialized_fields():
    prior = tg.priors.SarimaTree(period=12, max_order=5, max_seasonal_order=1)
    for key, shape in {"phi": (5,), "theta": (5,), "Phi": (1,), "Theta": (1,),
                       "season0": (12,), "p": (), "mean": ()}.items():
        assert prior[key].shape == shape
        assert set(vars(prior[key])) == {"joint", "key"}
    assert tg.priors.Garch11()["alpha"].shape == ()
    assert tg.ets("AAA", period=4).params()["alpha"].shape == ()


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_joint_vector_targets_equal_the_simulated_padded_coefficients(backend):
    model = tg.worlds.low_informative("quarterly")
    problem = model.estimate(tg.params("phi", "theta", "Phi", "Theta"))
    h, slots, width = problem.layout()
    assert h == 0 and width == 10
    assert [size for _, size in slots.values()] == [3, 3, 2, 2]
    cfg = TrainConfig(lengths=(24, 32), batch_size=8, budget=16, backend=backend)
    _, actual = training_batch(problem, cfg, h, slots, np.random.default_rng(7),
                               torch.Generator().manual_seed(7), torch.device("cpu"), 8)
    sim = tg.simulate(model, n=8, lengths=cfg.lengths, seed=7, backend=backend)
    expected = np.concatenate([sim.params[path] for path in slots], axis=1)
    np.testing.assert_array_equal(actual.numpy(), expected.astype(np.float32))
    for coef, order in [("phi", "p"), ("theta", "q"), ("Phi", "P"), ("Theta", "Q")]:
        values = sim.params[f"arima.{coef}"]
        inactive = np.arange(values.shape[1])[None, :] >= sim.params[f"arima.{order}"][:, None]
        assert np.all(values[inactive] == 0)


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_vector_parameter_estimation_trains_and_roundtrips(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    problem = tg.worlds.low_informative("quarterly").estimate(tg.params("phi"))
    est = problem.pretrain(lengths=(24, 32), budget=32, batch_size=8, chunk=16,
                           net=tg.ResMLP(width=8, depth=1), backend=backend,
                           device="cpu", verbose=False)
    values = est.predict([np.arange(24.)])
    assert values.shape == (1, 3) and np.isfinite(values).all()
    again = library.load(est.key, problem=problem)
    np.testing.assert_array_equal(again.predict([np.arange(24.)]), values)


def _collision_problems():
    a, b = tg.ar(1, phi=[.2]), tg.ar(1, phi=[.8])
    return ((a + b + a).estimate(tg.forecast(h=1)),
            (b + a + b).estimate(tg.forecast(h=1)))


def _estimator(problem, config):
    h, slots, outputs = problem.layout()
    return Estimator(problem, config.net.build(config.lengths[1], outputs), config,
                     h, slots, [], key=library.key(problem, config))


def _files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_legacy_collision_cannot_relabel_overwrite_or_silently_retrain(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    p, q = _collision_problems()
    settings = dict(lengths=(8, 16), budget=16, batch_size=8, device="cpu",
                    net=tg.ResMLP(width=8, depth=1))
    cfg = TrainConfig(**settings)
    first, second = _estimator(p, cfg), _estimator(q, cfg)
    # Legacy keys deliberately remain unchanged; this is a fail-closed guard.
    assert first.key == second.key
    library.save(first, name="original")
    before = _files(tmp_path)

    def forbidden_train(*a, **kw):
        pytest.fail("cache rejection must happen before training")

    monkeypatch.setattr(importlib.import_module("timegram.train"), "train", forbidden_train)
    attempts = [lambda: library.load(first.key, problem=q),
                lambda: tg.pretrained("original", problem=q),
                lambda: library.save(second, name="wrong"),
                lambda: q.pretrain(verbose=False, **settings),
                lambda: q.pretrain(force_train=True, verbose=False, **settings)]
    for attempt in attempts:
        with pytest.raises(library.CacheCollisionError, match="refusing to reuse or overwrite"):
            attempt()
        assert _files(tmp_path) == before

    fresh_p, _ = _collision_problems()
    for loaded in [tg.pretrained("original"), library.load(first.key, problem=fresh_p),
                   fresh_p.pretrain(verbose=False, **settings)]:
        np.testing.assert_array_equal(loaded.forecast([np.arange(12.)]),
                                      first.forecast([np.arange(12.)]))
    assert _files(tmp_path) == before


def test_guard_rejects_rounded_float_collision_and_conservative_reordering(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    cfg = TrainConfig(lengths=(8, 16), budget=8, batch_size=8, net=tg.ResMLP(width=8, depth=1))
    p = tg.ar(1, phi=[.2]).estimate(tg.forecast(h=1))
    q = tg.ar(1, phi=[.2 + 1e-14]).estimate(tg.forecast(h=1))
    assert library.key(p, cfg) == library.key(q, cfg)
    library.save(_estimator(p, cfg))
    with pytest.raises(library.CacheCollisionError):
        library.load(library.key(p, cfg), problem=q)
    a, b = tg.ar(1, phi=[.2]), tg.ma(1, theta=[.3])
    left, right = (a + b).estimate(tg.forecast(h=1)), (b + a).estimate(tg.forecast(h=1))
    assert library.key(left, cfg) == library.key(right, cfg)
    library.save(_estimator(left, cfg))
    with pytest.raises(library.CacheCollisionError, match="reordered"):
        library.load(library.key(left, cfg), problem=right)
