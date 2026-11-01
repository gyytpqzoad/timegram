"""Replicated simulation and the decision-theoretic objectives (timegram.objective)."""
import math
import pickle

import numpy as np
import pytest
import torch

import timegram as tg
from timegram import library
from timegram.objective import Mean, Objective, Outputs, Worst, _squared_mean, hard_coverage
from timegram.train import TrainConfig, _backward, _observed, outputs

# the AR worlds here have a known zero mean on purpose; the equivariance warning is expected
pytestmark = pytest.mark.filterwarnings("ignore::timegram.equivariance.EquivarianceWarning")


# ---------------------------------------------------------------------------------------
# replicates
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["reference", "fast"])
@pytest.mark.parametrize("make", [lambda: tg.ar(2), lambda: tg.ets("AAA", period=4),
                                  lambda: tg.select(tg.ets("ANN"), tg.ar(1))])
def test_replicates_share_the_process_not_the_noise(backend, make):
    sim = tg.simulate(make(), n=5, replicates=4, lengths=(16, 48), horizon=3, seed=2, backend=backend)
    assert sim.context.shape == (20, 48) and sim.replicates == 4
    lengths = sim.lengths.reshape(5, 4)
    assert (lengths == lengths[:, :1]).all()                       # the length belongs to the process
    for v in sim.params.values():
        v = np.nan_to_num(np.asarray(v, float).reshape(5, 4, -1))
        assert np.allclose(v, v[:, :1])                            # so do the parameters
    target = sim.target.reshape(5, 4, 3)
    assert not np.allclose(target[:, 0], target[:, 1])             # the innovations do not


def test_one_replicate_is_ordinary_simulation():
    a = tg.simulate(tg.ets("AAN"), n=6, lengths=(16, 32), seed=3)
    b = tg.simulate(tg.ets("AAN"), n=6, replicates=1, lengths=(16, 32), seed=3)
    np.testing.assert_array_equal(a.context, b.context)


# ---------------------------------------------------------------------------------------
# estimators of process-level quantities
# ---------------------------------------------------------------------------------------
def test_debiased_squared_mean_is_unbiased_and_plug_in_is_not():
    g = torch.Generator().manual_seed(0)
    mu, R = 0.3, 4
    x = mu + torch.randn(200_000, R, generator=g)                  # Var = 1
    assert abs(_squared_mean(x, debiased=True).mean().item() - mu ** 2) < 0.01
    assert abs(_squared_mean(x, debiased=False).mean().item() - (mu ** 2 + 1 / R)) < 0.01


def test_worst_is_a_smooth_maximum_and_weights_are_its_gradient():
    c = torch.tensor([0.1, 0.5, 0.45, 0.2], requires_grad=True)
    for agg in (Mean(), Worst(psi=20.0)):
        v = agg.value(c)
        (grad,) = torch.autograd.grad(v, c)
        torch.testing.assert_close(grad, agg.weights(c))
    v = Worst(psi=20.0).value(c).item()
    assert 0.5 <= v <= 0.5 + math.log(4) / 20
    assert Worst(psi=100.0, anneal=0.5).psi_at(0.25) == pytest.approx(10.0)


def test_miscoverage_with_a_sharp_surrogate_is_the_hard_coverage_error():
    g = torch.Generator().manual_seed(1)
    J, R = 3, 50
    z = torch.randn(J * R, 1, generator=g)
    q = torch.full((J * R, 1, 1), 1.0)                             # q = 1 for every process
    levels = torch.tensor([0.9])
    out = Outputs(torch.zeros(J * R, 1), R, quantiles=q, target=z, levels=levels)
    cov = hard_coverage(q, z, R)[:, 0, 0]
    plug_in = tg.miscoverage(kappa=1e6, debiased=False, smoothing="fixed").per_process(out)
    torch.testing.assert_close(plug_in, (cov - 0.9) ** 2, atol=1e-5, rtol=0)


def test_bias_ignores_estimates_a_process_does_not_have():
    e = torch.tensor([[0.2, float("nan")], [0.4, float("nan")], [-0.1, 1.0], [0.1, 3.0]])
    out = Outputs(torch.zeros(4, 2), 2, errors=e)
    b = tg.bias(debiased=False).per_process(out)
    torch.testing.assert_close(b, torch.tensor([0.3 ** 2, (0.0 + 2.0 ** 2) / 2]))


# ---------------------------------------------------------------------------------------
# training: the default is ERM, accumulation is exact
# ---------------------------------------------------------------------------------------
def _setup(objective, replicates=4, rows=32, seed=0):
    problem = tg.ar(1).estimate(tg.forecast(h=2), tg.params("phi")).loss(tg.absolute(scale="relative"))
    if objective is not None:
        problem = problem.objective(*objective)
    cfg = TrainConfig(lengths=(8, 16), batch_size=rows, replicates=replicates, device="cpu",
                      net=tg.ResMLP(width=16, depth=1))
    torch.manual_seed(seed)
    net = cfg.net.build(16, problem.layout()[2])
    sim = tg.simulate(problem.model, n=rows // replicates, replicates=replicates, lengths=(8, 16), horizon=2, seed=seed)
    ctx = torch.as_tensor(sim.context, dtype=torch.float32)
    tgt = torch.cat([torch.as_tensor(sim.target, dtype=torch.float32),
                     torch.as_tensor(sim.params["ar.phi"], dtype=torch.float32).reshape(rows, 1)], 1)
    gen = torch.Generator().manual_seed(0)
    return problem, cfg, net, _observed(problem, ctx, tgt, gen)


def _grad(problem, net, group, cfg):
    net.zero_grad()
    _backward(problem, net, group, 2, cfg, progress=1.0)
    return torch.cat([p.grad.flatten() for p in net.parameters()])


def test_default_objective_is_the_mean_loss():
    problem, cfg, net, batch = _setup(None, replicates=1)
    x, m, loc, scale, tgt = batch
    out = outputs(problem, net(x, m), tgt, loc, scale, 2, cfg, 1)
    (r,) = problem.objective_spec.per_process(out)
    torch.testing.assert_close(problem.objective_spec.value([r]), out.losses.mean())


@pytest.mark.parametrize("objective", [
    [tg.risk()], [tg.risk(), tg.bias(weight=2.0)], [tg.risk(over=tg.worst(psi=5.0))],
    [tg.risk(weight=0.5), tg.bias(weight=1.0, over=tg.worst(psi=3.0))],
])
def test_accumulated_gradient_equals_one_big_batch(objective):
    problem, cfg, net, batch = _setup(objective)
    halves = [tuple(t[:16] for t in batch), tuple(t[16:] for t in batch)]
    torch.testing.assert_close(_grad(problem, net, halves, cfg), _grad(problem, net, [batch], cfg),
                               atol=1e-6, rtol=1e-4)


def test_quantile_forecaster_trains_end_to_end_and_never_crosses(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    problem = (tg.ar(1).estimate(tg.forecast(h=2, quantiles=[0.1, 0.5, 0.9])).loss(tg.pinball("relative"))
               .objective(tg.risk(weight=0.5), tg.miscoverage(weight=0.5, levels=[0.9])))
    est = problem.pretrain(lengths=(8, 16), budget=2048, batch_size=256, replicates=8, accumulate=2,
                           net=tg.ResMLP(width=16, depth=1), device="cpu", verbose=False)
    q = est.forecast([np.random.default_rng(0).normal(size=12) for _ in range(5)])
    assert q.shape == (5, 2, 3) and (np.diff(q, axis=-1) > 0).all()
    assert list(est(np.arange(12.0)).forecast.columns) == ["period", "q0.1", "q0.5", "q0.9"]
    again = library.load(est.key)                                   # saved and reloaded
    np.testing.assert_allclose(again.forecast([np.arange(12.0)]), est.forecast([np.arange(12.0)]), rtol=1e-5)


def test_warm_start_continues_from_a_library_estimator(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    small = dict(lengths=(8, 16), budget=512, batch_size=128, net=tg.ResMLP(width=16, depth=1),
                 device="cpu", verbose=False)
    problem = tg.ar(1).estimate(tg.params("phi"))
    base = problem.pretrain(**small)
    debiased = problem.objective(tg.risk(), tg.bias()).pretrain(init=base.key, replicates=4, **small)
    assert debiased.key != base.key and debiased.config.init == base.key


# ---------------------------------------------------------------------------------------
# specification checks and library keys
# ---------------------------------------------------------------------------------------
def test_criteria_must_fit_the_estimands():
    with pytest.raises(ValueError, match="quantile"):
        tg.ar(1).estimate(tg.forecast(h=1)).objective(tg.miscoverage())
    with pytest.raises(ValueError, match="pinball"):
        tg.ar(1).estimate(tg.forecast(h=1)).loss(tg.pinball())
    with pytest.raises(ValueError, match="point"):
        tg.ar(1).estimate(tg.forecast(h=1, quantiles=[.5])).loss(tg.pinball()).objective(tg.bias())
    with pytest.raises(ValueError, match="levels"):
        tg.ar(1).estimate(tg.forecast(h=1, quantiles=[.5])).loss(tg.pinball()).objective(tg.miscoverage(levels=[.9]))
    with pytest.raises(ValueError, match="increase"):
        tg.forecast(h=1, quantiles=[.9, .1])
    with pytest.raises(ValueError, match="replicates"):
        tg.ar(1).estimate(tg.params()).objective(tg.bias()).pretrain(budget=64, batch_size=32, verbose=False,
                                                                     device="cpu", save=False)
    with pytest.raises(ValueError, match="multiple"):
        TrainConfig(batch_size=100, replicates=8)


def test_default_objective_and_settings_keep_library_keys():
    problem = tg.ets("AAN").estimate(tg.forecast(h=4))
    assert "objective" not in library.canonical(problem)
    assert library.key(problem, TrainConfig()) == library.key(problem.objective(tg.risk()), TrainConfig(replicates=1))
    assert library.key(problem, TrainConfig()) != library.key(problem.objective(tg.risk(over=tg.worst())), TrainConfig())
    assert library.key(problem, TrainConfig()) != library.key(problem, TrainConfig(replicates=2))


def test_problems_pickled_before_the_objective_existed_still_load():
    problem = tg.ar(1).estimate(tg.forecast(h=2))
    state = dict(problem.__dict__)
    state.pop("objective_spec")
    old = object.__new__(type(problem))
    old.__setstate__(state)
    assert old.objective_spec == Objective()
    assert pickle.loads(pickle.dumps(problem)).objective_spec == Objective()


def test_spread_smoothing_removes_the_surrogate_bias_of_a_narrow_forecast():
    """With a fixed kappa, a narrow predictive distribution (small spread on the normalized
    scale) makes the smoothed coverage of an upper quantile read low; measuring q - z in
    units of the forecast's own spread removes that bias."""
    g = torch.Generator().manual_seed(0)
    J, R, sd = 4, 20_000, 0.02                                    # spread ~ the fixed smoothing width
    z = sd * torch.randn(J * R, 1, generator=g)
    levels = torch.tensor([0.1, 0.9])
    q = torch.tensor([-1.2816, 1.2816]).mul(sd).expand(J * R, 1, 2)   # the exact quantiles
    out = Outputs(torch.zeros(J * R, 1), R, quantiles=q, target=z, levels=levels)
    fixed, spread = tg.miscoverage(levels=[0.9], smoothing="fixed"), tg.miscoverage(levels=[0.9])
    upper = lambda m: torch.sigmoid(m.kappa * ((q[..., 1] - z) / ((q[..., 1] - q[..., 0])
                                    if m.smoothing == "spread" else 1))).mean().item()
    assert abs(upper(spread) - 0.9) < 0.003 < abs(upper(fixed) - 0.9)
    assert spread.per_process(out).mean() < fixed.per_process(out).mean()
    with pytest.raises(ValueError, match="two quantile levels"):
        tg.ar(1).estimate(tg.forecast(h=1, quantiles=[.9])).loss(tg.pinball()).objective(
            tg.miscoverage(smoothing="spread"))


def test_spread_smoothing_is_the_default_and_earlier_models_keep_their_meaning():
    import pickle
    assert tg.miscoverage().smoothing == "spread"
    assert "smoothing" not in repr(tg.miscoverage(smoothing="fixed"))     # original description, original keys
    saved = tg.miscoverage(smoothing="fixed")
    state = dict(saved.__dict__)
    state.pop("smoothing")                                                # pickled before the option existed
    old = object.__new__(type(saved))
    old.__setstate__(state)
    assert old.smoothing == "fixed" and pickle.loads(pickle.dumps(tg.miscoverage())).smoothing == "spread"



def test_which_estimand_learns_order_probabilities(tmp_path, monkeypatch):
    """Model selection: the network outputs one probability per operand of the selection,
    trained with the log score; the target is the drawn index."""
    monkeypatch.setenv("TIMEGRAM_HOME", str(tmp_path))
    world = tg.select(tg.ar(1, phi=tg.stationary(1, last_min=0.5)), tg.ar(3, phi=tg.stationary(3, last_min=0.5)))
    problem = world.estimate(tg.which())
    h, slots, out = problem.layout()
    assert h == 0 and out == 2 and problem.categorical() == {"select.which": 2}
    est = problem.pretrain(lengths=(40, 40), budget=4096, batch_size=256, net=tg.ResMLP(width=16, depth=1),
                           device="cpu", verbose=False)
    p_ = est.predict([np.random.default_rng(0).normal(size=40) for _ in range(5)])
    np.testing.assert_allclose(p_.sum(1), 1, rtol=1e-5)
    assert (p_ >= 0).all()
    with pytest.raises(ValueError, match="select"):
        tg.ar(1).estimate(tg.which()).layout()


def test_stationary_with_a_distinguishable_order():
    from timegram.dists import pacf_to_coefs
    d = tg.stationary(4, last_min=0.3)
    rng = np.random.default_rng(0)
    draws = np.array([d.sample(rng, None) for _ in range(2000)])
    assert (np.abs(draws[:, -1]) >= 0.3 - 1e-12).all()                 # phi_k equals the last PACF
    fast = d.sample_batch(torch.Generator().manual_seed(0), 2000, "cpu", None).numpy()
    assert (np.abs(fast[:, -1]) >= 0.3 - 1e-6).all()
    with pytest.raises(ValueError):
        tg.stationary(2, last_min=1.2)
    with pytest.raises(ValueError, match="k >= 1"):
        tg.stationary(0, last_min=0.5)
    assert tg.stationary(0).sample(rng, None).shape == (0,)            # plain AR(0) stays allowed
