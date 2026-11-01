"""Checked exploration contracts and bounded simulation evidence; no training."""
import copy
import importlib
import pickle

import numpy as np
import pytest
import torch

import timegram as tg
from timegram.dists import Distribution
from timegram.model import Block, Combine, Feed, OneHot
from timegram.moves import DEFAULT_MOVE_IDS, _feed_innovations, moves


def objects(root):
    """All graph/parameter identities, including ties hidden in compound trees."""
    seen, found, pending = set(), set(), [root]
    while pending:
        x = pending.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, (Block, Distribution, OneHot)):
            found.add(id(x))
        if isinstance(x, (Block, Distribution)) or hasattr(x, "params"):
            pending.extend(vars(x).values())
        elif isinstance(x, dict):
            pending.extend(x.values())
        elif isinstance(x, (list, tuple)):
            pending.extend(x)
    return found


@pytest.mark.parametrize("factory", [lambda: tg.ets("AAN"), tg.worlds.backbone], ids=["holt", "backbone"])
@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_nine_neighbours_are_keyed_checked_and_simulable(factory, backend):
    result = moves(factory())
    assert len(result) == 9 and not result.skipped
    assert [c.move_ids for c in result] == [(name,) for name in DEFAULT_MOVE_IDS]
    assert len({c.world_key for c in result}) == 9
    for c in result:
        assert c.identity.status == "supported" and not c.report.has_errors
        simulated = c.world.simulate(n=4, length=128, horizon=8, seed=20261029, backend=backend)
        assert simulated.context.shape == (4, 128)
        assert simulated.target.shape == (4, 8)
        assert np.isfinite(simulated.context).all(), c.move_ids
        assert np.isfinite(simulated.target).all(), c.move_ids


def test_enumeration_is_pure_and_candidates_are_independent(monkeypatch):
    model = tg.worlds.backbone()
    legacy = tg.library.canonical(model.estimate(tg.forecast(h=8)))
    # Legacy canonical() itself populates parameter-path metadata. Snapshot after
    # that existing call, and separately check a never-inspected parent below.
    before = pickle.dumps(model)
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state().clone()

    def forbidden(*a, **kw):
        raise AssertionError("enumeration must not simulate, train, or write the library")

    monkeypatch.setattr(importlib.import_module("timegram.simulate"), "simulate", forbidden)
    monkeypatch.setattr(importlib.import_module("timegram.train"), "train", forbidden)
    monkeypatch.setattr(tg.library, "save", forbidden)
    monkeypatch.setattr(tg.library, "load", forbidden)
    monkeypatch.setattr(tg.library, "alias", forbidden)
    result, again = moves(model), moves(model)
    assert len(result) == 9 and not result.skipped
    assert [c.world_key for c in result] == [c.world_key for c in again]
    assert pickle.dumps(model) == before
    assert tg.library.canonical(model.estimate(tg.forecast(h=8))) == legacy
    now = np.random.get_state()
    assert now[0] == np_state[0] and np.array_equal(now[1], np_state[1]) and now[2:] == np_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    fresh = tg.worlds.backbone()
    fresh_before = pickle.dumps(fresh)
    moves(fresh)
    assert pickle.dumps(fresh) == fresh_before
    occupied = objects(model)
    for candidate in result:
        ids = objects(candidate.world)
        assert occupied.isdisjoint(ids)
        occupied |= ids


def test_recursive_selection_feed_preserves_choice_and_parameter_ties():
    nested = tg.select(tg.ets("ANN"), tg.ets("ANN"), probs=[0.2, 0.8])
    index = nested.schedule.index
    alpha = 0.2 + 0.2 * index
    nested.operands = [tg.ets("ANN", alpha=alpha), tg.ets("ANN", alpha=alpha)]
    original = tg.select(nested, tg.ets("AAN"), probs=[0.3, 0.7])
    before = pickle.dumps(original)
    result = moves(original, move_ids=("innovations.student",))
    assert len(result) == 1 and not result.skipped
    root = result[0].world
    inner = root.operands[0]
    np.testing.assert_array_equal(root.schedule.index.probs, original.schedule.index.probs)
    np.testing.assert_array_equal(inner.schedule.index.probs, index.probs)
    assert inner.schedule.index is not index
    a, b = inner.operands
    assert isinstance(a, Feed) and isinstance(b, Feed)
    assert a.up is not b.up and a.up is not root.operands[1].up
    assert a.down.alpha is b.down.alpha
    assert id(inner.schedule.index) in objects(a.down.alpha)
    assert id(index) not in objects(root)
    assert pickle.dumps(original) == before


@pytest.mark.parametrize("factory", [lambda: tg.convex(tg.ets("AAN"), tg.ets("ANN")),
                                    lambda: tg.ets("AAN") + tg.ets("ANN"),
                                    lambda: tg.select(tg.ets("ANN"), tg.noise())])
def test_feed_rule_is_not_distributed_through_simultaneous_or_invalid_operands(factory):
    result = moves(factory(), move_ids=DEFAULT_MOVE_IDS[:4])
    assert not result.candidates
    assert [s.reason for s in result.skipped] == ["root_does_not_accept_input"] * 4


def test_reviewed_defaults_and_parent_mixture_keep_order():
    root = tg.ets("AAN")
    candidates = {c.move_ids[0]: c for c in moves(root)}
    ar_feed = candidates["innovations.ar1"].world.up
    assert (ar_feed.down.phi.low, ar_feed.down.phi.high) == (0.3, 0.95)
    assert candidates["innovations.student"].world.up.shape.df == 4
    assert candidates["innovations.garch11"].world.up.down.burn_in == 200
    long = candidates["innovations.long_memory"].world.up.down
    assert (long.d.low, long.d.high) == (0.1, 0.45)
    damped = candidates["add.damped_trend"].world.operands[1]
    assert (damped.b.low, damped.b.high) == (-1, 1)
    assert (damped.phi.low, damped.phi.high) == (0.8, 0.98)
    alternative = candidates["combine.alternative.damped_ets"].world
    assert len(alternative.operands[1].operands) == 3
    mix = candidates["combine.mix_self"].world
    assert objects(mix.operands[0]).isdisjoint(objects(mix.operands[1]))
    wrapped = moves(root, keep_parent=0.5, move_ids=("positive.exp",))[0].world
    np.testing.assert_array_equal(wrapped.schedule.index.probs, [0.5, 0.5])
    assert tg.identity.world(wrapped.operands[0]).digest == tg.identity.world(root).digest
    assert tg.identity.world(wrapped.operands[1]).digest == candidates["positive.exp"].world_key
    assert objects(wrapped.operands[0]).isdisjoint(objects(wrapped.operands[1]))
    assert moves(root, keep_parent=0)[0].world_key == moves(root, keep_parent=None)[0].world_key


def test_supported_duplicate_keys_merge_provenance_and_unknown_identities_are_skipped():
    result = moves(tg.ets("AAN"), move_ids=("positive.exp", "positive.exp"))
    assert len(result) == 1 and result[0].move_ids == ("positive.exp", "positive.exp")

    class Unknown(tg.Block):
        pass

    result = moves(Unknown())
    assert len(result) == 0 and len(result.skipped) == 9
    assert all(s.reason == "identity_unverified" for s in result.skipped)


@pytest.mark.parametrize("prob", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_invalid_parent_probability_is_rejected(prob):
    with pytest.raises(ValueError, match="keep_parent"):
        moves(tg.ets("AAN"), keep_parent=prob)


def test_moves_reject_tasks_and_unknown_move_names():
    with pytest.raises(TypeError, match="world Block"):
        moves(tg.ets("AAN").estimate(tg.forecast(h=1)))
    with pytest.raises(ValueError, match="unknown move IDs"):
        moves(tg.ets("AAN"), move_ids=("invented",))


def test_shared_and_fresh_selection_innovations_have_consistent_marginal_statistics():
    """Finite two-sample evidence, not a proof of law equality or seed equivalence."""
    selection = tg.select(tg.ar(1, phi=[0.2], init="zero"),
                          tg.ar(1, phi=[0.7], init="zero"), probs=[0.3, 0.7])
    common = tg.noise()
    shared = Combine([common >> leaf for leaf in selection.operands], selection.schedule)
    fresh = _feed_innovations(copy.deepcopy(selection), tg.noise)
    features = []
    for model, seed in [(shared, 101), (fresh, 202)]:
        x = model.simulate(n=1024, length=32, seed=seed, backend="fast").context
        features.append(np.stack([x.mean(1), (x*x).mean(1), (x[:, 1:]*x[:, :-1]).mean(1)], axis=1))
    left, right = features
    se = np.sqrt(left.var(0, ddof=1)/len(left) + right.var(0, ddof=1)/len(right))
    assert np.all(np.abs(left.mean(0)-right.mean(0)) <= 6*se)


def test_fresh_vs_shared_innovations_are_not_interchangeable_in_a_convex_sum():
    """Numerical counterexample to extending the selection rule to active pairs."""
    common = tg.noise()
    shared = tg.convex(common, common, weight=0.5)
    fresh = tg.convex(tg.noise(), tg.noise(), weight=0.5)
    x = shared.simulate(n=512, length=32, seed=303, backend="fast").context
    y = fresh.simulate(n=512, length=32, seed=404, backend="fast").context
    assert 0.85 < np.var(x) < 1.15
    assert 0.4 < np.var(y) < 0.6
