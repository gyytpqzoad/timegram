"""Inspection must not change existing stochastic or cache behavior."""
import pickle

import numpy as np
import pytest
import torch

import timegram as tg
from timegram.witnesses import ObservationDesign
from timegram.dists import Expr, Fixed


def _case(kind):
    a, b = tg.uniform(-2, 2), tg.uniform(-2, 2)
    if kind == "levels":
        return tg.constant(a) + tg.constant(b), (a, b)
    if kind == "scales":
        a, b = tg.uniform(0, 3), tg.uniform(0, 3)
        return tg.noise(scale=a) + tg.noise(scale=b), (a, b)
    if kind == "tied":
        return tg.constant(a) + tg.trend(a), (a,)
    if kind == "backbone":
        return tg.worlds.backbone(), (a,)
    return tg.ets("AAA", period=4), (a,)


@pytest.mark.parametrize("kind", ["levels", "scales", "tied", "backbone", "ets"])
def test_opt_in_reports_leave_existing_world_keys_and_rng_unchanged(kind):
    world, targets = _case(kind)
    problem = world.estimate(tg.forecast(h=4))
    config = tg.TrainConfig()
    key = tg.library.key(problem, config)
    canonical = tg.library.canonical(problem)
    identity = tg.identity.world(world).digest
    frozen = pickle.dumps(world)
    numpy_state = pickle.dumps(np.random.get_state())
    torch_state = torch.get_rng_state().clone()

    for _ in range(2):
        tg.witnesses(world, targets=targets, observation=ObservationDesign(history_length=128))
        tg.complexity(world)

    assert pickle.dumps(world) == frozen
    assert pickle.dumps(np.random.get_state()) == numpy_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert tg.library.key(problem, config) == key
    assert tg.library.canonical(problem) == canonical
    assert tg.identity.world(world).digest == identity


def test_unknown_callable_hooks_are_not_executed_by_any_inspection():
    class Opaque:
        def __hash__(self):
            raise AssertionError("inspection must not hash unknown callables")

        def __eq__(self, other):
            raise AssertionError("inspection must not compare unknown callables")

        def __call__(self, *args):
            raise AssertionError("inspection must not evaluate unknown callables")

    a, b = tg.uniform(-2, 2), tg.uniform(-2, 2)
    world = tg.constant(a) + tg.constant(b) + tg.constant(Expr(Opaque(), (Fixed(1),), "opaque"))
    assert tg.identity.world(world).status == "unverified"
    assert tg.complexity(world).structural.status == "unknown"
    assert tg.witnesses(world, targets=(a, b), observation=ObservationDesign(8)).status == "unverified"
