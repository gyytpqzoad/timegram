"""Inspection regressions: inspect, never train."""
import pickle

import numpy as np
import pytest

import timegram as tg
from timegram.blocks import Input
from timegram.dists import Fixed, Joint


@pytest.mark.parametrize("mutation", ["tree", "input", "stage_binding", "copied_tree"])
@pytest.mark.parametrize("fed", [False, True])
def test_inconsistent_composite_has_no_world_or_problem_identity(mutation, fed):
    chain = tg.ar(1, phi=[.3], init="zero") >> tg.ma(1, theta=[.2])
    if mutation == "tree":
        chain.tree = tg.constant(17.)
    elif mutation == "input":
        chain.input = Input()
    elif mutation == "stage_binding":
        chain.tree.down = chain.stages[-1].copy()
    else:
        chain.tree = chain.tree.copy()
    world = tg.noise() >> chain if fed else chain
    before = pickle.dumps(world)
    records = [tg.identity.world(world), tg.identity.problem(world.estimate(tg.forecast(h=1)))]
    for record in records:
        assert record.status == "unverified"
        assert record.digest is None
        assert any("Composite" in why for why in record.unsupported)
    assert pickle.dumps(world) == before
    assert tg.complexity(world).structural.status == "unknown"


def test_identity_collision_is_rejected_while_both_worlds_remain_simulable():
    original = tg.ar(1, phi=[.3], init="zero") >> tg.ma(1, theta=[.2])
    modified = original.copy()
    modified.tree = tg.constant(17.)
    assert np.var(original.simulate(length=16, seed=12).context) > 0
    np.testing.assert_array_equal(modified.simulate(length=16).context, np.full((1, 16), 17.))
    assert tg.identity.world(original).status == "supported"
    assert tg.identity.world(modified).digest is None


@pytest.mark.parametrize("random_order", [False, True])
@pytest.mark.parametrize("fed", [False, True])
def test_fractional_difference_is_supported_and_explorable(random_order, fed):
    order = tg.uniform(.1, .3) if random_order else .2
    difference = tg.fractional_difference(order)
    negative_integral = tg.fractional_integrate(-order)
    if fed:
        difference = tg.noise() >> difference
        negative_integral = tg.noise() >> negative_integral
    before = pickle.dumps(difference)
    record = tg.identity.world(difference)
    assert record.status == "supported", record.unsupported
    assert record.digest == tg.identity.world(negative_integral).digest
    assert tg.identity.problem(difference.estimate(tg.forecast(h=1))).status == "supported"
    # Only a transform or a selection of transforms receives innovation-feed moves.
    # A fed process still admits the other five moves.
    candidates = tg.moves(difference)
    assert len(candidates) == (5 if fed else 9)
    assert not any(s.reason == "identity_unverified" for s in candidates.skipped)
    assert tg.complexity(difference).structural.status == "exact"
    assert pickle.dumps(difference) == before


@pytest.mark.parametrize("name", ["unreviewed_filter", None])
def test_fractional_name_exception_does_not_allow_arbitrary_fields(name):
    world = tg.fractional_difference(.2)
    world.name = name
    assert tg.identity.world(world).status == "unverified"


class VectorOrder(Joint):
    keys = ("d",)

    def shape_for(self, key):
        return (2,)

    def own_support(self, context):
        return context.joint({"d": {"integer": True, "interval": (1, 2)}})

    def sample(self, *args):
        raise AssertionError("typechecking must not draw the order")


@pytest.mark.parametrize("joint", [False, True])
def test_vector_integration_order_blocks_dependent_scalar_claims(joint):
    if joint:
        world = tg.integrate(VectorOrder()["d"])
    else:
        world = tg.integrate(1)
        world.d = Fixed([1, 2])
    before = pickle.dumps(world)
    report = world.typecheck()
    root = [c for c in report.claims if c.subject_ref == report.root_ref]
    shape = next(c for c in root if c.claim == "shape:d")
    assert report.has_errors and shape.status == "refuted"
    for name in ("nonnegative_integer:d", "event_shape", "integration_upper_bound"):
        conclusion = next(c for c in root if c.claim == name)
        assert conclusion.status == "unverified"
        assert shape.claim_id in conclusion.premises
    assert pickle.dumps(world) == before
