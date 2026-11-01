"""The approved dag-v3 fragment, not general equality of process laws."""
import copy
import json
import operator
import pickle

import numpy as np
import pytest
import torch

import timegram as tg
from timegram import identity, library
from timegram.dists import Expr, Fixed, StationaryOrder, as_dist
from timegram.train import TrainConfig


def world_key(world):
    record = identity.world(world)
    assert record.status == "supported", record.unsupported
    assert len(record.digest) == 64
    return record.digest


def problem_key(problem, config=None):
    record = identity.problem(problem, config)
    assert record.status == "supported", record.unsupported
    return record.digest


def test_independent_addition_commutes_but_repeated_draw_is_not_independent():
    a, b = tg.ar(1, phi=tg.uniform(-.5, .5)), tg.ma(1, theta=tg.uniform(-.4, .4))
    assert world_key(a + b) == world_key(b + a)
    assert problem_key((a+b).estimate(tg.forecast(h=4))) == problem_key((b+a).estimate(tg.forecast(h=4)))
    c = tg.constant(1.)
    assert world_key((a+b)+c) == world_key(a+(b+c))
    e = tg.noise()
    assert world_key(e + e) != world_key(e + e.copy())


def test_explicit_bindings_separate_the_actual_legacy_sharing_collision():
    a, b = tg.ar(1, phi=[.2]), tg.ar(1, phi=[.8])
    p = (a+b+a).estimate(tg.forecast(h=2))
    c, d = tg.ar(1, phi=[.2]), tg.ar(1, phi=[.8])
    q = (d+c+d).estimate(tg.forecast(h=2))
    assert library.key(p, TrainConfig()) == library.key(q, TrainConfig())
    assert problem_key(p) != problem_key(q)
    assert problem_key(p) == problem_key(copy.deepcopy(p))
    # No sorting of shared sums is promised by v1.
    assert world_key(a+b+a) != world_key(b+a+b)


def test_associative_composition_preserves_resolved_parameter_head():
    x, ma, ar = tg.noise(), tg.ma(1, theta=[.4]), tg.ar(1, phi=[.6])
    forms = [(x >> ma) >> ar, x >> (ma >> ar), tg.compose(x, ma, ar)]
    assert len({world_key(m) for m in forms}) == 1
    tasks = [m.estimate(tg.forecast(h=2), tg.params()) for m in forms]
    assert len({problem_key(p) for p in tasks}) == 1
    assert list(tasks[0].layout()[1]) == ["ar.phi", "ar.ma.theta"]
    left = forms[0].estimate(tg.params("phi", "theta"))
    right = forms[0].estimate(tg.params("theta", "phi"))
    assert problem_key(left) != problem_key(right)


def test_standalone_composite_and_explicit_feed_match():
    ma, ar = tg.ma(1, theta=[.4]), tg.ar(1, phi=[.6])
    assert world_key(ma >> ar) == world_key(tg.model.Feed(ma, ar))


@pytest.mark.parametrize("left,right", [
    (lambda: tg.normalize("std"), lambda: tg.normalize("minmax")),
    (lambda: tg.lag(4), lambda: tg.lag(12)),
    (lambda: tg.integrate(1), lambda: tg.integrate(2)),
    (lambda: tg.ar(1, phi=[.5], init="zero"), lambda: tg.ar(1, phi=[.5])),
    (lambda: tg.exp(.1), lambda: tg.exp(.2)),
])
def test_fed_transform_settings_are_part_of_identity(left, right):
    assert world_key(tg.noise() >> left()) != world_key(tg.noise() >> right())
    assert world_key(tg.noise() >> left()) == world_key(tg.noise() >> left())


def test_named_family_and_operator_roles_are_not_erased():
    assert world_key(tg.ets("AAA", period=4)) == world_key(tg.ets(trend="additive", season=4))
    assert world_key(tg.ets("AAA", period=4)) != world_key(tg.ets("AAA", period=12))
    assert world_key(tg.ets("ANN", min_length=9)) != world_key(tg.ets("ANN", min_length=10))
    a,b=tg.ets("ANN"),tg.ets("AAN")
    assert world_key(tg.convex(a,b)) != world_key(tg.select(a,b))
    variants = [{}, {"power":1}, {"gamma":[.1]}, {"burn_in":2000}, {"initial":2.}]
    assert len({world_key(tg.garch(**v)) for v in variants}) == len(variants)


def test_parameter_binding_and_which_class_order_are_observable():
    a,b=tg.ar(1,phi=[.2]),tg.ma(1,theta=[.8])
    left=(a+b).estimate(tg.params())
    right=(b+a).estimate(tg.params())
    assert problem_key(left) != problem_key(right)
    p=tg.select(a,b).estimate(tg.which())
    q=tg.select(b,a).estimate(tg.which())
    assert p.layout()[2] == q.layout()[2] == 2
    assert problem_key(p) != problem_key(q)
    assert world_key(p.model) != world_key(q.model)
    assert world_key(tg.select(a,b,probs=[.2,.8])) != world_key(tg.select(a,b,probs=[.8,.2]))


def test_exact_scalars_and_constructor_normalization():
    assert world_key(tg.constant(as_dist(.9))) == world_key(tg.constant(Fixed(.9)))
    assert world_key(tg.constant(.9)) == world_key(tg.constant(Fixed(.9)))
    a,b=tg.constant(.12345678901231),tg.constant(.12345678901232)
    assert library.canonical(a.estimate(tg.forecast(h=2))) == library.canonical(b.estimate(tg.forecast(h=2)))
    assert world_key(a) != world_key(b)


def test_joint_dependence_and_shapes_are_explicit():
    joint=tg.priors.SarimaTree(period=4)
    shared=tg.ar(3,phi=joint["phi"]) + tg.ma(3,theta=joint["theta"])
    independent=tg.ar(3,phi=joint["phi"]) + tg.ma(3,theta=tg.priors.SarimaTree(period=4)["theta"])
    assert world_key(shared) != world_key(independent)
    record=identity.world(shared)
    doc=json.loads(record.canonical_json)
    phi=[d for d in doc["definitions"] if d["type"]=="JointKey" and d["fields"]["key"]=="phi"]
    assert phi[0]["fields"]["event_shape"] == {"tuple":[{"integer":"3"}]}
    assert record.subject_ref(joint["phi"]).document_digest == record.digest


def test_transform_definition_reuse_is_not_output_reuse():
    x,y=tg.noise(),tg.noise()
    filt=tg.ar(1,phi=tg.uniform(.1,.8))
    p=tg.convex(x >> filt,y >> filt)
    q=tg.convex(x >> filt,y >> filt.copy())
    assert world_key(p) != world_key(q)           # tied versus independent filter parameter
    applied=x >> filt
    assert world_key(tg.convex(applied,applied)) != world_key(tg.convex(x >> filt,y >> filt))
    assert world_key(p) == world_key(copy.deepcopy(p))


def test_unknown_subclasses_callables_and_fields_have_no_digest():
    class CustomNormal(tg.dists.Normal):
        pass
    for m in [tg.noise(CustomNormal()), tg.constant(Expr(lambda x:x,(Fixed(1),),"add"))]:
        r=identity.world(m)
        assert r.status=="unverified" and r.digest is None and r.canonical_json is None
        assert r.unsupported
    good=tg.constant(Expr(operator.add,(Fixed(1),Fixed(2)),"+"))
    assert identity.world(good).status=="supported"
    unknown=tg.noise()
    unknown.unreviewed_setting=7
    assert identity.world(unknown).digest is None


def test_unknown_task_extension_is_not_called_or_copied():
    class UnknownObservation:
        def __deepcopy__(self, memo):
            raise AssertionError("unknown extension hook must not execute")
    p=tg.ar(1).observe(UnknownObservation()).estimate(tg.forecast(h=2))
    assert identity.problem(p).status=="unverified"


def test_inactive_default_source_does_not_contaminate_a_fed_filter():
    class Custom(tg.model.Source):
        pass
    filt=tg.ar(1,phi=[.4])
    filt.source=Custom()
    assert identity.world(filt).status=="unverified"
    assert world_key(tg.noise() >> filt)==world_key(tg.noise() >> tg.ar(1,phi=[.4]))


def test_cycle_invalid_task_and_nonfinite_fields_return_unverified():
    m=tg.ar(1)
    m.source=m
    for result in [identity.world(m),identity.world(tg.constant(float("nan"))),
                   identity.world(tg.noise(),schema="not-a-schema"),identity.problem(None)]:
        assert result.status=="unverified" and result.digest is None and result.unsupported


def test_world_problem_training_domains_and_task_fields():
    m=tg.ar(1)
    p=m.estimate(tg.forecast(h=2))
    assert len({world_key(m),problem_key(p),problem_key(p,TrainConfig())})==3
    assert problem_key(p)!=problem_key(p.observe())
    assert problem_key(p)!=problem_key(p.loss(tg.squared()))
    assert problem_key(p)!=problem_key(m.estimate(tg.forecast(h=3)))
    assert problem_key(p,TrainConfig())!=problem_key(p,TrainConfig(seed=1))
    assert problem_key(p,TrainConfig(backend="fast"))==problem_key(p,TrainConfig())
    quantiles=m.estimate(tg.forecast(h=2,quantiles=[.1,.9])).loss(tg.pinball())
    fixed=quantiles.objective(tg.miscoverage(smoothing="fixed"))
    spread=quantiles.objective(tg.miscoverage(smoothing="spread"))
    assert problem_key(fixed)!=problem_key(spread)


@pytest.mark.parametrize("factory", [
    tg.worlds.backbone,tg.worlds.m1_ets,tg.worlds.ets_family,tg.worlds.ets_variants,
    lambda:tg.worlds.low_informative("yearly"),lambda:tg.worlds.low_informative("quarterly"),
    lambda:tg.worlds.low_informative("monthly"),lambda:tg.arfima(d=.2),
    lambda:tg.noise() >> tg.normalize("std") >> tg.exp(),
    lambda:tg.noise(tg.mixture(tg.normal(),tg.student_t(4))),
    lambda:tg.ar(2,phi=StationaryOrder(2,.1)),
])
def test_named_worlds_and_move_building_blocks_are_supported(factory):
    model=factory()
    assert world_key(model)==world_key(copy.deepcopy(model))
    assert problem_key(model.estimate(tg.forecast(h=4)))


def test_joint_garch_prior_codec():
    g=tg.priors.Garch11()
    assert world_key(tg.garch(omega=g["omega"],alpha=g["alpha"],beta=g["beta"]))


def test_identity_is_pure_and_does_not_change_legacy_keys(monkeypatch):
    m=tg.worlds.backbone()
    p=m.estimate(tg.forecast(h=4))
    old=library.key(p,TrainConfig())
    before=pickle.dumps(p)
    numpy_state=pickle.dumps(np.random.get_state())
    torch_state=torch.random.get_rng_state().clone()
    def forbidden(*a,**kw):
        raise AssertionError("identity must not simulate, draw, train or touch a cache")
    monkeypatch.setattr(tg.model.Block,"simulate",forbidden)
    monkeypatch.setattr(library,"home",forbidden)
    assert identity.world(m).digest and identity.problem(p,TrainConfig()).digest
    assert pickle.dumps(p)==before
    assert library.key(p,TrainConfig())==old
    assert pickle.dumps(np.random.get_state())==numpy_state
    assert torch.equal(torch.random.get_rng_state(),torch_state)


def test_fresh_joint_cache_is_unchanged_when_resolving_parameter_targets():
    m=tg.worlds.low_informative("quarterly")
    p=m.estimate(tg.forecast(h=2),tg.params())
    before=pickle.dumps(p)
    assert identity.world(m).digest
    record=identity.problem(p)
    assert record.digest and record.subject_ref(m) is not None
    assert pickle.dumps(p)==before


def test_external_binding_prevents_unsafe_sorting_of_equal_components():
    a,b=tg.ar(1),tg.ar(1)
    # The components look identical in isolation, but only a is reused outside the sum.
    left=tg.convex(a+b,a)
    right=tg.convex(b+a,b)
    assert world_key(left)==world_key(right)   # complete rooted graphs are isomorphic
    shared=tg.convex(a+a,b)
    assert world_key(left)!=world_key(shared)
    assert world_key(left)==world_key(copy.deepcopy(left))


def test_custom_numeric_and_array_subclasses_are_unverified_without_conversion_hooks():
    class IntegerHook(int):
        def __int__(self):
            raise AssertionError("identity must not convert custom integer subclasses")
    class FloatHook(float):
        def __float__(self):
            raise AssertionError("identity must not convert custom float subclasses")
    class ArrayHook(np.ndarray):
        def tolist(self):
            raise AssertionError("identity must not call custom array methods")
    for value in (IntegerHook(1), FloatHook(1), np.array([1.]).view(ArrayHook)):
        world = tg.constant([0.] if isinstance(value, np.ndarray) else 0.)
        world.c.value = value
        record = identity.world(world)
        assert record.status == "unverified" and record.digest is None
        assert identity.problem(world.estimate(tg.forecast(h=2))).status == "unverified"


@pytest.mark.parametrize("value", [1, 1., np.int8(1), np.int64(1), np.longlong(1),
                                  np.uint64(1), np.ulonglong(1), np.float16(1),
                                  np.float32(1), np.float64(1), np.bool_(True), np.array([1.])])
def test_exact_builtin_numeric_and_numpy_types_remain_supported(value):
    assert world_key(tg.constant(value))
