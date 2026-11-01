"""Scoped typecheck contracts, negative cases, and actual named-world inventory."""
import ast
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

import timegram as tg
from timegram.blocks import Centered
from timegram.dists import Fixed, Joint
from timegram.model import Source
from timegram.typecheck import ClaimRecord, implied_status


def claims(report, name, *, root=False):
    return [c for c in report.claims if c.claim == name and (not root or c.subject_ref == report.root_ref)]


def test_unknown_extension_and_failed_premise_are_not_refutations():
    class Mystery(Source):
        def run(self, state):
            raise AssertionError("inspection must never execute simulation")
    report = Mystery().typecheck()
    assert report.has_unverified and not report.has_errors
    true = ClaimRecord('a', {}, 'premise', 'established')
    false = ClaimRecord('b', {}, 'premise', 'refuted')
    unknown = ClaimRecord('c', {}, 'premise', 'unverified')
    assert implied_status([true]) == 'established'
    assert implied_status([true, false]) == implied_status([unknown]) == 'unverified'


def test_custom_subclass_does_not_inherit_a_builtin_certificate():
    from timegram.blocks import AR
    class DifferentAR(AR):
        def apply(self, state, values):
            raise AssertionError('different recurrence')
    report = DifferentAR(1).typecheck()
    assert report.has_unverified
    assert not claims(report, 'stationary_coefficients')


def test_rule_failure_returns_a_diagnostic():
    class Broken(Source):
        def own_type_rules(self, context):
            raise RuntimeError('broken custom contract')
    report = Broken().typecheck()
    assert claims(report, 'rule_execution')[0].status == 'unverified'


@pytest.mark.parametrize('prior', [tg.stationary(2), tg.l1_ball(2)])
def test_declared_ar_regions_and_fixed_counterexample(prior):
    report = tg.ar(2, phi=prior).typecheck()
    assert claims(report, 'stationary_coefficients')[0].status == 'established'
    bad = tg.ar(1, phi=[1.2]).typecheck()
    assert claims(bad, 'stationary_coefficients')[0].status == 'refuted'
    assert claims(bad, 'integration_upper_bound', root=True)[0].status == 'unverified'
    boundary = tg.ar(1, phi=[1.0]).typecheck()
    assert claims(boundary, 'stationary_coefficients')[0].status == 'refuted'


def test_ma_sign_convention_and_stationarity_outside_l1():
    assert claims(tg.ar(2, phi=[1.5, -.7]).typecheck(), 'stationary_coefficients')[0].status == 'established'
    assert claims(tg.ma(2, theta=[-1.5, .7]).typecheck(), 'invertible_coefficients')[0].status == 'established'
    assert claims(tg.ma(2, theta=[1.5, -.7]).typecheck(), 'invertible_coefficients')[0].status == 'refuted'
    assert claims(tg.ma(2).typecheck(), 'invertible_coefficients')[0].status == 'established'


def test_joint_shape_for_and_bad_shape_are_checked_without_drawing():
    j = tg.priors.SarimaTree(period=4)
    assert not tg.ar(j.max_order, phi=j['phi']).typecheck().has_errors
    assert not tg.pattern(4, values=j['season0']).typecheck().has_errors
    class Wrong(Joint):
        keys = ('phi',)
        def shape_for(self, key):
            return (2,)
        def sample(self, *args):
            raise AssertionError('no draws')
    report = tg.ar(3, phi=Wrong()['phi']).typecheck()
    assert claims(report, 'shape:phi')[0].status == 'refuted'
    assert report.has_errors
    invalid = tg.ar(1, phi=j['not-a-key']).typecheck()
    assert claims(invalid, 'joint_key')[0].status == 'refuted'


@pytest.mark.parametrize('order', [1, tg.choice([0, 2]), tg.integers(0, 2), tg.uniform(1, 1)])
def test_integer_orders_and_refuted_malformed_definition(order):
    model = tg.integrate(order)
    assert claims(model.typecheck(), 'nonnegative_integer:d')[0].status == 'established'
    model.d = Fixed(1.5)  # inspect a mutated definition without changing constructors
    report = model.typecheck()
    assert claims(report, 'nonnegative_integer:d')[0].status == 'refuted'
    assert report.has_errors


def test_expr_region_is_unverified_and_broadcast_errors_are_reported():
    model = tg.ar(1, phi=tg.uniform(.2, .3) + .1)
    assert claims(model.typecheck(), 'stationary_coefficients')[0].status == 'unverified'
    bad = tg.ar(2)
    bad.phi = Fixed([1., 2.]) + Fixed([1., 2., 3.])
    assert claims(bad.typecheck(), 'broadcast_shape')[0].status == 'refuted'


def test_simplex_rule_scalar_pair_and_invalid_vector():
    good = tg.convex(tg.noise(), tg.noise(), tg.noise(), weight=tg.dirichlet(3))
    assert claims(good.typecheck(), 'simplex_weights')[0].status == 'established'
    pair = tg.convex(tg.noise(), tg.noise(), weight=tg.uniform())
    assert claims(pair.typecheck(), 'simplex_weights')[0].status == 'established'
    good.schedule.w = Fixed([1., 1., -1.])
    assert claims(good.typecheck(), 'simplex_weights')[0].status == 'refuted'
    assert good.typecheck().has_errors


def test_period_warning_is_a_disabled_policy_not_invalid_process():
    world = tg.pattern(12)
    report = world.typecheck(history_length=9)
    warning = claims(report, 'history_covers_declared_cycle')[0]
    assert warning.scope == 'policy' and warning.diagnostic_severity == 'warning'
    assert not report.has_errors
    assert not claims(world.typecheck(history_length=9, policies=()), 'history_covers_declared_cycle')
    assert not claims(world.typecheck(history_length=128), 'history_covers_declared_cycle')


def test_integration_bound_and_cancellation_flag_do_not_rewrite():
    world = tg.noise() >> tg.integrate(2)
    assert claims(world.typecheck(), 'integration_upper_bound', root=True)[0].value == {1: 2}
    world = world >> tg.difference(1)
    before = pickle.dumps(world)
    report = world.typecheck()
    assert claims(report, 'possible_cancellation')
    assert claims(report, 'integration_upper_bound', root=True)[0].value == {1: 2}
    assert pickle.dumps(world) == before
    nonlinear = (tg.noise() >> tg.exp()).typecheck()
    assert claims(nonlinear, 'integration_upper_bound', root=True)[0].status == 'unverified'


@pytest.mark.parametrize('source', [lambda: tg.constant(tg.uniform()), lambda: tg.noise(tg.student_t(1))])
def test_shared_zero_witness_does_not_inherit_infinite_variance(source):
    e = source()
    report = (e - e).typecheck()
    assert claims(report, 'zero_witness', root=True)[0].status == 'established'
    assert not any(c.status == 'refuted' for c in claims(report, 'finite_second_moment', root=True))
    assert not claims((e - e.copy()).typecheck(), 'zero_witness', root=True)


def test_sort_cycles_and_unused_default_input():
    filt = tg.ar(1)
    filt.source = filt
    assert filt.typecheck().has_errors
    fed = tg.noise() >> filt
    assert not fed.typecheck().has_errors  # cyclic default source is unused when fed
    fed.down = tg.constant(1)
    assert claims(fed.typecheck(), 'feed_sort')[0].status == 'refuted'
    with pytest.raises(TypeError):
        tg.noise() >> tg.constant(1)


@pytest.mark.parametrize('factory', [lambda: tg.worlds.low_informative('monthly'), tg.worlds.backbone, tg.worlds.m1_ets])
def test_typechecking_fresh_worlds_is_pure(factory, monkeypatch):
    world = factory()
    before = pickle.dumps(world)
    numpy_before = pickle.dumps(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    from timegram.train import TrainConfig
    legacy_before = tg.library.key(world.estimate(tg.forecast(4)), TrainConfig(budget=32))
    def forbidden(*args, **kwargs):
        raise AssertionError('inspection must not train or simulate')
    monkeypatch.setattr(tg, 'simulate', forbidden)
    report = world.typecheck(history_length=128)
    # The legacy key check itself can populate joint caches, so compare its state
    # separately; also test an actually fresh instance below.
    assert not report.has_errors
    assert tg.library.key(world.estimate(tg.forecast(4)), TrainConfig(budget=32)) == legacy_before
    fresh = factory()
    fresh_before = pickle.dumps(fresh)
    fresh.typecheck()
    assert pickle.dumps(fresh) == fresh_before
    assert pickle.dumps(np.random.get_state()) == numpy_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)


ROOT = Path(__file__).resolve().parents[1]


def selected_functions(relative, names=None, globals_=None):
    """Compile world-building function definitions, never script entry points: only `def`
    statements are executed, so no module-level code runs. names=None takes every function
    defined at module level (experiment scripts grow helpers as candidates are added)."""
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    if names is not None:
        assert {n.name for n in selected} == set(names)
    for fn in selected:                   # annotations may name the script's imports (e.g. Dataset)
        fn.returns = None
        for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs, fn.args.vararg, fn.args.kwarg]:
            if arg is not None:
                arg.annotation = None
    namespace = {'tg': tg, 'Centered': Centered, **(globals_ or {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, 'exec'), namespace)
    return tree, namespace


def experiment_module(relative):
    """Import an experiment script whose module level only defines things (its entry point is
    behind `if __name__ == "__main__"`), under a private module name."""
    import importlib.util
    import sys
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("_inventory_" + path.parent.name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _dict_literals(tree, variable):
    """The dict literal assigned to `variable` and those passed to `variable.update(...)`."""
    for n in tree.body:
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets)):
            yield n.value
        elif (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
              and n.value.func.attr == 'update' and isinstance(n.value.func.value, ast.Name)
              and n.value.func.value.id == variable):
            yield from (a for a in n.value.args if isinstance(a, ast.Dict))


def world_lambdas(tree, variable, namespace, keyword=None):
    literals = list(_dict_literals(tree, variable))
    assert literals, variable
    for node in literals:
        for key, value in zip(node.keys, node.values):
            if keyword is not None:
                value = next(kw.value for kw in value.keywords if kw.arg == keyword)
            yield ast.literal_eval(key), eval(compile(ast.Expression(value), variable, 'eval'), namespace)


def test_named_world_and_experiment_inventory_has_no_structural_errors():
    worlds = [(f'low_informative:{f}', tg.worlds.low_informative(f)) for f in tg.worlds.LOW_INFORMATIVE]
    worlds += [(name, getattr(tg.worlds, name)()) for name in ('backbone', 'ets_variants', 'ets_family', 'm1_ets')]
    worlds += [('ets_family:mix', tg.worlds.ets_family(mix_share=.5)), ('AR5', tg.ar(5)), ('ETS_parameters', tg.ets('AAA', period=4))]
    tree, ns = selected_functions('experiments/m1_ets/run.py', ['paperlike', 'ets_family', 'mixed_pair', 'paper_mix'])
    worlds += [('m1:' + name, factory()) for name, factory in world_lambdas(tree, 'VARIANTS', ns, 'world')]
    accuracy = experiment_module('experiments/accuracy/run.py')     # import-safe: guarded entry point
    worlds += [('accuracy:' + name, factory()) for name, factory in accuracy.worlds().items()]
    worlds += [('accuracy:' + name, factory()) for name, factory in accuracy.CANDIDATES.items()]
    _, ns = selected_functions('experiments/tourism_calibration/run.py', ['world'])
    worlds.append(('Tourism', ns['world']()))
    tree = ast.parse((ROOT / 'experiments/ar_order/run.py').read_text(encoding='utf-8'))
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('ORDERS', 'LAST_MIN'):
            constants[n.targets[0].id] = ast.literal_eval(n.value)
    _, ns = selected_functions('experiments/ar_order/run.py', ['world'], constants)
    worlds.append(('AR_order', ns['world']()))
    assert len(worlds) >= 45
    for name, world in worlds:
        report = world.typecheck(history_length=128)
        assert not report.has_errors, (name, report.diagnostics)
        assert not claims(report, 'rule_execution'), (name, report.diagnostics)


def test_false_shape_premises_block_output_shape_and_order():
    malformed = tg.noise() * np.ones(3)
    report = malformed.typecheck()
    shape = claims(report, 'shape:scale', root=True)[0]
    assert shape.status == 'refuted'
    for name in ('event_shape', 'integration_upper_bound'):
        output = claims(report, name, root=True)[0]
        assert output.status == 'unverified'
        assert shape.claim_id in output.premises
    ar = tg.ar(2)
    ar.phi = tg.stationary(3)
    report = ar.typecheck()
    assert claims(report, 'shape:phi')[0].status == 'refuted'
    assert claims(report, 'stationary_coefficients')[0].status == 'unverified'
    assert claims((malformed + tg.noise()).typecheck(), 'event_shape', root=True)[0].status == 'unverified'


def test_wrong_input_sort_does_not_get_a_feed_or_combination_proof():
    from timegram.model import Feed
    report = Feed(Fixed(1), tg.ar(1)).typecheck()
    assert claims(report, 'feed_sort')[0].status == 'refuted'
    combined = tg.noise() + tg.noise()
    combined.operands[0] = Fixed(1)
    report = combined.typecheck()
    assert claims(report, 'combination_sort')[0].status == 'refuted'
    assert claims(report, 'event_shape', root=True)[0].status == 'unverified'
    nonsampler = tg.noise(tg.stationary(2)).typecheck()
    assert claims(nonsampler, 'iid_sampler')[0].status == 'unverified'
    assert claims(nonsampler, 'event_shape', root=True)[0].status == 'unverified'


def test_custom_scale_cannot_forge_a_zero_witness():
    from timegram.model import Scale
    class DifferentScale(Scale):
        def run(self, state):
            return state.value(self.block) + 17
    e = tg.noise()
    report = (e + DifferentScale(e, -1)).typecheck()
    assert not claims(report, 'zero_witness', root=True)
    class Forged(Source):
        def own_linear_terms(self):
            return [(e, -1)]
    assert not claims((e + Forged()).typecheck(), 'zero_witness', root=True)


def test_nonfinite_fixed_values_block_zero_witness():
    e = tg.constant(float('inf'))
    report = (e - e).typecheck()
    assert report.has_errors
    assert all(c.status == 'unverified' for c in claims(report, 'zero_witness', root=True))
    assert not any(c.status == 'established' for c in claims(report, 'zero_witness', root=True))


def test_invalid_declared_integer_prior_is_not_certified():
    prior = tg.priors.SarimaTree(d_choices=(0, .5))
    report = tg.integrate(prior['d']).typecheck()
    assert claims(report, 'nonnegative_integer:d')[0].status == 'refuted'
