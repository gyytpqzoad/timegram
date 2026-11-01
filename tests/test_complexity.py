"""Descriptive counts and conditional-coordinate contracts, without sampling."""
import copy
import pickle

import numpy as np
import pytest
import torch

import timegram as tg
from timegram.blocks import Centered
from timegram.complexity import complexity
from timegram.dists import Distribution, Fixed, Joint
from timegram.model import Source


def test_fixed_ar_counts_replay_declared_fields_and_exclude_unused_default_noise():
    world = tg.noise() >> tg.ar(1, phi=[.5], init='zero')
    report = complexity(world)
    # Feed, Noise, AR; noise.scale and AR.phi; one IID Normal law. Numeric atoms:
    # scale=1, Normal mu/sd=0/1, p/period=1/1 and fixed coefficient=.5.
    assert dict(report.counts) == dict(blocks=3, parameters=2, schedules=0, innovation_laws=1,
                                      references=5, reused_references=0, numeric_atoms=6,
                                      string_atoms=1, null_atoms=0, operator_atoms=0)
    assert report.structural.status == 'exact'
    assert report.dimensions.minimum == report.dimensions.mean == report.dimensions.maximum == 0
    assert report.trace_entropy.bits == 0
    assert report.L_G is None and report.weight_table is None
    assert complexity(tg.noise()).dimensions.mean == 0


def test_shared_and_independent_sums_have_different_counts_and_dimensions():
    a = tg.constant(tg.uniform(-1, 1))
    shared, independent = complexity(a+a), complexity(a+a.copy())
    assert shared.counts['blocks'] == 2 and shared.counts['parameters'] == 1
    assert shared.counts['reused_references'] == 1
    assert independent.counts['blocks'] == 3 and independent.counts['parameters'] == 2
    assert independent.counts['reused_references'] == 0
    assert shared.dimensions.mean == 1
    assert independent.dimensions.mean == 2
    assert shared.world_key != independent.world_key


def test_tied_parameter_is_not_counted_twice_or_confused_with_identification():
    u, v = tg.uniform(-1, 1), tg.uniform(-1, 1)
    tied = complexity(tg.constant(u) + tg.trend(u))
    free = complexity(tg.constant(u) + tg.constant(v))
    assert tied.dimensions.mean == 1 and tied.counts['reused_references'] == 1
    assert free.dimensions.mean == 2  # W1 does not reduce the generative dimension


def test_selection_trace_differs_from_structure_entropy():
    a = tg.constant(tg.uniform(-1, 1))
    report = complexity(tg.select(a, a, probs=[.5, .5]))
    assert report.trace_entropy.bits == 1
    assert report.structure_entropy.status == 'exact' and report.structure_entropy.bits == 0
    assert report.dimensions.minimum == report.dimensions.mean == report.dimensions.maximum == 1
    assert len(report.dimensions.cases) == 2
    assert [c.probability for c in report.dimensions.cases] == [.5, .5]
    assert all(c.k == 1 for c in report.dimensions.cases)


def test_conditional_dimensions_do_not_sum_inactive_branches():
    a = tg.constant(tg.uniform())
    b = tg.constant(tg.normal()) + tg.trend(tg.uniform())
    report = complexity(tg.select(a, b, probs=[.25, .75]))
    assert [c.k for c in report.dimensions.cases] == [1, 2]
    assert (report.dimensions.minimum, report.dimensions.mean, report.dimensions.maximum) == (1, 1.75, 2)
    assert report.trace_entropy.bits == pytest.approx(-.25*np.log2(.25)-.75*np.log2(.75))


def test_nested_selection_entropy_is_weighted_by_activation():
    a, b = tg.constant(1), tg.trend(1)
    nested = tg.select(tg.select(a, b), a, probs=[.25, .75])
    report = complexity(nested)
    h_outer = -.25*np.log2(.25)-.75*np.log2(.75)
    assert report.trace_entropy.bits == pytest.approx(h_outer + .25)
    assert sorted(c.probability for c in report.dimensions.cases) == [.125, .125, .75]
    assert report.structure_entropy.status == 'unknown'


def test_zero_probability_branch_does_not_add_coordinates_or_entropy():
    class Custom(Source):
        pass
    world = tg.select(tg.constant(tg.uniform()), Custom(), probs=[1, 0])
    report = complexity(world)
    assert report.structural.status == 'unknown'  # full configured syntax still contains it
    assert report.dimensions.status == 'exact' and report.dimensions.mean == 1
    assert report.trace_entropy.status == 'exact' and report.trace_entropy.bits == 0
    assert len(report.dimensions.cases) == 1


def test_reused_same_select_counts_its_draw_once_distinct_tied_consumers_are_unknown():
    a, b = tg.constant(1), tg.trend(1)
    selected = tg.select(a, b)
    assert complexity(selected + selected).trace_entropy.bits == 1
    second = tg.select(a, b)
    second.schedule.index = selected.schedule.index
    report = complexity(selected + second)
    assert report.trace_entropy.status == 'unknown' and report.trace_entropy.bits is None
    assert report.dimensions.status == 'unknown'
    assert any('distinct consumers' in reason for reason in report.unsupported)


def test_independent_structural_draws_sum_entropies_and_discrete_orders_count():
    a, b = tg.select(tg.constant(1), tg.trend(1)), tg.select(tg.constant(2), tg.trend(2))
    assert complexity(a+b).trace_entropy.bits == 2
    order = tg.noise() >> tg.integrate(tg.choice([0, 1, 2], probs=[.5, .25, .25]))
    assert complexity(order).trace_entropy.bits == 1.5
    assert complexity(order).dimensions.mean == 0
    assert complexity(tg.integrate(tg.choice([1, 1]))).trace_entropy.bits == 0
    assert complexity(tg.constant(tg.choice([1, 2]))).trace_entropy.bits == 0


@pytest.mark.parametrize('k', [1, 2, 4])
def test_simplex_dimension_is_constrained_not_number_of_weights(k):
    world = tg.convex(*[tg.noise() for _ in range(k)], weight=tg.dirichlet(k))
    assert complexity(world).dimensions.mean == k-1
    fixed = tg.convex(*[tg.noise() for _ in range(k)], weight=np.full(k, 1/k))
    assert complexity(fixed).dimensions.mean == 0


def test_expr_dependencies_count_once_without_guessing_output_rank():
    u = tg.uniform(-1, 1)
    report = complexity(tg.constant(u-u))
    assert report.dimensions.mean == 1
    assert report.dimensions.output_parameterization[0][1] == 'unknown'
    assert report.counts['operator_atoms'] == 1
    assert complexity(tg.constant(Fixed(0))).dimensions.mean == 0


def test_centered_output_bound_does_not_invent_a_latent_dimension():
    report = complexity(tg.pattern(4, values=Centered(tg.normal(), 4)))
    assert report.dimensions.status == 'unknown'
    assert report.dimensions.output_parameterization[0][1:3] == ('upper_bound', 3)
    assert report.trace_entropy.status == 'exact' and report.trace_entropy.bits == 0
    assert report.counts['innovation_laws'] == 1


def test_unknown_custom_and_dependent_priors_keep_known_partial_coordinates():
    class Custom(Distribution):
        pass
    known = tg.constant(tg.uniform())
    report = complexity(known + tg.constant(Custom()))
    assert report.structural.status == report.dimensions.status == 'unknown'
    assert report.counts['parameters'] == 1
    assert report.dimensions.mean is None
    assert report.dimensions.cases[0].known_coordinates == 1
    class Dependent(Joint):
        keys = ('d',)
    report = complexity(tg.integrate(Dependent()['d']))
    assert report.trace_entropy.status == 'unknown' and report.trace_entropy.bits is None


def test_extra_builtin_attributes_prevent_exact_complete_counts():
    world = tg.constant(tg.uniform())
    world.c.secret_dependency = tg.normal()
    report = complexity(world)
    assert report.structural.status == 'unknown'
    assert report.dimensions.status == 'unknown'
    assert report.world_key is None
    assert any('extra=' in item for item in report.unsupported)


def test_named_configuration_counts_and_expanded_view_are_separate():
    world = tg.ets('AAN')
    report = complexity(world)
    assert report.structural.status == 'exact'
    assert report.counts['blocks'] == 2  # one named production plus its actual default input
    assert report.dimensions.status == 'unknown'
    expanded = complexity(world, view='expanded')
    assert expanded.structural.status == 'unknown' and expanded.counts['blocks'] == 0
    assert any('expanded view not implemented' in reason for reason in expanded.unsupported)


def test_composite_does_not_count_inactive_stage_default_sources():
    world = tg.ar(1, phi=[.5]) >> tg.ma(1, theta=[.2])
    report = complexity(world)
    assert report.structural.status == 'exact'
    assert report.counts['innovation_laws'] == 1


def test_case_budget_fails_closed_without_sampling():
    report = complexity(tg.integrate(tg.integers(0, 100)), max_cases=4)
    assert report.dimensions.status == report.trace_entropy.status == 'unknown'
    assert any('budget' in reason for reason in report.unsupported)


@pytest.mark.parametrize('factory', [lambda: tg.select(tg.constant(tg.uniform()), tg.trend(tg.uniform())),
                                    tg.worlds.backbone, tg.worlds.m1_ets])
def test_inspection_is_idempotent_pure_and_leaves_rng_and_legacy_keys_unchanged(factory):
    world = factory()
    before = pickle.dumps(world)
    numpy_state = pickle.dumps(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()
    first = complexity(world)
    second = complexity(world)
    assert first == second
    assert pickle.dumps(world) == before
    assert pickle.dumps(np.random.get_state()) == numpy_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    baseline = copy.deepcopy(world).estimate(tg.forecast(4))
    checked = world.estimate(tg.forecast(4))
    assert tg.library.canonical(baseline) == tg.library.canonical(checked)


def test_bad_arguments_are_explicit():
    with pytest.raises(ValueError):
        complexity(tg.noise(), convention='invented')
    with pytest.raises(ValueError):
        complexity(tg.noise(), max_cases=0)


@pytest.mark.parametrize('law,attr,value', [(tg.normal(), 'sd', -1), (tg.uniform(), 'low', 2),
                                           (tg.dirichlet(3), 'concentration', -1)])
def test_mutated_invalid_parameter_domains_do_not_get_exact_dimensions(law, attr, value):
    setattr(law, attr, value)
    report = complexity(tg.constant(law))
    assert report.structural.status == 'unknown' and report.dimensions.status == 'unknown'
    assert report.dimensions.mean is None
    assert any('invalid declared' in reason for reason in report.unsupported)


def test_reference_scope_and_identity_correspondence_are_explicit():
    u = tg.uniform()
    report = complexity(tg.constant(u) + tg.trend(u))
    local = report.dimensions.cases[0].coordinates[0][0]
    reference = dict(report.identity_bindings)[local]
    assert reference.document_digest == report.world_key
    assert 'local to this report' in report.reference_scope
    assert 'not unique' in report.reference_scope


def test_deep_graph_returns_unknown_partial_counts_instead_of_recursion_error():
    world = tg.noise()
    for _ in range(1200):
        world = world * 1.
    report = complexity(world)
    assert report.structural.status == 'unknown'
    assert report.counts['blocks'] > 0
    assert report.dimensions.status == 'unknown'
    assert any('recursion' in reason or 'recursive' in reason for reason in report.unsupported)


@pytest.mark.parametrize('factory,field', [(tg.normal, 'sd'), (lambda: Fixed(1), 'value'),
                                          (lambda: tg.choice([1, 2]), 'probs'),
                                          (lambda: tg.choice([1, 2]), 'values'),
                                          (lambda: tg.stationary(2), 'shape'),
                                          (lambda: Centered(tg.normal(), 4), 'k')])
def test_custom_numeric_fields_are_unknown_without_executing_conversion_hooks(factory, field):
    class Opaque:
        def __array__(self, *args, **kwargs):
            raise AssertionError('custom array executed')
        def __float__(self):
            raise AssertionError('custom float executed')
        def __eq__(self, other):
            raise AssertionError('custom equality executed')
    law = factory()
    setattr(law, field, Opaque())
    report = complexity(tg.constant(law))
    assert report.structural.status == report.dimensions.status == 'unknown'
    assert report.dimensions.mean is None
