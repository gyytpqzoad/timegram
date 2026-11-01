"""Analytic witnesses need the complete observation law and a changing target."""
import pickle
from fractions import Fraction

import numpy as np
import pytest
import torch

import timegram as tg
from timegram.blocks import Noise
from timegram.dists import Fixed
from timegram.witnesses import ObservationDesign, Target, witnesses


def record(report, rule):
    return next(r for r in report.records if r.rule==f"timegram.witnesses.{rule}/1")


def levels(extra=None):
    a,b=tg.uniform(-2,2),tg.uniform(-2,2)
    world=tg.constant(a)+tg.constant(b)
    if extra is not None:
        world=world+extra(a,b)
    return world,a,b


def noises(extra=None, scale_form=False):
    a,b=tg.uniform(0,3),tg.uniform(0,3)
    world=a*tg.noise()+b*tg.noise() if scale_form else tg.noise(scale=a)+tg.noise(scale=b)
    if extra is not None:
        world=world+extra(a,b)
    return world,a,b


@pytest.mark.parametrize("length",[1,8,128])
def test_w1_exact_level_witness_and_complete_numerical_law(length):
    world,a,b=levels()
    report=witnesses(world,targets=(a,b),observation=ObservationDesign(length))
    result=record(report,"W1")
    assert report.status==result.status=="non_identified_witness"
    assert result.witness['delta_exact']=='1/2'
    assert result.witness['allowable_delta']=={'lower':'-2','upper':'2','bounds':'open','excludes':'0'}
    assert all(t['changes'] for t in result.targets)
    assert result.numerical_check['mean_max_abs']==0
    assert result.numerical_check['covariance_max_abs']==0
    assert result.numerical_check['history_length']==length
    assert result.numerical_check['implementation']=='timegram.witnesses.raw_iid_moments/1'
    assert all(c['ref'] is not None for c in result.coordinates)
    premise_ids={c.claim_id for c in report.premise_records if c.status=='established'}
    assert set(result.premises)<=premise_ids
    assert 'all positive' in result.scope


@pytest.mark.parametrize("scale_form",[False,True])
def test_w2_exact_variance_witness_with_radical_scale_points(scale_form):
    world,a,b=noises(scale_form=scale_form)
    report=witnesses(world,targets=(a,Target('variance',(b,))),observation=ObservationDesign(128))
    result=record(report,"W2")
    assert result.status=='non_identified_witness'
    assert result.witness['coordinates_used']=='variances'
    assert result.coordinates[0]['point_values_exact']==('sqrt(9/2)','sqrt(45/8)')
    assert result.coordinates[1]['point_values_exact']==('sqrt(9/2)','sqrt(27/8)')
    assert result.targets[0]['changes'] and result.targets[1]['changes']
    assert result.numerical_check['matches'] and result.numerical_check['covariance_max_abs']<1e-12


@pytest.mark.parametrize("factory,rule",[(levels,'W1'),(noises,'W2')])
def test_independent_fixed_gaussian_remainder_preserves_each_witness(factory,rule):
    remainder=tg.noise(tg.normal(mu=2,sd=3),scale=2)
    # Internal remainder sharing is fixed and accounted for, not silently assumed absent.
    world,a,b=factory(lambda a,b: remainder+remainder+tg.constant(4))
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),rule)
    assert result.status=='non_identified_witness'
    assert len(result.witness['fixed_remainder'])==3
    assert result.numerical_check['matches']


@pytest.mark.parametrize('scale',[-2,2])
def test_fixed_gaussian_remainder_reports_nonnegative_sd_and_signed_amplitude(scale):
    world,a,b=noises(lambda a,b:tg.noise(tg.normal(mu=2,sd=3),scale=scale))
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),"W2")
    assert result.status=='non_identified_witness'
    assert result.witness['fixed_remainder']==({'kind':'fixed_Gaussian','mean':str(2*scale),
                                             'sd':'6','signed_amplitude':str(3*scale)},)


@pytest.mark.parametrize("extra",[lambda a,b:tg.trend(a),lambda a,b:tg.constant(a),
                                  lambda a,b:tg.constant(a+b),lambda a,b:tg.noise(scale=a)])
def test_w1_extra_root_dependency_blocks_child_witness(extra):
    world,a,b=levels(extra)
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),"W1")
    assert result.status=='unverified' and result.open_obligations


@pytest.mark.parametrize("extra",[lambda a,b:tg.constant(a),lambda a,b:tg.trend(b),
                                  lambda a,b:tg.constant(a+b)])
def test_w2_extra_root_dependency_blocks_child_witness(extra):
    world,a,b=noises(extra)
    assert record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),"W2").status=='unverified'


def test_shared_innovation_is_not_an_independent_gaussian_variance_witness():
    a,b=tg.uniform(0,3),tg.uniform(0,3)
    common=tg.noise()
    world=a*common+b*common
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),"W2")
    assert result.status=='unverified'
    assert 'independent' in result.open_obligations[0]


def test_remainder_reusing_a_changed_innovation_is_rejected():
    a,b=tg.uniform(0,3),tg.uniform(0,3)
    e1,e2=tg.noise(),tg.noise()
    world=a*e1+b*e2+e1
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),"W2")
    assert result.status=='unverified' and 'innovation' in result.open_obligations[0]


@pytest.mark.parametrize("factory,rule",[(levels,'W1'),(noises,'W2')])
def test_non_gaussian_remainder_does_not_receive_the_gaussian_fragment_proof(factory,rule):
    world,a,b=factory(lambda a,b:tg.noise(tg.student_t(4)))
    assert record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),rule).status=='unverified'


def test_non_gaussian_scale_components_are_not_moment_only_witnesses():
    a,b=tg.uniform(0,3),tg.uniform(0,3)
    world=tg.noise(tg.student_t(4),scale=a)+tg.noise(scale=b)
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),"W2")
    assert result.status=='unverified'


@pytest.mark.parametrize("factory,rule,kind",[(levels,'W1','sum'),(noises,'W2','sum_of_variances')])
def test_unchanged_target_function_is_not_declared_nonidentified(factory,rule,kind):
    world,a,b=factory()
    report=witnesses(world,targets=(Target(kind,(a,b)),),observation=ObservationDesign(8))
    assert record(report,rule).status=='unverified'
    assert 'target unchanged' in record(report,rule).open_obligations[0]
    assert record(report,rule).targets[0]['kind']==kind
    assert record(report,rule).targets[0]['changes'] is False
    assert len(record(report,rule).coordinates)==2
    assert record(report,rule).coordinates[0]['point_values_exact']
    # A changing coordinate in the same vector is enough, but the invariant is labeled.
    report=witnesses(world,targets=(a,Target(kind,(a,b))),observation=ObservationDesign(8))
    result=record(report,rule)
    assert result.status=='non_identified_witness'
    assert [t['changes'] for t in result.targets]==[True,False]


@pytest.mark.parametrize("rule",['W1','W2'])
def test_fixed_tied_and_degenerate_parameters_are_not_free_witness_directions(rule):
    constructor=tg.constant if rule=='W1' else lambda p:tg.noise(scale=p)
    a=tg.uniform(1,2)
    for b in (a,Fixed(1.5),tg.uniform(1.5,1.5)):
        world=constructor(a)+constructor(b)
        result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8)),rule)
        assert result.status=='unverified'
        assert len(result.targets)==2 and len(result.coordinates)==2
        assert result.targets[1]['parameter_details'][0]['domain'] is not None
        if type(b) is Fixed:
            assert result.targets[1]['parameter_details'][0]['role']=='fixed'
        elif a is b:
            assert result.targets[0]['parameters']==result.targets[1]['parameters']


@pytest.mark.parametrize("delta",[0,100,float('nan'),True])
def test_bad_displacements_block_the_witness(delta):
    for factory,rule in ((levels,'W1'),(noises,'W2')):
        world,a,b=factory()
        result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8),delta=delta),rule)
        assert result.status=='unverified'


@pytest.mark.parametrize('factory,rule',[(levels,'W1'),(noises,'W2')])
def test_exact_points_that_collapse_in_float64_do_not_pass_numeric_validation(factory,rule):
    world,a,b=factory()
    # A nonzero center avoids a representable tiny level displacement from zero.
    if rule=='W1':
        a,b=tg.uniform(1,2),tg.uniform(1,2)
        world=tg.constant(a)+tg.constant(b)
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8),delta=1e-30),rule)
    assert result.status=='unverified' and 'collapse' in result.open_obligations[0]
    assert all(t['changes'] for t in result.targets)
    assert result.numerical_check['parameter_points'][0]==result.numerical_check['parameter_points'][1]


@pytest.mark.parametrize('factory,rule',[(levels,'W1'),(noises,'W2')])
def test_nonfinite_raw_moments_do_not_pass_even_when_inf_equals_inf(factory,rule):
    world,a,b=factory(lambda a,b:tg.constant(1e308)+tg.constant(1e308))
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(1)),rule)
    assert result.status=='unverified'
    assert result.numerical_check['matches'] is False


@pytest.mark.parametrize("design",[
    ObservationDesign(8,representation='normalized'),ObservationDesign(8,missingness='stochastic'),
    ObservationDesign(8,time_origin=0),ObservationDesign(8,presample=1),
    ObservationDesign(8,initialization='stationary'),ObservationDesign(0),None,
])
def test_unsupported_observation_does_not_inherit_a_raw_witness(design):
    world,a,b=levels()
    report=witnesses(world,targets=(a,b),observation=design)
    assert report.status=='unverified'


def test_normalization_custom_parent_and_unsupported_targets_are_unverified():
    world,a,b=levels()
    for parent in (world>>tg.normalize('std'),2*world,tg.select(world,tg.noise())):
        assert witnesses(parent,targets=(a,b),observation=ObservationDesign(8)).status=='unverified'
    class Unknown(Noise):
        pass
    assert witnesses(world+Unknown(),targets=(a,b),observation=ObservationDesign(8)).status=='unverified'
    for targets in ((),('a','b'),(tg.uniform(-2,2),),(Target('custom',(a,b)),)):
        assert witnesses(world,targets=targets,observation=ObservationDesign(8)).status=='unverified'


def test_explicit_fraction_delta_is_exact_and_unknown_request_hooks_are_not_called():
    world,a,b=levels()
    result=record(witnesses(world,targets=(a,b),observation=ObservationDesign(8),delta=Fraction(1,3)),"W1")
    assert result.status=='non_identified_witness' and result.witness['delta_exact']=='1/3'
    class Opaque:
        def __eq__(self,other):
            raise AssertionError('unknown request field must not be compared')
        def __float__(self):
            raise AssertionError('unknown request field must not be converted')
    assert witnesses(world,targets=(a,b),observation=ObservationDesign(8,representation=Opaque())).status=='unverified'
    assert witnesses(world,targets=(Target(Opaque(),(a,b)),),observation=ObservationDesign(8)).status=='unverified'
    assert witnesses(world,targets=(a,b),observation=ObservationDesign(8),delta=Opaque()).status=='unverified'


def test_pure_inspection_no_simulation_training_library_or_random_draws(monkeypatch):
    def forbidden(*args,**kw):
        raise AssertionError('witness inspection must not draw or train')
    world,a,b=noises()
    before=pickle.dumps(world)
    old=tg.library.key(world.estimate(tg.forecast(h=2)),tg.TrainConfig())
    np_state=pickle.dumps(np.random.get_state());torch_state=torch.random.get_rng_state().clone()
    monkeypatch.setattr(tg.Block,'simulate',forbidden)
    monkeypatch.setattr(tg.library,'home',forbidden)
    monkeypatch.setattr(type(a),'sample',forbidden)
    report=witnesses(world,targets=(a,b),observation=ObservationDesign(128))
    assert report.status=='non_identified_witness'
    assert pickle.dumps(world)==before
    assert tg.library.key(world.estimate(tg.forecast(h=2)),tg.TrainConfig())==old
    assert pickle.dumps(np.random.get_state())==np_state
    assert torch.equal(torch.random.get_rng_state(),torch_state)
