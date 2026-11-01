"""Independent analytic/finite-difference checks for the opt-in research illustration."""
import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parents[1] / "experiments/identification/sensitivity.py"
_spec = importlib.util.spec_from_file_location("timegram_sensitivity_illustration", _path)
sensitivity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sensitivity)


def point(*values):
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize("theta", [point(.2,.8,1,1), point(.5,.5001,1,1), point(.5,.5,1,1),
                                   point(0,.6,1,1), point(-.4,.3,.7,1.3)])
def test_autodiff_matches_independent_analytic_and_several_fd_steps(theta):
    n=16
    jac=torch.func.jacfwd(lambda x:sensitivity.moment_map(x,n))(theta)
    analytic=sensitivity.analytic_jacobian(theta,n)
    assert torch.isfinite(jac).all()
    torch.testing.assert_close(jac,analytic,atol=1e-10,rtol=1e-9)
    errors=[sensitivity.error_summary(sensitivity.central_difference(theta,h,length=n),jac)
            for h in sensitivity.STEPS]
    assert min(e["relative_frobenius"] for e in errors)<1e-6


def test_full_map_shape_and_zero_start_boundary_are_explicit():
    theta=point(.2,.8,1,1)
    g=sensitivity.moment_map(theta)
    assert g.shape==(8384,)
    assert torch.equal(g[:128],torch.zeros(128,dtype=torch.float64))
    c=sensitivity.covariance(theta,3)
    assert c[0,0]==2
    torch.testing.assert_close(c[1,0],point(1)[0])
    torch.testing.assert_close(c[2,0],point(.68)[0])
    # Stationary initialisation has a different first variance; it cannot be mixed in.
    stationary_variance=1/(1-.2**2)+1/(1-.8**2)
    assert abs(float(c[0,0])-stationary_variance)>1


def test_coincidence_has_two_first_order_null_directions_but_only_variance_witness():
    theta=point(.5,.5,.7,1.3)
    jac=torch.func.jacfwd(lambda x:sensitivity.moment_map(x,16))(theta)
    s=torch.linalg.svdvals(jac)
    assert sensitivity.numerical_rank(s)==2
    torch.testing.assert_close(jac @ point(0,0,1,-1), torch.zeros(jac.shape[0],dtype=torch.float64),atol=1e-12,rtol=0)
    torch.testing.assert_close(jac @ point(1.3,-.7,0,0), torch.zeros(jac.shape[0],dtype=torch.float64),atol=1e-12,rtol=0)
    # Equal/opposite persistence is not the null direction for unequal variances.
    assert torch.linalg.vector_norm(jac @ point(1,-1,0,0))>.1
    record=sensitivity.counterexamples()
    assert record["variance_witness"]["numerical_max_covariance_error"]<1e-12
    p=record["persistence_null_is_not_a_witness"]
    assert p["max_covariance_change"]>1e-5
    assert p["lag_two_entry_change"]==pytest.approx(p["analytic_lag_two_change"],abs=1e-14)


def test_fixed_metric_does_not_differentiate_away_variance_information():
    theta=point(.5,.6,1,2)
    n=8
    fixed=sensitivity.fixed_row_scale(theta,n)
    good=torch.func.jacfwd(lambda x:fixed*sensitivity.moment_map(x,n))(theta)
    bad=torch.func.jacfwd(lambda x:sensitivity.fixed_row_scale(x,n)*sensitivity.moment_map(x,n))(theta)
    # fixed_row_scale explicitly detaches its scale; the public helper remains safe
    # even if called inside a derivative transform.
    torch.testing.assert_close(good[n,2:],point(1/3,1/3),atol=1e-12,rtol=0)
    torch.testing.assert_close(bad,good,atol=1e-12,rtol=0)
    self_normalized=torch.func.jacfwd(lambda x:sensitivity.moment_map(x,n)/sensitivity.covariance(x,n)[0,0])(theta)
    torch.testing.assert_close(self_normalized[n,2:],point(0,0),atol=1e-12,rtol=0)


def test_law_and_path_derivatives_and_injective_singular_map_are_distinguished():
    r=sensitivity.counterexamples()
    ranks=r["path_vs_law"]["ranks_at_rtol_1e-9"]
    assert ranks=={"law_variance_coordinates":1,"law_scale_coordinates":1,"fixed_path_scale_coordinates":2}
    assert r["injective_cube"]["derivative_at_zero"]==[0.0,0.0]
    assert r["injective_cube"]["derivative_at_one"]==[3.0,0.0]
    assert r["label_swap"]["numerical_max_covariance_error"]<1e-12


def test_separated_full_local_rank_does_not_remove_label_swap():
    theta=point(.2,.8,1,1)
    jac=torch.func.jacfwd(lambda x:sensitivity.moment_map(x,16))(theta)
    assert sensitivity.numerical_rank(torch.linalg.svdvals(jac))==4
    torch.testing.assert_close(sensitivity.moment_map(theta,16),sensitivity.moment_map(theta[[1,0,3,2]],16))


def test_rank_thresholds_and_zero_spectrum_are_reported_as_conventions():
    s=point(1,1e-5,1e-10,0)
    assert sensitivity.numerical_rank(s,1e-12)==3
    assert sensitivity.numerical_rank(s,1e-9)==2
    assert sensitivity.numerical_rank(s,1e-3)==1
    assert sensitivity.numerical_rank(point(0,0))==0


@pytest.mark.parametrize("theta", [point(1,.2,1,1),point(.2,.3,0,1),point(.2,.3,-1,1),point(float('nan'),.2,1,1)])
def test_invalid_parameter_points_are_rejected(theta):
    with pytest.raises(ValueError,match="positive innovation"):
        sensitivity.validate_point(theta)


def test_domain_perturbations_and_numeric_conventions_are_not_silently_changed():
    with pytest.raises(ValueError):
        sensitivity.central_difference(point(.9999,.2,1,1),.01,length=8)
    with pytest.raises(ValueError,match="float64"):
        sensitivity.moment_map(torch.tensor([.2,.3,1,1],dtype=torch.float32))
    with pytest.raises(ValueError,match="positive integer"):
        sensitivity.moment_map(point(.2,.3,1,1),0)
