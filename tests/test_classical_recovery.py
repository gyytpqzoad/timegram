"""Opt-in long-series recovery checks. No neural training."""
import pytest
pytest.importorskip("statsmodels")
from validation.classical_recovery import all_cases,recovery


@pytest.mark.recovery
@pytest.mark.parametrize("case",all_cases(),ids=lambda c:c["name"])
def test_atomic_grammar_parameters_recovered_by_classical_mle(case):
    result = recovery(case,nobs=12000,seed=11)
    assert result["converged"],result
    for check in result["checks"].values():
        assert check["passed"],result
