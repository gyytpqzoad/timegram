"""Opt-in long-series recovery, independent arch MLE, no neural training."""
import pytest
pytest.importorskip("arch")
from validation.volatility_recovery import cases, recovery


@pytest.mark.recovery
@pytest.mark.parametrize("case",cases(),ids=lambda case:case["name"])
def test_grammar_volatility_parameters_recovered_by_arch_mle(case):
    result=recovery(case,seed=11)
    assert result["passed"],result
