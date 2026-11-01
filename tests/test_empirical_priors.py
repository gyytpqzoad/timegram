"""Experiment helper: ETS form labels from statsforecast fits are marked damped once."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "experiments" / "accuracy"


def _module():
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location("_empirical_priors", HERE / "empirical_priors.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(HERE))
    return mod


def test_form_label_is_idempotent():
    label = _module().form_label
    assert label("AAN", True) == "AAdN"            # early fits: damping only visible through phi
    assert label("AAdN", True) == "AAdN"           # already marked: not marked twice
    assert label("ANN", True) == "ANN"             # no trend, nothing to damp
    assert label("AAA", False) == "AAA"
