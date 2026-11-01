"""Independent closed-form check of the research evaluator's finite-window law."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_path = Path(__file__).resolve().parents[1] / "experiments/identification/sensitivity.py"
_spec = importlib.util.spec_from_file_location("sensitivity_review_subject", _path)
subject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(subject)


@pytest.mark.parametrize("theta", [(.2, .8, 1, 1), (.5, .5001, 1, 1),
                                   (.5, .5, .7, 1.3), (0, 0, 1, 2),
                                   (-.7, .3, .9, 1.1), (.99, -.95, 2, .4)])
def test_covariance_matches_independent_geometric_series_formula(theta):
    length = 128
    t = np.arange(1, length + 1)
    lag = np.abs(t[:, None] - t[None, :])
    shared_history = np.minimum(t[:, None], t[None, :])
    expected = sum(v * phi ** lag * (1 - phi ** (2 * shared_history)) / (1 - phi**2)
                   for phi, v in zip(theta[:2], theta[2:]))
    actual = subject.covariance(torch.tensor(theta, dtype=torch.float64), length).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
