"""Real benchmark data and the evaluation protocol of the experiments.

Quarterly sets: M1, M3, M4 and Tourism (via `fcompdata`). Each series' history is its
official training part and the test window its official horizon (8 for all four).
MASE uses the full-history seasonal-naive scale (lag = period when the history covers
more than two periods, else lag 1); sMAPE is in percent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FREQ = {1: "yearly", 4: "quarterly", 12: "monthly"}


@dataclass
class Dataset:
    name: str
    period: int
    horizon: int
    histories: list
    futures: np.ndarray

    def __len__(self):
        return len(self.histories)


def load(name: str, period: int = 4) -> Dataset:
    """name in m1, m3, m4, tourism."""
    from fcompdata import load_m1, load_m3, load_m4, load_tourism
    freq = FREQ[period]
    if name == "m1":
        series = [s for s in load_m1() if s.period == period]
    elif name == "m3":
        series = [s for s in load_m3() if s.period == period and s.type == freq]
    elif name == "m4":
        series = list(load_m4(freq))
    elif name == "tourism":
        series = [s for s in load_tourism() if s.period == period]
    else:
        raise ValueError(name)
    horizon = len(series[0].xx)
    return Dataset(f"{name.upper() if name != 'tourism' else 'Tourism'}-{freq[0].upper()}", period, horizon,
                   [np.asarray(s.x, float) for s in series], np.stack([np.asarray(s.xx, float) for s in series]))


def mase_scale(history: np.ndarray, period: int, eps: float = 1e-8) -> float:
    lag = period if len(history) > 2 * period else 1
    return max(float(np.mean(np.abs(history[lag:] - history[:-lag]))) if len(history) > lag else 0.0, eps)


def seasonal_naive(histories, period, horizon) -> np.ndarray:
    return np.stack([np.resize(h[-period:], horizon) if len(h) >= period else np.full(horizon, h[-1])
                     for h in histories])


def score(ds: Dataset, forecasts: np.ndarray) -> dict:
    """Mean MASE and sMAPE over the dataset, plus the per-series MASE."""
    f, y = forecasts[:, :ds.horizon], ds.futures
    scale = np.array([mase_scale(h, ds.period) for h in ds.histories])
    mase = np.mean(np.abs(f - y), 1) / scale
    smape = np.mean(200 * np.abs(f - y) / (np.abs(f) + np.abs(y) + 1e-8), 1)
    return {"mase": float(mase.mean()), "smape": float(smape.mean()), "per_series_mase": mase}


def evaluate(estimator, names=("m1", "m3", "m4", "tourism"), period=4) -> "pd.DataFrame":
    """MASE / sMAPE of an estimator's point forecasts on each quarterly set, with seasonal
    naive for reference."""
    import pandas as pd
    rows = []
    for name in names:
        ds = load(name, period)
        fc = estimator.forecast(ds.histories)
        sc, sn = score(ds, fc), score(ds, seasonal_naive(ds.histories, period, ds.horizon))
        rows.append({"dataset": ds.name, "series": len(ds), "MASE": sc["mase"], "sMAPE": sc["smape"],
                     "MASE sNaive": sn["mase"], "rel. MASE": sc["mase"] / sn["mase"]})
    return pd.DataFrame(rows)
