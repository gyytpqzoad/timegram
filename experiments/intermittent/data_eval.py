"""The four intermittent datasets of fable.intermittent and their evaluation protocol.

Data: the package's own datasets (CRAN fable.intermittent 0.3.0, exported to CSV by
data/export.R). Protocol of Damato et al. (2026), Section 5.1:
- two expanding windows per series, T = L - j h, j = 1, 2; h = 6 (Auto), 12 (RAF),
  28 (Pasta, TinyM5);
- RMSSE of the forecast mean, scaled by the in-sample mean squared one-step difference;
- scaled quantile score sQS_tau at tau = 0.5, 0.75, 0.835, 0.975, 0.995: the mean quantile score
  over the horizon divided by the in-sample mean absolute one-step difference;
- per series, the mean over the two windows; per dataset, the mean over series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
H = {"auto": 6, "raf": 12, "pasta": 28, "tinym5": 28, "carparts": 12}
PERIOD = {"auto": 12, "raf": 12, "pasta": 7, "tinym5": 7, "carparts": 12}
TAUS = (0.5, 0.75, 0.835, 0.975, 0.995)


def load(name: str) -> list[np.ndarray]:
    if name == "carparts":          # Hyndman et al. (2008), via R expsmooth (data/carparts.R); complete series only
        t = pd.read_csv(DATA / "carparts.csv")
        return [t[c].to_numpy(float) for c in t.columns if t[c].notna().all()]
    if name in ("auto", "raf"):
        t = pd.read_csv(DATA / f"{name}.csv")
        key, time = "series_id", "index"
    elif name == "pasta":
        t = pd.read_csv(DATA / "pasta.csv")
        t["key"] = t.brand.astype(str) + "/" + t["product"].astype(str)
        key, time = "key", "index"
    else:
        t = pd.read_csv(DATA / "tinyM5.csv", usecols=["item_id", "store_id", "value", "date"])
        t["key"] = t.item_id.astype(str) + "/" + t.store_id.astype(str)
        key, time = "key", "date"
    t = t.sort_values([key, time])
    return [g.value.to_numpy(float) for _, g in t.groupby(key, sort=True)]


def windows(name: str, series):
    """(series index, window j, history, future) for j = 1, 2."""
    h = H[name]
    for i, y in enumerate(series):
        for j in (1, 2):
            T = len(y) - j * h
            yield i, j, y[:T], y[T:T + h]


def scores(history, future, mean, quantiles) -> dict:
    """quantiles: {tau: array [h]}. Returns RMSSE and sQS_tau for one window."""
    d = np.diff(history)
    out = {"RMSSE": np.sqrt(np.mean((future - mean) ** 2) / np.mean(d ** 2))}
    mad = np.mean(np.abs(d))
    for tau in TAUS:
        e = future - quantiles[tau]
        out[f"sQS{tau:g}"] = np.mean(np.maximum(tau * e, (tau - 1) * e)) / mad
    return out


def evaluate(name, series, forecaster) -> pd.DataFrame:
    """forecaster(history, h) -> (mean [h], {tau: [h]}). Mean over windows, then over series."""
    rows = []
    for i, j, hist, fut in windows(name, series):
        mean, q = forecaster(hist, len(fut))
        rows.append({"series": i, "window": j, **scores(hist, fut, mean, q)})
    return pd.DataFrame(rows)


def summarize(per_window: pd.DataFrame) -> pd.Series:
    per_series = per_window.drop(columns="window").groupby("series").mean()
    finite = np.isfinite(per_series).all(axis=1)
    s = per_series[finite].mean()
    s["series used"] = int(finite.sum())
    return s


def empsd(history, h):
    """EMPSD: the empirical distribution of the history (R quantile type 7)."""
    q = {tau: np.full(h, np.quantile(history, tau)) for tau in TAUS}
    return np.full(h, history.mean()), q


if __name__ == "__main__":
    for name in ("auto", "pasta", "raf", "tinym5"):
        series = load(name)
        print(name, len(series), "series; lengths", {len(s) for s in series} if len(series) < 10 else
              (min(map(len, series)), max(map(len, series))), flush=True)
        print(summarize(evaluate(name, series, empsd)).round(3).to_string(), flush=True)
