"""Predictive check of the order probabilities on real population records.

No one knows the true order of a real population, but better probabilities over the order
should give better forecasts when they weight the forecasts of the candidate autoregressions.
For each GPDD record (ecology.py) and each of its last five years with at least 15 years before
it, AR(1)-AR(5) are fitted by least squares to the preceding years and their one-step Gaussian
predictive densities are averaged with each method's order probabilities. Methods are scored by
the log predictive density of the held-out year (a proper score) and the squared error of the
mixture mean, averaged per record, with a paired bootstrap over records.

    python experiments/ar_order/ecology_predictive.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from timegram import library

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ecology  # noqa: E402
import run  # noqa: E402

MIN_HISTORY, ORIGINS, MAX_P = 15, 5, 5


def ar_forecasts(x: np.ndarray):
    """One-step predictive means and variances of AR(1)..AR(5) fitted by least squares on the
    demeaned history (common estimation sample, as for the information criteria)."""
    mu = x.mean()
    z = x - mu
    n = len(z)
    target = z[MAX_P:]
    means, variances = [], []
    for p in range(1, MAX_P + 1):
        X = np.stack([z[MAX_P - k:n - k] for k in range(1, p + 1)], -1)
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        resid = target - X @ coef
        sigma2 = max(resid @ resid / max(len(target) - p, 1), 1e-8)
        means.append(mu + coef @ z[::-1][:p])
        variances.append(sigma2)
    return np.array(means), np.array(variances)


def mixture_scores(w, means, variances, y):
    dens = np.exp(-0.5 * (y - means) ** 2 / variances) / np.sqrt(2 * np.pi * variances)
    return float(np.log(max(w @ dens, 1e-300))), float((w @ means - y) ** 2)


def main():
    warnings.filterwarnings("ignore")
    series = ecology.gpdd_series()
    est = library.pretrained(run.NAME)
    cases = []                                            # (record, history, next value)
    for i, s in enumerate(series):
        x = np.asarray(s["series"], float)
        for t in range(max(MIN_HISTORY, len(x) - ORIGINS), len(x)):
            cases.append((i, x[:t], x[t]))
    histories = [h for _, h, _ in cases]
    probs = {"timegram": np.asarray(est.predict(histories))[:, :MAX_P]}
    for name in ("AIC", "BIC"):
        probs[name] = np.stack([run.weights(run.information_criteria((h - h.mean())[None, :])[name])[0]
                                for h in histories])
    probs["uniform"] = np.full((len(cases), MAX_P), 1 / MAX_P)
    probs["AR(1) only"] = np.eye(MAX_P)[np.zeros(len(cases), int)]
    probs["AIC best"] = np.eye(MAX_P)[probs["AIC"].argmax(1)]
    rows = []
    for c, (i, h, y) in enumerate(cases):
        means, variances = ar_forecasts(h)
        r = {"record": i, "n": len(h)}
        for k, w in probs.items():
            r[f"{k} logscore"], r[f"{k} sq"] = mixture_scores(w[c], means, variances, y)
        rows.append(r)
    t = pd.DataFrame(rows)
    per = t.groupby("record").mean()                      # each record weighs the same
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(per), (2000, len(per)))
    out = {"records": int(len(per)), "forecasts": int(len(t))}
    print(f"{len(per)} records, {len(t)} one-step forecasts")
    for k in probs:
        d = (per[f"timegram logscore"] - per[f"{k} logscore"]).to_numpy()
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        mse = per[f"{k} sq"].mean()
        out[k] = {"mean log score": float(per[f"{k} logscore"].mean()), "mse": float(mse),
                  "timegram minus this (log score)": float(d.mean()), "95% interval": [float(lo), float(hi)]}
        print(f"  {k:11s} log score {per[f'{k} logscore'].mean():8.4f}   MSE {mse:.4f}   "
              f"timegram - {k}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (ecology.RESULTS / "gpdd_predictive.json").write_text(json.dumps(out, indent=1))
    t.to_csv(ecology.RESULTS / "gpdd_predictive.csv", index=False)


if __name__ == "__main__":
    main()
