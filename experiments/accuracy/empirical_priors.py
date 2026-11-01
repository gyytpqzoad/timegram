"""Empirical ETS priors fitted to real training histories (reference worlds of the search).

Fit additive-error ETS (statsforecast AutoETS, model "AZZ": trend N/A/Ad, season N/A,
chosen by AICc) to real histories, then summarize what the data say about the world:
which ETS forms win, the smoothing parameters, and the size of the slope relative to the
noise. Only the validation split's histories are used (each training series without its
last h points), so nothing from the validation or test windows enters a prior.

    python experiments/accuracy/empirical_priors.py fit         # CPU, about 10 minutes
    python experiments/accuracy/empirical_priors.py summarize
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from timegram.data import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

FITS = run.RESULTS / "ets_fits.csv"
PER_SET = 1500                      # at most this many series per dataset and frequency
SEED = 0


def fit():
    from statsforecast.models import AutoETS
    warnings.filterwarnings("ignore")
    run.RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(SEED)
    for name, period in run.SETS:
        ds = run.validation(load(name, period))
        idx = np.arange(len(ds))
        if len(idx) > PER_SET:
            idx = np.sort(rng.choice(idx, PER_SET, replace=False))
        for i in idx:
            y = np.asarray(ds.histories[i], float)[-run.CONTEXT:]          # what the network sees
            season = period if period > 1 and len(y) >= 2 * period + 2 else 1
            try:
                m = AutoETS(season_length=season, model="AZZ")
                m.fit(y)
                fm = m.model_
            except Exception:                                              # noqa: BLE001
                continue
            comp, par = fm["components"], np.asarray(fm["par"], float)
            sigma = float(np.sqrt(fm["sigma2"]))
            trend, damped = comp[1], comp[3] == "D"          # statsforecast: components[3] "D" = damped
            states = np.asarray(fm["states"], float)
            slope = states[:, 1] if trend != "N" else np.zeros(len(states))
            rows.append({"dataset": name, "period": period, "series": int(i), "length": len(y),
                         "form": f"A{trend}{'d' if damped else ''}{comp[2]}",
                         "alpha": par[0], "beta": par[1], "gamma": par[2], "phi": par[3],
                         "slope_to_noise": float(np.mean(slope[1:])) / sigma if sigma > 0 else np.nan,
                         "sigma_to_level": sigma / max(abs(float(np.mean(y))), 1e-9)})
        print(f"{name}-{period}: {sum(r['dataset'] == name and r['period'] == period for r in rows)} fits",
              flush=True)
    pd.DataFrame(rows).to_csv(FITS, index=False)


def form_label(form: str, damped: bool) -> str:
    """The ETS form with damping marked once ("AAdN"): idempotent, so labels already marked
    (fits from the corrected extractor) are not marked twice. Early fits labelled damped
    models without the "d"; their non-missing phi still identifies them."""
    if "d" in form or not damped or form[1] != "A":
        return form
    return form[:2] + "d" + form[2:]


def summarize():
    f = pd.read_csv(FITS)
    pd.set_option("display.width", 200)
    print("== winning additive form, share by frequency")
    print(pd.crosstab(f.period, f.form, normalize="index").round(3).to_string())
    print("\n== smoothing parameters (10/50/90%) by frequency")
    q = lambda s: s.dropna().quantile([.1, .5, .9]).round(3).tolist()        # noqa: E731
    for period, g in f.groupby("period"):
        print(f"  period {period}: alpha {q(g.alpha)}  beta/alpha {q(g.beta / g.alpha)}  "
              f"gamma/(1-alpha) {q(g.gamma / (1 - g.alpha))}  phi {q(g.phi)}")
        t = g[g.form.str[1] == "A"]
        print(f"      trended series: slope/noise {q(t.slope_to_noise)}, "
              f"share positive {np.mean(t.slope_to_noise > 0):.2f} (n={len(t)})")


PRIORS = run.HERE / "ets_priors.json"
EPS = 1e-3
COORDS = ("alpha", "beta_frac", "gamma_frac", "phi_frac")


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def stylize():
    """Clean parametric priors per frequency, from the fits:
    - (alpha, beta/alpha, gamma/(1-alpha), (phi-0.8)/0.18): a multivariate normal on the logit
      scale (the coordinates are clearly dependent; pairwise-complete covariance, projected to
      the nearest positive-definite matrix);
    - the starting slope in noise units: an independent location-scale Student-t (heavy tails;
      nearly uncorrelated with the smoothing parameters);
    - the ETS forms: their observed shares."""
    import json
    from scipy import stats
    f = pd.read_csv(FITS)
    f["damped"] = f.phi.notna()
    out = {}
    for period, g in f.groupby("period"):
        z = pd.DataFrame({"alpha": _logit(g.alpha), "beta_frac": _logit(g.beta / g.alpha),
                          "gamma_frac": _logit(g.gamma / (1 - g.alpha)),
                          "phi_frac": _logit((g.phi - 0.8) / 0.18)})
        mean = z.mean().fillna(0.0)
        cov = z.cov().fillna(0.0).to_numpy(copy=True)      # owned and writable (pandas 3)
        for i, c in enumerate(COORDS):                       # unused coordinates (e.g. no season)
            if z[c].notna().sum() < 10:
                cov[i, :], cov[:, i], cov[i, i] = 0.0, 0.0, 1.0
        w, v = np.linalg.eigh((cov + cov.T) / 2)
        cov = (v * np.clip(w, 1e-4, None)) @ v.T
        trended = g[g.form.str[1] == "A"].slope_to_noise.dropna()
        df, loc, scale = stats.t.fit(trended)
        shares = g.apply(lambda r: form_label(r.form, r.damped), axis=1).value_counts(normalize=True)
        shares = shares.round(4).to_dict()
        out[str(period)] = {"logit_mean": mean.round(4).tolist(), "logit_cov": np.round(cov, 4).tolist(),
                            "coords": list(COORDS), "slope_t": {"df": round(df, 3), "loc": round(loc, 4),
                                                                "scale": round(scale, 4)},
                            "forms": shares, "n": int(len(g))}
        print(f"period {period}: forms {shares}\n  slope t(df={df:.2f}) loc {loc:.3f} scale {scale:.3f}\n"
              f"  logit mean {mean.round(2).tolist()}")
    PRIORS.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"fit": fit, "summarize": summarize, "stylize": stylize}[sys.argv[1]]()
