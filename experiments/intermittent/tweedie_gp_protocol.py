"""Car Parts with the protocol of Damato, Azzimonti and Corani (2025, IJF; arXiv 2502.19086):
the series with at least one zero, the first 45 months as training, the last 6 as test; RMSSE of the
forecast mean; quantile losses at 0.5, 0.8, 0.9, 0.95, 0.99 scaled by the in-sample loss of the
empirical quantile; srps0.5+ over the levels 0.5, 0.55, ..., 0.95, 0.99, scaled the same way.
Their Table 2 (Car Parts) is in PUBLISHED; the empirical-quantile row checks this implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T, H = 45, 6
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
RPS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
COLS = ["rmsse"] + [f"sQ{q:g}" for q in LEVELS] + ["srps0.5+"]
PUBLISHED = pd.DataFrame({        # Table 2, Car Parts block
    "EmpQuant": [0.66, 1.13, 1.18, 1.25, 1.32, 1.86, 1.19],
    "WSS": [0.78, 1.15, 1.33, 1.45, 1.63, 2.00, 1.29],
    "ADIDA": [0.60, 1.20, 1.14, 1.16, 1.34, 3.71, 1.25],
    "iETS": [0.59, 1.10, 1.15, 1.24, 1.46, 3.03, 1.18],
    "NegBinGP": [0.61, 1.10, 1.09, 1.15, 1.23, 1.64, 1.11],
    "TweedieGP": [0.61, 1.11, 1.09, 1.13, 1.20, 1.55, 1.10],
}, index=COLS).T


def qloss(q, f, y):
    e = y - f
    return np.maximum(q * e, (q - 1) * e)


def select(series_list):
    """Series complete over the 51 months with at least one zero (ADI > 1)."""
    return [y for y in series_list if np.isfinite(y).all() and len(y) == T + H and (y == 0).any()]


def scores(y, mean, quant) -> dict:
    """quant: {level: [H]} for every level in RPS."""
    tr, te = y[:T], y[T:T + H]
    out = {"rmsse": np.sqrt(np.mean((te - mean) ** 2) / np.mean(np.diff(tr) ** 2))}
    emp = {q: np.quantile(tr, q) for q in RPS}
    for q in LEVELS:
        out[f"sQ{q:g}"] = qloss(q, quant[q], te).mean() / qloss(q, emp[q], tr).mean()
    num = np.mean([qloss(q, quant[q], te).mean() for q in RPS])
    den = np.mean([qloss(q, emp[q], tr).mean() for q in RPS])
    out["srps0.5+"] = num / den
    return out


def evaluate(series, forecaster) -> pd.Series:
    rows = []
    for y in series:
        mean, quant = forecaster(y[:T])
        rows.append(scores(y, mean, quant))
    t = pd.DataFrame(rows)
    t = t[np.isfinite(t).all(axis=1)]
    s = t.mean()
    s["series"] = len(t)
    return s


def empquant(history):
    return np.full(H, history.mean()), {q: np.full(H, np.quantile(history, q)) for q in RPS}


if __name__ == "__main__" and len(sys.argv) == 1:
    sys.path.insert(0, str(HERE))
    import data_eval
    raw = pd.read_csv(data_eval.DATA / "carparts.csv")
    series = select([raw[c].to_numpy(float) for c in raw.columns])
    print(len(series), "series")
    print("EmpQuant, ours:     ", evaluate(series, empquant).round(2).to_dict())
    print("EmpQuant, published:", PUBLISHED.loc["EmpQuant"].to_dict())


# ---------------------------------------------------------------------------------------------
# timegram: the world selected by search.py ("combined3"),
# a quantile network for the eleven RPS levels (50M series, as run.py's networks), quantiles
# rounded to integers as selected; the mean is search.py's combined3 mean network. One run,
# reported against every published row.
def train_quantiles():
    import timegram as tg
    from timegram import library
    import run
    import search
    opts = {}
    for c in ("spread", "drift", "integer"):
        opts.update({**search.MOVES, **search.ROUND2}[c])
    prob = (search.world(**opts).estimate(tg.forecast(h=run.H, quantiles=RPS))
            .loss(tg.pinball(scale="relative")))
    est = prob.pretrain(**run.TRAIN)
    library.alias("intermittent_rps_quantile", est.key)
    print("rps quantile", est.key, flush=True)


def timegram_forecaster():
    from timegram import library
    q_est = library.pretrained("intermittent_rps_quantile").to("cuda")
    m_est = library.pretrained("intermittent_s_combined3_mean").to("cuda")

    def f(history):
        q = np.sort(np.asarray(q_est.forecast([history[-128:]]))[0, :H, :], axis=-1)
        q = np.round(np.clip(q, 0, None))
        mean = np.clip(np.asarray(m_est.forecast([history[-128:]]))[0, :H], 0, None)
        return mean, {lev: q[:, i] for i, lev in enumerate(RPS)}
    return f


def run_timegram():
    import data_eval
    raw = pd.read_csv(data_eval.DATA / "carparts.csv")
    series = select([raw[c].to_numpy(float) for c in raw.columns])
    table = PUBLISHED.copy()
    table.loc["timegram"] = evaluate(series, timegram_forecaster())[COLS]
    table.round(3).to_csv(HERE / "results" / "carparts_tweedie_gp_protocol.csv")
    print(table.round(2).to_string())


if __name__ == "__main__" and len(sys.argv) > 1:
    sys.path.insert(0, str(HERE))
    {"train": train_quantiles, "evaluate": run_timegram}[sys.argv[1]]()
