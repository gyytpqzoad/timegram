"""Calibrated selection of the order of a short autoregression.

World: one of AR(1), ..., AR(5) with equal probability, coefficients uniform over the
stationary region in partial autocorrelations, with the last one at least 0.2 in magnitude
(so the order is distinguishable). Estimand: which order, as probabilities, trained with
the log score; a strictly proper scoring rule, so under the world the optimum is the
posterior probability of each order given the normalized history. Normalization removes
location and scale, which the raw-series posterior could use (the stationary variance depends
on the coefficients): the target is the posterior given the series' shape. The information
criteria below are scale invariant but use the known zero mean, which the network does not see.

Classical comparison: AIC, AICc and BIC from least-squares fits of AR(1..5) on a common
estimation sample (first 5 observations held back), with Akaike / Schwarz weights
w_p ~ exp(-IC_p / 2) as probabilities. Evaluation at fixed short lengths.

    python experiments/ar_order/run.py train
    python experiments/ar_order/run.py evaluate
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import timegram as tg
from timegram import library

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ORDERS = (1, 2, 3, 4, 5)
LAST_MIN = 0.2
LENGTHS = (15, 100)
NS = (15, 20, 30, 50, 100)
NAME = "ar_order"


def world():
    return tg.select(*[tg.ar(p, phi=tg.stationary(p, last_min=LAST_MIN)) for p in ORDERS])


def train():
    est = world().estimate(tg.which()).pretrain(lengths=LENGTHS, budget=30_000_000, batch_size=4096,
                                                chunk=16384, backend="fast", log_every=1000)
    library.alias(NAME, est.key)
    print(f"{NAME}: {est.key}", flush=True)


# ---------------------------------------------------------------------------------------
def information_criteria(series: np.ndarray, max_p: int = 5) -> dict[str, np.ndarray]:
    """AIC, AICc, BIC of least-squares AR(p) fits without intercept (the world has zero
    mean), p = 1..max_p, all on the same sample t = max_p+1..n. series: [m, n]."""
    m, n = series.shape
    target = series[:, max_p:]
    n_eff = n - max_p
    ic = {k: np.empty((m, max_p)) for k in ("AIC", "AICc", "BIC")}
    for p in range(1, max_p + 1):
        X = np.stack([series[:, max_p - k:n - k] for k in range(1, p + 1)], -1)        # [m, n_eff, p]
        coef = np.linalg.solve(np.einsum("mtp,mtq->mpq", X, X) + 1e-10 * np.eye(p),
                               np.einsum("mtp,mt->mp", X, target)[..., None])[..., 0]
        rss = ((target - np.einsum("mtp,mp->mt", X, coef)) ** 2).sum(1)
        loglik = -0.5 * n_eff * (np.log(2 * np.pi * rss / n_eff) + 1)
        k = p + 1                                                                    # coefficients + variance
        ic["AIC"][:, p - 1] = -2 * loglik + 2 * k
        ic["AICc"][:, p - 1] = ic["AIC"][:, p - 1] + 2 * k * (k + 1) / max(n_eff - k - 1, 1)
        ic["BIC"][:, p - 1] = -2 * loglik + k * np.log(n_eff)
    return ic


def weights(ic: np.ndarray) -> np.ndarray:
    z = -0.5 * (ic - ic.min(1, keepdims=True))
    w = np.exp(z)
    return w / w.sum(1, keepdims=True)


def scores(prob: np.ndarray, order: np.ndarray, bins: int = 10) -> dict:
    """Selection accuracy, log score, multiclass Brier score, and the expected calibration
    error of the top-label confidence."""
    m = len(order)
    p_true = prob[np.arange(m), order]
    top = prob.argmax(1)
    conf = prob.max(1)
    onehot = np.eye(prob.shape[1])[order]
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    ece = sum(abs(conf[which == b].mean() - (top == order)[which == b].mean()) * (which == b).mean()
              for b in range(bins) if (which == b).any())
    return {"accuracy": float((top == order).mean()), "log score": float(-np.log(np.clip(p_true, 1e-12, 1)).mean()),
            "Brier": float(((prob - onehot) ** 2).sum(1).mean()), "ECE": float(ece)}


def reliability(prob: np.ndarray, order: np.ndarray, bins: int = 10):
    """All (series, order) pairs: predicted probability against observed frequency."""
    p = prob.ravel()
    hit = (np.eye(prob.shape[1])[order]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [(float(p[which == b].mean()), float(hit[which == b].mean()), int((which == b).sum()))
            for b in range(bins) if (which == b).any()]


def evaluate(processes: int = 4000):
    RESULTS.mkdir(parents=True, exist_ok=True)
    est = library.pretrained(NAME).to("cuda")
    rows, curves = [], {}
    for n in NS:
        sim = tg.simulate(world(), n=processes, lengths=(n, n), seed=500 + n, backend="fast", device="cuda")
        series = sim.context[:, -n:]
        order = np.asarray(sim.params["select.which"], int).ravel()
        probs = {"timegram": est.predict(list(series))}
        for name, ic in information_criteria(series).items():
            probs[name] = weights(ic)
        for name, prob in probs.items():
            rows.append({"n": n, "method": name, **scores(prob, order)})
            curves[f"{name} n={n}"] = reliability(prob, order)
        print(pd.DataFrame([r for r in rows if r["n"] == n]).round(4).to_string(index=False), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "selection.csv", index=False)
    (RESULTS / "reliability.json").write_text(json.dumps(curves, indent=1))
    return table


def figure():
    import matplotlib.pyplot as plt
    table = pd.read_csv(RESULTS / "selection.csv")
    curves = json.loads((RESULTS / "reliability.json").read_text())
    colors = {"timegram": "#1b9e77", "AIC": "#d55e00", "AICc": "#e6ab02", "BIC": "#4269d0"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for method, c in colors.items():
        t = table[table.method == method]
        axes[0].plot(t.n, t.accuracy, marker="o", color=c, label=method)
        axes[1].plot(t.n, t["log score"], marker="o", color=c, label=method)
        pts = np.array(curves[f"{method} n=20"])
        axes[2].plot(pts[:, 0], pts[:, 1], marker="o", color=c, label=method)
    axes[0].set(xlabel="series length n", ylabel="selection accuracy", xscale="log", xticks=NS, xticklabels=NS)
    axes[1].set(xlabel="series length n", ylabel="log score (lower is better)", xscale="log", xticks=NS, xticklabels=NS)
    axes[2].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[2].set(xlabel="predicted probability of an order", ylabel="observed frequency", title="reliability, n = 20")
    from matplotlib.ticker import NullFormatter
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:2]:
        ax.xaxis.set_minor_formatter(NullFormatter())          # only the evaluated lengths
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "selection.pdf")
    fig.savefig(HERE / "selection.png", dpi=140)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    {"train": train, "evaluate": lambda: (evaluate(), figure())}[sys.argv[1]]()
