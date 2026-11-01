"""Per-process ("uniform") calibration of AR(5) predictive quantiles (the Simulacrum, sec. 6.2).

World (as in the Simulacrum paper): AR(5), partial autocorrelations uniform on (-1, 1)^5, N(0, 1)
innovations, no constant, observed lengths uniform on 12..64. Estimand: the 0.05, 0.5
and 0.95 quantiles of y_{T+1}. Every estimator here is ONE network.

    unif         pinball only (ERM): the stand-in for any average-risk forecaster
    unif_cont    unif trained further, pinball only, with replicates: the control for
                 "more training"
    calib_<w>    unif trained further with (1 - w) * risk + w * miscoverage, debiased
    plugin_<w>   the same with the Simulacrum paper's plug-in penalty (debiased=False)

    python experiments/ar5_calibration/run.py train unif
    python experiments/ar5_calibration/run.py train calib_0.97
    python experiments/ar5_calibration/run.py evaluate unif unif_cont calib_0.97 ...

Evaluation is on fresh processes (other seeds than training), with many replicates
each, and the exact coverage indicator.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import timegram as tg
from timegram import library
from timegram.equivariance import EquivarianceWarning

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LEVELS = (0.05, 0.5, 0.95)
LENGTHS = (12, 64)
warnings.filterwarnings("ignore", category=EquivarianceWarning)       # a zero-mean AR world, by design

BASE = dict(lengths=LENGTHS, backend="fast", chunk=16384, batch_size=4096, log_every=1000)
NAME = "ar5cal"                 # library aliases ar5cal_<variant>
BASE_SEED = 2                   # seed 0's library key belongs to a diverged run from before the fast-AR fix
BASE_BUDGET = 200_000_000
TUNE = dict(budget=50_000_000, replicates=32, warmup=0)


def problem(objective=()):
    p = (tg.ar(5)
         .estimate(tg.forecast(h=1, quantiles=LEVELS))
         .loss(tg.pinball(scale="relative")))
    return p.objective(*objective) if objective else p


def variant(name: str):
    """(problem, training settings) of a named variant."""
    if name == "unif":
        return problem(), dict(BASE, budget=BASE_BUDGET, seed=BASE_SEED)
    tune = dict(BASE, **TUNE, init=library.resolve(f"{NAME}_unif"))
    if name == "unif_cont":
        return problem([tg.risk()]), tune | {"seed": 1}
    kind, w = name.rsplit("_", 1)
    w = float(w)
    debiased = {"calib": True, "plugin": False}[kind]
    return problem([tg.risk(weight=1 - w), tg.miscoverage(weight=w, debiased=debiased, smoothing="fixed")]), tune | {"seed": 1}


def train(name: str):
    p, settings = variant(name)
    est = p.pretrain(**settings)
    library.alias(f"{NAME}_{name}", est.key)
    print(f"{name}: {est.key}")


# ---------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------
def evaluation_set(processes: int, replicates: int, lengths, seed: int, chunk: int = 16384):
    """Fresh processes with `replicates` series each, simulated a chunk of processes at a
    time (all at once would need tens of GB on the GPU)."""
    from timegram.simulate import Simulation
    gen = torch.Generator(device="cuda").manual_seed(seed)
    per_call = max(1, chunk // replicates)
    parts = [tg.simulate(tg.ar(5), n=min(per_call, processes - start), replicates=replicates, lengths=lengths,
                         horizon=1, backend="fast", device="cuda", gen=gen)
             for start in range(0, processes, per_call)]
    sim = Simulation(np.concatenate([q.context for q in parts]), np.concatenate([q.target for q in parts]),
                     np.concatenate([q.lengths for q in parts]),
                     {"ar.phi": np.concatenate([q.params["ar.phi"] for q in parts])}, replicates)
    histories = [sim.context[i, -n:] for i, n in enumerate(sim.lengths)]
    return sim, histories


def ols_quantiles(histories, levels=LEVELS, p: int = 5) -> np.ndarray:
    """Classical plug-in: OLS AR(p) without intercept, Gaussian quantiles from the
    residual variance (parameter uncertainty ignored). [n, Q]."""
    out = np.empty((len(histories), len(levels)))
    z = stats.norm.ppf(levels)
    for n in sorted({len(h) for h in histories}):
        idx = [i for i, h in enumerate(histories) if len(h) == n]
        y = np.stack([histories[i] for i in idx])                       # [m, n]
        X = np.stack([y[:, p - k - 1:n - k - 1] for k in range(p)], -1)  # [m, n - p, p]
        target = y[:, p:]
        XtX = np.einsum("mtp,mtq->mpq", X, X) + 1e-8 * np.eye(p)
        phi = np.linalg.solve(XtX, np.einsum("mtp,mt->mp", X, target)[..., None])[..., 0]
        resid = target - np.einsum("mtp,mp->mt", X, phi)
        dof = max(n - 2 * p, 1)
        sigma = np.sqrt((resid ** 2).sum(1) / dof)
        mean = (y[:, ::-1][:, :p] * phi).sum(1)
        out[idx] = mean[:, None] + sigma[:, None] * z
    return out


def summarize(quantiles, sim, replicates: int, levels=LEVELS) -> dict:
    """Marginal and per-process coverage, pinball, binomial rejection rate."""
    y = sim.target[:, 0]
    q = quantiles.reshape(len(y), len(levels))
    scale = np.nanstd(sim.context, axis=1) + 1e-9                        # the network's own scale
    hit = (y[:, None] <= q).astype(float)
    J = len(y) // replicates
    cov = hit.reshape(J, replicates, -1).mean(1)                         # [J, Q]
    tau = np.array(levels)
    err = y[:, None] - q
    pin = np.maximum(tau * err, (tau - 1) * err) / scale[:, None]        # relative scale
    lo = stats.binom.cdf(np.round(cov * replicates), replicates, tau)
    hi = stats.binom.sf(np.round(cov * replicates) - 1, replicates, tau)
    rejected = (2 * np.minimum(lo, hi) < 0.05)
    floor = np.sqrt(2 * tau * (1 - tau) / (math.pi * replicates))       # E|cov - tau| if calibrated
    return {"marginal_coverage": hit.mean(0).tolist(), "per_process_abs_gap": np.abs(cov - tau).mean(0).tolist(),
            "abs_gap_floor": floor.tolist(), "rejected_share": rejected.mean(0).tolist(),
            "pinball_relative": pin.mean(0).tolist(), "coverage": cov}


def coverage_curve(cov, phi, column: int, bins: int = 20):
    """Mean coverage of the processes in each bin of a coefficient (the Simulacrum paper's 'hump')."""
    edges = np.linspace(-1, 1, bins + 1) if phi.min() >= -1 else np.quantile(phi, np.linspace(0, 1, bins + 1))
    which = np.clip(np.digitize(phi, edges) - 1, 0, bins - 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    means = [float(cov[which == b, column].mean()) if (which == b).any() else float("nan") for b in range(bins)]
    return centers.tolist(), means


def evaluate(names, processes=1000, replicates=400):
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"levels": LEVELS, "processes": processes, "replicates": replicates, "sets": {}}
    sets = {"lengths 12-64": LENGTHS, "length 16": (16, 16), "length 32": (32, 32), "length 64": (64, 64)}
    for k, (label, lengths) in enumerate(sets.items()):
        sim, histories = evaluation_set(processes, replicates, lengths, seed=1000 + k)
        phi = np.asarray(sim.params["ar.phi"]).reshape(processes, replicates, 5)[:, 0]
        rows = {}
        for name in names:                                   # one network on the GPU at a time
            est = library.pretrained(f"{NAME}_{name}").to("cuda")
            rows[name] = summarize(est.forecast(histories)[:, 0, :], sim, replicates)
            del est
            torch.cuda.empty_cache()
        rows["ols_plugin"] = summarize(ols_quantiles(histories), sim, replicates)
        out = {}
        for name, r in rows.items():
            cov = r.pop("coverage")
            r["curve_phi1_q95"] = coverage_curve(cov, phi[:, 0], LEVELS.index(0.95))
            r["curve_phi1_q05"] = coverage_curve(cov, phi[:, 0], LEVELS.index(0.05))
            out[name] = r
        report["sets"][label] = out
        print(f"\n{label}  ({processes} processes x {replicates} replicates)")
        print(f"  {'estimator':14s} {'cov q05/q50/q95':>22s} {'|gap| q05/q95':>16s} {'rejected q05/q95':>18s} {'pinball':>8s}")
        for name, r in out.items():
            c, g, rj = r["marginal_coverage"], r["per_process_abs_gap"], r["rejected_share"]
            print(f"  {name:14s} {c[0]:6.3f} {c[1]:6.3f} {c[2]:6.3f}   {g[0]:6.4f} {g[2]:6.4f}   "
                  f"{rj[0]:7.1%} {rj[2]:7.1%}   {np.mean(r['pinball_relative']):.4f}")
        print(f"  floor |gap| at R={replicates}: {out[names[0]]['abs_gap_floor'][0]:.4f} / "
              f"{out[names[0]]['abs_gap_floor'][2]:.4f}", flush=True)
        del sim, histories, rows                             # one evaluation set in memory at a time
        gc.collect()
    (RESULTS / "evaluation.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "evaluate"])
    ap.add_argument("names", nargs="+")
    ap.add_argument("--processes", type=int, default=1000)
    ap.add_argument("--replicates", type=int, default=400)
    a = ap.parse_args()
    if a.command == "train":
        for name in a.names:
            train(name)
    else:
        evaluate(a.names, a.processes, a.replicates)
