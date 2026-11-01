"""Intermittent demand: one pretrained forecaster against the fable.intermittent benchmark.

The world is written with plain blocks: an occurrence process times a positive size process,
the structure of Croston's method and of TWEES (Damato et al., 2026):

    occurrence = tg.select(*[tg.noise(tg.bernoulli(p)) for p in RATES])     # demand or not
    size = (slow log level + spread * noise [+ weekly pattern]) >> tg.exp()  # positive sizes
    demand = occurrence * size

Setup:
- world: half of the series without and half with a weekly pattern on the log size; the
  occurrence probability p is one of RATES, equally likely; log level a random walk with
  step scale U(0, 0.1); size spread U(0.2, 1.2); weekly amplitude U(0, 0.5).
- two networks, contexts of 10 to 128 observations, 28 steps ahead, 50M series each:
  "quantile": quantiles TAUS, pinball loss on the relative scale;
  "mean": the forecast mean, squared loss on the relative scale.
  Normalized targets and losses are clipped at 50 (the default 10 would cut demand spikes).
- secondary: "quantile_worst", the quantile network fine-tuned 10M series with the smooth
  worst case over processes, tg.risk(over=tg.worst(PSI)), R = 32; and "quantile_ctrl", the same
  fine-tune with the mean (equal-budget control).
- forecasting: the last 128 observations; the first h outputs; forecasts below zero set to zero;
  quantiles sorted across levels.
- evaluation: data_eval.py, the benchmark's protocol (its EMPSD rows are reproduced exactly);
  RMSSE uses the "mean" network, sQS the quantile network, against the published table of
  all eleven methods.

    python experiments/intermittent/run.py train
    python experiments/intermittent/run.py evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import timegram as tg
from timegram import library

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_eval  # noqa: E402

RESULTS = HERE / "results"
NAME = "intermittent"
RATES = (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
TAUS = data_eval.TAUS
H, CONTEXT = 28, 128
PSI = 30.0
TRAIN = dict(lengths=(10, CONTEXT), budget=50_000_000, chunk=16384, backend="fast", log_every=2000, seed=0,
             target_clip=50.0, loss_clip=50.0)
TUNE = dict(TRAIN, budget=10_000_000, warmup=1_000_000, seed=1, replicates=32,
            optimizer=tg.MuonAdamW(lr=1e-4, muon_lr=1e-3))


def demand(weekly: bool):
    occurrence = tg.select(*[tg.noise(tg.bernoulli(p)) for p in RATES])
    log_size = tg.uniform(0, 0.1) * (tg.noise() >> tg.integrate()) + tg.uniform(0.2, 1.2) * tg.noise()
    if weekly:
        log_size = log_size + tg.uniform(0, 0.5) * tg.pattern(7)
    return occurrence * (log_size >> tg.exp())


def world():
    return tg.select(demand(False), demand(True))


def problem(kind, objective=None):
    if kind == "quantile":
        p = world().estimate(tg.forecast(h=H, quantiles=TAUS)).loss(tg.pinball(scale="relative"))
    else:
        p = world().estimate(tg.forecast(h=H)).loss(tg.squared(scale="relative"))
    return p if objective is None else p.objective(objective)


def train(names):
    for kind in ("quantile", "mean"):
        if kind in names:
            est = problem(kind).pretrain(**TRAIN)
            library.alias(f"{NAME}_{kind}", est.key)
            print(kind, est.key, flush=True)
    for name, obj in (("quantile_ctrl", tg.risk()), ("quantile_worst", tg.risk(over=tg.worst(PSI, anneal=0.5)))):
        if name in names:
            est = problem("quantile", obj).pretrain(**TUNE, init=library.resolve(f"{NAME}_quantile"))
            library.alias(f"{NAME}_{name}", est.key)
            print(name, est.key, flush=True)


def forecasts(est, histories, h):
    out = np.asarray(est.forecast([x[-CONTEXT:] for x in histories]))
    return np.clip(out[:, :h], 0, None)


def evaluate(quantile_names=("quantile", "quantile_ctrl", "quantile_worst")):
    RESULTS.mkdir(parents=True, exist_ok=True)
    mean_est = library.pretrained(f"{NAME}_mean").to("cuda")
    q_ests = {}
    for n in quantile_names:
        try:
            q_ests[n] = library.pretrained(f"{NAME}_{n}").to("cuda")
        except Exception as e:                                  # a variant not trained (yet)
            print(f"skipping {n}: {e}", flush=True)
    rows = []
    for ds in ("auto", "pasta", "raf", "tinym5"):
        series = data_eval.load(ds)
        cases = list(data_eval.windows(ds, series))
        hist = [c[2] for c in cases]
        h = data_eval.H[ds]
        means = forecasts(mean_est, hist, h)
        for qn, qe in q_ests.items():
            q = np.sort(_quantiles(qe, hist, h), axis=-1)
            per = []
            for k, (i, j, x, y) in enumerate(cases):
                per.append({"series": i, "window": j, **data_eval.scores(
                    x, y, means[k], {tau: q[k, :, m] for m, tau in enumerate(TAUS)})})
            s = data_eval.summarize(pd.DataFrame(per))
            rows.append({"dataset": ds, "model": qn, **s.to_dict()})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "timegram_scores.csv", index=False)
    print(t.round(3).to_string(index=False))


def _quantiles(est, histories, h):
    out = np.asarray(est.forecast([x[-CONTEXT:] for x in histories]))      # [n, H, Q]
    return np.clip(out[:, :h, :], 0, None)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2:] or ["quantile", "mean", "quantile_ctrl", "quantile_worst"])
    else:
        evaluate()
