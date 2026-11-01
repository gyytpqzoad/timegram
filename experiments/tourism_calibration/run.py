"""Per-series calibration of one-step quantiles on Tourism monthly.

World: ETS(A,A,A) or ETS(A,N,A) with equal probability, period 12, admissible smoothing
parameters, level0 ~ U(-10, 10), slope0 ~ U(-1, 1), season0 a zero-sum U(-1, 1) pattern times
an amplitude ~ U(0, 6), N(0, 1) innovations. Observed lengths 24..96; the estimand is the
quantiles 0.1, ..., 0.9 of y_{T+1}, with pinball loss on the normalized scale.

    pinball      pinball only, from scratch (82M series)
    control      pinball fine-tuned for 20M series at a tenth of the learning rate, pinball only
    calibrated   the same fine-tune with tg.risk(weight=0.1) + tg.miscoverage(weight=0.9) of
                 q0.9 (debiased, spread smoothing), 128 replicates per process

    python experiments/tourism_calibration/run.py train pinball control calibrated
    python experiments/tourism_calibration/run.py own pinball calibrated       # coverage by length
    python experiments/tourism_calibration/run.py tourism val pinball:log control:log calibrated:log autoets chronos2
    python experiments/tourism_calibration/run.py tourism test pinball:log control:log calibrated:log autoets chronos2
    python experiments/tourism_calibration/run.py report

Tourism: rolling one-step origins on the full monthly series (test: the last 150 origins;
validation: the 60 origins before them). ":log" forecasts log y for positive series and
exponentiates the quantiles.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binom

import timegram as tg
from timegram import library
from timegram.blocks import Centered

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
P, K9 = 0.9, 8                                   # the level under study and its column
TEST_ORIGINS, VAL_ORIGINS = 150, 60
NAME = "tourcal"                                  # library aliases tourcal_<variant>

TRAIN = dict(lengths=(24, 96), budget=20_000 * 4096, batch_size=4096, replicates=32, chunk=16384,
             warmup=1000 * 4096, net=tg.ResMLP(width=256, depth=4), optimizer=tg.AdamW(lr=1e-3),
             backend="fast", log_every=1000)
TUNE = dict(TRAIN, budget=20_000_000, warmup=1_000_000, seed=3, optimizer=tg.AdamW(lr=1e-4))
FINE_TUNED = {   # name: (criteria, replicates per process)
    "control": ((tg.risk(),), 32),
    "calibrated": ((tg.risk(weight=0.1), tg.miscoverage(weight=0.9, levels=[P], smoothing="spread")), 128),
}


def world():
    def ets(spec):
        season0 = Centered(tg.uniform(-1, 1), 12) * tg.uniform(0, 6)
        trend = {"slope0": tg.uniform(-1, 1)} if spec == "AAA" else {}
        return tg.ets(spec, period=12, level0=tg.uniform(-10, 10), season0=season0, **trend)
    return tg.select(ets("AAA"), ets("ANA"))


def problem(*criteria):
    p = (world().observe()
         .estimate(tg.forecast(h=1, quantiles=LEVELS))
         .loss(tg.pinball(scale="relative")))
    return p.objective(*criteria) if criteria else p


def train(name):
    if name in FINE_TUNED:
        criteria, R = FINE_TUNED[name]
        est = problem(*criteria).pretrain(**dict(TUNE, replicates=R, init=library.resolve(f"{NAME}_pinball")))
    else:
        est = problem().pretrain(**TRAIN)
    library.alias(f"{NAME}_{name}", est.key)
    print(f"{name}: {est.key}", flush=True)


class Forecaster:
    """A trained timegram estimator as a backtest forecaster (one-step quantiles)."""

    def __init__(self, spec: str):
        variant, _, mode = spec.partition(":")
        self.log = mode == "log"
        self.name = f"timegram-{variant}" + ("-log" if self.log else "")
        self.estimator = library.pretrained(f"{NAME}_{variant}").to("cuda")

    def predict(self, contexts, horizon, quantiles):
        assert horizon == 1 and np.allclose(quantiles, LEVELS)
        use_log = np.array([self.log and bool(np.all(np.asarray(c) > 0)) for c in contexts])
        series = [np.log(c) if u else np.asarray(c, float) for c, u in zip(contexts, use_log)]
        q = self.estimator.forecast(series)[:, 0, :]
        q[use_log] = np.exp(q[use_log])
        return q[:, None, :]


def model(spec: str):
    if spec == "autoets":
        return backtest.AutoETS()
    if spec == "chronos2":
        return backtest.Chronos2()
    return Forecaster(spec)


# ---------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------
def rejected(cov, n: int) -> np.ndarray:
    """Exact two-sided binomial test of coverage 0.9 at nominal 5%."""
    k = np.round(np.asarray(cov) * n).astype(int)
    return np.minimum(1, 2 * np.minimum(binom.cdf(k, n, P), binom.sf(k - 1, n, P))) < 0.05


def own_world(variants, ns=(36, 48, 72, 96), J=1000, R=400):
    """Per-process coverage of q0.9 on the training world at fixed lengths (J processes x R
    replicates each)."""
    forecasters = {v: Forecaster(v) for v in variants}
    out = RESULTS / "own"
    out.mkdir(parents=True, exist_ok=True)
    for n in ns:
        gen = torch.Generator(device="cuda").manual_seed(20_000 + n)
        rows = {v: [] for v in variants}
        per_call = 25
        for _ in range(0, J, per_call):
            sim = tg.simulate(world(), n=per_call, replicates=R, lengths=(n, n), horizon=1, backend="fast",
                              device="cuda", gen=gen, track_params=False)
            contexts = [row[-n:] for row in sim.context]
            z = sim.target[:, 0]
            for v, f in forecasters.items():
                q90 = f.predict(contexts, 1, LEVELS)[:, 0, K9]
                rows[v].append((z <= q90).reshape(per_call, R).mean(1))
        for v, r in rows.items():
            cov = np.concatenate(r)
            pd.DataFrame({"n": n, "cov90": cov}).to_csv(out / f"n{n}_{v}.csv", index=False)
            print(f"n={n:3d} {v:12s} mean cov {cov.mean():.4f}  |cov-0.9| {np.abs(cov - P).mean():.4f}  "
                  f"rejected {rejected(cov, R).mean():.3f}", flush=True)


def tourism(split, specs):
    full = backtest.load_tourism_monthly()
    if split == "val":
        series = {k: v[:-TEST_ORIGINS] for k, v in full.items() if len(v) >= 48 + VAL_ORIGINS + TEST_ORIGINS}
        origins = VAL_ORIGINS
    else:
        series, origins = full, TEST_ORIGINS
    config = backtest.BacktestConfig(n_origins=origins, quantiles=LEVELS)
    per = backtest.summarize(backtest.run_backtest(series, [model(s) for s in specs], config))
    out = RESULTS / "tourism"
    out.mkdir(parents=True, exist_ok=True)
    for name, d in per[per["quantile"].round(2) == P].groupby("model"):
        d[["series_id", "coverage", "n", "pinball_loss"]].to_csv(out / f"{split}_{name}.csv", index=False)
    print(f"tourism {split} done", flush=True)


def paired(x, y, seed=0):
    """Mean of |x - 0.9| - |y - 0.9| over matched series, with a bootstrap 95% interval."""
    d = np.abs(np.asarray(x) - P) - np.abs(np.asarray(y) - P)
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return d.mean(), *np.percentile(boot, [2.5, 97.5])


def report():
    pd.set_option("display.width", 160)
    for split, n in (("val", VAL_ORIGINS), ("test", TEST_ORIGINS)):
        runs = {f.stem.split("_", 1)[1]: pd.read_csv(f).sort_values("series_id")
                for f in sorted((RESULTS / "tourism").glob(f"{split}_*.csv"))}
        if not runs:
            continue
        rows = [{"model": m, "series": len(d), "mean cov": d.coverage.mean(),
                 "mean |cov-0.9|": np.abs(d.coverage - P).mean(), "rejected": rejected(d.coverage, n).mean(),
                 "pinball 0.9": d.pinball_loss.mean()} for m, d in runs.items()]
        print(f"\n== Tourism {split}: per-series coverage of q0.9 over {n} one-step origins")
        print(pd.DataFrame(rows).set_index("model").sort_values("mean |cov-0.9|").round(4).to_string())
        calibrated = runs.get("timegram-calibrated-log")
        for other in ("timegram-pinball-log", "Chronos-2"):
            if calibrated is not None and other in runs:
                m, lo, hi = paired(calibrated.coverage, runs[other].coverage)
                print(f"calibrated - {other}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    command, args = sys.argv[1], sys.argv[2:]
    if command == "train":
        for v in args:
            train(v)
    elif command == "own":
        own_world(args)
    elif command == "tourism":
        tourism(args[0], args[1:])
    else:
        report()
