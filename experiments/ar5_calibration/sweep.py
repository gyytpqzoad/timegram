"""Can the calibration penalty beat pinball-only training per process? A targeted sweep.

The first AR(5) run (run.py) fine-tuned at the full learning rate, penalized all three
levels with a mean over processes and 32 replicates, and lost to pinball. Here every
variant fine-tunes the same pinball base (ar5cal_unif) for 20M series at a tenth of the
learning rate with a warmup, penalizes the tails only (q0.05, q0.95), and varies one
thing at a time:

    ctrl           pinball only (the control for the fine-tuning itself)
    mean_w<w>      mean over processes of the debiased miscoverage, weight w
    mean_w<w>_R128   ... with 128 replicates per process (32 processes per batch)
    worst_w<w>_R128  ... smooth worst case over processes (log-sum-exp, psi 1000)
    plug_w<w>      the plug-in penalty
    mean_w<w>_k200 a sharper surrogate indicator (kappa 200)

    python experiments/ar5_calibration/sweep.py train            # all variants
    python experiments/ar5_calibration/sweep.py evaluate         # with the pinball base
"""
from __future__ import annotations

import sys

import timegram as tg
from timegram import library

import run

TAILS = [0.05, 0.95]
TUNE = dict(run.BASE, budget=20_000_000, warmup=1_000_000, seed=3,
            optimizer=tg.MuonAdamW(lr=1e-4, muon_lr=1e-3))


def penalty(w, **kw):
    kw.setdefault("smoothing", "fixed")              # run before spread smoothing became the default
    return (tg.risk(weight=1 - w), tg.miscoverage(weight=w, levels=TAILS, **kw))


VARIANTS = {
    "ctrl": ((tg.risk(),), 32),
    "mean_w0.9": (penalty(0.9), 32),
    "mean_w0.97": (penalty(0.97), 32),
    "mean_w0.9_R128": (penalty(0.9), 128),
    "worst_w0.9_R128": (penalty(0.9, over=tg.worst(psi=1000)), 128),
    "worst_w0.97_R128": (penalty(0.97, over=tg.worst(psi=1000)), 128),
    "plug_w0.9": (penalty(0.9, debiased=False), 32),
    "mean_w0.9_k200": (penalty(0.9, kappa=200), 32),
}


def train(name):
    criteria, R = VARIANTS[name]
    est = run.problem(list(criteria)).pretrain(**TUNE, replicates=R, init=library.resolve(f"{run.NAME}_unif"))
    library.alias(f"{run.NAME}_sw_{name}", est.key)
    print(f"{name}: {est.key}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        for name in sys.argv[2:] or VARIANTS:
            train(name)
    else:
        run.RESULTS = run.HERE / "results" / "sweep"
        run.evaluate(["unif"] + [f"sw_{n}" for n in VARIANTS])
