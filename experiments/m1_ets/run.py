"""M1 with the additive ETS family (Simulacrum paper, Section "Real Data: Forecasting
Results of Neural Exponential Smoothing"): one network for yearly, quarterly and
monthly M1, trained only on simulated ETS series.

    python run.py train <variant> [--budget N] [--seed S]
    python run.py report

Selection protocol: variants are compared on a VALIDATION split (each series' last h
training observations, forecast from the rest); the M1 test windows are scored for every
run.

Metrics (M1 style, APE = 100 |y - f| / |y|):
  MAPE all        mean APE over every (series, horizon) pair (Hyndman et al. 2002, Table 3)
  MAPE trim       the same after dropping the 3 series with the largest mean APE (paper: n = 998)
  median APE @h   median over series at horizon h, h in 1..6, 8, 12, 15, 18 (M1 table)
  mean APE @h     same, mean (trimmed)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import timegram as tg

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
H = 18
HORIZONS = (1, 2, 3, 4, 5, 6, 8, 12, 15, 18)


# ---------------------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------------------
def paperlike(spec, period=None, **kw):
    """The Simulacrum paper's sampler: alpha, beta, gamma ~ U(0, 1) independently (no textbook box, no
    admissibility check), starting states in +-10."""
    from timegram.blocks import Centered
    return tg.ets(spec, period=period, alpha=tg.uniform(0, 1), beta=tg.uniform(0, 1), gamma=tg.uniform(0, 1),
                  phi=tg.uniform(0.8, 0.98), level0=tg.uniform(-10, 10), slope0=tg.uniform(-10, 10),
                  season0=Centered(tg.uniform(-10, 10), period) if period else None, **kw)


def ets_family(nonseasonal=("ANN", "AAN", "AAdN"), seasonal=("ANA", "AAA", "AAdA"), periods=(4, 12),
               weights=None, make=tg.ets):
    """Selection over ETS variants. weights: (non-seasonal group, period-4 group,
    period-12 group) shares; each group splits its share equally among its variants."""
    groups = [[make(s, min_length=9) for s in nonseasonal]]
    for m in periods:
        groups.append([make(s, period=m) for s in seasonal])
    w = weights or [1.0] * len(groups)
    ops, probs = [], []
    for g, wg in zip(groups, w):
        ops += g
        probs += [wg / len(g)] * len(g)
    return tg.select(*ops, probs=probs)


def mixed_pair(kind):
    """Two independent draws from the nine-model family, each normalized, convex-combined
    with w ~ U(0, 1)."""
    return tg.mix(ets_family() >> tg.normalize(kind), ets_family() >> tg.normalize(kind))


def paper_mix(kind="minmax"):
    """The Simulacrum paper's mixed world: SES, Holt, Holt-Winters 4 and 12, each normalized, combined
    with weights uniform on the simplex."""
    ops = [tg.ets("ANN", min_length=9), tg.ets("AAN", min_length=9), tg.ets("AAA", period=4),
           tg.ets("AAA", period=12)]
    return tg.mix(*[o >> tg.normalize(kind) for o in ops])


VARIANTS = {
    # the Simulacrum paper's world: Holt-Winters at periods 1 (Holt), 4 and 12, equally likely
    "paper3": dict(world=lambda: ets_family(("AAN",), ("AAA",)), loss="squared"),
    "paper3_l1": dict(world=lambda: ets_family(("AAN",), ("AAA",))),
    # every additive variant, groups equally likely
    "full9": dict(world=lambda: ets_family()),
    "full9_mse": dict(world=lambda: ets_family(), loss="squared"),
    # half the series non-seasonal
    "full9_ns": dict(world=lambda: ets_family(weights=[2, 1, 1])),
    # no damped trend
    "full6": dict(world=lambda: ets_family(("ANN", "AAN"), ("ANA", "AAA"))),
    # longer window (monthly M1 histories reach 132)
    "full9_L128": dict(world=lambda: ets_family(), lengths=(9, 128)),
    "full9_ns_L128": dict(world=lambda: ets_family(weights=[2, 1, 1]), lengths=(9, 128)),
    "full9_L128_mse": dict(world=lambda: ets_family(), lengths=(9, 128), loss="squared"),
    "full9_L128_ema": dict(world=lambda: ets_family(), lengths=(9, 128), ema=0.999),
    # regularization of the single network
    "full9_L128_obs02": dict(world=lambda: ets_family(), lengths=(9, 128), missing=0.2),
    "full9_L128_bdrop": dict(world=lambda: ets_family(), lengths=(9, 128), net=tg.ResMLP(dropout=0.1)),
    # mixed worlds: normalized operands, convex combination
    "mix4_minmax": dict(world=lambda: paper_mix("minmax"), lengths=(9, 128)),
    "mix2_minmax": dict(world=lambda: mixed_pair("minmax"), lengths=(9, 128)),
    "mix2_std": dict(world=lambda: mixed_pair("std"), lengths=(9, 128)),
    "mixhalf_minmax_obs02": dict(world=lambda: tg.select(ets_family(), mixed_pair("minmax"), probs=[0.5, 0.5]),
                                 lengths=(9, 128), missing=0.2),
    "mixhalf_minmax_bdrop": dict(world=lambda: tg.select(ets_family(), mixed_pair("minmax"), probs=[0.5, 0.5]),
                                 lengths=(9, 128), net=tg.ResMLP(dropout=0.1)),
    "mixhalf_minmax": dict(world=lambda: tg.select(ets_family(), mixed_pair("minmax"), probs=[0.5, 0.5]), lengths=(9, 128)),
    "mixhalf_std": dict(world=lambda: tg.select(ets_family(), mixed_pair("std"), probs=[0.5, 0.5]), lengths=(9, 128)),
    # capacity
    "full9_w768": dict(world=lambda: ets_family(), net=tg.ResMLP(width=768, depth=4)),
    "full9_d8": dict(world=lambda: ets_family(), net=tg.ResMLP(width=384, depth=8)),
    # the Simulacrum paper's sampler (no admissibility, U(0,1) parameters, states in +-10)
    "paperlike3": dict(world=lambda: ets_family(("AAN",), ("AAA",), make=paperlike)),
    # the Simulacrum paper's model set: SES, Holt, Holt-Winters at period 4 or 12, one drawn per series
    "paper4": dict(world=lambda: tg.select(tg.ets("ANN", min_length=9), tg.ets("AAN", min_length=9),
                                           tg.ets("AAA", period=4), tg.ets("AAA", period=12))),
    "paperlike4": dict(world=lambda: tg.select(paperlike("ANN", min_length=9), paperlike("AAN", min_length=9),
                                               paperlike("AAA", 4), paperlike("AAA", 12))),
    "paperlike9": dict(world=lambda: ets_family(make=paperlike)),
    # training setup
    "full9_b4096": dict(world=lambda: ets_family(), batch_size=4096),
    "full9_lr2x": dict(world=lambda: ets_family(), optimizer=tg.MuonAdamW(lr=2e-3, muon_lr=0.02)),
    "full9_lrhalf": dict(world=lambda: ets_family(), optimizer=tg.MuonAdamW(lr=5e-4, muon_lr=0.005)),
}


def problem(v):
    loss = tg.squared(scale="relative") if v.get("loss") == "squared" else tg.absolute(scale="relative")
    return v["world"]().observe(tg.missing(rate=v.get("missing", 0.1))).estimate(tg.forecast(h=H)).loss(loss)


# ---------------------------------------------------------------------------------------
# data and metrics
# ---------------------------------------------------------------------------------------
def m1(split="test"):
    """(histories, futures (NaN-padded to H), horizons, periods)."""
    from fcompdata import load_m1
    hist, fut, hs, ps = [], [], [], []
    for s in load_m1():
        x, xx, h = np.asarray(s.x, float), np.asarray(s.xx, float), len(s.xx)
        if split == "val":
            if len(x) - h < 6:
                continue
            x, xx = x[:-h], x[-h:]
        f = np.full(H, np.nan)
        f[:h] = xx
        hist.append(x), fut.append(f), hs.append(h), ps.append(s.period)
    return hist, np.stack(fut), np.array(hs), np.array(ps)


def metrics(fc, fut, ps):
    ape = 100 * np.abs(fut - fc) / np.abs(fut)                     # NaN beyond each series' horizon
    per_series = np.nanmean(ape, 1)
    keep = np.argsort(per_series)[:-3]                             # drop the 3 worst series
    out = {"MAPE_all": float(np.nanmean(ape)), "MAPE_trim": float(np.nanmean(ape[keep])),
           "n_trim": int(len(keep)),
           "median_ape": {h: float(np.nanmedian(ape[:, h - 1])) for h in HORIZONS},
           "mean_ape_trim": {h: float(np.nanmean(ape[keep, h - 1])) for h in HORIZONS},
           "sMAPE": float(np.nanmean(200 * np.abs(fut - fc) / (np.abs(fut) + np.abs(fc))))}
    for p, name in ((1, "Y"), (4, "Q"), (12, "M")):
        out[f"MAPE_{name}"] = float(np.nanmean(ape[ps == p]))
    out["median_ape_mean"] = float(np.mean(list(out["median_ape"].values())))
    return out


def evaluate(est, split):
    hist, fut, _, ps = m1(split)
    res = {}
    for mode in ("plain", "log"):
        if mode == "log":
            fc = np.exp(est.forecast([np.log(h) for h in hist]))     # M1 is strictly positive
        else:
            fc = est.forecast(hist)
        res[mode] = metrics(fc[:, :H], fut, ps)
    return res


# ---------------------------------------------------------------------------------------
def train(name, budget, seed):
    v = VARIANTS[name]
    t0 = time.time()
    est = problem(v).pretrain(lengths=v.get("lengths", (9, 64)), budget=budget,
                              batch_size=v.get("batch_size", 2048), chunk=16384,
                              net=v.get("net", tg.ResMLP()), optimizer=v.get("optimizer", tg.MuonAdamW()),
                              ema=v.get("ema", 0.0), backend="fast", seed=seed, log_every=1000)
    tag = f"{name}_{budget // 1_000_000}M_s{seed}"
    tg.library.alias(f"m1_{tag}", est.key)
    est.to("cuda")
    rec = {"variant": name, "budget": budget, "seed": seed, "key": est.key,
           "minutes": round((time.time() - t0) / 60, 1),
           "val": evaluate(est, "val"), "test": evaluate(est, "test"), "history": est.history}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    v_, t_ = rec["val"]["plain"], rec["test"]["plain"]
    print(f"{tag}: val MAPE_trim {v_['MAPE_trim']:.2f} med {v_['median_ape_mean']:.2f} | "
          f"test MAPE_trim {t_['MAPE_trim']:.2f} med {t_['median_ape_mean']:.2f}  ({rec['minutes']} min)")


def ensemble(tags: list[str], name: str):
    """Average the forecasts of several trained runs (library aliases m1_<tag>), in both
    modes; scored like a run and saved as results/<name>.json."""
    ests = [tg.pretrained(f"m1_{t}").to("cuda") for t in tags]
    rec = {"variant": "ensemble", "members": tags, "val": {}, "test": {}}
    for split in ("val", "test"):
        hist, fut, _, ps = m1(split)
        for mode in ("plain", "log"):
            if mode == "log":
                fc = np.mean([np.exp(e.forecast([np.log(h) for h in hist])) for e in ests], 0)
            else:
                fc = np.mean([e.forecast(hist) for e in ests], 0)
            rec[split][mode] = metrics(fc[:, :H], fut, ps)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    v, t = rec["val"]["log"], rec["test"]["log"]
    print(f"{name} (log): val MAPE_trim {v['MAPE_trim']:.2f} med {v['median_ape_mean']:.2f} | "
          f"test MAPE_trim {t['MAPE_trim']:.2f} med {t['median_ape_mean']:.2f}")


def report(split="val"):
    rows = []
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        for mode in ("plain", "log"):
            m = r[split][mode]
            rows.append((f.stem, mode, m["MAPE_trim"], m["MAPE_all"], m["median_ape_mean"], m["MAPE_Y"],
                         m["MAPE_Q"], m["MAPE_M"], m["sMAPE"]))
    print(f"[{split}] {'run':28s} {'mode':5s} {'MAPEtrim':>8s} {'MAPEall':>8s} {'medAPE':>7s} "
          f"{'Y':>6s} {'Q':>6s} {'M':>6s} {'sMAPE':>6s}")
    for r in sorted(rows, key=lambda r: r[2]):
        print(f"        {r[0]:28s} {r[1]:5s} " + " ".join(f"{x:8.2f}" if i < 3 else f"{x:6.2f}"
                                                         for i, x in enumerate(r[2:])))


PAPER_MEDIAN = {  # Simulacrum paper, Table "Median APE: all M1 Competition data (1001)"
    "Neural-ETS (paper)": [4.3, 5.1, 5.6, 6.9, 7.9, 8.9, 9.0, 9.1, 10.8, 11.9],
    "COMBINING A": [4.2, 5.4, 5.9, 7.1, 8.0, 9.7, 9.3, 9.5, 11.8, 12.6],
    "WINTERS": [4.4, 5.4, 5.5, 6.6, 7.9, 9.6, 9.7, 10.1, 11.0, 12.2],
    "Chronos-2": [5.2, 6.5, 6.6, 8.5, 9.7, 10.9, 9.6, 10.1, 10.9, 12.2],
}
PAPER_MEAN = {"Neural-ETS (paper, n=998)": [8.4, 11.1, 12.3, 13.6, 16.8, 19.1, 18.7, 16.9, 21.4, 25.8]}


def compare(run: str, mode: str = "plain"):
    """Per-horizon median and mean APE of a run's TEST forecasts against the Simulacrum paper."""
    r = json.loads((OUT / f"{run}.json").read_text())["test"][mode]
    head = "".join(f"{h:>6d}" for h in HORIZONS) + "   mean"
    print(f"median APE        {head}")
    rows = {**PAPER_MEDIAN, f"{run} ({mode})": [r["median_ape"][str(h)] if str(h) in r["median_ape"]
                                                 else r["median_ape"][h] for h in HORIZONS]}
    for k, v in rows.items():
        print(f"{k[:18]:18s}" + "".join(f"{x:6.1f}" for x in v) + f"{np.mean(v):7.2f}")
    print()
    print(f"mean APE (trimmed){head}")
    rows = {**PAPER_MEAN, f"{run} ({mode})": [r["mean_ape_trim"][str(h)] if str(h) in r["mean_ape_trim"]
                                              else r["mean_ape_trim"][h] for h in HORIZONS]}
    for k, v in rows.items():
        print(f"{k[:18]:18s}" + "".join(f"{x:6.1f}" for x in v) + f"{np.mean(v):7.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("variant", nargs="?")
    ap.add_argument("--budget", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.variant, a.budget, a.seed)
    elif a.cmd == "ensemble":
        ensemble(a.variant.split(","), a.split)
    elif a.cmd == "compare":
        compare(a.variant, a.split if a.split in ("plain", "log") else "plain")
    else:
        report(a.split)
