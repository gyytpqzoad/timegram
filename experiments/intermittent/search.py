"""A small world search for intermittent demand, selected on validation windows only.

- Validation: RAF (the two windows before the test windows, T = L - j h for j = 3, 4) and Car
  Parts (Hyndman et al. 2008; the series complete over all 51 months; j = 3).
- Score: the mean over datasets and the 6 metrics (RMSSE, sQS at 5 levels) of the ratio to the
  control's validation score; lower is better.
- Candidates are warm-started from run.py's networks (intermittent_quantile and intermittent_mean),
  20M series each, and each changes the world in one way:
  control      the unchanged world (equal-budget control)
  spikes       log sizes driven by mixture(normal(0, 1), normal(0, 3), weights [.95, .05])
  spread       size spread U(0.2, 1.6) instead of U(0.2, 1.2)
  sparse       occurrence rates add 0.03 and 0.05 (very sparse series)
  drift        log level step scale U(0, 0.3) instead of U(0, 0.1)
  persistent   half of the occurrence processes from a persistent latent process,
               (U(-1.5, 1.5) + U(0.3, 1) * AR(1) with phi ~ U(0.5, 0.95)) >> tg.indicator()
  weekly_occ   as persistent, and in the weekly half the latent process has a weekly pattern
  integer      sizes on a level exp(U(0, 3)), demand >> tg.counts(): integer counts
- Every candidate is also scored with its quantile forecasts rounded to integers.
- Combination: every move that beats the control on validation, in one world ("combined3").
- Car Parts against the Monash archive's published MASE: the selected world fine-tuned once
  more, 10M series, with .observe(tg.missing(0.1)); all 2,674 series, the last 12 months as
  test, MASE with the archive's scaling (monash_carparts.py), the median as point forecast;
  histories with gaps kept missing and zero-filled.

    python experiments/intermittent/search.py train [names]
    python experiments/intermittent/search.py combine3
    python experiments/intermittent/search.py validate
    python experiments/intermittent/search.py test --final
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
import run  # noqa: E402

RESULTS = HERE / "results" / "search"
VAL_SETS = ("raf", "carparts")
VAL_JS = {"raf": (3, 4), "carparts": (3,)}
TEST_SETS = ("auto", "raf", "carparts")
BASE = dict(rates=run.RATES, spread=(0.2, 1.2), drift=(0.0, 0.1), spikes=False, latent=False, weekly_occ=False,
            integer=False)
MOVES = {
    "control": {},
    "spikes": dict(spikes=True),
    "spread": dict(spread=(0.2, 1.6)),
    "sparse": dict(rates=(0.03, 0.05) + run.RATES),
    "drift": dict(drift=(0.0, 0.3)),
}
ROUND2 = {
    "persistent": dict(latent=True),
    "weekly_occ": dict(latent=True, weekly_occ=True),
    "integer": dict(integer=True),
}
TUNE = dict(run.TRAIN, budget=20_000_000, warmup=1_000_000, seed=2,
            optimizer=tg.MuonAdamW(lr=1e-4, muon_lr=1e-3))


def latent_occurrence(weekly):
    z = tg.constant(tg.uniform(-1.5, 1.5)) + tg.uniform(0.3, 1.0) * (tg.noise() >> tg.ar(1, phi=tg.uniform(0.5, 0.95)))
    if weekly:
        z = z + tg.uniform(0, 1.0) * tg.pattern(7)
    return z >> tg.indicator()


def demand(weekly, rates, spread, drift, spikes, latent, weekly_occ, integer):
    occurrence = tg.select(*[tg.noise(tg.bernoulli(p)) for p in rates])
    if latent:
        occurrence = tg.select(occurrence, latent_occurrence(weekly and weekly_occ))
    shock = tg.noise(tg.mixture(tg.normal(0, 1), tg.normal(0, 3), weights=[.95, .05])) if spikes else tg.noise()
    log_size = tg.uniform(*drift) * (tg.noise() >> tg.integrate()) + tg.uniform(*spread) * shock
    if weekly:
        log_size = log_size + tg.uniform(0, 0.5) * tg.pattern(7)
    if integer:
        return (occurrence * ((log_size + tg.constant(tg.uniform(0, 3))) >> tg.exp())) >> tg.counts()
    return occurrence * (log_size >> tg.exp())


def world(**opts):
    o = dict(BASE, **opts)
    return tg.select(demand(False, **o), demand(True, **o))


def problem(kind, opts):
    w = world(**opts)
    if kind == "quantile":
        return w.estimate(tg.forecast(h=run.H, quantiles=run.TAUS)).loss(tg.pinball(scale="relative"))
    return w.estimate(tg.forecast(h=run.H)).loss(tg.squared(scale="relative"))


def train(name, opts):
    for kind in ("quantile", "mean"):
        est = problem(kind, opts).pretrain(**TUNE, init=library.resolve(f"{run.NAME}_{kind}"))
        library.alias(f"{run.NAME}_s_{name}_{kind}", est.key)
        print(name, kind, est.key, flush=True)


def cases(ds, js):
    h = data_eval.H[ds]
    for i, y in enumerate(data_eval.load(ds)):
        for j in js:
            T = len(y) - j * h
            yield i, j, y[:T], y[T:T + h]


def score_sets(name, sets, js) -> pd.DataFrame:
    q_est = library.pretrained(f"{run.NAME}_s_{name}_quantile").to("cuda")
    m_est = library.pretrained(f"{run.NAME}_s_{name}_mean").to("cuda")
    rows = []
    for ds in sets:
        cs = list(cases(ds, VAL_JS[ds] if js is None else js))
        hist = [c[2] for c in cs]
        h = data_eval.H[ds]
        means = run.forecasts(m_est, hist, h)
        q = np.sort(run._quantiles(q_est, hist, h), axis=-1)
        for rounded in (False, True):
            qq = np.round(q) if rounded else q
            per = [{"series": i, "window": j, **data_eval.scores(x, y, means[k], {t: qq[k, :, m] for m, t in enumerate(run.TAUS)})}
                   for k, (i, j, x, y) in enumerate(cs)]
            s = data_eval.summarize(pd.DataFrame(per))
            rows.append({"candidate": name, "rounded": rounded, "dataset": ds, **s.to_dict()})
    return pd.DataFrame(rows)


def relative(t: pd.DataFrame) -> pd.Series:
    cols = ["RMSSE"] + [f"sQS{tau:g}" for tau in run.TAUS]
    ref = t[(t.candidate == "control") & ~t.rounded].set_index("dataset")[cols]
    return t.groupby(["candidate", "rounded"]).apply(
        lambda g: float((g.set_index("dataset")[cols] / ref.loc[g.dataset.values].values).to_numpy().mean()))


def trained():
    out = []
    for name in list(MOVES) + list(ROUND2) + ["combined", "combined3"]:
        if all((library.home() / library.resolve(f"{run.NAME}_s_{name}_{k}") / "estimator.pt").exists()
               for k in ("quantile", "mean")):
            out.append(name)
    return out


def validate():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t = pd.concat([score_sets(n, VAL_SETS, None) for n in trained()])
    t.to_csv(RESULTS / "validation.csv", index=False)
    rel = relative(t).sort_values()
    rel.to_csv(RESULTS / "validation_relative.csv")
    print(rel.round(4).to_string(), flush=True)
    return rel


def combine():
    rel = pd.read_csv(RESULTS / "validation_relative.csv").set_index(["candidate", "rounded"]).iloc[:, 0]
    raw = {c: rel[(c, False)] for c in MOVES if c != "control"}
    good = [c for c, v in raw.items() if v < rel[("control", False)]]
    print("moves that beat the control:", good, flush=True)
    (RESULTS / "combined_moves.json").write_text(json.dumps(good))
    if len(good) >= 2:
        opts = {}
        for c in good:
            opts.update(MOVES[c])
        train("combined", opts)


def test():
    rel = validate()
    choice = rel.index[0]
    (RESULTS / "choice.json").write_text(json.dumps({"candidate": choice[0], "rounded": bool(choice[1])}))
    print("validation choice:", choice, flush=True)
    t = pd.concat([score_sets(n, TEST_SETS, (1, 2)) for n in trained()])
    t.to_csv(RESULTS / "test_all.csv", index=False)
    print(t.round(3).to_string(index=False))


def final_carparts():
    """The selected world, fine-tuned with missing values, on Car Parts against the Monash archive."""
    import monash_carparts as mc
    choice = json.loads((RESULTS / "choice.json").read_text())
    name, rounded = choice["candidate"], choice["rounded"]
    moves = {**MOVES, **ROUND2}
    if name.startswith("combined"):
        opts = {}
        for c in json.loads((RESULTS / f"{name}_moves.json").read_text()):
            opts.update(moves[c])
    else:
        opts = moves[name]
    key = f"{run.NAME}_s_final_missing"
    if not (library.home() / library.resolve(key) / "estimator.pt").exists():
        w = world(**opts).observe(tg.missing(0.1))
        prob = w.estimate(tg.forecast(h=run.H, quantiles=run.TAUS)).loss(tg.pinball(scale="relative"))
        est = prob.pretrain(**dict(TUNE, budget=10_000_000, seed=3),
                            init=library.resolve(f"{run.NAME}_s_{name}_quantile"))
        library.alias(key, est.key)
    est = library.pretrained(key).to("cuda")
    filled, gaps = mc.load()
    out = {"candidate": name, "rounded": rounded}
    for label, hist in (("gaps as missing", [g[:-mc.H] for g in gaps]), ("zero-filled", [f[:-mc.H] for f in filled])):
        q = np.asarray(est.forecast([x[-run.CONTEXT:] for x in hist]))[:, :mc.H, :]
        med = np.clip(np.sort(q, axis=-1)[:, :, 0], 0, None)
        med = np.round(med) if rounded else med
        out[label] = mc.mase(filled, list(med))[0]
    (RESULTS / "carparts_monash.json").write_text(json.dumps(out, indent=1))
    print(out, flush=True)


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "train":
        for n in names or MOVES:
            train(n, {**MOVES, **ROUND2}[n])
    elif cmd == "combine3":
        rel = validate()
        ctrl = rel[("control", False)]
        good = [c for c in list(MOVES)[1:] + list(ROUND2) if (c, False) in rel and rel[(c, False)] < ctrl]
        print("round 3 combines:", good, flush=True)
        (RESULTS / "combined3_moves.json").write_text(json.dumps(good))
        opts = {}
        for c in good:
            opts.update({**MOVES, **ROUND2}[c])
        if len(good) >= 2:
            train("combined3", opts)
    elif cmd == "validate":
        validate()
    elif cmd == "combine":
        combine()
    elif "--final" in names:
        test()
        final_carparts()
    else:
        test()
