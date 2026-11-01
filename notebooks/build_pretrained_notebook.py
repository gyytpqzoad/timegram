"""Build notebooks/pretrained_networks.ipynb: the trained networks of the experiments, ready to use.

    python notebooks/build_pretrained_notebook.py
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# timegram: the trained networks of the experiments

This notebook uses the networks that ship with the repository in `pretrained/`: no training is
needed, and everything runs on a CPU in a few minutes (a GPU is used when available).

1. one pretrained forecaster for yearly, quarterly and monthly data, on M1, M3 and Tourism;
2. forecasting your own series;
3. one-step quantiles calibrated for every series, on monthly Tourism;
4. probabilities over the order of an autoregression, for a short series."""),
    md("""## Setup

On Colab the first cell clones the repository (networks included); alternatively upload it as
`timegram.zip` and set `REPO_URL = ""`. Run locally, the notebook finds the repository it is in."""),
    code("""REPO_URL = "https://github.com/gyytpqzoad/timegram"

import os, subprocess, sys, zipfile
from pathlib import Path

def find_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "src" / "timegram").exists() and (p / "pretrained").exists():
            return p
    if Path("timegram/src/timegram").exists():
        return Path("timegram").resolve()
    if REPO_URL:
        subprocess.run(["git", "clone", "-q", REPO_URL, "timegram"], check=True)
        return Path("timegram").resolve()
    if Path("timegram.zip").exists():
        zipfile.ZipFile("timegram.zip").extractall(".")
        found = next(Path(".").glob("**/src/timegram")).parents[1]
        return found.resolve()
    raise FileNotFoundError("set REPO_URL or upload timegram.zip")

ROOT = find_root()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fcompdata"], check=True)
sys.path.insert(0, str(ROOT / "src"))
os.environ["TIMEGRAM_HOME"] = str(ROOT / "pretrained")     # the library of trained networks

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import timegram as tg
from timegram import library
from timegram.data import load, score, seasonal_naive

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("repository:", ROOT, "| device:", DEVICE)
import json
print("trained networks:", sorted(json.loads((ROOT / "pretrained" / "names.json").read_text())))"""),
    md("""## 1. One pretrained forecaster for M1, M3 and Tourism

`acc_r6_natural_forms` is the network trained on the world selected by the grammar-guided search:
one network for yearly, quarterly and monthly series, forecasting 24 steps from the last 128
observations. Each data set uses its own horizon and official test window; MASE is scaled by the
in-sample seasonal naive error. Printing the estimator shows the world it was trained on."""),
    code("""forecaster = library.pretrained("acc_r6_natural_forms").to(DEVICE)
print(forecaster.problem.describe()[:1500], "...")"""),
    code("""PAPER = {"M1-Y": 3.492, "M1-Q": 1.647, "M1-M": 1.030, "M3-Y": 2.583, "M3-Q": 1.085, "M3-M": 0.825,
         "M4-Y": 2.979, "M4-Q": 1.134, "M4-M": 0.908, "Tourism-Y": 2.693, "Tourism-Q": 1.512, "Tourism-M": 1.466}
RUN_M4 = False      # M4 has 100,000 series: a few more minutes

rows = []
for name in ["m1", "m3"] + (["m4"] if RUN_M4 else []) + ["tourism"]:
    for period in (1, 4, 12):
        ds = load(name, period)
        sc = score(ds, forecaster.forecast(ds.histories))
        sn = score(ds, seasonal_naive(ds.histories, period, ds.horizon))
        rows.append({"set": ds.name, "series": len(ds), "horizon": ds.horizon, "MASE": sc["mase"],
                     "paper": PAPER.get(ds.name), "seasonal naive": sn["mase"]})
pd.DataFrame(rows).round(3)"""),
    code("""ds = load("m3", 12)
fc = forecaster.forecast(ds.histories)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
for ax, i in zip(axes, (0, 500, 1000)):
    h, n = ds.histories[i], len(ds.histories[i])
    ax.plot(range(n), h, color="0.3", lw=1, label="history")
    ax.plot(range(n, n + ds.horizon), ds.futures[i], color="0.6", lw=1, ls="--", label="actual")
    ax.plot(range(n, n + ds.horizon), fc[i, :ds.horizon], color="tab:blue", lw=1.5, label="timegram")
    ax.set_title(f"M3 monthly, series {i}")
axes[0].legend(frameon=False)
plt.tight_layout()"""),
    md("""## 2. Forecasting your own series

An estimator applies to any series: give it an array or a pandas Series. With a `PeriodIndex`
the forecast is labelled by period."""),
    code("""from fcompdata import load_tourism
s = next(x for x in load_tourism() if x.period == 12)
y = pd.Series(np.asarray(s.x, float), index=pd.period_range("2000-01", periods=len(s.x), freq="M"))
result = forecaster(y)
result.forecast.head(12)"""),
    md("""## 3. Quantiles calibrated for every series (monthly Tourism)

Three networks share the same exponential-smoothing world and estimate the quantiles
0.1, ..., 0.9 of the next value: `pinball` (pinball loss only), `control` (the same fine-tuning
with pinball loss only) and `calibrated` (fine-tuned with the per-process miscoverage objective).
They are evaluated on the last 150 one-step origins of each Tourism monthly series; for each
series, the coverage of the 0.9 quantile is compared with 0.9, and "rejected" is the share of
series whose coverage an exact binomial test rejects at 5%. Positive series are forecast on the
log scale."""),
    code("""sys.path.insert(0, str(ROOT / "experiments" / "tourism_calibration"))
import backtest
from scipy.stats import binom

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

class LogForecaster:
    def __init__(self, variant):
        self.name = variant
        self.est = library.pretrained(f"tourcal_{variant}").to(DEVICE)
    def predict(self, contexts, horizon, quantiles):
        pos = np.array([bool(np.all(np.asarray(c) > 0)) for c in contexts])
        q = self.est.forecast([np.log(c) if p else np.asarray(c, float) for c, p in zip(contexts, pos)])[:, 0, :]
        q[pos] = np.exp(q[pos])
        return q[:, None, :]

series = backtest.load_tourism_monthly()
config = backtest.BacktestConfig(n_origins=150, quantiles=LEVELS)
models = [LogForecaster(v) for v in ("pinball", "control", "calibrated")]
per = backtest.summarize(backtest.run_backtest(series, models, config))
q90 = per[per["quantile"].round(2) == 0.9]

def rejected(cov, n=150, p=0.9):
    k = np.round(np.asarray(cov) * n).astype(int)
    return np.minimum(1, 2 * np.minimum(binom.cdf(k, n, p), binom.sf(k - 1, n, p))) < 0.05

PAPER_T = {"pinball": (0.027, 0.132), "control": (0.027, 0.127), "calibrated": (0.023, 0.061)}
pd.DataFrame([{"model": m, "series": len(d), "mean |cov - 0.9|": np.abs(d.coverage - 0.9).mean(),
               "rejected": rejected(d.coverage).mean(), "pinball (q0.9)": d.pinball_loss.mean(),
               "paper": PAPER_T[m]} for m, d in q90.groupby("model")]).round(3)"""),
    md("""## 4. Probabilities over the order of a short autoregression

`ar_order` returns probabilities that a series of 15 to 100 observations was generated by an
AR(1), ..., AR(5). On simulated series the true order is known; the Canadian lynx record (log10,
1821-1934) is the classic example of a population with delayed density dependence."""),
    code("""selector = library.pretrained("ar_order").to(DEVICE)

world = tg.select(*[tg.ar(p, phi=tg.stationary(p, last_min=.2)) for p in range(1, 6)])
sim = tg.simulate(world, n=6, lengths=(20, 20), seed=1)
true_order = np.asarray(sim.params["select.which"], int).ravel() + 1
probs = selector.predict(list(sim.context[:, -20:]))[:, :5]
table = pd.DataFrame(probs.round(3), columns=[f"AR({p})" for p in range(1, 6)])
table.insert(0, "true order", true_order)
table"""),
    code("""lynx = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv")["value"]
y = np.log10(lynx.to_numpy(float))[-100:]
p = selector.predict([y])[0, :5]
print("lynx:", {f"AR({k})": round(float(v), 3) for k, v in zip(range(1, 6), p)},
      "| probability of order two or more:", round(float(p[1:].sum()), 3))"""),
    md("""## More

Other networks in `pretrained/`: `acc_backbone` (the base world of the search),
`acc_r8_intermittent` (the selected world with intermittent demand added) and
`m1_mixhalf_minmax_50M_s0` (the additive exponential-smoothing world for M1). The scripts in
`experiments/` find them by the same names when `TIMEGRAM_HOME` points to `pretrained/`."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
})
out = Path(__file__).resolve().parent / "pretrained_networks.ipynb"
nbf.write(nb, out)
print("wrote", out)
