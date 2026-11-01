"""Example 3 applied: how much memory do animal populations have?

Delayed density dependence, a population's growth depending on its abundance two or more years
back, appears as an autoregression of order two or more in log abundance (Turchin 1990; Royama
1992). Ecologists decide the order from short annual records, often with information criteria
(Brook and Bradshaw 2006). This script applies the calibrated order-selection estimator of
run.py (trained on AR(1)-AR(5) worlds) to the Canadian lynx series and to annual series from the
Global Population Dynamics Database (GPDD, via the rOpenSci rgpdd package, CC0), next to AIC and
BIC weights on the same series.

    python experiments/ar_order/ecology.py

Data are downloaded on first use into experiments/ar_order/data/ (not versioned).
"""
from __future__ import annotations

import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from timegram import library

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
RGPDD = "https://raw.githubusercontent.com/ropensci/rgpdd/master/data/{}.rda"
LYNX = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv"
MIN_N, MAX_N = 15, 100                         # the lengths the estimator was trained on


def fetch(url, path):
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def gpdd_tables():
    """GPDD data and metadata tables (rgpdd .rda files, read with the `rdata` package)."""
    out = {}
    for name in ("gpdd_data", "gpdd_main", "gpdd_taxon"):
        csv = DATA / f"{name}.csv"
        if not csv.exists():                      # first use: convert the R data file
            import rdata
            rda = fetch(RGPDD.format(name), DATA / f"{name}.rda")
            df = pd.DataFrame(list(rdata.read_rda(str(rda), default_encoding="utf-8").values())[0])
            df.to_csv(csv, index=False)
        out[name] = pd.read_csv(csv)
    return out


def gpdd_series():
    """Annual count series with reliability >= 2, consecutive years, at most 10% zeros, and at
    least MIN_N points; log(N + 1), the last MAX_N years."""
    t = gpdd_tables()
    main = t["gpdd_main"]
    main = main[(main.SamplingFrequency.astype(str) == "1") & (main.Reliability >= 2)]
    taxa = t["gpdd_taxon"].set_index("TaxonID")
    data = t["gpdd_data"]
    data = data[data.MainID.isin(main.MainID)].sort_values(["MainID", "SampleYear"])
    out = []
    for mid, g in data.groupby("MainID"):
        years = g.SampleYear.to_numpy()
        n = g.Population.to_numpy(float)
        if len(n) < MIN_N or np.any(np.diff(years) != 1) or np.isnan(n).any() or (n < 0).any():
            continue
        if np.mean(n == 0) > 0.1:
            continue
        x = np.log(n + 1.0)[-MAX_N:]
        if np.std(x) < 1e-8:
            continue
        row = main[main.MainID == mid].iloc[0]
        tax = taxa.loc[row.TaxonID] if row.TaxonID in taxa.index else None
        out.append({"MainID": int(mid), "series": x, "n": len(x),
                    "taxon": None if tax is None else tax.TaxonName,
                    "class": None if tax is None else tax.TaxonomicClass})
    return out


def lynx():
    y = pd.read_csv(fetch(LYNX, DATA / "lynx.csv"))["value"].to_numpy(float)
    return np.log10(y)[-MAX_N:]


def probabilities(series):
    """Order probabilities from the pretrained estimator and from AIC/BIC weights (on
    demeaned series: the criteria in run.py fit autoregressions without an intercept)."""
    est = library.pretrained(run.NAME)
    g = np.asarray(est.predict([np.asarray(s, float) for s in series]))[:, :len(run.ORDERS)]
    out = {"timegram": g, "AIC": [], "BIC": []}
    for s in series:
        z = (np.asarray(s, float) - np.mean(s))[None, :]
        ic = run.information_criteria(z)
        out["AIC"].append(run.weights(ic["AIC"])[0])
        out["BIC"].append(run.weights(ic["BIC"])[0])
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    warnings.filterwarnings("ignore")
    RESULTS.mkdir(parents=True, exist_ok=True)
    L = lynx()
    pl = probabilities([L])
    print("Canadian lynx (log10, last 100 years): order probabilities")
    for k, v in pl.items():
        print(f"  {k:8s} " + "  ".join(f"AR({p}) {v[0, p - 1]:.2f}" for p in run.ORDERS))
    series = gpdd_series()
    P = probabilities([s["series"] for s in series])
    rows = []
    for i, s in enumerate(series):
        r = {"MainID": s["MainID"], "taxon": s["taxon"], "class": s["class"], "n": s["n"]}
        for k, v in P.items():
            r[f"{k} P(order>=2)"] = float(v[i, 1:].sum())
            r[f"{k} order"] = int(v[i].argmax() + 1)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "gpdd_order.csv", index=False)
    print(f"\nGPDD: {len(t)} annual series, length {t.n.min()}-{t.n.max()} (median {t.n.median():.0f})")
    for k in P:
        p = t[f"{k} P(order>=2)"]
        print(f"  {k:8s} mean P(delayed density dependence) {p.mean():.2f};"
              f" decisive (P<0.1 or P>0.9) for {np.mean((p < 0.1) | (p > 0.9)):.0%} of series;"
              f" favours order >= 2 for {np.mean(p > 0.5):.0%}")
    bins = pd.cut(t.n, [14, 20, 30, 100])
    print("\n  by length: share of series where the method is decisive (P<0.1 or P>0.9)")
    print(t.groupby(bins, observed=True)[[f"{k} P(order>=2)" for k in P]]
          .agg(lambda p: np.mean((p < 0.1) | (p > 0.9))).round(2).to_string())
    (RESULTS / "lynx_order.json").write_text(json.dumps({k: v[0].tolist() for k, v in pl.items()}, indent=1))


def figure():
    """Left: how often each method is decisive about delayed density dependence, by record
    length. Right: the distribution of P(order >= 2) over the GPDD series."""
    import matplotlib.pyplot as plt
    t = pd.read_csv(RESULTS / "gpdd_order.csv")
    colors = {"timegram": "#1b9e77", "AIC": "#d55e00", "BIC": "#4269d0"}
    edges = [14, 20, 30, 100]
    mids = ["15-20", "21-30", "31-100"]
    bins = pd.cut(t.n, edges)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, c in colors.items():
        p = t[f"{k} P(order>=2)"]
        dec = ((p < 0.1) | (p > 0.9)).groupby(bins, observed=True).mean()
        axes[0].plot(mids, dec.to_numpy(), marker="o", color=c, label=k)
        axes[1].hist(p, bins=np.linspace(0, 1, 21), histtype="step", lw=1.6, color=c, label=k)
    counts = t.groupby(bins, observed=True).size().to_numpy()
    axes[0].set_xticks(range(3), [f"{m}\n({n})" for m, n in zip(mids, counts)])
    axes[0].set(xlabel="record length in years (series)", ylabel="share decisive (P < 0.1 or P > 0.9)")
    axes[1].set(xlabel="probability of delayed density dependence", ylabel="series")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "ecology.pdf")
    fig.savefig(HERE / "ecology.png", dpi=150)


if __name__ == "__main__":
    main()
    figure()
