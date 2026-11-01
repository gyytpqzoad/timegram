"""The ecology application in one multi-panel figure .

a  three records: the Canadian lynx; the longest GPDD record on which all three methods are
   decisive for delayed density dependence; the 15-20-year record on which AIC weights most
   exceed the calibrated probability
b  their return maps x_{t-1} -> x_t, a loop for delayed dependence
c  calibration in simulation: expected calibration error by record length
d  P(delayed density dependence) across the 701 records; AIC weights cannot fall below ~0.36
e  share of records on which each method is decisive (P < 0.1 or P > 0.9), by length
f  predictive check: log-score gain of the calibrated probabilities over each alternative
g  the lynx: probability of each order

Reads results/ (ecology.py, ecology_predictive.py, run.py); no model is evaluated.

    python experiments/ar_order/ecology_figure.py
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ecology  # noqa: E402

R = HERE / "results"
INK, MUTED, GRID = "#1f2933", "#616e7c", "#d9dee3"
C = {"timegram": "#0072B2", "AIC": "#E69F00", "AICc": "#CC79A7", "BIC": "#009E73"}   # Okabe-Ito
LABEL = {"timegram": "calibrated (ours)", "AIC": "AIC", "AICc": "AICc", "BIC": "BIC"}
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})


def clean(ax, grid_y=False):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_y:
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def letter(ax, s, dx=-0.18, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


def examples():
    """(title, years, series, {method: P(delayed)}) for the three records of panel a."""
    t = pd.read_csv(R / "gpdd_order.csv")
    series = {s["MainID"]: s for s in ecology.gpdd_series()}
    lynx = json.loads((R / "lynx_order.json").read_text())
    out = [("Canadian lynx, 1835-1934 (log$_{10}$)", ecology.lynx(),
            {k: float(np.sum(v[1:])) for k, v in lynx.items()})]
    p = {k: t[f"{k} P(order>=2)"] for k in ("timegram", "AIC", "BIC")}
    agree = t[(t.n >= 30) & (p["timegram"] > 0.9) & (p["AIC"] > 0.9) & (p["BIC"] > 0.9)
              & (t["class"] == "Mammalia") & (t.taxon != "Lynx canadensis")]
    long_ = agree.sort_values("n").iloc[-1]                  # longest decisive mammal besides the lynx
    hare = t[(t.taxon == "Lepus americanus") & (t.n <= 20)].sort_values("n").iloc[0]   # the lynx's prey
    for row, name, what in ((long_, "Fox ($\\it{Vulpes}$)", "all methods decisive"),
                            (hare, "Snowshoe hare ($\\it{Lepus~americanus}$)", "a short record")):
        s = series[int(row.MainID)]
        out.append((f"{name}, {int(row.n)} years ({what})", s["series"],
                    {k: float(row[f"{k} P(order>=2)"]) for k in ("timegram", "AIC", "BIC")}))
    return out


def main():
    warnings.filterwarnings("ignore")
    fig = plt.figure(figsize=(7.2, 7.4))
    outer = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 0.9], hspace=0.55, wspace=0.45,
                             left=0.07, right=0.98, top=0.97, bottom=0.06)

    # a + b: records and return maps
    ex = examples()
    ga = outer[0, :2].subgridspec(3, 2, width_ratios=[5, 1.1], hspace=0.65, wspace=0.08)
    gb = outer[0, 2].subgridspec(3, 1, hspace=0.65)
    for i, (title, x, probs) in enumerate(ex):
        ax = fig.add_subplot(ga[i, 0])
        ax.plot(np.arange(len(x)), x, color=INK, lw=0.8)
        ax.set_title(title, loc="left", pad=2, color=INK)
        ax.set_xlim(-1, len(x))
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, integer=True))
        ax.set_yticks([])
        clean(ax)
        ax.spines["left"].set_visible(False)
        if i == 2:
            ax.set_xlabel("year of record")
        if i == 0:
            letter(ax, "a", dx=-0.05)
        bx = fig.add_subplot(ga[i, 1])                            # P(delayed) by method
        names = ["timegram", "AIC", "BIC"]
        bx.barh(range(3), [probs[k] for k in names], color=[C[k] for k in names], height=0.62)
        bx.set_xlim(0, 1)
        bx.set_ylim(2.6, -0.6)
        bx.set_yticks(range(3), ["ours", "AIC", "BIC"])
        bx.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
        bx.axvline(0.5, color=MUTED, lw=0.5, ls=":")
        bx.tick_params(axis="y", length=0, pad=1.5)
        clean(bx)
        if i == 0:
            bx.set_title("P(delayed)", pad=2, color=INK)
        cx = fig.add_subplot(gb[i])                               # return map
        pts = np.column_stack([x[:-1], x[1:]])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="Greys", norm=plt.Normalize(-0.3 * len(segs), len(segs)), lw=0.7)
        lc.set_array(np.arange(len(segs)))
        cx.add_collection(lc)
        cx.scatter(pts[:, 0], pts[:, 1], s=2.5, color=INK, zorder=3, lw=0)
        lo, hi = x.min(), x.max()
        pad = 0.06 * (hi - lo)
        cx.set_xlim(lo - pad, hi + pad)
        cx.set_ylim(lo - pad, hi + pad)
        cx.plot([lo, hi], [lo, hi], color=GRID, lw=0.6, zorder=0)
        cx.set_xticks([])
        cx.set_yticks([])
        cx.set_aspect("equal", adjustable="box")
        cx.set_ylabel("$x_t$", labelpad=1)
        if i == 2:
            cx.set_xlabel("$x_{t-1}$", labelpad=1)
        clean(cx)
        if i == 0:
            letter(cx, "b", dx=-0.45)
            cx.set_title("return map", pad=2, color=INK)

    # c: calibration in simulation
    sel = pd.read_csv(R / "selection.csv")
    ax = fig.add_subplot(outer[1, 0])
    for m in ("AICc", "AIC", "BIC", "timegram"):
        d = sel[sel.method == m]
        ax.plot(d.n, d.ECE, marker="o", ms=3, lw=1.1 if m == "timegram" else 0.9, color=C[m], label=LABEL[m],
                ls="--" if m == "AICc" else "-", zorder=3 if m == "timegram" else 2)
    ax.set_xscale("log")
    ax.set_xticks([15, 20, 30, 50, 100], ["15", "20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set(xlabel="record length (years)", ylabel="gap: stated vs actual probability", ylim=(0, 0.42))
    ax.set_title("calibration in simulation (lower is better)", loc="left", pad=2)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    clean(ax, grid_y=True)
    letter(ax, "c")

    # d: distribution over the database
    t = pd.read_csv(R / "gpdd_order.csv")
    ax = fig.add_subplot(outer[1, 1])
    bins = np.linspace(0, 1, 21)
    for m in ("AIC", "BIC", "timegram"):
        p = t[f"{m} P(order>=2)"]
        short = "ours" if m == "timegram" else m
        ax.hist(p, bins=bins, histtype="step", lw=1.1, color=C[m], label=f"{short}: {np.mean(p > 0.5):.0%}")
    top = ax.get_ylim()[1] * 1.55
    ax.set_ylim(0, top)
    ax.axvline(0.36, color=C["AIC"], lw=0.6, ls=":")
    ax.text(0.34, top * 0.97, "AIC floor", color=C["AIC"], fontsize=6, va="top", ha="right", rotation=90)
    ax.set(xlabel="P(delayed density dependence)", ylabel="records")
    ax.set_title("701 GPDD records", loc="left", pad=2)
    ax.legend(frameon=False, loc="upper right", title="favour delayed", title_fontsize=6.5,
              alignment="left")
    clean(ax)
    letter(ax, "d")

    # e: share decisive by length
    ax = fig.add_subplot(outer[1, 2])
    edges, mids = [14, 20, 30, 100], ["15-20", "21-30", "31-100"]
    grp = pd.cut(t.n, edges)
    counts = t.groupby(grp, observed=True).size().to_numpy()
    w = 0.26
    for j, m in enumerate(("timegram", "AIC", "BIC")):
        p = t[f"{m} P(order>=2)"]
        dec = ((p < 0.1) | (p > 0.9)).groupby(grp, observed=True).mean().to_numpy()
        ax.bar(np.arange(3) + (j - 1) * w, dec, width=w - 0.03, color=C[m], label=LABEL[m])
        if m == "timegram":
            for k, v in enumerate(dec):
                ax.text(k - w, v + 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=5.8, color=INK)
    ax.set_xticks(range(3), [f"{a}\n(n={c})" for a, c in zip(mids, counts)])
    ax.set(xlabel="record length (years)", ylabel="share decisive", ylim=(0, 0.6))
    ax.set_title("decisive: P < 0.1 or P > 0.9", loc="left", pad=2)
    clean(ax, grid_y=True)
    letter(ax, "e")

    # f: predictive check
    pr = json.loads((R / "gpdd_predictive.json").read_text())
    ax = fig.add_subplot(outer[2, :2].subgridspec(1, 2, width_ratios=[0.22, 1], wspace=0)[1])
    rows = [("AIC weights", "AIC", C["AIC"]), ("BIC weights", "BIC", C["BIC"]),
            ("AIC best order only", "AIC best", C["AIC"]), ("equal weights", "uniform", MUTED),
            ("AR(1) only", "AR(1) only", MUTED)]
    for i, (lab, k, col) in enumerate(rows):
        d, (lo, hi) = pr[k]["timegram minus this (log score)"], pr[k]["95% interval"]
        ax.plot([lo, hi], [i, i], color=col, lw=1.4, solid_capstyle="round")
        ax.plot(d, i, "o", ms=4.5, color=col, mec="white", mew=0.6, zorder=3)
        ax.text(hi + 0.012, i, f"{d:+.3f}", va="center", fontsize=6.2, color=INK)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(-0.06, 0.72)
    ax.set_xlabel("log-score gain of the calibrated probabilities (one-step forecasts, 95% interval)")
    ax.set_title(f"predictive check: {pr['forecasts']:,} held-out years in {pr['records']} records",
                 loc="left", pad=2)
    ax.tick_params(axis="y", length=0)
    ax.text(0.715, len(rows) - 0.45, "better than the alternative →", ha="right", va="bottom",
            fontsize=6, color=MUTED)
    clean(ax)
    ax.spines["left"].set_visible(False)
    letter(ax, "f", dx=-0.29)

    # g: lynx order probabilities
    ly = json.loads((R / "lynx_order.json").read_text())
    ax = fig.add_subplot(outer[2, 2])
    w = 0.26
    for j, m in enumerate(("timegram", "AIC", "BIC")):
        ax.bar(np.arange(1, 6) + (j - 1) * w, ly[m], width=w - 0.03, color=C[m], label=LABEL[m])
    ax.set_xticks(range(1, 6), [f"AR({p})" for p in range(1, 6)])
    ax.set(ylabel="probability", ylim=(0, 1))
    ax.set_title("lynx: probability of each order", loc="left", pad=2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    clean(ax, grid_y=True)
    letter(ax, "g")

    fig.savefig(HERE / "ecology_panel.pdf")
    fig.savefig(HERE / "ecology_panel.png", dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
