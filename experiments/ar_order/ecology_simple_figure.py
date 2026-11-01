"""The ecology application in two plain panels .

Left: calibration in simulation, the gap between the stated probability of an order and how
often it is right, by series length. Right: the predictive check on the 701 population records,
the mean log score of one-step forecasts that weight AR(1)-AR(5) by each method's probabilities.

    python experiments/ar_order/ecology_simple_figure.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "results"
C = {"timegram": "#0072B2", "AIC": "#E69F00", "BIC": "#009E73", "other": "#8a949e"}
NAME = {"timegram": "ours", "AIC": "AIC", "BIC": "BIC"}


def main():
    sel = pd.read_csv(R / "selection.csv")
    pr = json.loads((R / "gpdd_predictive.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1, 1.1]})

    for m in ("AIC", "BIC", "timegram"):
        d = sel[sel.method == m]
        a.plot(d.n, d.ECE, marker="o", ms=4, lw=1.4, color=C[m], label=NAME[m])
    a.set_xscale("log")
    a.set_xticks([15, 20, 30, 50, 100], ["15", "20", "30", "50", "100"])
    a.minorticks_off()
    a.set(xlabel="series length (years)", ylabel="gap between stated\nand actual probability", ylim=(0, 0.2))
    a.set_title("calibration in simulation (lower is better)", loc="left", fontsize=9)
    a.legend(frameon=False)

    rows = [("ours", "timegram", C["timegram"]), ("AIC weights", "AIC", C["AIC"]), ("BIC weights", "BIC", C["BIC"]),
            ("equal weights", "uniform", C["other"]), ("AR(1) only", "AR(1) only", C["other"])]
    vals = [pr[k]["mean log score"] for _, k, _ in rows]
    y = np.arange(len(rows))
    b.barh(y, vals, color=[c for *_, c in rows], height=0.6)
    for yi, v in zip(y, vals):
        b.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8)
    b.set_yticks(y, [r[0] for r in rows])
    b.invert_yaxis()
    b.set_xlim(0, 0.32)
    b.set_xlabel("mean log score (higher is better)")
    b.set_title(f"forecasting {pr['forecasts']:,} held-out years", loc="left", fontsize=9)

    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "ecology_simple.pdf")
    fig.savefig(HERE / "ecology_simple.png", dpi=170)


if __name__ == "__main__":
    main()
