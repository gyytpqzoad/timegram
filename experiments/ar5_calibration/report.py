"""Figures and the results table from results/evaluation.json.

    python experiments/ar5_calibration/report.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"unif": "Neural-Unif (pinball)", "unif_cont": "Neural-Unif, trained further",
          "calib_0.97": "Neural-Calibr, debiased, w=0.97", "calib_0.9": "Neural-Calibr, debiased, w=0.9",
          "plugin_0.97": "Neural-Calibr, plug-in, w=0.97", "ols_plugin": "AR-OLS plug-in"}
COLORS = {"unif": "#4269d0", "unif_cont": "#97bbf5", "calib_0.97": "#d55e00", "calib_0.9": "#f4a261",
          "plugin_0.97": "#6a3d9a", "ols_plugin": "#7a7a7a"}


def coverage_figure(report, path):
    sets = report["sets"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, (key, tau) in zip(axes, [("curve_phi1_q95", 0.95), ("curve_phi1_q05", 0.05)]):
        for name, r in sets["lengths 12-64"].items():
            x, y = r[key]
            ax.plot(x, y, color=COLORS.get(name, "k"), lw=1.8 if name.startswith("calib") else 1.2,
                    label=LABELS.get(name, name))
        ax.axhline(tau, color="k", lw=0.8, ls=":")
        ax.set(xlabel=r"$\phi_1$ of the process", ylabel=f"coverage of $q_{{{tau}}}$",
               title=f"Per-process coverage of the {tau} quantile")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def table(report) -> str:
    lines = []
    for label, rows in report["sets"].items():
        floor = next(iter(rows.values()))["abs_gap_floor"]
        lines += [f"\n**{label}** ({report['processes']} processes x {report['replicates']} replicates; "
                  f"|gap| of a perfectly calibrated estimator ≈ {floor[0]:.4f})\n",
                  "| estimator | coverage q0.05 / q0.5 / q0.95 | per-process abs gap q0.05 / q0.95 | "
                  "processes rejected q0.05 / q0.95 | pinball (relative) |",
                  "|---|---|---|---|---|"]
        for name, r in rows.items():
            c, g, rj = r["marginal_coverage"], r["per_process_abs_gap"], r["rejected_share"]
            lines.append(f"| {LABELS.get(name, name)} | {c[0]:.3f} / {c[1]:.3f} / {c[2]:.3f} | "
                         f"{g[0]:.4f} / {g[2]:.4f} | {rj[0]:.1%} / {rj[2]:.1%} | {np.mean(r['pinball_relative']):.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    report = json.loads((HERE / "results" / "evaluation.json").read_text())
    (HERE / "figures").mkdir(exist_ok=True)
    coverage_figure(report, HERE / "figures" / "coverage_by_phi1.png")
    print(table(report))
