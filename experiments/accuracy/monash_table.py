"""The selected world against the Monash archive: results/monash_comparison.csv and a LaTeX table.

Archive scores are the published mean MASE of its twelve methods:
- monash_published.csv: the spreadsheet (forecastingdata.org, Monash_Benchmarks.xlsx), which
  has no M4 Yearly row;
- M4_YEARLY below: Table 5 ("Mean MASE results") of the archive paper's supplementary material
  (Godahewa et al. 2021, NeurIPS Datasets and Benchmarks), where M4 Yearly is reported for the
  seven classical and machine-learning methods and not for the four deep-learning ones nor FFNN.
timegram scores are the test scores of `run.py test backbone r6_natural_forms`. Chronos-2
(chronos_test.py) is added as a column once all twelve sets are scored.

    python experiments/accuracy/monash_table.py [path/to/tables/monash.tex]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METHODS = ["SES", "Theta", "TBATS", "ETS", "(DHR-)ARIMA", "PR", "CatBoost", "FFNN", "DeepAR", "N-BEATS",
           "WaveNet", "Transformer"]
M4_YEARLY = dict(zip(METHODS, [3.981, 3.375, 3.437, 3.444, 3.876, 3.625, 3.649] + [np.nan] * 5))
SETS = ["M1-Y", "M1-Q", "M1-M", "M3-Y", "M3-Q", "M3-M", "M4-Y", "M4-Q", "M4-M", "Tourism-Y", "Tourism-Q", "Tourism-M"]
NAMES = {"Y": "Yearly", "Q": "Quarterly", "M": "Monthly"}
SHOWN = ["ETS", "Theta", "TBATS", "(DHR-)ARIMA", "N-BEATS", "DeepAR"]


def comparison() -> pd.DataFrame:
    archive = pd.read_csv(HERE / "monash_published.csv").set_index("Dataset")
    archive.loc["M4 Yearly"] = pd.Series(M4_YEARLY)
    archive.index = [f"{d.split()[0]}-{d.split()[1][0]}" for d in archive.index]
    test = pd.read_csv(RESULTS / "test_backbone_r6_natural_forms.csv").set_index("dataset")
    out = archive.loc[SETS, METHODS].astype(float)
    out["timegram (selected)"] = test.loc[SETS, "r6_natural_forms MASE"]
    out["timegram base"] = test.loc[SETS, "backbone MASE"]
    chronos = [RESULTS / "chronos2" / f"{s}.json" for s in SETS]
    if all(f.exists() for f in chronos):
        out["Chronos-2"] = [json.loads(f.read_text())["mase"] for f in chronos]
    return out


def latex(t: pd.DataFrame) -> str:
    best = t[METHODS].min(axis=1)                          # best of the reported methods
    complete = t[METHODS].notna().all(axis=1)
    rel = lambda col: (col[complete] / t.loc[complete, "ETS"]).mean()     # noqa: E731
    chronos = "Chronos-2" in t
    rivals = (["Chronos-2"] if chronos else []) + ["timegram (selected)"]
    rows = []
    for s in SETS:
        r = t.loc[s]
        cells = ["--" if np.isnan(r[m]) else f"{r[m]:.3f}" for m in SHOWN]
        last = [best[s]] + [r[c] for c in rivals]              # bold a rival that beats the archive
        win = int(np.argmin(last))
        last = [(r"\textbf{%.3f}" if i == win and i > 0 else "%.3f") % v for i, v in enumerate(last)]
        rows.append(" & ".join([s] + cells + last) + r" \\")
    avg = [rel(t[m]) for m in SHOWN] + [rel(best)] + [rel(t[c]) for c in rivals]
    top = min(avg)
    avg = [(r"\textbf{%.3f}" if a == top else "%.3f") % a for a in avg]
    rows += [r"\midrule", " & ".join(["relative to ETS"] + avg) + r" \\"]
    return "\n".join([
        r"\begin{table}[tbp]",
        r"\centering\tableformat",
        r"\caption{Test MASE against the Monash archive's published results \citep{godahewa2021monash}. Six of its"
        r" twelve methods are shown; ``best'' is the best of all twelve for each set. For M4 Yearly the archive"
        r" reports only its classical and machine-learning methods (--: not reported), so ``best'' there is the best"
        r" of those seven. The archive's methods are fitted to each dataset; \gt\ is one network pretrained on"
        r" simulated worlds, the selected world of Section~\ref{sec:search}. The last row averages each column"
        r" relative to ETS over the 11 sets where every method is reported."
        + (r" Chronos-2 \citep{ansari2025chronos2} was run here on the same windows, with full"
           r" histories and median forecasts; its pretraining data likely include these datasets."
           r" Bold marks the best of the archive, Chronos-2 and \gt."
           if chronos else "") + "}",
        r"\label{tab:monash}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrrrr" + ("r" if chronos else "") + r"r@{}}",
        r"\toprule",
        r" & \multicolumn{7}{c}{Monash archive} & " + ("& " if chronos else "") + r"\\",
        r"\cmidrule(lr){2-8}",
        r"Set & ETS & Theta & TBATS & ARIMA & N-BEATS & DeepAR & best & " + ("Chronos-2 & " if chronos else "")
        + r"\gt \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])


if __name__ == "__main__":
    t = comparison()
    t.to_csv(RESULTS / "monash_comparison.csv")
    best = t[METHODS].min(axis=1)
    ours = t["timegram (selected)"]
    print("first on", int((ours < best).sum()), "of", len(t), "sets;",
          "second or worse on", list(t.index[ours >= best]))
    complete = t[METHODS].notna().all(axis=1)
    print("relative to ETS (11 complete sets): selected %.4f, base %.4f, best %.4f" % (
        (ours[complete] / t.loc[complete, "ETS"]).mean(), (t.loc[complete, "timegram base"] / t.loc[complete, "ETS"]).mean(),
        (best[complete] / t.loc[complete, "ETS"]).mean()))
    print("relative to ETS (12 sets): selected %.4f" % (ours / t["ETS"]).mean())
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(latex(t), encoding="utf-8")
        print("wrote", sys.argv[1])
