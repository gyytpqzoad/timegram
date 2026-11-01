"""Score the R runs of fable.intermittent's methods on Car Parts (baselines.R; our runs, not
published results): the Monash archive's MASE (median forecast, all 2,674 series) and the
benchmark's protocol (two windows, h = 12, the series complete over 51 months)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_eval  # noqa: E402
import monash_carparts as mc  # noqa: E402

raw = pd.read_csv(data_eval.DATA / "carparts.csv")
complete = [c for c in raw.columns if raw[c].notna().all()]
filled, _ = mc.load()
rows = []
for m in ("TWEES", "NEGBINES", "PARAMSD"):
    f = pd.read_csv(HERE / "results" / f"carparts_{m}.csv")
    f["id"] = f["id"].astype(str)
    mon = f[f.window == "monash"].sort_values(["id", "step"])
    order = [str(c) for c in raw.columns]
    med = np.stack([mon[mon.id == k]["q0.5"].to_numpy() for k in order])
    r = {"method": m, "Monash MASE": mc.mase(filled, list(med))[0]}
    per = []
    for i, k in enumerate(complete):
        y = raw[k].to_numpy(float)
        for j, w in ((1, "w1"), (2, "w2")):
            g = f[(f.id == str(k)) & (f.window == w)].sort_values("step")
            T = len(y) - j * 12
            q = {t: g[f"q{t:g}"].to_numpy() for t in data_eval.TAUS}
            per.append({"series": i, "window": j, **data_eval.scores(y[:T], y[T:T + 12], g["mean"].to_numpy(), q)})
    r.update(data_eval.summarize(pd.DataFrame(per)).to_dict())
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(HERE / "results" / "carparts_r_baselines_scored.csv", index=False)
print(t.round(3).to_string(index=False))
