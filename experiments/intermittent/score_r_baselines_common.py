import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, data_eval, run
from timegram import library

raw = pd.read_csv(data_eval.DATA / "carparts.csv")
complete = [c for c in raw.columns if raw[c].notna().all()]
d = Path(__file__).resolve().parent / "results"
per = {}
for m in ("TWEES", "NEGBINES", "PARAMSD"):
    f = pd.read_csv(d / f"carparts_{m}.csv"); f["id"] = f["id"].astype(str)
    rows = []
    for i, k in enumerate(complete):
        y = raw[k].to_numpy(float)
        for j, w in ((1, "w1"), (2, "w2")):
            g = f[(f.id == str(k)) & (f.window == w)].sort_values("step"); T = len(y) - j * 12
            rows.append({"series": i, "window": j, **data_eval.scores(y[:T], y[T:T+12], g["mean"].to_numpy(), {t: g[f"q{t:g}"].to_numpy() for t in data_eval.TAUS})})
    per[m] = pd.DataFrame(rows).drop(columns="window").groupby("series").mean()
q = library.pretrained("intermittent_s_combined3_quantile").to("cuda"); me = library.pretrained("intermittent_s_combined3_mean").to("cuda")
rows = []
hist = []; fut = []
for i, k in enumerate(complete):
    y = raw[k].to_numpy(float)
    for j in (1, 2):
        T = len(y) - j * 12; hist.append(y[:T]); fut.append((i, j, y[:T], y[T:T+12]))
qq = np.round(np.sort(run._quantiles(q, hist, 12), axis=-1)); mm = run.forecasts(me, hist, 12)
for n, (i, j, x, y) in enumerate(fut):
    rows.append({"series": i, "window": j, **data_eval.scores(x, y, mm[n], {t: qq[n, :, a] for a, t in enumerate(data_eval.TAUS)})})
per["timegram"] = pd.DataFrame(rows).drop(columns="window").groupby("series").mean()
ok = np.ones(len(complete), bool)
for t in per.values():
    ok &= np.isfinite(t).all(axis=1).reindex(range(len(complete)), fill_value=False).to_numpy()
out = pd.DataFrame({m: t[ok].mean() for m, t in per.items()}).T
print(int(ok.sum()), "common series"); print(out.round(3).to_string())
