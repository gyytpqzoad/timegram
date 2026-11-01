"""Tables: every single-network run on validation and test (plain forecasts)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for f in sorted((HERE / "results").glob("*.json")):
    r = json.loads(f.read_text())
    if r.get("variant") == "ensemble":
        continue
    v, t = r["val"]["plain"], r["test"]["plain"]
    rows.append((f.stem, v["MAPE_trim"], v["median_ape_mean"], t["MAPE_trim"], t["median_ape_mean"],
                 r.get("minutes", 0)))
print("| run | val MAPE (trim) | val median APE | test MAPE (trim) | test median APE | minutes |")
print("|---|---|---|---|---|---|")
for name, a, b, c, d, m in sorted(rows, key=lambda x: (x[2], x[1])):
    print(f"| {name} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {m:.0f} |")
