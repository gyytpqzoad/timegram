"""The search's forecasters on monthly count data against the Monash archive's published MASE.

Car Parts (2,674 intermittent series; the archive's zero-filled version) and Hospital (767 series
of patient counts), both from R expsmooth: horizon 12, the last 12 months as test, MASE scaled by
the in-sample seasonal naive error (m = 12), matching the archive's mean seasonal error (checked
below). The validation window is the 12 months before the test window. Point forecasts are set to
zero where negative (counts); the forecasters are run as trained, from the last 128 observations.

    python experiments/accuracy/monash_counts.py r6_natural_forms r8_intermittent
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from timegram import library

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "intermittent"))
import monash_carparts as mc  # noqa: E402

DATA = HERE.parent / "intermittent" / "data"
H = 12
SETS = {"carparts": ("car_parts_dataset_without_missing_values.tsf", 0.7396),
        "hospital": ("hospital_dataset.tsf", 27.8511)}


def mase(series, forecasts, cut):
    """cut = 0: test (last H); cut = H: validation (the H before)."""
    vals = []
    for y, f in zip(series, forecasts):
        hist = y[:len(y) - H - cut]
        target = y[len(y) - H - cut:len(y) - cut]
        s = mc.seasonal_scale(hist)
        if s > 0:
            vals.append(np.mean(np.abs(target - f)) / s)
    return float(np.mean(vals))


def main(names):
    out = {}
    ests = {n: library.pretrained(f"acc_{n}").to("cuda") for n in names}
    for ds, (fname, archive_scale) in SETS.items():
        series = mc.read_tsf(DATA / fname)
        scale = np.mean([mc.seasonal_scale(y[:-H]) for y in series])
        print(f"{ds}: {len(series)} series; mean seasonal error {scale:.4f} (archive {archive_scale})", flush=True)
        for split, cut in (("validation", H), ("test", 0)):
            hist = [y[:len(y) - H - cut][-128:] for y in series]
            for n, e in ests.items():
                fc = np.clip(np.asarray(e.forecast(hist))[:, :H], 0, None)
                out[f"{ds} {split} {n}"] = mase(series, list(fc), cut)
                print(f"  {split:10s} {n:20s} MASE {out[f'{ds} {split} {n}']:.4f}", flush=True)
    (HERE / "results" / f"monash_counts_{'_'.join(names)}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
