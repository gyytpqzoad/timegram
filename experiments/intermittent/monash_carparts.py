"""Car Parts against the Monash archive's published MASE (Godahewa et al. 2021).

The archive scores the version whose missing values are replaced by zeros (the .tsf header says
so): horizon 12, the last 12 months as test, MASE scaled by the in-sample seasonal naive error
(m = 12). timegram reads the version with the gaps kept missing, which its training handles
(`.observe(tg.missing(...))`), and is scored on the zero-filled targets like every archive method.

    python experiments/intermittent/monash_carparts.py check     # the protocol against the archive
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
H, M = 12, 12


def read_tsf(path: Path) -> list[np.ndarray]:
    out, body = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().lower() == "@data":
            body = True
            continue
        if body and line.strip():
            values = line.split(":")[-1].split(",")
            out.append(np.array([np.nan if v.strip() == "?" else float(v) for v in values]))
    return out


def load():
    filled = read_tsf(DATA / "car_parts_dataset_without_missing_values.tsf")
    gaps = read_tsf(DATA / "car_parts_dataset_with_missing_values.tsf")
    assert len(filled) == len(gaps) == 2674
    return filled, gaps


def seasonal_scale(train: np.ndarray) -> float:
    return float(np.mean(np.abs(train[M:] - train[:-M])))


def mase(filled, forecasts) -> tuple[float, int]:
    """Mean MASE over series with a positive scale, and how many series that is."""
    vals = []
    for y, f in zip(filled, forecasts):
        s = seasonal_scale(y[:-H])
        if s > 0:
            vals.append(np.mean(np.abs(y[-H:] - f)) / s)
    return float(np.mean(vals)), len(vals)


if __name__ == "__main__":
    filled, gaps = load()
    scales = np.array([seasonal_scale(y[:-H]) for y in filled])
    print(f"mean seasonal error {scales.mean():.4f} (archive: 0.7396); zero scales: {(scales == 0).sum()}")
    print("missing values in the gapped version:", int(sum(np.isnan(g).sum() for g in gaps)))
    zero = [np.zeros(H) for _ in filled]
    print("zero forecast MASE %.3f over %d series" % mase(filled, zero))
    naive = [np.repeat(y[-H - 1], H) for y in filled]
    print("naive MASE %.3f" % mase(filled, naive)[0])
