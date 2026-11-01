"""Chronos-2 on the twelve test sets, for the Monash comparison table.

amazon/chronos-2 (cached weights), each series forecast on its own (cross-learning
off), from its full history (context up to 2048), at each dataset's horizon; the point forecast
is the median. MASE is computed by the same `timegram.data.score` as the timegram models, on the
same official test windows.

    python experiments/accuracy/chronos_test.py [set ...]      # default: all twelve
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from timegram.data import load, score

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "chronos2"
SETS = [(name, period) for name in ("m1", "m3", "m4", "tourism") for period in (1, 4, 12)]
CONTEXT, BATCH = 2048, 64


def pipeline():
    from chronos import Chronos2Pipeline
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", torch_dtype=dtype)


def median_forecast(pipe, histories, horizon):
    preds, _ = pipe.predict_quantiles(
        inputs=[np.asarray(h[-CONTEXT:], dtype=np.float32) for h in histories],
        prediction_length=horizon, quantile_levels=[0.5], batch_size=BATCH,
        context_length=CONTEXT, cross_learning=False)
    out = np.stack([p.detach().float().cpu().numpy()[0, :, 0] for p in preds])
    assert out.shape == (len(histories), horizon)
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    RESULTS.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]
    pipe = pipeline()
    for name, period in SETS:
        ds = load(name, period)
        if wanted and ds.name not in wanted:
            continue
        path = RESULTS / f"{ds.name}.json"
        if path.exists():
            print("done already:", ds.name, flush=True)
            continue
        t0 = time.time()
        fc = median_forecast(pipe, ds.histories, ds.horizon)
        sc = score(ds, fc)
        np.save(RESULTS / f"{ds.name}_median.npy", fc)
        row = {"dataset": ds.name, "series": len(ds), "horizon": ds.horizon, "mase": float(sc["mase"]),
               "smape": float(sc["smape"]), "seconds": round(time.time() - t0, 1)}
        path.write_text(json.dumps(row, indent=1))
        print(row, flush=True)
