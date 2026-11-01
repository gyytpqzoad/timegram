"""Chronos-2 on Car Parts with the Monash archive's protocol (monash_carparts.py):
each series on its own (cross-learning off), full history, median forecast, last 12 months as test.
Scored on the archive's zero-filled targets, with zero-filled histories and with gaps kept missing.

    python experiments/intermittent/chronos_carparts.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import monash_carparts as mc  # noqa: E402


def median(pipe, histories):
    preds, _ = pipe.predict_quantiles(inputs=[np.asarray(h, dtype=np.float32) for h in histories],
                                      prediction_length=mc.H, quantile_levels=[0.5], batch_size=64,
                                      cross_learning=False)
    return [np.clip(p.detach().float().cpu().numpy()[0, :, 0], 0, None) for p in preds]


if __name__ == "__main__":
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.float32)
    filled, gaps = mc.load()
    out = {}
    for label, series in (("zero-filled", filled), ("gaps as missing", gaps)):
        out[label] = mc.mase(filled, median(pipe, [s[:-mc.H] for s in series]))[0]
    (HERE / "results" / "carparts_chronos2.json").write_text(json.dumps(out, indent=1))
    print(out)
