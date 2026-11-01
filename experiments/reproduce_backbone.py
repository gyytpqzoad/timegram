"""Train the backbone world with timegram, then score the quarterly sets (M1, M3, M4,
Tourism). Hand-written reference implementation, seeds 0/1:
M1-Q 1.709/1.704, M3-Q 1.136/1.138, M4-Q 1.148/1.146, Tourism-Q 1.538/1.542.

    python experiments/reproduce_backbone.py [--budget 300000000] [--seed 0] [--name ...]
"""
import argparse
import json
import time
from pathlib import Path

import timegram as tg
from timegram.data import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=300_000_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", default=None)
ap.add_argument("--log-every", type=int, default=500)
a = ap.parse_args()

problem = (tg.worlds.backbone()
           .observe(tg.missing(rate=0.1))
           .estimate(tg.forecast(h=24))
           .loss(tg.absolute(scale="relative")))
t0 = time.time()
estimator = problem.pretrain(lengths=(9, 64), budget=a.budget, batch_size=2048, chunk=16384,
                             net=tg.ResMLP(width=384, depth=4), optimizer=tg.MuonAdamW(),
                             backend="fast", seed=a.seed, log_every=a.log_every)
name = a.name or f"backbone_q_{a.budget // 1_000_000}M_s{a.seed}"
tg.library.alias(name, estimator.key)
table = evaluate(estimator.to("cuda"))
print(table.to_string(index=False))
out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
(out / f"{name}.json").write_text(json.dumps({
    "key": estimator.key, "budget": a.budget, "seed": a.seed, "minutes": round((time.time() - t0) / 60, 1),
    "results": table.to_dict(orient="records"), "history": estimator.history}, indent=1))
