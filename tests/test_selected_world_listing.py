"""The readable listing of the selected forecasting world is the trained world.

The listing nests selections where the training script uses one flat selection, so the two
are compared as distributions: every non-selection component with its overall probability.
"""
import ast
from collections import defaultdict
from pathlib import Path

import timegram as tg
from timegram.model import OneHot

ROOT = Path(__file__).resolve().parents[1]


def trained_world():
    """mix(lambda: backbone9_natural(simple=2)), compiled from the script's own definitions."""
    relative = "experiments/accuracy/run.py"
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("backbone9_natural", "mix")]
    namespace = {"tg": tg}
    exec(compile(ast.Module(body=body, type_ignores=[]), relative, "exec"), namespace)
    return namespace["mix"](lambda: namespace["backbone9_natural"](simple=2))


# ------------------------------------------------------------------ the listing, verbatim
up = lambda: tg.uniform(-.25, 1)       # a prior: trend slopes lean upward  # noqa: E731


def smoothing(period=None):            # one seasonality, three trend forms
    s = "A" if period else "N"
    return tg.select(tg.ets(f"AN{s}", period=period),
                     tg.ets(f"AA{s}", period=period, slope0=up()),
                     tg.ets(f"AAd{s}", period=period, slope0=up()),
                     probs=[2, 1, 1])  # no trend twice as likely


def world():
    arima = tg.select(*[tg.worlds.low_informative(f)
                        for f in ("yearly", "quarterly", "monthly")])
    ets = tg.select(smoothing(), smoothing(4), smoothing(12))
    drift = (tg.noise() >> tg.integrate()) + tg.trend(up())  # random walk + drift
    return tg.select(arima, ets, drift, probs=[3, 4, 1])     # selections nest


def listing_world():
    # each call to world() builds an independent copy
    blend = tg.convex(world() >> tg.normalize("minmax"),         # weight w ~ U(0, 1),
                      world() >> tg.normalize("minmax"))         # drawn per series
    return tg.select(world(), blend)                             # half pure, half blended
# ------------------------------------------------------------------


def flatten(block):
    """Flatten nested selections, also inside every other combinator: a world whose
    selections are all at the top, as (identity digest -> probability)."""
    sched = getattr(block, "schedule", None)
    if isinstance(sched, OneHot):
        probs = [float(p) for p in sched.index.probs]
        out = defaultdict(float)
        for op, q in zip(block.operands, probs):
            for k, v in flatten(op).items():
                out[k] += q / sum(probs) * v
        return dict(out)
    return {tg.identity.world(block).digest: 1.0}


def components(block):
    return {k: round(v, 12) for k, v in flatten(block).items()}


def test_world_component_matches_training_script():
    namespace = {"tg": tg}
    relative = "experiments/accuracy/run.py"
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "backbone9_natural"]
    exec(compile(ast.Module(body=body, type_ignores=[]), relative, "exec"), namespace)
    assert components(world()) == components(namespace["backbone9_natural"](simple=2))


def test_final_world_shape_matches_training_script():
    trained, listed = trained_world(), listing_world()
    assert [float(p) for p in trained.schedule.index.probs] == [float(p) for p in listed.schedule.index.probs]
    pure_t, blend_t = trained.operands
    pure_l, blend_l = listed.operands
    assert components(pure_t) == components(pure_l)
    assert type(blend_t.schedule) is type(blend_l.schedule)
    assert blend_t.schedule.w.describe() == blend_l.schedule.w.describe()      # w ~ U(0, 1)
    assert len(blend_t.operands) == len(blend_l.operands) == 2
    for a, b in zip(blend_t.operands, blend_l.operands):     # world() >> normalize("minmax"), twice
        assert components(a.up) == components(b.up)
        assert tg.identity.world(a.down).digest == tg.identity.world(b.down).digest
    copies = [pure_l, *(b.up for b in blend_l.operands)]     # three independent copies, none shared
    assert len({id(c) for c in copies}) == 3
