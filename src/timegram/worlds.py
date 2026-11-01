"""Ready-made worlds: complete models with their parameter distributions, ready to
pretrain on. ("World" is the Simulacrum paper's word for a model plus its parameter
rule.) The joint distributions they are built from live in `tg.priors`.

    tg.worlds.low_informative("quarterly")    curated SARIMA tree prior
    tg.worlds.backbone()                      the seven-part backbone mixture
    tg.worlds.ets_family(mix_share=0.5)       the additive ETS family, optionally mixed
    tg.worlds.m1_ets()                        the additive ETS world used for M1
"""
from __future__ import annotations

from .blocks import normalize
from .compounds import arima, ets
from .model import convex, select
from .priors import SarimaTree

LOW_INFORMATIVE = {
    # freq: (period, floor, SarimaTree settings): curated presets
    "yearly": (1, 9, dict(seasonal_prob=0.0, l1_prob=0.5, const_prob=1.0, warm_max=0)),
    "quarterly": (4, 10, dict(l1_prob=0.5, const_prob=0.5)),
    "monthly": (12, 30, dict(l1_prob=0.0, const_prob=0.5)),
}


def low_informative(freq: str, **overrides):
    """The curated low-informative ARIMA (yearly) or SARIMA (quarterly, monthly) prior:
    equal weight at every structural decision, auto.arima's rules as constraints."""
    period, floor, cfg = LOW_INFORMATIVE[freq]
    prior = SarimaTree(period=period, **{**cfg, **overrides})
    seasonal = (prior.max_seasonal_order, 1, prior.max_seasonal_order) if period > 1 else (0, 0, 0)
    return arima(prior.max_order, max(prior.d_choices), prior.max_order, seasonal=seasonal,
                 period=period, prior=prior, min_length=floor)


def backbone():
    """The backbone world: seven equally likely components, the curated
    ARIMA/SARIMA priors at periods 1, 4 and 12, SES, Holt and Holt-Winters at 4 and 12."""
    return select(
        low_informative("yearly"), low_informative("quarterly"), low_informative("monthly"),
        ets("ANN", min_length=9), ets("AAN", min_length=9),
        ets("AAA", period=4, min_length=10), ets("AAA", period=12, min_length=30),
    )


NONSEASONAL = ("ANN", "AAN", "AAdN")
SEASONAL = ("ANA", "AAA", "AAdA")


def ets_variants(periods=(4, 12), nonseasonal=NONSEASONAL, seasonal=SEASONAL, weights=None):
    """Selection over additive ETS variants. Groups (non-seasonal, then one per period) get
    `weights` (equal by default), split equally among their variants."""
    groups = [[ets(spec, min_length=9) for spec in nonseasonal]]
    groups += [[ets(spec, period=m) for spec in seasonal] for m in periods]
    w = weights or [1.0] * len(groups)
    operands = [op for g in groups for op in g]
    probs = [wg / len(g) for g, wg in zip(groups, w) for _ in g]
    return select(*operands, probs=probs)


def ets_family(periods=(4, 12), mix_share: float = 0.0, normalize_kind: str = "minmax", **kw):
    """The additive ETS family. With mix_share > 0, that share of series is a convex
    combination of two independent family draws, each normalized on its observed window
    (weight w ~ U(0, 1)); the rest are pure."""
    pure = ets_variants(periods, **kw)
    if mix_share <= 0:
        return pure
    pair = convex(ets_variants(periods, **kw) >> normalize(normalize_kind),
                  ets_variants(periods, **kw) >> normalize(normalize_kind))
    return select(pure, pair, probs=[1 - mix_share, mix_share])


def m1_ets():
    """The additive ETS world used for M1 (experiments/m1_ets): nine additive ETS
    variants, half of the series a min-max-normalized convex mix of two. Train with
    contexts up to 128 and an 18-step horizon."""
    return ets_family(periods=(4, 12), mix_share=0.5, normalize_kind="minmax")
