"""One pretrained forecaster for yearly, quarterly and monthly data, from a world found with
the grammar.

One network forecasts 24 steps from the last <= 128 observations; each dataset uses its own
horizon (Tourism-Y 4, yearly 6, quarterly 8, monthly 18, Tourism-M 24). Worlds are compared
on a validation split, the last h points of each series' training part, averaged over the
twelve sets (M1, M3, M4, Tourism x yearly, quarterly, monthly). Only the selected world is
scored on the official test windows. MASE uses the protocol of `timegram.data` (the backbone
experiments), to be compared with the Monash archive's published results.

The search: a context-128 base trained from scratch on the backbone world (150M series), then
single grammar moves added to that world and warm-started from the base at a fixed screening
budget (CANDIDATES, 25M series each), all against ft_backbone, which only continues training on
the base world. Selection uses the validation split only. Moves that help are then combined.

    python experiments/accuracy/run.py train backbone               # the base, from scratch
    python experiments/accuracy/run.py train ft_backbone r6_natural_forms   # warm-started candidates
    python experiments/accuracy/run.py validate backbone ft_backbone r6_natural_forms
    python experiments/accuracy/run.py test backbone r6_natural_forms
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import timegram as tg
from timegram import library
from timegram.data import Dataset, load, mase_scale, score, seasonal_naive

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NAME = "acc"                                   # library aliases acc_<world>
SETS = [(name, period) for name in ("m1", "m3", "m4", "tourism") for period in (1, 4, 12)]
H, CONTEXT = 24, 128
TRAIN = dict(lengths=(7, CONTEXT), budget=150_000_000, chunk=16384, backend="fast", log_every=1000, seed=0)
FREQ = {1: "yearly", 4: "quarterly", 12: "monthly"}


# ---------------------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------------------
def puzzle():
    """ETS and ARIMA driven by common shocks and mixed with a random weight, at each period:
    the generative world of forecast combination (Simulacrum, Neural-Puzzle), extended to
    monthly data."""
    parts = []
    for period in (1, 4, 12):
        shock = tg.noise()
        ets = tg.ets("AAN", min_length=9) if period == 1 else tg.ets("AAA", period=period)
        arima = tg.worlds.low_informative(FREQ[period])
        parts.append(tg.convex(shock >> ets, shock >> arima))
    return tg.select(*parts)


def worlds():
    return {
        "backbone": lambda: tg.worlds.backbone(),
        "puzzle": puzzle,
        "backbone_puzzle": lambda: tg.select(tg.worlds.backbone(), puzzle()),
        "mixhalf": lambda: tg.select(
            tg.worlds.backbone(),
            tg.convex(tg.worlds.backbone() >> tg.normalize("minmax"), tg.worlds.backbone() >> tg.normalize("minmax"))),
    }


def multiplicative(world):
    """A positive version of a world with multiplicative dynamics: standardize, then
    exponentiate (the world on the log scale; not the classical multiplicative ETS)."""
    return world >> tg.normalize("std") >> tg.exp()


def smooth():
    """Persistent growth: ETS driven by AR(1) innovations with positive coefficient (momentum,
    a lower noise-to-signal ratio), non-seasonal or seasonal at periods 4 and 12."""
    def momentum():
        return tg.noise() >> tg.ar(1, phi=tg.uniform(0.3, 0.95))
    return tg.select(momentum() >> tg.ets("AAN", min_length=9),
                     momentum() >> tg.ets("AAA", period=4, min_length=10),
                     momentum() >> tg.ets("AAA", period=12, min_length=30))


def components():
    """The seven components of tg.worlds.backbone(), as separate blocks."""
    lo = tg.worlds.low_informative
    return [lo("yearly"), lo("quarterly"), lo("monthly"),
            tg.ets("ANN", min_length=9), tg.ets("AAN", min_length=9),
            tg.ets("AAA", period=4, min_length=10), tg.ets("AAA", period=12, min_length=30)]


def fed(innovations):
    """The backbone with each component driven by fresh innovations from `innovations()`
    instead of its own Gaussian noise: one grammar move, `innovations >> component`."""
    return tg.select(*[innovations() >> c for c in components()])


def damped():
    """Damped-trend exponential smoothing, non-seasonal and at periods 4 and 12."""
    return tg.select(tg.ets("AAdN", min_length=9), tg.ets("AAdA", period=4, min_length=10),
                     tg.ets("AAdA", period=12, min_length=30))


def trended():
    """Each component plus a deterministic damped trend: stronger, smoother growth."""
    return tg.select(*[c + tg.damped_trend(b=tg.uniform(-1, 1), phi=tg.uniform(0.8, 0.98))
                       for c in components()])


def backbone9():
    """The backbone with its four ETS components replaced by the M1 world's nine additive
    variants (ANN, AAN, AAdN; ANA, AAA, AAdA at periods 4 and 12); ARIMA keeps its 3/7."""
    lo = tg.worlds.low_informative
    return tg.select(tg.select(lo("yearly"), lo("quarterly"), lo("monthly")), tg.worlds.ets_variants(),
                     probs=[3 / 7, 4 / 7])


def mixhalf(world):
    """The M1 recipe applied to a world factory: half of the series pure, half a convex
    combination of two independent min-max-normalized draws (weight U(0, 1))."""
    return tg.select(world(), tg.convex(world() >> tg.normalize("minmax"), world() >> tg.normalize("minmax")))


def components9():
    """backbone9() as (component, probability) pairs: three ARIMA priors at 1/7 each, nine ETS
    variants at 4/63 each."""
    lo = tg.worlds.low_informative
    arima = [(lo(f), 1 / 7) for f in ("yearly", "quarterly", "monthly")]
    ets = [(tg.ets(s, min_length=9), 4 / 63) for s in tg.worlds.NONSEASONAL]
    ets += [(tg.ets(s, period=m), 4 / 63) for m in (4, 12) for s in tg.worlds.SEASONAL]
    return arima + ets


def fed9(innovations):
    """backbone9() with every component driven by `innovations()` (call it once per component;
    returning the same block every time makes the shocks shared)."""
    pairs = components9()
    return tg.select(*[innovations() >> c for c, _ in pairs], probs=[p for _, p in pairs])


def mix(world, share=0.5, k=2, kind="minmax"):
    """A share of the series a convex combination of k independent normalized draws of
    `world()` (weights uniform on the simplex), the rest pure draws."""
    blend = tg.convex(*[world() >> tg.normalize(kind) for _ in range(k)])
    return tg.select(world(), blend, probs=[1 - share, share])


import json as _json
import math as _math

import torch as _torch

from timegram.compounds import EtsSmoothing, LogitNormalEts  # noqa: F401  (stable, importable home)
from timegram.dists import numpy_rng as _numpy_rng


def ets_priors():
    """Parametric priors stylized from ETS fits on real histories (empirical_priors.py stylize):
    per frequency, a multivariate normal on the logit scale of (alpha, beta/alpha,
    gamma/(1-alpha), (phi-0.8)/0.18), a Student-t for the starting slope in noise units, and the
    shares of the ETS forms."""
    return _json.loads((HERE / "ets_priors.json").read_text())


def slope_prior(period):
    """Starting slope in noise units: a location-scale Student-t fitted to real trended series."""
    t = ets_priors()[str(period)]["slope_t"]
    return t["loc"] + t["scale"] * tg.student_t(t["df"])


def ets_fitted(spec, period, prior_period=None, **kw):
    """Additive ETS with the fitted parametric priors of frequency `prior_period` (default:
    `period`, or yearly for non-seasonal forms without one)."""
    pp = prior_period or period or 1
    pr = ets_priors()[str(pp)]
    key = spec.upper()
    trend, seasonal, damped = key[1] == "A", key[-1] == "A", "AD" in key
    joint = LogitNormalEts(trend, seasonal, period if seasonal else None, damped,
                           pr["logit_mean"], pr["logit_cov"])
    params = {"alpha": joint["alpha"]}
    if trend:
        params |= {"beta": joint["beta"], "slope0": slope_prior(pp)}
    if seasonal:
        params["gamma"] = joint["gamma"]
    if damped:
        params["phi"] = joint["phi"]
    floor = 9 if not seasonal else (10 if period == 4 else 30)
    return tg.ets(spec, period=period if seasonal else None, min_length=floor, **params, **kw)


def ets_empirical():
    """One group per frequency (equal weight), the ETS forms in their observed shares, each
    with that frequency's fitted priors."""
    groups = []
    for period, pr in ets_priors().items():
        period = int(period)
        forms = pr["forms"]
        groups.append(tg.select(*[ets_fitted(f, period if f[-1] == "A" else None, prior_period=period)
                                  for f in forms], probs=list(forms.values())))
    return tg.select(*groups)


def backbone_emp(rwd=False):
    """backbone9 with the ETS part replaced by ets_empirical(); optionally a random walk with
    the yearly fitted drift as a further alternative."""
    lo = tg.worlds.low_informative
    parts = [tg.select(lo("yearly"), lo("quarterly"), lo("monthly")), ets_empirical()]
    probs = [3 / 7, 4 / 7]
    if rwd:
        parts.append((tg.noise() >> tg.integrate()) + tg.trend(slope_prior(1)))
        probs = [3 / 8, 4 / 8, 1 / 8]
    return tg.select(*parts, probs=probs)


def backbone9_drift():
    """backbone9 with only the drift changed: trended ETS starting slopes from the fitted
    Student-t (yearly fit for non-seasonal forms, quarterly / monthly for periods 4 / 12)."""
    lo = tg.worlds.low_informative
    ets = [tg.ets(s, min_length=9, **({"slope0": slope_prior(1)} if s != "ANN" else {}))
           for s in tg.worlds.NONSEASONAL]
    ets += [tg.ets(s, period=m, **({"slope0": slope_prior(m)} if s != "ANA" else {}))
            for m in (4, 12) for s in tg.worlds.SEASONAL]
    probs = [1 / 7] * 3 + [4 / 63] * 9
    return tg.select(lo("yearly"), lo("quarterly"), lo("monthly"), *ets, probs=probs)


def backbone9_natural(simple=1):
    """The lesson of the empirical fits in plain grammar: trended ETS slopes lean upward,
    U(-0.25, 1) (80% up), and a random walk with drift joins the alternatives. `simple` is the
    weight of the untrended ETS forms (ANN, ANA) relative to each trended one, within the ETS
    share; simple=2 puts half of each ETS group on the untrended form."""
    lo = tg.worlds.low_informative
    up = lambda: tg.uniform(-0.25, 1)                                       # noqa: E731
    ets = [tg.ets(s, min_length=9, **({"slope0": up()} if s != "ANN" else {})) for s in tg.worlds.NONSEASONAL]
    ets += [tg.ets(s, period=m, **({"slope0": up()} if s != "ANA" else {}))
            for m in (4, 12) for s in tg.worlds.SEASONAL]
    rwd = (tg.noise() >> tg.integrate()) + tg.trend(up())
    w = [simple if s in ("ANN", "ANA") else 1 for s in tg.worlds.NONSEASONAL]
    w += [simple if s == "ANA" else 1 for m in (4, 12) for s in tg.worlds.SEASONAL]
    ets_probs = [4 / 8 * wi / sum(w) for wi in w]
    return tg.select(lo("yearly"), lo("quarterly"), lo("monthly"), *ets, rwd,
                     probs=[1 / 8] * 3 + ets_probs + [1 / 8])


def stochastic_volatility():
    """Shocks with stochastic volatility, e_t * exp(h_t / 2), the log-volatility h an AR(1)
    with persistence U(0.8, 0.98): a product of two processes."""
    log_vol = tg.noise(scale=tg.uniform(0.1, 0.4)) >> tg.ar(1, phi=tg.uniform(0.8, 0.98))
    return tg.noise() * (log_vol >> tg.exp(rate=0.5))


def seasonal_factor(period):
    """1 + c * s_t, with s_t a slowly evolving zero-sum seasonal and c ~ U(0.05, 0.4): a
    multiplicative seasonal factor around one."""
    s = tg.pattern(period) + (tg.noise(scale=tg.uniform(0, 0.1)) >> tg.difference() >> tg.integrate(period=period))
    return tg.constant(1.0) + tg.uniform(0.05, 0.4) * s


def uc(irregular, level_shocks, slope=False, season=None, cycle=False):
    """An unobserved-components model from blocks: a positive starting level plus a random-
    walk level (optionally a slope with its own shocks), an irregular, optionally a stochastic
    cycle (an AR(2) component) and a season. season = (mode, period), mode "add", "mult"
    (level times a seasonal factor) or "hybrid" (both). Shock factories are called once per use."""
    level = tg.constant(tg.uniform(5, 20)) + (level_shocks() >> tg.integrate())
    if slope:
        level = level + tg.trend(tg.uniform(-1, 1)) + (tg.noise(scale=tg.uniform(0, 0.3))
                                                      >> tg.integrate(2) >> tg.lag(1))
    if season is not None and season[0] in ("mult", "hybrid"):
        level = level * seasonal_factor(season[1])
    y = level + irregular()
    if season is not None and season[0] in ("add", "hybrid"):
        y = y + tg.uniform(0.2, 1) * tg.seasonal(season[1])
    if cycle:
        y = y + (tg.noise(scale=tg.uniform(0, 1)) >> tg.ar(2))
    # length floor: two full seasons plus the parameter count plus one (9 without a season)
    k = 3 + 2 * slope + (2 if season else 0) + 2 * cycle
    return tg.at_least(y, max(9, 2 * season[1] + k + 1) if season else max(9, 2 * k + 1))


def uc_menu(irregular, level_shocks, modes=("add",), cycle=False):
    """Local level or local trend, without a season or with each seasonal mode at periods 4
    and 12 (equal weights), optionally each with or without a cycle."""
    ops = [uc(irregular, level_shocks, slope, season, c)
           for slope in (False, True)
           for season in [None, *[(mode, m) for mode in modes for m in (4, 12)]]
           for c in ((False, True) if cycle else (False,))]
    return tg.select(*ops)


def uc_rich():
    """The unobserved-components family: Gaussian, stochastic-volatility or Student-t
    irregulars, Gaussian or stochastic-volatility level shocks, additive, multiplicative or
    hybrid seasons, with or without a cycle."""
    return uc_menu(lambda: tg.select(gaussian(), sv_scaled(), tg.noise(tg.student_t(4), scale=tg.uniform(0, 1))),
                   lambda: tg.select(gaussian(), sv_scaled()), ("add", "mult", "hybrid"), cycle=True)


def gaussian():
    return tg.noise(scale=tg.uniform(0, 1))


def sv_scaled():
    return tg.uniform(0.2, 1) * stochastic_volatility()


def ets_multiplicative():
    """Exponential smoothing (positive level) times a seasonal factor: multiplicative
    seasonality built from blocks, at periods 4 and 12."""
    return tg.select(*[tg.ets(s, min_length=9 if m == 4 else 30, level0=tg.uniform(5, 20)) * seasonal_factor(m)
                       for s in tg.worlds.NONSEASONAL for m in (4, 12)])


ARIMA_ORDERS = [(1, 0, 0), (2, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 0),
                (0, 1, 2), (0, 2, 2)]


def arima_blocks(period=1):
    """ARIMA (noise >> ma >> ar >> integrate) with coefficients uniform in the L1 ball, over a
    grid of orders (equal weights); at a seasonal period also with no season, a seasonal AR(1)
    or a seasonal (0,1,1). Each alternative has its own independent noise. The named
    constructor sets each order's length floor: 2k + 1 points after differencing, and two
    full seasons for a seasonal model."""
    def one(p, d, q, season):
        P, D, Q = {None: (0, 0, 0), "ar": (1, 0, 0), "ima": (0, 1, 1)}[season]
        coef = {}
        if p:
            coef["phi"] = tg.l1_ball(p)
        if q:
            coef["theta"] = tg.l1_ball(q)
        if P:
            coef["seasonal_phi"] = tg.l1_ball(P)
        if Q:
            coef["seasonal_theta"] = tg.l1_ball(Q)
        return tg.arima(p, d, q, seasonal=(P, D, Q), period=period if season else 1, **coef)
    seasons = [None] if period == 1 else [None, "ar", "ima"]
    return tg.select(*[one(p, d, q, s) for p, d, q in ARIMA_ORDERS for s in seasons])


def backbone9_with(arima_part):
    """backbone9 with its ARIMA part (3/7 of the series) replaced."""
    return tg.select(arima_part, tg.worlds.ets_variants(), probs=[3 / 7, 4 / 7])


def arima_curated_and_blocks():
    """Per frequency, half the curated low-informative prior and half the block-built family."""
    lo = tg.worlds.low_informative
    return tg.select(*[tg.select(lo(f), arima_blocks(m)) for f, m in (("yearly", 1), ("quarterly", 4),
                                                                      ("monthly", 12))])


def innovation_menu():
    """Innovations whose law is drawn once per series: Gaussian (40%), Student-t(4), Laplace,
    occasional large shocks, or GARCH(1,1) volatility (15% each)."""
    return tg.select(tg.noise(), tg.noise(tg.student_t(4)), tg.noise(tg.laplace(0, 1)),
                     tg.noise(tg.mixture(tg.normal(0, 1), tg.normal(0, 6), weights=[0.95, 0.05])),
                     tg.noise() >> tg.garch(1, 1, burn_in=200),
                     probs=[0.4, 0.15, 0.15, 0.15, 0.15])


def structural():
    """Unobserved-components (basic structural) models: local level or local linear trend,
    without a season or with a dummy season at period 4 or 12, each with its own noises."""
    def level_or_trend():
        return tg.select(tg.local_level(), tg.local_trend())
    return tg.select(level_or_trend(), level_or_trend() + tg.seasonal(4), level_or_trend() + tg.seasonal(12))


def mixed():
    """Convex combinations of two independent backbone draws, each min-max normalized."""
    return tg.convex(tg.worlds.backbone() >> tg.normalize("minmax"),
                     tg.worlds.backbone() >> tg.normalize("minmax"))


# Candidates warm-started from the context-128 base (acc_backbone), same budget each. Each
# adds one grammar move to the backbone (half of the series stay from the backbone), so that
# its effect can be read against ft_backbone, which continues on the base world alone.
# Round 1a, from diagnostics of the base world (diagnose.py): real series are smoother, with
# persistent growth, less seasonality, and seasonal swings that grow with the level.
# Round 1b, the other moves of the grammar: innovations (heavy tails, volatility clustering,
# long memory), components (damped trends, a deterministic trend), and combination.
CANDIDATES = {
    "ft_backbone": lambda: tg.worlds.backbone(),
    "ft_multiplicative": lambda: tg.select(tg.worlds.backbone(), multiplicative(tg.worlds.backbone())),
    "ft_smooth": lambda: tg.select(tg.worlds.backbone(), smooth()),
    "ft_combined": lambda: tg.select(tg.worlds.backbone(), multiplicative(tg.worlds.backbone()),
                                     smooth(), multiplicative(smooth())),
    "ft_heavy": lambda: tg.select(tg.worlds.backbone(), fed(lambda: tg.noise(tg.student_t(4)))),
    "ft_garch": lambda: tg.select(tg.worlds.backbone(), fed(lambda: tg.noise() >> tg.garch(1, 1, burn_in=200))),
    "ft_longmem": lambda: tg.select(tg.worlds.backbone(),
                                    fed(lambda: tg.noise() >> tg.fractional_integrate(tg.uniform(0.1, 0.45)))),
    "ft_damped": lambda: tg.select(tg.worlds.backbone(), damped()),
    "ft_trended": lambda: tg.select(tg.worlds.backbone(), trended()),
    "ft_mixed": lambda: tg.select(tg.worlds.backbone(), mixed()),
    # Round 1c: the ideas of the M1 world (tg.worlds.m1_ets), whose 18-step model
    # beats the base on 9 of the 11 validation sets it covers: the full nine-variant ETS menu,
    # and half of the series a convex mix of two min-max-normalized draws.
    "ft_m1world": lambda: tg.select(tg.worlds.backbone(), tg.worlds.m1_ets()),      # select, 1:1
    "ft_m1only": lambda: tg.worlds.m1_ets(),                                       # no ARIMA
    "ft_ets9": lambda: backbone9(),                                                # menu only
    "ft_mixhalf9": lambda: mixhalf(backbone9),                                     # M1 recipe
    "ft_convex_m1": lambda: tg.select(                                             # cross-world mix
        tg.worlds.backbone(),
        tg.convex(tg.worlds.backbone() >> tg.normalize("minmax"), tg.worlds.m1_ets() >> tg.normalize("minmax"))),
}
CANDIDATES.update({
    # Round 1d: exotic moves. Unobserved components (several independent noises, unlike the
    # single source of error of ETS), alone and mixed with the backbone; volatility as a
    # heteroskedastic irregular and as a smaller share of GARCH-fed series; level shifts.
    "ft_uc": lambda: tg.select(tg.worlds.backbone(), structural()),
    "ft_uc_mix": lambda: tg.select(tg.worlds.backbone(), tg.convex(tg.worlds.backbone() >> tg.normalize("minmax"),
                                                                   structural() >> tg.normalize("minmax"))),
    "ft_garch_irregular": lambda: tg.select(tg.worlds.backbone(),
                                            tg.select(*[c + (tg.noise() >> tg.garch(1, 1, burn_in=200))
                                                        for c in components()])),
    "ft_garch20": lambda: tg.select(tg.worlds.backbone(), fed(lambda: tg.noise() >> tg.garch(1, 1, burn_in=200)),
                                    probs=[0.8, 0.2]),
    "ft_shifts": lambda: tg.select(tg.worlds.backbone(), fed(lambda: tg.noise(
        tg.mixture(tg.normal(0, 1), tg.normal(0, 6), weights=[0.95, 0.05])))),
})
CANDIDATES.update({
    # Round 1e: mixing in the innovation distribution itself. Between series: each series'
    # shocks follow one law from a menu (Gaussian 40%). Within series: every shock is drawn
    # from a mixture of laws (a contaminated distribution). Applied to every component.
    "ft_innov_menu": lambda: fed(innovation_menu),
    "ft_innov_within": lambda: fed(lambda: tg.noise(tg.mixture(tg.normal(0, 1), tg.laplace(0, 1), tg.student_t(3),
                                                               weights=[0.6, 0.2, 0.2]))),
})
CANDIDATES.update({
    # Round 2a: variations on the round-1 leader, mixhalf(backbone9) (= mix(backbone9)), using
    # different composition modes of the grammar.
    "r2_mix25": lambda: mix(backbone9, share=0.25),                       # select proportion
    "r2_mix75": lambda: mix(backbone9, share=0.75),
    "r2_mix3": lambda: mix(backbone9, k=3),                               # Dirichlet over three
    "r2_mixstd": lambda: mix(backbone9, kind="std"),                      # another normalization
    "r2_mixtrend": lambda: mix(lambda: tg.select(                         # nest round 1b's best
        backbone9(), backbone9() + tg.damped_trend(b=tg.uniform(-1, 1), phi=tg.uniform(0.8, 0.98))), share=0.5),
    "r2_commonshock": lambda: tg.select(                                  # mixed pair, shared shocks
        backbone9(), (lambda shock: tg.convex(fed9(lambda: shock) >> tg.normalize("minmax"),
                                              fed9(lambda: shock) >> tg.normalize("minmax")))(tg.noise())),
    "r2_crossfamily": lambda: tg.select(                                  # always ARIMA with ETS
        backbone9(), tg.convex(tg.select(*[tg.worlds.low_informative(f) for f in ("yearly", "quarterly", "monthly")])
                               >> tg.normalize("minmax"), tg.worlds.ets_variants() >> tg.normalize("minmax"))),
})
CANDIDATES.update({
    # Round 3: empirical priors. Forecast-vs-actual diagnostics (diagnose_forecasts.py) show
    # forecasts too flat and too low at long horizons; ETS fits on real histories show trends
    # mostly upward and strong relative to the noise. Each keeps the round-1 recipe mix().
    "r3_drift": lambda: mix(backbone9_drift),                     # drift asymmetry only
    "r3_emp": lambda: mix(backbone_emp),                          # stylized empirical ETS
    "r3_emp_rwd": lambda: mix(lambda: backbone_emp(rwd=True)),    # + random walk with drift
    # The same lesson with natural blocks and round-number settings: eligible as a final world.
    "r3_natural": lambda: mix(backbone9_natural),
    # Round 6: the "simpler ETS forms more often" lesson of the fits, in plain grammar.
    "r6_natural_forms": lambda: mix(lambda: backbone9_natural(simple=2)),
})
CANDIDATES.update({
    # Round 4: unobserved components and their variations, each a third of the series next
    # to the round-1 leader mix(backbone9). Trained twice as long (BUDGET), with the leader
    # itself trained as long as the control.
    "r4_leader50": lambda: mix(backbone9),
    "r4_uc_sv": lambda: tg.select(mix(backbone9), uc_menu(sv_scaled, gaussian), probs=[0.7, 0.3]),
    "r4_uc_hybrid": lambda: tg.select(mix(backbone9), uc_menu(gaussian, gaussian, ("add", "mult", "hybrid")),
                                      probs=[0.7, 0.3]),
    "r4_ets_mult": lambda: tg.select(mix(backbone9), ets_multiplicative(), probs=[0.7, 0.3]),
    "r4_uc_rich": lambda: tg.select(mix(backbone9), uc_rich(), probs=[0.7, 0.3]),
    "r4_uc": lambda: tg.select(mix(backbone9), uc_menu(gaussian, gaussian), probs=[0.7, 0.3]),
    "r4_uc_mult": lambda: tg.select(mix(backbone9), uc_menu(gaussian, gaussian, ("mult",)), probs=[0.7, 0.3]),
    "r4_uc_sv2": lambda: tg.select(mix(backbone9), uc_menu(sv_scaled, sv_scaled), probs=[0.7, 0.3]),
})
CANDIDATES.update({
    # Round 5: a broader ARIMA side, mixed with the curated prior (screening budget).
    "r5_arima_mix": lambda: mix(lambda: backbone9_with(arima_curated_and_blocks())),
    "r5_arima_blocks": lambda: mix(lambda: backbone9_with(tg.select(arima_blocks(1), arima_blocks(4),
                                                                    arima_blocks(12)))),
})
CANDIDATES.update({
    # Round 7: the two clean gains together, and a second seed of the clean leader.
    "r7_natural_uc": lambda: tg.select(mix(lambda: backbone9_natural(simple=2)), uc_rich(), probs=[0.7, 0.3]),
    "r6_natural_forms_s2": lambda: mix(lambda: backbone9_natural(simple=2)),
})



def intermittent_monthly():
    """Monthly intermittent demand, as selected by the intermittent search (experiments/
    intermittent/search.py, "combined3", without its weekly half): occurrence * size, integer
    counts."""
    occurrence = tg.select(*[tg.noise(tg.bernoulli(p)) for p in (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)])
    log_size = tg.uniform(0, 0.3) * (tg.noise() >> tg.integrate()) + tg.uniform(0.2, 1.6) * tg.noise()
    return (occurrence * ((log_size + tg.constant(tg.uniform(0, 3))) >> tg.exp())) >> tg.counts()


CANDIDATES.update({
    # The selected world extended with monthly intermittent demand as one more alternative,
    # share 1/8; scored on the twelve sets and on Car Parts (monash_counts.py).
    "r8_intermittent": lambda: tg.select(mix(lambda: backbone9_natural(simple=2)), intermittent_monthly(),
                                         probs=[7 / 8, 1 / 8]),
})
SEED = {"r6_natural_forms_s2": 2}                      # training seed if not TUNE's
ROUND_4 = ["r4_leader50", "r4_uc_sv", "r4_uc_hybrid", "r4_ets_mult", "r4_uc_rich", "r4_uc", "r4_uc_mult",
           "r4_uc_sv2"]
BUDGET = {name: 50_000_000 for name in ROUND_4}           # harder worlds: twice the screening budget
ROUND_1B = ["ft_heavy", "ft_garch", "ft_longmem", "ft_damped", "ft_trended", "ft_mixed"]
ROUND_3 = ["r3_drift", "r3_emp", "r3_emp_rwd"]
ROUND_2A = ["r2_mix25", "r2_mix75", "r2_mix3", "r2_mixstd", "r2_mixtrend", "r2_commonshock", "r2_crossfamily"]
ROUND_1E = ["ft_innov_menu", "ft_innov_within"]
ROUND_1D = ["ft_uc", "ft_uc_mix", "ft_garch_irregular", "ft_garch20", "ft_shifts"]
ROUND_1C = ["ft_m1world", "ft_m1only", "ft_ets9", "ft_mixhalf9", "ft_convex_m1"]
TUNE = dict(TRAIN, budget=25_000_000, warmup=2_000_000, seed=1)       # screening budget


def problem(world):
    return (world.observe(tg.missing(0.1))
            .estimate(tg.forecast(h=H))
            .loss(tg.absolute(scale="relative")))


def train(name):
    if name in CANDIDATES:
        tune = dict(TUNE, budget=BUDGET.get(name, TUNE["budget"]), seed=SEED.get(name, TUNE["seed"]))
        est = problem(CANDIDATES[name]()).pretrain(**tune, init=library.resolve(f"{NAME}_backbone"))
    else:
        est = problem(worlds()[name]()).pretrain(**TRAIN)
    library.alias(f"{NAME}_{name}", est.key)
    print(f"{name}: {est.key}", flush=True)


# ---------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------
def validation(ds: Dataset) -> Dataset:
    """The last h points of each training part as the target; series too short to keep at
    least three observations are left out."""
    keep = [h for h in ds.histories if len(h) >= ds.horizon + 3
            and mase_scale(h[:-ds.horizon], ds.period) > 1e-6]          # a constant history has no MASE scale
    return Dataset(ds.name + " (val)", ds.period, ds.horizon, [h[:-ds.horizon] for h in keep],
                   np.stack([h[-ds.horizon:] for h in keep]))


def forecast(est, histories, log: bool = False) -> np.ndarray:
    """Point forecasts; with log=True strictly positive series are forecast on the log scale
    and mapped back (a forecasting-time transform, not part of the world)."""
    if not log:
        return est.forecast(histories)
    positive = np.array([bool(np.all(h > 0)) for h in histories])
    series = [np.log(h) if p else h for h, p in zip(histories, positive)]
    fc = est.forecast(series)
    fc[positive] = np.exp(fc[positive])
    return fc


def evaluate(names, split: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    estimators = {n: library.pretrained(f"{NAME}_{n}").to("cuda") for n in names}
    for set_name, period in SETS:
        ds = load(set_name, period)
        if split == "val":
            ds = validation(ds)
        naive = score(ds, seasonal_naive(ds.histories, period, ds.horizon))
        row = {"dataset": ds.name, "series": len(ds), "snaive MASE": naive["mase"]}
        for n, est in estimators.items():
            sc = score(ds, est.forecast(ds.histories))
            row[f"{n} MASE"], row[f"{n} sMAPE"] = sc["mase"], sc["smape"]
        rows.append(row)
        print(row, flush=True)
    table = pd.DataFrame(rows).set_index("dataset")
    rel = {n: float((table[f"{n} MASE"] / table["snaive MASE"]).mean()) for n in names}
    stem = f"{split}_{'_'.join(names)}"
    if len(stem) > 120:                                   # Windows path limit: name long lists by hash
        import hashlib
        stem = f"{split}_{len(names)}models_{hashlib.sha256(' '.join(names).encode()).hexdigest()[:8]}"
    table.to_csv(RESULTS / f"{stem}.csv")
    (RESULTS / f"{stem}_summary.json").write_text(json.dumps(
        {"models": list(names), "mean MASE relative to seasonal naive": rel,
         "mean MASE": {n: float(table[f"{n} MASE"].mean()) for n in names}}, indent=1))
    print(table.round(3).to_string())
    print("mean MASE relative to seasonal naive:", {k: round(v, 4) for k, v in rel.items()})


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    command, args = sys.argv[1], sys.argv[2:]
    if command == "train":
        for n in args:
            train(n)
    elif command in ("validate", "test"):
        evaluate(args, "val" if command == "validate" else "test")
