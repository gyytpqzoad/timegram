"""Does per-series normalization cost anything for this world?

Estimators standardize every history (subtract its mean, divide by its standard
deviation) and map forecasts back. So forecasts are location-scale equivariant: shifting
a history by c shifts the forecast by c, whatever the network learned. For a world with
a known fixed mean (a zero-mean AR, GARCH innovations) the optimal forecast is not
shift-equivariant, and the network cannot represent it. Where the level is unknown and
arbitrary (an integrated component, a random level), equivariance restricts nothing
obvious. This is a structural screen, not a theorem: a bounded random level does not
make the simulation design translation invariant, and a zero-start integrated family
need not be closed under level shifts.

`level_status(model)` classifies a world structurally:

    "free"        a component with an unknown level (random level, integration) is present
                  in every realized series and is not shared with another term that could
                  cancel it: no restriction detected
    "fixed"       the level is known (e.g. zero mean): normalization restricts forecasts
    "partly"      some realizations have a fixed level (e.g. a stationary ARIMA branch
                  whose constant is exactly zero)
    "unverified"  a block of unknown kind is involved, or terms that carry the free level
                  share a component and could cancel (e.g. level - level)

`pretrain` warns (once, non-blocking) unless the status is "free".
"""
from __future__ import annotations

import warnings

from .dists import Fixed, JointKey


class EquivarianceWarning(UserWarning):
    """Per-series normalization may remove information this world's forecasts need."""


def _combine_select(statuses):
    if all(s == "free" for s in statuses):
        return "free"
    if all(s == "fixed" for s in statuses):
        return "fixed"
    return "unverified" if "unverified" in statuses else "partly"


def _combine_sum(statuses):
    for s in ("free", "partly", "unverified"):
        if s in statuses:
            return s
    return "fixed"


def _point_mass(dist) -> bool:
    """A level that is known in advance: fixed, or a law with zero spread."""
    from .blocks import _point_mass
    return isinstance(dist, Fixed) or (_point_mass(dist) is not None)


def level_status(block, bound=None) -> str:
    """Structural classification of the level of `block` (see the module docstring)."""
    return _level(block, bound or {})[0]


def _level(block, bound) -> tuple[str, frozenset]:
    """(status, witnesses): the witnesses are the blocks that make the level free."""
    from .blocks import (AR, MA, Constant, DampedTrend, Difference, Input, Integrate, Lag,
                         Noise, Normalize, Pattern, Trend)
    from .compounds import Compound
    from .model import Combine, Feed, OneHot, Scale, Transform

    def transform(down, up):
        status, witnesses = up
        if isinstance(down, Compound):
            return _level(down.tree, {**bound, id(down.input): up})
        if isinstance(down, Integrate):
            if isinstance(down.d, Fixed):
                return ("free", witnesses | {id(down)}) if float(down.d.value) >= 1 else up
            return ("free", witnesses | {id(down)}) if status == "free" else ("partly", witnesses | {id(down)})
        if isinstance(down, (AR, MA, Lag, Normalize)):
            return up                                                # preserve (or, Normalize: remove
        if isinstance(down, Difference):                             # equivariantly) the input's level
            return "fixed", frozenset()
        known = {"garch": up, "fractional_integrate": up, "fractional_difference": ("fixed", frozenset())}
        return known.get(down.name, ("unverified", witnesses))

    if id(block) in bound:
        return bound[id(block)]
    if isinstance(block, Input):
        return "fixed", frozenset()
    if isinstance(block, Feed):
        return transform(block.down, _level(block.up, bound))
    if isinstance(block, Combine):
        parts = [_level(o, bound) for o in block.operands]
        statuses = [s for s, _ in parts]
        witnesses = frozenset().union(*(w for _, w in parts))
        if isinstance(block.schedule, OneHot):
            return _combine_select(statuses), witnesses
        status = _combine_sum(statuses)
        shared = sum(len(w) for _, w in parts) > len(witnesses)      # a witness in two terms
        if shared and status in ("free", "partly"):
            return "unverified", witnesses                            # the terms could cancel
        return status, witnesses
    if isinstance(block, Scale):
        if isinstance(block.c, Fixed) and float(block.c.value) == 0:
            return "fixed", frozenset()                               # the zero process
        return _level(block.block, bound)
    if isinstance(block, Constant):
        if _point_mass(block.c):
            return "fixed", frozenset()
        if isinstance(block.c, JointKey) and getattr(block.c.joint, "may_be_zero", lambda k: False)(block.c.key):
            return "partly", frozenset({id(block)})
        return "free", frozenset({id(block)})
    if isinstance(block, (Noise, Pattern, Trend, DampedTrend)):
        return "fixed", frozenset()
    if isinstance(block, Transform):                                 # standalone: fed by its own noise
        return transform(block, _level(block.source, bound))
    return "unverified", frozenset()


MESSAGES = {
    "fixed": "this world has a known, fixed level (e.g. a zero mean). Centering can remove "
             "information needed for optimal forecasts.",
    "partly": "part of this world has a known, fixed level (e.g. a stationary component whose "
              "mean is exactly zero). For those series, centering can remove information needed "
              "for optimal forecasts.",
    "unverified": "whether this world's level is free could not be verified (it contains blocks "
                  "of unknown kind, or terms carrying the level that could cancel).",
}


def warn_if_restricted(model, stacklevel: int = 3) -> str:
    status = level_status(model)
    if status != "free":
        warnings.warn("Per-series centering and scaling restrict this forecaster to location-scale "
                      f"equivariant predictions; {MESSAGES[status]} Training continues with "
                      "normalization enabled.", EquivarianceWarning, stacklevel=stacklevel)
    return status
