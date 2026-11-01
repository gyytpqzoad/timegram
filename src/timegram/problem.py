"""A specification: model + observation + estimands + loss + objective,
and its solution by Neural Synthetic Pre-training.

    problem = (tg.ets("AAA", period=4)
               .observe(tg.missing(rate=0.1))
               .estimate(tg.forecast(h=8), tg.params("alpha", "beta", "gamma"))
               .loss(tg.absolute(scale="relative")))
    estimator = problem.pretrain(lengths=(10, 64), budget=2_000_000)

The objective says what the loss is averaged into. By default, the expected loss over
the world (ERM). `.objective(...)` asks for properties at each process instead: bias
control, per-process calibration, the worst case (`timegram.objective`).

Problems are immutable: every method returns a new problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from .model import Block, Combine, OneHot, named_params
from .objective import Criterion, Objective, _restore


# ---------------------------------------------------------------------------------------
# observation mechanism
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Missing:
    """Values hidden completely at random; the network receives a mask. The most recent
    observation is always kept. (In training this is also a regularizer, "observation
    dropout", which improves real-data accuracy.)"""

    rate: float = 0.1

    def apply(self, m: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        if self.rate == 0:
            return m
        keep = (torch.rand(m.shape, generator=gen, device=m.device) >= self.rate).float()
        keep[:, -1] = 1.0
        return m * keep


def missing(rate=0.1):
    return Missing(rate)


# ---------------------------------------------------------------------------------------
# estimands
# ---------------------------------------------------------------------------------------
# Parameters that change with the series' level or scale. Estimating them needs
# de-normalization rules (second stage); invariant parameters can be estimated now.
NOT_SCALE_FREE = {"level", "slope", "season", "level0", "slope0", "season0", "mean", "c", "b", "values", "scale",
                  "omega", "initial"}          # GARCH: omega and initial scale as |y|**power


@dataclass(frozen=True, repr=False)
class Forecast:
    """y_{T+1..T+h}: a point forecast, or the given quantiles of each future value
    (trained with the pinball loss)."""

    h: int | None = None
    quantiles: tuple[float, ...] = ()
    name = "forecast"
    __setstate__ = _restore

    def __repr__(self):                    # part of library keys: unchanged for point forecasts
        return f"Forecast(h={self.h}" + (f", quantiles={self.quantiles})" if self.quantiles else ")")

    def __post_init__(self):
        q = self.quantiles
        if q and not (all(0 < x < 1 for x in q) and all(a < b for a, b in zip(q, q[1:]))):
            raise ValueError(f"quantile levels must increase strictly inside (0, 1), not {q}")

    @property
    def width(self) -> int:
        """Network outputs this forecast takes."""
        return (self.h or 0) * max(len(self.quantiles), 1)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """Network outputs [N, width] -> forecasts [N, h, Q] (Q = 1 for a point forecast).
        Quantiles cannot cross: the lowest level plus positive increments."""
        n = raw.shape[0]
        if not self.quantiles:
            return raw.reshape(n, self.h, 1)
        raw = raw.reshape(n, self.h, len(self.quantiles))
        steps = torch.nn.functional.softplus(raw[..., 1:])
        return torch.cat([raw[..., :1], raw[..., :1] + steps.cumsum(-1)], -1)


@dataclass(frozen=True)
class Params:
    """Model parameters, by name or path suffix (all scale-free ones by default)."""

    names: tuple[str, ...] = ()
    name = "params"

    def resolve(self, model: Block) -> dict[str, object]:
        all_params = named_params(model)
        if not self.names:
            return {p: d for p, d in all_params.items() if p.rsplit(".", 1)[-1] not in NOT_SCALE_FREE}
        out = {}
        for want in self.names:
            hits = {p: d for p, d in all_params.items() if p == want or p.endswith("." + want)}
            if not hits:
                raise KeyError(f"no parameter {want!r}; the model has {list(all_params)}")
            for p, d in hits.items():
                if p.rsplit(".", 1)[-1] in NOT_SCALE_FREE:
                    raise NotImplementedError(
                        f"{p} depends on the series' level or scale; estimating it needs "
                        "de-normalization rules (second stage)")
                out[p] = d
        return out


@dataclass(frozen=True)
class Which:
    """Which operand of the model's top-level selection generated the series. The network
    outputs one probability per operand, trained with the log score: a strictly proper
    scoring rule, so under the world its optimum is the posterior probability of each
    operand given what the network sees, the normalized history and its mask. That equals
    the posterior given the raw series only when normalization discards nothing informative
    about the operand: selecting between N(0, 1) and N(0, 4) noise, x and 2x look the same
    after standardization although their raw-series posteriors differ. Finite training does
    not guarantee calibration either."""

    name = "which"

    def resolve(self, model: Block) -> tuple[str, int]:
        """(parameter path of the selection index, number of operands)."""
        if not (isinstance(model, Combine) and isinstance(model.schedule, OneHot)):
            raise ValueError("which() needs a model whose top block is tg.select(...)")
        path = next(p for p, d in named_params(model).items() if d is model.schedule.index)
        return path, len(model.operands)


def which():
    return Which()


def forecast(h=None, quantiles=None):
    return Forecast(h, tuple(float(q) for q in quantiles) if quantiles is not None else ())


def params(*names):
    return Params(tuple(names))


# ---------------------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Loss:
    """`scale="native"` (default) evaluates on the series' own scale. "relative" evaluates
    on the normalized scale (context mean / standard deviation), which weights every
    series equally: a relative-error criterion. "pinball" is the quantile
    loss, for quantile forecasts."""

    kind: str = "absolute"
    scale: str = "native"

    def __post_init__(self):
        if self.kind not in ("absolute", "squared", "pinball"):
            raise ValueError(f"unknown loss {self.kind!r}")
        if self.scale not in ("native", "relative"):
            raise ValueError(f'scale must be "native" or "relative", not {self.scale!r}')

    def __call__(self, pred, target, levels=None):
        """Elementwise loss. For pinball: pred [..., Q] quantiles of target [...]."""
        if self.kind == "pinball":
            err = target.unsqueeze(-1) - pred
            return torch.maximum(levels * err, (levels - 1) * err)
        err = pred - target
        return err.abs() if self.kind == "absolute" else err * err


def absolute(scale="native"):
    return Loss("absolute", scale)


def squared(scale="native"):
    return Loss("squared", scale)


def pinball(scale="native"):
    return Loss("pinball", scale)


# ---------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Problem:
    model: Block
    observation: tuple = (Missing(0.1),)
    estimands: tuple = ()
    loss_fn: Loss = field(default_factory=absolute)
    objective_spec: Objective = field(default_factory=Objective)
    __setstate__ = _restore

    def observe(self, *mechanisms):
        return replace(self, observation=tuple(mechanisms))

    def estimate(self, *estimands):
        return replace(self, estimands=tuple(estimands))._checked()

    def loss(self, loss: Loss):
        return replace(self, loss_fn=loss)._checked()

    def objective(self, *criteria: Criterion):
        """The training objective: a weighted sum of criteria, each aggregated over the
        world's processes (`timegram.objective`). Default: tg.risk(), i.e. ERM."""
        return replace(self, objective_spec=Objective(tuple(criteria)))._checked()

    def _checked(self, complete: bool = False) -> "Problem":
        """Estimands, loss and objective must fit together. Checked whenever one changes,
        as far as the parts given so far allow; `complete=True` (before training) also
        requires a loss for quantile forecasts."""
        f = self.forecast_estimand()
        quantiles = bool(f and f.quantiles)
        if quantiles and self.loss_fn.kind != "pinball" and (complete or self.loss_fn != Loss()):
            raise ValueError("quantile forecasts are trained with tg.pinball()")
        if self.loss_fn.kind == "pinball" and (complete or self.estimands):
            if not quantiles:
                raise ValueError("the pinball loss needs quantile forecasts: tg.forecast(h, quantiles=[...])")
            if any(isinstance(e, Params) for e in self.estimands):
                raise NotImplementedError("the pinball loss applies to quantile forecasts; "
                                          "estimate parameters in a separate problem")
        for c in self.objective_spec.criteria:
            c.check(self)
        return self

    def forecast_estimand(self) -> Forecast | None:
        """The (resolved) forecast estimand, if any."""
        est = self.resolved_estimands(self.default_horizon())
        return next((e for e in est if isinstance(e, Forecast)), None)

    def point_estimates(self) -> bool:
        """Whether the network outputs any point estimate (a parameter, a point forecast)."""
        est = self.resolved_estimands(self.default_horizon())
        return any(isinstance(e, Params) or (isinstance(e, Forecast) and e.h and not e.quantiles) for e in est)

    # -- resolved views -------------------------------------------------------------------
    def resolved_estimands(self, horizon_default: int):
        """Estimands with defaults filled in: if none were given, all
        (scale-free) parameters plus a forecast of two seasonal periods (or 8)."""
        est = self.estimands or (Forecast(), Params())
        return tuple(replace(e, h=horizon_default) if isinstance(e, Forecast) and e.h is None else e
                     for e in est)

    def default_horizon(self) -> int:
        periods = [getattr(b, "period", 1) or 1 for b in _walk(self.model)]
        return 2 * max(periods) if max(periods) > 1 else 8

    def layout(self):
        """(horizon, {param path: (start, size)}, output size) of the network head. The
        forecast takes the first h * Q outputs (Q quantile levels, 1 for a point forecast)."""
        est = self.resolved_estimands(self.default_horizon())
        f = next((e for e in est if isinstance(e, Forecast)), None)
        h = f.h if f else 0
        slots, k = {}, f.width if f else 0
        for e in est:
            if isinstance(e, Params):
                if any(isinstance(b, Combine) and isinstance(b.schedule, OneHot) for b in _walk(self.model)):
                    raise NotImplementedError("params of selected sub-models are second stage")
                for path, d in e.resolve(self.model).items():
                    size = int(np.prod(d.shape)) if d.shape else 1
                    slots[path] = (k, size)
                    k += size
            elif isinstance(e, Which):
                path, K = e.resolve(self.model)
                slots[path] = (k, K)
                k += K
        return h, slots, k

    def categorical(self) -> dict[str, int]:
        """Slots that are categorical (a `which()` estimand): path -> number of classes. Their
        target is one column, the drawn index; their output one probability per class."""
        est = self.resolved_estimands(self.default_horizon())
        return dict(e.resolve(self.model) for e in est if isinstance(e, Which))

    def describe(self) -> str:
        from .model import render_tree
        est = ", ".join(_describe(e) for e in self.resolved_estimands(self.default_horizon()))
        obs = ", ".join(repr(o) for o in self.observation) or "complete"
        text = (f"{render_tree(self.model)}\nobserve: {obs}\nestimate: {est}\n"
                f"loss: {self.loss_fn.kind} ({self.loss_fn.scale} scale)")
        if not self.objective_spec.is_default:
            text += f"\nobjective: {self.objective_spec!r}"
        return text

    def __str__(self):
        return self.describe()

    def pretrain(self, force_train: bool = False, save: bool = True, verbose: bool = True, **config):
        """Library lookup, else Neural Synthetic Pre-training.
        `config` overrides `TrainConfig` fields (lengths, budget, backend, ...)."""
        from . import library
        from .train import TrainConfig, train
        from .equivariance import warn_if_restricted
        cfg = TrainConfig(**config)
        warn_if_restricted(self.model)               # once, non-blocking; see timegram.equivariance
        key = library.key(self, cfg)
        if not force_train and library.exists(key):
            if verbose:
                print(f"timegram: found in the library ({key})")
            return library.load(key, problem=self)
        if force_train and save:
            library.check_compatible(key, self, cfg)  # fail before training, not after it
        estimator = train(self, cfg, verbose=verbose)
        estimator.key = key
        if save:
            library.save(estimator)
        return estimator


def _describe(e) -> str:
    if isinstance(e, Forecast):
        return f"forecast(h={e.h}" + (f", quantiles={list(e.quantiles)})" if e.quantiles else ")")
    if isinstance(e, Params):
        return f"params{e.names}"
    if isinstance(e, Which):
        return "which()"
    return type(e).__name__.lower()


def _walk(block: Block):
    seen, stack = set(), [block]
    while stack:
        b = stack.pop()
        if id(b) in seen:
            continue
        seen.add(id(b))
        yield b
        stack.extend(c for _, c in b.inputs())
        if hasattr(b, "down"):                       # a fed block's own definition (e.g. its period)
            stack.append(b.down)
        if hasattr(b, "tree"):
            stack.append(b.tree)
