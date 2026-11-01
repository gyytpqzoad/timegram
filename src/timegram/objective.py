"""Decision-theoretic objectives: properties of an estimator at each process, trained for.

A training batch holds J processes, each drawn once (structure, parameters, starting
states, length) and simulated R times with fresh innovations (`replicates`, see
`timegram.simulate`). Over a process's R replicates, properties of the estimator *at that
process* become measurable, and so trainable (the Simulacrum, sec. 3):

    risk         r_j = mean over the replicates of the loss
    bias         ||b_j||^2, b_j = mean over the replicates of the signed error of each
                 point estimate (a parameter, a point forecast). Forecast errors are on
                 the normalized scale and clipped like the loss (targets to +-target_clip,
                 errors to +-loss_clip): a robust bias, not the ordinary mean error
    miscoverage  (cov_j - tau)^2, cov_j = share of the replicates whose target falls
                 below the predicted tau-quantile

Each criterion is aggregated over the J processes, by its mean (the average over the
world) or by a smooth worst case, the log-sum-exp (1/psi) log sum_j exp(psi c_j), whose
gap to the maximum is at most log(J)/psi (the sampled minimax). The objective is the
weighted sum of the aggregated criteria: a Lagrangian relaxation of

    min_F  risk(F)   subject to   bias(omega, F) <= delta_b,  miscoverage(omega, F) <= delta_c.

    problem.objective(tg.risk())                                  # ERM: the default
    problem.objective(tg.risk(over=tg.worst(psi=10)))             # minimax
    problem.objective(tg.risk(), tg.bias(weight=1.0))             # bias control
    problem.objective(tg.risk(weight=0.03), tg.miscoverage(weight=0.97))

**Debiasing.** The plug-in estimate of a squared mean, (mean_r x_r)^2, is biased upward
by Var(x)/R. For coverage, that extra term rewards an indicator that varies less across
replicates, which widens intervals rather than calibrating them. By default both bias and
miscoverage use the unbiased
U-statistic (mean_r x_r)^2 - s_x^2 / R = mean over pairs r != r' of x_r x_r'.
`debiased=False` gives the plug-in form.

The coverage indicator 1{z <= q} has no gradient; training uses sigmoid(kappa (q - z)) on
the normalized scale. Evaluation must use the hard indicator (`hard_coverage`). A fixed
smoothing width is biased where the predictive spread is small on the normalized scale
(e.g. long integrated series, whose context spread grows with length while the one-step
noise does not): smoothing an upper-tail coverage lowers it, so the penalty pushes the
quantile out and the true coverage overshoots. So by default (`smoothing="spread"`) q - z
is measured in units of each forecast's own predicted spread; `smoothing="fixed"` is the
original form (kappa on the normalized scale).

**Recommended recipe** (experiments/ar5_calibration and experiments/tourism_calibration):
train with pinball first, then fine-tune gently (a tenth of the learning rate, a short
warmup, ~20M series) with tg.risk(weight=0.1) + tg.miscoverage(weight=0.9, levels=...)
on the levels of interest, debiased, spread smoothing, ~128 replicates per process.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields

import torch


def _restore(self, state: dict):
    """Unpickle an object saved before one of its fields existed: the field gets its
    default (library entries outlive timegram versions)."""
    for f in fields(self):
        if f.name not in state:
            state[f.name] = f.default if f.default is not MISSING else f.default_factory()
    self.__dict__.update(state)


# ---------------------------------------------------------------------------------------
# aggregation over processes
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Mean:
    """The average over the sampled processes."""

    def value(self, c: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        return c.mean()

    def weights(self, c: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        """d value / d c_j (what exact gradient accumulation needs)."""
        return torch.full_like(c, 1.0 / len(c))

    def __repr__(self):
        return "mean"


@dataclass(frozen=True)
class Worst:
    """Smooth worst case over the sampled processes: (1/psi) log sum_j exp(psi c_j).

    `anneal` > 0 grows psi geometrically from 1 to `psi` over that fraction of training
    (a moderate psi first, then sharpening toward the worst processes)."""

    psi: float = 10.0
    anneal: float = 0.0

    def __post_init__(self):
        if not self.psi > 0:
            raise ValueError(f"psi must be positive, not {self.psi}")
        if not 0 <= self.anneal <= 1:
            raise ValueError(f"anneal is a fraction of training, not {self.anneal}")

    def psi_at(self, progress: float) -> float:
        if self.anneal == 0 or progress >= self.anneal:
            return self.psi
        return self.psi ** (progress / self.anneal)

    def value(self, c, progress=1.0):
        psi = self.psi_at(progress)
        return torch.logsumexp(psi * c, 0) / psi

    def weights(self, c, progress=1.0):
        return torch.softmax(self.psi_at(progress) * c.detach(), 0)

    def __repr__(self):
        return f"worst(psi={self.psi:g}" + (f", anneal={self.anneal:g})" if self.anneal else ")")


def mean() -> Mean:
    return Mean()


def worst(psi: float = 10.0, anneal: float = 0.0) -> Worst:
    return Worst(psi, anneal)


# ---------------------------------------------------------------------------------------
# what a criterion sees: one batch, group-major
# ---------------------------------------------------------------------------------------
@dataclass
class Outputs:
    """Per-row training quantities of a batch of J processes x R replicates.

    losses     [N, K]       elementwise loss over every output (0 where a target is missing)
    errors     [N, P]       signed errors of the point estimates (NaN where missing), or None
    quantiles  [N, h, Q]    predicted quantiles (normalized scale), or None
    target     [N, h]       the realized future (normalized scale), for the quantiles
    levels     [Q]          the quantile levels
    """

    losses: torch.Tensor
    replicates: int
    errors: torch.Tensor | None = None
    quantiles: torch.Tensor | None = None
    target: torch.Tensor | None = None
    levels: torch.Tensor | None = None

    @property
    def processes(self) -> int:
        return self.losses.shape[0] // self.replicates

    def grouped(self, x: torch.Tensor) -> torch.Tensor:
        """[N, ...] -> [J, R, ...]."""
        return x.reshape(self.processes, self.replicates, *x.shape[1:])


def _squared_mean(x: torch.Tensor, debiased: bool) -> torch.Tensor:
    """(mean_r x)^2 over the replicate axis 1, or its unbiased U-statistic."""
    R = x.shape[1]
    m = x.mean(1)
    if not debiased:
        return m * m
    return m * m - x.var(1, unbiased=True) / R


# ---------------------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Criterion:
    weight: float = 1.0
    over: Mean | Worst = field(default_factory=Mean)

    name = "criterion"
    needs_replicates = False

    def per_process(self, out: Outputs) -> torch.Tensor:
        """The criterion at each process of the batch, [J]."""
        raise NotImplementedError

    def check(self, problem) -> None:
        """Raise if the criterion does not apply to `problem`'s estimands."""

    def __repr__(self):
        extra = "".join(f", {k}={v!r}" for k, v in self._extra().items())
        return f"{self.name}(weight={self.weight:g}, over={self.over!r}{extra})"

    def _extra(self) -> dict:
        return {}


@dataclass(frozen=True, repr=False)
class Risk(Criterion):
    """Expected loss at each process: r_j = mean over replicates and outputs."""

    name = "risk"

    def per_process(self, out):
        return out.grouped(out.losses.mean(1)).mean(1)


@dataclass(frozen=True, repr=False)
class Bias(Criterion):
    """Squared bias of the point estimates at each process, averaged over the estimates.
    Parameters as they are; point forecasts on the normalized scale, with the training
    clips applied (targets to +-target_clip, errors to +-loss_clip). So for forecasts
    this is a robust, clipped-error bias: it equals the ordinary bias only while errors
    stay inside the clips. Measure the ordinary bias separately when it matters."""

    debiased: bool = True

    name = "bias"
    needs_replicates = True

    def per_process(self, out):
        e = out.grouped(out.errors)                                  # [J, R, P]
        ok = torch.isfinite(e).all(1)                                # an estimate this process has
        sq = _squared_mean(torch.where(torch.isfinite(e), e, torch.zeros_like(e)), self.debiased)
        return torch.where(ok, sq, torch.zeros_like(sq)).sum(1) / ok.sum(1).clamp_min(1)

    def check(self, problem):
        if not problem.point_estimates():
            raise ValueError("bias needs point estimates (parameters or a point forecast), "
                             "not only quantiles")

    def _extra(self):
        return {} if self.debiased else {"debiased": False}


@dataclass(frozen=True, repr=False)
class Miscoverage(Criterion):
    """Squared deviation of each process's coverage from nominal, averaged over horizons
    and the quantile levels `levels` (default: all the forecast's levels). Coverage is
    of the normalized target clipped to +-target_clip, which changes it only for
    quantiles beyond the clip."""

    levels: tuple[float, ...] | None = None
    kappa: float = 50.0
    debiased: bool = True
    smoothing: str = "spread"        # "spread": kappa per unit of the forecast's own spread
                                     # (lowest to highest level); "fixed": kappa on the normalized scale

    name = "miscoverage"
    needs_replicates = True

    def __setstate__(self, state):
        state.setdefault("smoothing", "fixed")             # saved before smoothing existed: fixed
        _restore(self, state)

    def __post_init__(self):
        if not self.kappa > 0:
            raise ValueError(f"kappa must be positive, not {self.kappa}")
        if self.smoothing not in ("fixed", "spread"):
            raise ValueError(f'smoothing must be "fixed" or "spread", not {self.smoothing!r}')

    def per_process(self, out):
        cols = self._columns(out.levels)
        q, tau = out.quantiles[..., cols], out.levels[cols]
        gap = q - out.target.unsqueeze(-1)                                          # [N, h, Q]
        if self.smoothing == "spread":                                              # a fixed share of
            spread = out.quantiles[..., -1:] - out.quantiles[..., :1]               # the predictive
            gap = gap / spread.detach().clamp_min(1e-3)                             # spread, any length
        ind = torch.sigmoid(self.kappa * gap)
        dev = out.grouped(ind - tau)                                                # [J, R, h, Q]
        return _squared_mean(dev, self.debiased).flatten(1).mean(1)

    def _columns(self, levels: torch.Tensor) -> list[int]:
        if self.levels is None:
            return list(range(len(levels)))
        have = [round(float(x), 6) for x in levels]
        missing = [x for x in self.levels if round(float(x), 6) not in have]
        if missing:
            raise ValueError(f"miscoverage levels {missing} are not forecast quantiles {have}")
        return [have.index(round(float(x), 6)) for x in self.levels]

    def check(self, problem):
        f = problem.forecast_estimand()
        if f is None or not f.quantiles:
            raise ValueError("miscoverage needs quantile forecasts: tg.forecast(h, quantiles=[...])")
        if self.levels is not None:
            self._columns(torch.tensor(f.quantiles))
        if self.smoothing == "spread" and len(f.quantiles) < 2:
            raise ValueError('smoothing="spread" (the default) needs at least two quantile levels to '
                             'measure the spread; forecast more levels or use smoothing="fixed"')

    def _extra(self):
        extra = {} if self.levels is None else {"levels": tuple(self.levels)}
        if self.kappa != 50.0:
            extra["kappa"] = self.kappa
        if not self.debiased:
            extra["debiased"] = False
        if self.smoothing != "fixed":          # the original form keeps its original description,
            extra["smoothing"] = self.smoothing   # so library keys from before the option stay valid
        return extra


def risk(weight: float = 1.0, over: Mean | Worst | None = None) -> Risk:
    return Risk(weight, over or Mean())


def bias(weight: float = 1.0, over: Mean | Worst | None = None, debiased: bool = True) -> Bias:
    return Bias(weight, over or Mean(), debiased)


def miscoverage(weight: float = 1.0, over: Mean | Worst | None = None, levels=None,
                kappa: float = 50.0, debiased: bool = True, smoothing: str = "spread") -> Miscoverage:
    return Miscoverage(weight, over or Mean(), None if levels is None else tuple(levels), kappa, debiased,
                       smoothing)


# ---------------------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Objective:
    """sum_k weight_k * over_k( criterion_k at each process )."""

    criteria: tuple[Criterion, ...] = (Risk(),)

    def __post_init__(self):
        if not self.criteria:
            raise ValueError("an objective needs at least one criterion")

    @property
    def is_default(self) -> bool:
        return self.criteria == (Risk(),)

    @property
    def needs_replicates(self) -> bool:
        return any(c.needs_replicates for c in self.criteria)

    @property
    def separable(self) -> bool:
        """Whether it is a mean over processes (so a batch's gradient splits into chunks
        without a first pass)."""
        return all(isinstance(c.over, Mean) for c in self.criteria)

    def per_process(self, out: Outputs) -> list[torch.Tensor]:
        return [c.per_process(out) for c in self.criteria]

    def value(self, per_process: list[torch.Tensor], progress: float = 1.0) -> torch.Tensor:
        return sum(c.weight * c.over.value(v, progress) for c, v in zip(self.criteria, per_process))

    def weights(self, per_process: list[torch.Tensor], progress: float = 1.0) -> list[torch.Tensor]:
        """d value / d (criterion k at process j): the exact gradient of the objective is
        sum_kj weights_kj * d c_kj / d theta, which lets it accumulate over chunks."""
        return [c.weight * c.over.weights(v.detach(), progress) for c, v in zip(self.criteria, per_process)]

    def stats(self, per_process: list[torch.Tensor], progress: float = 1.0) -> dict[str, float]:
        out = {}
        for c, v in zip(self.criteria, per_process):
            v = v.detach()
            out[c.name] = float(c.over.value(v, progress))
            if len(self.criteria) > 1 or not isinstance(c.over, Mean):
                out[f"{c.name}_max"] = float(v.max())
        return out

    def __repr__(self):
        return " + ".join(repr(c) for c in self.criteria)


# ---------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------
def hard_coverage(quantiles: torch.Tensor, target: torch.Tensor, replicates: int) -> torch.Tensor:
    """Coverage of each process with the exact indicator: [N, h, Q], [N, h] -> [J, h, Q]."""
    ind = (target.unsqueeze(-1) <= quantiles).float()
    return ind.reshape(-1, replicates, *ind.shape[1:]).mean(1)
