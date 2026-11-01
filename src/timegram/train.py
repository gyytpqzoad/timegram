"""Neural Synthetic Pre-training: simulate, observe, compute targets, fit.

Every step draws fresh series from the model (unless the batch-reuse trick is on), so
the network never sees the same series twice. The defaults are the recipe used for the
backbone world, except the budget, which is small so that the default (reference) simulator
finishes in minutes; the quarterly notebook sets `backend="fast"` and the full budget.

Validation contract. The definition is checked once, when training starts; batches
assume it does not change. Built-in blocks never modify a model during simulation, but a
custom block could (e.g. by rewiring its inputs). `check_batches=True` re-checks the
definition before every batch, for debugging such extensions.

Batches and the objective. A batch is `batch_size` series: batch_size / replicates
processes with `replicates` series each (replicates = 1, the default, is ordinary i.i.d.
sampling). The problem's objective (`timegram.objective`) turns the per-series losses into
per-process criteria and aggregates them over the processes. With `accumulate` = k, one
optimizer step uses k batches; its gradient is exactly the gradient of the objective
over all of them (a worst-case objective does not split over batches, so it takes one
extra forward pass to fix the per-process gradient weights first).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral

import numpy as np
import torch

from .nets import ResMLP
from .objective import Outputs
from .optim import MuonAdamW, cosine_with_warmup
from .problem import Problem
from .simulate import Simulation, simulate, simulate_batch


@dataclass(frozen=True)
class TrainConfig:
    lengths: tuple[int, int] = (16, 128)    # observed lengths; the network's window is lengths[1]
    budget: int = 2_000_000                  # training series (samples)
    batch_size: int = 2048                   # series per batch (processes x replicates)
    chunk: int | None = None                 # series per simulator call (default: one batch); larger
                                             # chunks amortize the fast simulator's fixed cost.
                                             # Not a trick: every series is still used once.
    net: ResMLP = field(default_factory=ResMLP)
    optimizer: MuonAdamW = field(default_factory=MuonAdamW)
    warmup: int = 1_000_000                  # samples of linear learning-rate warmup (capped at 10 % of budget)
    grad_clip: float = 1.0
    target_clip: float = 10.0                # clip normalized targets to +-target_clip (relative scale)
    loss_clip: float = 5.0                   # clip each element's loss (relative scale)
    seed: int = 0
    backend: str = "reference"               # "reference" (default) or "fast"
    device: str = "auto"
    threads: int = 2                         # CPU threads for torch (GPU training needs few)
    reuse: int = 1                           # TRICK: train on each simulated batch `reuse` times
    ema: float = 0.0                         # > 0: the estimator is the exponential moving average of
                                             # the weights (decay `ema` per step); still one network
    log_every: int = 100                     # steps
    replicates: int = 1                      # series per process (the objective's R)
    accumulate: int = 1                      # batches per optimizer step (exact accumulation)
    init: str | None = None                  # library key or alias of an estimator to start from
    check_batches: bool = False              # debugging: re-validate the definition before every batch

    def __post_init__(self):
        for name in ("budget", "batch_size", "replicates", "accumulate", "chunk"):
            n = getattr(self, name)
            if name == "chunk" and n is None:
                continue
            if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
                raise ValueError(f"{name} must be a positive integer, not {n!r}")
        for name in ("batch_size", "chunk"):
            n = getattr(self, name)
            if n is not None and n % self.replicates:
                raise ValueError(f"{name}={n} must be a multiple of replicates={self.replicates}")
        if self.chunk is not None and self.chunk < self.batch_size:
            raise ValueError(f"chunk={self.chunk} must be >= batch_size={self.batch_size}; "
                             "smaller chunks cannot yield a complete training batch")

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def step_size(self) -> int:
        """Series per optimizer step."""
        return self.batch_size * self.accumulate

    def key_dict(self) -> dict:
        """What determines the trained estimator (not where or how fast it was computed).
        Fields added after the first release are left out at their defaults, so earlier
        library keys stay valid."""
        d = asdict(self)
        for k in ("backend", "device", "log_every", "threads", "check_batches"):
            d.pop(k)
        for f in fields(self):
            if f.name in _ADDED and d[f.name] == f.default:
                d.pop(f.name)
        d["net"], d["optimizer"] = self.net.to_dict(), self.optimizer.to_dict()
        return d


_ADDED = {"replicates", "accumulate", "init"}


# ---------------------------------------------------------------------------------------
# normalization: location-scale equivariance
# ---------------------------------------------------------------------------------------
def normalize(context: torch.Tensor):
    """context [B, L] with NaN where unobserved -> x (standardized, 0 where unobserved),
    mask, loc [B, 1], scale [B, 1]."""
    m = torch.isfinite(context)
    mf = m.float()
    c0 = torch.where(m, context, torch.zeros_like(context))
    n = mf.sum(1, keepdim=True).clamp_min(1)
    loc = c0.sum(1, keepdim=True) / n
    scale = (((c0 - loc) * mf) ** 2).sum(1, keepdim=True).div(n).sqrt().clamp_min(1e-6)
    x = torch.where(m, (context - loc) / scale, torch.zeros_like(context))
    return x, mf, loc, scale


# ---------------------------------------------------------------------------------------
# training batches
# ---------------------------------------------------------------------------------------
def training_batch(problem: Problem, cfg: TrainConfig, h: int, slots: dict, rng, gen, device, n: int):
    """Simulate n series (n / replicates processes, group-major): (context [n, L]
    NaN-padded, targets [n, h + parameters] with NaN where a target does not exist)."""
    R = cfg.replicates
    if cfg.backend == "fast":
        rows, lengths, b = simulate_batch(problem.model, n, cfg.lengths, h, gen, device,
                                          validate=cfg.check_batches, replicates=R)
        context, fc = rows[:, :cfg.lengths[1]], rows[:, cfg.lengths[1]:]
        par = {p: b.drawn(d) for p, d in _slot_dists(problem, slots).items()}
    else:
        sim: Simulation = simulate(problem.model, n=n // R, lengths=cfg.lengths, horizon=h, rng=rng,
                                   validate=cfg.check_batches, replicates=R)
        context = torch.as_tensor(sim.context, dtype=torch.float32, device=device)
        fc = torch.as_tensor(sim.target, dtype=torch.float32, device=device)
        par = {p: torch.as_tensor(sim.params[p], dtype=torch.float32, device=device) for p in slots}
        lengths = torch.as_tensor(sim.lengths, device=device)
    _check_finite(context, lengths, fc)
    cols = [fc]
    categorical = problem.categorical()
    for p, (_, size) in slots.items():
        size = 1 if p in categorical else size                   # a categorical target is its index
        v = par[p]
        cols.append(v.reshape(len(v), size) if v is not None else torch.full((len(fc), size), float("nan"),
                                                                                device=device))
    return context, torch.cat(cols, 1)


def _check_finite(context, lengths, future):
    """Every observed value and every future target must be finite. NaN is expected only
    as padding, before a series' first observation (the context is right-aligned, so
    series i is observed in its last lengths[i] columns). One check per chunk."""
    L = context.shape[1]
    observed = torch.arange(L, device=context.device) >= (L - lengths).unsqueeze(1)
    if bool(((~torch.isfinite(context)) & observed).any() | (~torch.isfinite(future)).any()):
        raise FloatingPointError("the simulator produced non-finite values (an explosive or numerically "
                                 "unstable model?); training on them would give a corrupt input or a NaN loss")


def batches(problem, cfg, h, slots, rng, gen, device):
    """Endless stream of training batches, simulated `chunk` series at a time (batches
    are cut at multiples of `replicates`, so a process never spans two batches)."""
    B = cfg.batch_size
    while True:
        context, targets = training_batch(problem, cfg, h, slots, rng, gen, device, cfg.chunk or B)
        for i in range(0, len(context) - B + 1, B):
            yield context[i:i + B], targets[i:i + B]


def _slot_dists(problem, slots):
    from .model import named_params
    allp = named_params(problem.model)
    return {p: allp[p] for p in slots}


def outputs(problem: Problem, pred, targets, loc, scale, h, cfg, replicates: int = 1) -> Outputs:
    """What the objective sees, per series: elementwise losses over all estimands, signed
    errors of the point estimates, and the predicted quantiles. Forecasts are compared on
    the relative or native scale (the loss's choice), parameters as they are (they are
    scale-free). NaN targets are ignored."""
    f = problem.forecast_estimand()
    width = f.width if f else 0
    losses, errors = [], []
    quantiles = z = levels = None
    if h:
        y = targets[:, :h]
        z = ((y - loc) / scale).clamp(-cfg.target_clip, cfg.target_clip)       # normalized target
        q = f.decode(pred[:, :width])                                          # [N, h, Q], normalized
        if f.quantiles:
            levels = torch.tensor(f.quantiles, device=pred.device)
            quantiles = q
        if problem.loss_fn.scale == "relative":
            loss = problem.loss_fn(q if f.quantiles else q[..., 0], z, levels).clamp(max=cfg.loss_clip)
        else:
            native = q * scale.unsqueeze(-1) + loc.unsqueeze(-1)
            loss = problem.loss_fn(native if f.quantiles else native[..., 0], y, levels)
        losses.append(loss.reshape(len(pred), -1))
        if not f.quantiles:
            errors.append((q[..., 0] - z).clamp(-cfg.loss_clip, cfg.loss_clip))
    categorical, column = problem.categorical(), h
    for path, (start, size) in problem.layout()[1].items():
        p = pred[:, start:start + size]
        if path in categorical:                                  # log score of the drawn index
            index = targets[:, column].long()
            losses.append(torch.nn.functional.cross_entropy(p, index, reduction="none").unsqueeze(1))
            column += 1
            continue
        y = targets[:, column:column + size]
        column += size
        ok = torch.isfinite(y)
        losses.append(torch.where(ok, problem.loss_fn(p, torch.where(ok, y, p)), torch.zeros_like(p)))
        errors.append(torch.where(ok, p - y, torch.full_like(p, float("nan"))))
    return Outputs(torch.cat(losses, 1), replicates, torch.cat(errors, 1) if errors else None,
                   quantiles, z, levels)


# ---------------------------------------------------------------------------------------
def train(problem: Problem, cfg: TrainConfig, verbose: bool = True):
    from ._dependencies import validate_acyclic
    from .estimator import Estimator
    validate_acyclic(problem.model)                             # once; batches skip it
    problem._checked(complete=True)
    objective = problem.objective_spec
    if objective.needs_replicates and cfg.replicates < 2:
        raise ValueError(f"the objective ({objective!r}) is measured over replicates of each process: "
                         "set replicates >= 2")
    device = cfg.resolved_device()
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=device if cfg.backend == "fast" else "cpu")
    gen.manual_seed(cfg.seed)
    mgen = torch.Generator(device=device)                       # observation mechanism draws
    mgen.manual_seed(cfg.seed + 1)

    h, slots, out = problem.layout()
    L = cfg.lengths[1]
    net = cfg.net.build(L, out).to(device)
    if cfg.init:
        _start_from(net, cfg.init, problem)
    opts = cfg.optimizer.build(net)
    steps = max(1, cfg.budget // cfg.step_size)
    warmup = min(cfg.warmup, cfg.budget // 10)
    averaged = torch.optim.swa_utils.AveragedModel(
        net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.ema)) if cfg.ema else None
    data = batches(problem, cfg, h, slots, rng, gen, device)
    history, running, t0 = [], torch.zeros((), device=device), time.perf_counter()
    if verbose:
        n_par = sum(p.numel() for p in net.parameters())
        extra = "" if objective.is_default else f"; objective {objective!r}"
        print(f"timegram: pre-training {n_par:,} parameters on {steps * cfg.step_size:,} simulated series "
              f"({cfg.backend} simulator, {device}{extra})")
    group = []
    for step in range(1, steps + 1):
        seen = (step - 1) * cfg.step_size
        progress = seen / cfg.budget
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * cosine_with_warmup(seen, cfg.budget, warmup)
        if (step - 1) % cfg.reuse == 0:                         # TRICK when reuse > 1
            group = [_observed(problem, *next(data), mgen) for _ in range(cfg.accumulate)]
        for o in opts:
            o.zero_grad(set_to_none=True)
        criteria = _backward(problem, net, group, h, cfg, progress)
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        for o in opts:
            o.step()
        if averaged is not None:
            averaged.update_parameters(net)
        running += objective.value(criteria, progress).detach()            # no GPU sync per step
        if step % cfg.log_every == 0 or step == steps:
            k = (step - 1) % cfg.log_every + 1
            rec = {"step": step, "samples": step * cfg.step_size, "loss": float(running) / k,
                   "seconds": round(time.perf_counter() - t0, 1)}
            if not objective.is_default:
                rec.update(objective.stats(criteria, progress))             # this step's criteria
            history.append(rec)
            running.zero_()
            if verbose:
                more = "".join(f"  {k} {v:.4g}" for k, v in rec.items() if k not in ("step", "samples", "loss", "seconds"))
                print(f"  step {step:6d}/{steps}  {rec['samples'] / 1e6:7.1f}M series  loss {rec['loss']:.4f}{more}  "
                      f"{rec['samples'] / max(rec['seconds'], 1e-9):,.0f} series/s", flush=True)
    if averaged is not None:
        net.load_state_dict(averaged.module.state_dict())
    net.eval()
    return Estimator(problem, net.cpu(), cfg, h, slots, history)


def _observed(problem, context, targets, mgen):
    """Normalize a batch and apply the observation mechanism (once, so that every pass
    over the batch sees the same observations)."""
    x, m, loc, scale = normalize(context)
    for mech in problem.observation:
        m = mech.apply(m, mgen)
    return x * m, m, loc, scale, targets


def _backward(problem, net, group, h, cfg, progress) -> list[torch.Tensor]:
    """Accumulate the exact gradient of the objective over the batches of one step; return
    the per-process criteria (detached, all batches).

    d objective / d theta = sum_kj w_kj d c_kj / d theta, with w_kj = d objective / d c_kj.
    For a mean over processes w is known in advance (weight / J); for a worst case it
    depends on every process's c_kj, so with several batches a first pass without
    gradients fixes it."""
    objective = problem.objective_spec

    def criteria(batch):
        x, m, loc, scale, targets = batch
        return objective.per_process(outputs(problem, net(x, m), targets, loc, scale, h, cfg, cfg.replicates))

    weights = None
    if len(group) > 1 and not objective.separable:
        with torch.no_grad():
            first = [criteria(b) for b in group]
        weights = objective.weights([torch.cat(parts) for parts in zip(*first)], progress)
    collected, start = [], 0
    for batch in group:
        c = criteria(batch)
        J = len(c[0])
        if weights is None:                                  # a mean, or a single batch
            w = [wk / len(group) for wk in objective.weights(c, progress)]
        else:
            w = [wk[start:start + J] for wk in weights]
        sum((wk * ck).sum() for wk, ck in zip(w, c)).backward()
        collected.append([ck.detach() for ck in c])
        start += J
    return [torch.cat(parts) for parts in zip(*collected)]


def _start_from(net, init: str, problem: Problem):
    """Warm start from a library estimator with the same network and output layout."""
    from . import library
    base = library.load(library.resolve(init))
    if base.problem.layout() != problem.layout():
        raise ValueError(f"init {init!r} has another output layout: {base.problem.layout()} vs {problem.layout()}")
    net.load_state_dict(base.net.state_dict())
