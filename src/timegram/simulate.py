"""Simulating a model: the reference backend (default) and the fast backend (optional).

One simulated series, in three steps:

1. **Structure.** Draw whatever decides the realized structure (selection indices,
   random orders), and from it the minimum length `min_length` (an operational floor:
   enough observations to estimate the drawn structure, by the usual counting rules;
   not an identification theorem).
2. **Length.** Draw the observed length n uniformly on [max(lo, min_length), hi]: the
   model is drawn first, then a length at or above its floor.
3. **Values.** Simulate on the time axis t = 1 - presample, ..., n + horizon, where
   t = 1 is the first observation and t <= 0 is the pre-sample period that filters use
   to approach their steady state (a finite burn-in, not an exact stationary start).
   Return t = 1 .. n (observed) and n+1 .. n+horizon.

Parameters are drawn lazily and memoized by identity (`draw`), block outputs likewise
(`value`), so shared parameters and shared blocks share their realization.

**Replicates.** `replicates=R` draws each process once (structure, parameters, starting
states and length: everything `draw` returns) and simulates it R times with fresh
innovations. Rows come group-major: the R replicates of process 0, then of process 1, ...
Process-level properties of an estimator (its bias, coverage or risk at one process) are
measured over a group's replicates (as in the Simulacrum's two-stage simulation).

The reference backend (`Series`) does this for one series at a time in plain NumPy. The
fast backend (`Batch`) does the same for a whole batch in torch, on CPU or GPU.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import numpy as np
import torch

from .dists import Distribution, JointKey


class Series:
    """Simulation state of one series (reference backend)."""

    _batched = False

    def __init__(self, rng: np.random.Generator, process: "Series | None" = None):
        self.rng = rng
        self.t: np.ndarray | None = None          # time index of each simulated step
        self.horizon = 0                          # the last `horizon` steps are the future
        # a replicate of `process` shares its draws (the process), not its innovations
        self._draws: dict[int, object] = {} if process is None else process._draws
        self._keep: list = [] if process is None else process._keep  # keep drawn objects alive
        self._memo: dict[tuple, object] = {}
        self._bindings: dict[int, np.ndarray] = {}

    def draw(self, dist: Distribution):
        key = id(dist)
        if key not in self._draws:
            self._draws[key] = dist.sample(self.rng, self.draw)
            self._keep.append(dist)
        return self._draws[key]

    def drawn(self, dist: Distribution):
        """The draw of `dist` if it was used, else None."""
        return _drawn(self._draws, dist)

    def _memoized(self, kind, block, fn):
        key = (kind, id(block))
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def value(self, block) -> np.ndarray:
        return self._memoized("value", block, lambda: block.run(self))

    def min_length(self, block) -> int:
        return self._memoized("min_length", block, lambda: int(block.min_length(self)))

    def presample(self, block) -> int:
        return self._memoized("presample", block, lambda: int(block.presample(self)))

    def scope(self, bindings: dict) -> "Series":
        """Same series and parameters, fresh block memo, with placeholders bound (used by
        compound blocks to run their tree on a given input)."""
        child = Series.__new__(Series)
        child.rng, child.t, child.horizon = self.rng, self.t, self.horizon
        child._draws, child._keep = self._draws, self._keep
        child._memo, child._bindings = {}, {**self._bindings, **bindings}
        return child

    def bound(self, placeholder) -> np.ndarray:
        return self._bindings[id(placeholder)]


class Batch:
    """Simulation state of a batch of series (fast backend). Same interface as `Series`,
    with every value carrying a leading batch dimension."""

    _batched = True

    def __init__(self, n: int, gen: torch.Generator, device, root_model=None, replicates: int = 1):
        if n % replicates:
            raise ValueError(f"{n} rows do not split into groups of {replicates} replicates")
        self.n, self.gen, self.device = n, gen, torch.device(device)
        self.replicates = replicates              # rows per process (group-major)
        self.root_model = root_model              # for the shared-structure check of selections
        self.horizon = 0                          # the last `horizon` steps are the future
        self._t: torch.Tensor | None = None       # [n, W] per-series time index
        self._parent, self._rows = None, None     # set for a subset of a parent batch (weakref)
        self._draws: dict[int, torch.Tensor] = {}
        self._keep: list = []
        self._memo: dict[tuple, object] = {}
        self._bindings: dict[int, torch.Tensor] = {}

    @property
    def t(self) -> torch.Tensor | None:
        parent = self._parent() if self._parent is not None else None
        if self._t is None and parent is not None and parent.t is not None:
            self._t = parent.t[self._rows]
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    def subset(self, owner, k: int, rows: torch.Tensor) -> "Batch":
        """The rows of this batch that a selection assigned to operand k: a batch of their
        own, with their own draws (memoized, so every phase sees the same subset)."""
        key = ("subset", id(owner), k)
        if key not in self._memo:
            child = Batch(len(rows), self.gen, self.device, self.root_model, self.replicates)
            # a weak link back: parent -> child (memo) and child -> parent would be a
            # reference cycle, keeping every chunk's GPU tensors alive until a full gc
            child._parent, child._rows, child.horizon = weakref.ref(self), rows, self.horizon
            child._bindings = {i: v[rows] for i, v in self._bindings.items()}
            self._memo[key] = child
        return self._memo[key]

    def draw(self, dist: Distribution) -> torch.Tensor:
        key = id(dist)
        if key not in self._draws:
            R = self.replicates
            self._draws[key] = _per_row(dist.sample_batch(self.gen, self.n // R, self.device, self.draw),
                                        self.n // R, R)
            self._keep.append(dist)
        return self._draws[key]

    def per_process(self, values: torch.Tensor) -> torch.Tensor:
        """A per-process draw [n / R] repeated for its R replicate rows [n]."""
        return values.repeat_interleave(self.replicates, 0)

    def drawn(self, dist):
        return _drawn(self._draws, dist)

    def _memoized(self, kind, block, fn):
        key = (kind, id(block))
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def value(self, block) -> torch.Tensor:
        return self._memoized("value", block, lambda: block.run_batch(self))

    def min_length(self, block) -> torch.Tensor:
        return self._memoized("min_length", block, lambda: block.min_length_batch(self).float())

    def presample(self, block) -> int:
        return self._memoized("presample", block, lambda: int(block.presample_batch(self)))

    def scope(self, bindings: dict) -> "Batch":
        child = Batch(self.n, self.gen, self.device, self.root_model, self.replicates)
        child._t, child.horizon = self.t, self.horizon
        child._draws, child._keep = self._draws, self._keep
        child._bindings = {**self._bindings, **bindings}
        return child

    def bound(self, placeholder) -> torch.Tensor:
        return self._bindings[id(placeholder)]


def _per_row(v, n_processes: int, R: int):
    """Spread a draw over the replicate rows. A sampled draw has one value per process and
    is repeated R times; a derived one (an `Expr` or joint entry, computed from draws that
    are already per row) passes through."""
    if R == 1:
        return v
    if isinstance(v, dict):
        return {k: _per_row(x, n_processes, R) for k, x in v.items()}
    if torch.is_tensor(v) and v.dim() and v.shape[0] == n_processes:
        return v.repeat_interleave(R, 0)
    return v


def _drawn(draws, dist):
    """A parameter's draw if it was used; an entry of a joint draw counts as used."""
    if id(dist) in draws:
        return draws[id(dist)]
    if isinstance(dist, JointKey) and id(dist.joint) in draws:
        return draws[id(dist.joint)][dist.key]
    return None


# ---------------------------------------------------------------------------------------
@dataclass
class Simulation:
    """Simulated series, right-aligned: `context[i]` holds series i's observations in its
    last `lengths[i]` columns (NaN before), `target[i]` the following `horizon` values.
    `params[path]` holds each parameter's draw (NaN where a series did not use it).
    With `replicates` R > 1 the rows are R replicates of each process, group-major."""

    context: np.ndarray
    target: np.ndarray
    lengths: np.ndarray
    params: dict[str, np.ndarray] = field(default_factory=dict)
    replicates: int = 1

    def series(self, i: int) -> np.ndarray:
        """Observed values of series i followed by its target."""
        n = self.lengths[i]
        return np.concatenate([self.context[i, -n:], self.target[i]])


def simulate(model, n: int = 1, lengths=(64, 64), horizon: int = 0, seed=None, rng=None,
             backend: str = "reference", device="cpu", gen: torch.Generator | None = None,
             track_params: bool = True, validate: bool = True, replicates: int = 1) -> Simulation:
    """Simulate `n` processes of `model`, `replicates` series each (n * replicates rows).
    lengths = (lo, hi) range of observed lengths. `validate=False` skips the structural
    check, for callers that already made it (the training loop checks once, not once per
    batch)."""
    if validate:
        from ._dependencies import validate_acyclic
        validate_acyclic(model)
    if backend == "fast":
        return _simulate_fast(model, n * replicates, lengths, horizon, device, gen, seed, track_params,
                              replicates)
    if backend != "reference":
        raise ValueError(f"backend must be 'reference' or 'fast', not {backend!r}")
    from .model import named_params
    rng = rng if rng is not None else np.random.default_rng(seed)
    lo, hi = lengths
    paths = named_params(model) if track_params else {}
    rows = n * replicates
    ctx = np.full((rows, hi), np.nan)
    tgt = np.empty((rows, horizon))
    lens = np.empty(rows, dtype=int)
    params = {p: [] for p in paths}
    for i in range(rows):
        if i % replicates == 0:                              # a new process
            s = process = Series(rng)
            n_i = _draw_length(rng, lo, hi, s.min_length(model))
        else:                                                # a replicate: same draws, new noise
            s = Series(rng, process=process)
        pre = s.presample(model)
        s.t, s.horizon = np.arange(1 - pre, n_i + horizon + 1), horizon
        y = s.value(model)[pre:]
        ctx[i, hi - n_i:], tgt[i], lens[i] = y[:n_i], y[n_i:], n_i
        for p, d in paths.items():
            v = s.drawn(d)
            params[p].append(np.full(d.shape, np.nan) if v is None else np.asarray(v, float))
    return Simulation(ctx, tgt, lens, {p: _stack(v) for p, v in params.items()}, replicates)


def _stack(values: list) -> np.ndarray:
    """Per-series draws of one parameter as rows. A joint prior's entry can change length
    with the drawn structure (e.g. ARIMA coefficients, by order): pad with NaN."""
    shapes = {np.shape(v) for v in values}
    if len(shapes) == 1:
        return np.stack(values)
    width = max(np.size(v) for v in values)
    out = np.full((len(values), width), np.nan)
    for i, v in enumerate(values):
        flat = np.ravel(v)
        if not (np.ndim(v) == 0 and np.isnan(flat[0])):          # a scalar NaN marks "not used"
            out[i, :len(flat)] = flat
    return out


def _draw_length(rng, lo, hi, min_len):
    lo = max(lo, min_len)
    if lo > hi:
        raise ValueError(f"the drawn structure needs at least {min_len} observations, "
                         f"more than the maximum length {hi}")
    return int(rng.integers(lo, hi + 1))


FFT_PLAN_CACHE = 16


def _limit_fft_plan_cache(device):
    """Selection splits each batch into randomly sized subsets, so FFT filters see a new
    batch size almost every call. cuFFT caches a plan (with its workspace memory) per size,
    up to 4096 by default, which slowly fills the GPU and then spills into shared system
    memory. Keep only a few plans."""
    device = torch.device(device)
    if device.type == "cuda":
        cache = torch.backends.cuda.cufft_plan_cache[device.index if device.index is not None else 0]
        if cache.max_size != FFT_PLAN_CACHE:
            cache.max_size = FFT_PLAN_CACHE


def simulate_batch(model, n, lengths, horizon, gen, device, validate: bool = True, replicates: int = 1):
    """Fast backend core: n rows (n / replicates processes). Returns (rows [n, hi +
    horizon] right-aligned with NaN pads, lengths [n], Batch state)."""
    if validate:
        from ._dependencies import validate_acyclic
        validate_acyclic(model)
    _limit_fft_plan_cache(device)
    lo, hi = lengths
    b = Batch(n, gen, device, root_model=model, replicates=replicates)
    min_len = b.min_length(model).clamp_min(lo)
    if bool((min_len > hi).any()):
        raise ValueError(f"a drawn structure needs {int(min_len.max())} observations, "
                         f"more than the maximum length {hi}")
    u = b.per_process(torch.rand(n // replicates, generator=gen, device=b.device))   # length: per process
    n_i = (min_len + torch.floor(u * (hi - min_len + 1))).long()
    pre = b.presample(model)
    W = pre + hi + horizon
    col = torch.arange(W, device=b.device).unsqueeze(0)
    b.t = (col - (pre + hi - n_i).unsqueeze(1) + 1).float()
    b.horizon = horizon
    y = b.value(model)[:, pre:]
    rows = torch.where(b.t[:, pre:] >= 1, y, torch.full_like(y, float("nan")))
    return rows, n_i, b


def _simulate_fast(model, n, lengths, horizon, device, gen, seed, track_params, replicates=1):
    from .model import named_params
    if gen is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(0 if seed is None else seed)
    rows, n_i, b = simulate_batch(model, n, lengths, horizon, gen, device, validate=False,
                                  replicates=replicates)
    hi = lengths[1]
    params = {}
    if track_params:
        for p, d in named_params(model).items():
            v = b.drawn(d)
            if v is not None:
                params[p] = v.detach().cpu().numpy()
    rows = rows.cpu().numpy().astype(float)
    return Simulation(rows[:, :hi], rows[:, hi:], n_i.cpu().numpy(), params, replicates)
