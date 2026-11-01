"""The model graph: blocks, nesting (feed), and the combination operators.

Every block is a causal map from an input sequence to an output sequence.

* A `Source` has no input: noise, constant, trend, pattern.
* A `Transform` maps an input to an output: ar, ma, integrate, the compounds (ets,
  arima). Used on its own, it is fed by its own default noise source.
* `Feed(up, down)` is nesting, `up >> down`.
* `Combine(operands, schedule)` is every combination operator: addition, convex
  combination and selection. `Scale` multiplies a block by a parameter.

A block object is a node of a graph, not a value: using the same object twice means the
same realization (memoized by identity during simulation). `a.copy()` is an independent
copy.

Each block implements two simulation methods side by side:

* `run(s)` / `apply(s, x)`: the **reference** backend. One series, plain NumPy,
  written as directly from the definitions as possible. This is the default.
* `run_batch(b)` / `apply_batch(b, x)`: the optional **fast** backend. A batch of series
  in torch. It must agree with the reference (tests check this).

Both receive a simulation state (`simulate.Series` or `simulate.Batch`) that knows the
time axis, draws parameters (`s.draw(dist)`) and memoizes block outputs (`s.value(block)`).
Time is 1-based: t = 1 is the first observation; t <= 0 is the pre-sample period that
filters use to reach their steady state (see `presample`).
"""
from __future__ import annotations

import copy as _copy

import numpy as np
import torch

from .dists import Choice, Distribution, Fixed, as_dist


class Block:

    def typecheck(self, **kwargs):
        """Return opt-in structural diagnostics without sampling or training."""
        from .typecheck import typecheck
        return typecheck(self, **kwargs)

    def own_type_rules(self, context):
        return context.unknown()

    def own_linear_terms(self):
        """Exact direct-realization terms; this does not expand arbitrary filters."""
        return [(self, 1.0)]

    name = "block"

    # -- structure -----------------------------------------------------------------------
    def label(self) -> str:
        """One-line description for the tree printout."""
        return self.name

    def params(self) -> dict[str, Distribution]:
        """This block's own parameters, by name."""
        return {}

    def inputs(self) -> list[tuple[str, "Block"]]:
        """Child blocks, with the path segment that names them."""
        return []

    # -- simulation: identifiability and pre-sample (both backends) -----------------------
    def min_length(self, s) -> int:
        """Shortest series that identifies this block's realized structure."""
        return max((s.min_length(b) for _, b in self.inputs()), default=1)

    def presample(self, s) -> int:
        """Steps before t = 1 this block needs to reach its steady state."""
        return max((s.presample(b) for _, b in self.inputs()), default=0)

    def min_length_batch(self, b) -> torch.Tensor:
        vals = [b.min_length(c) for _, c in self.inputs()]
        return torch.stack(vals).amax(0) if vals else torch.ones(b.n, device=b.device)

    def presample_batch(self, b) -> int:
        return max((b.presample(c) for _, c in self.inputs()), default=0)

    # -- simulation: values ---------------------------------------------------------------
    def run(self, s) -> np.ndarray:
        raise NotImplementedError

    def run_batch(self, b) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no fast (batched) implementation")

    # -- grammar operators ------------------------------------------------------------------
    def __rshift__(self, other):
        return compose(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return add(self, scale(_as_block(other), -1.0))

    def __rsub__(self, other):
        return add(other, scale(self, -1.0))

    def __mul__(self, c):
        return product(self, c) if isinstance(c, Block) else scale(self, c)

    def __rmul__(self, c):
        return product(c, self) if isinstance(c, Block) else scale(self, c)

    def __neg__(self):
        return scale(self, -1.0)

    def __or__(self, other):
        return select(self, other, flatten=True)

    def copy(self) -> "Block":
        """An independent copy: same structure and distributions, new realizations."""
        return _copy.deepcopy(self)

    # -- problem builder: model.observe(...).estimate(...).loss(...) -----------
    def observe(self, *mechanisms):
        from .problem import Problem
        return Problem(self).observe(*mechanisms)

    def estimate(self, *estimands):
        from .problem import Problem
        return Problem(self).estimate(*estimands)

    def loss(self, loss):
        from .problem import Problem
        return Problem(self).loss(loss)

    def objective(self, *criteria):
        from .problem import Problem
        return Problem(self).objective(*criteria)

    def pretrain(self, **kwargs):
        from .problem import Problem
        return Problem(self).pretrain(**kwargs)

    def simulate(self, n: int = 1, length: int = 64, horizon: int = 0, seed: int = 0, **kw):
        """Convenience: simulate `n` series (reference backend unless backend="fast")."""
        from .simulate import simulate
        return simulate(self, n=n, lengths=(length, length), horizon=horizon, seed=seed, **kw)

    # -- display ------------------------------------------------------------------------------
    def __str__(self) -> str:
        return render_tree(self)

    def __repr__(self) -> str:
        return f"<{self.label()}>"

    def graph(self, expanded: bool = False):
        """The generating definition as a DAG (shared realizations are one node)."""
        from .graph import model_graph
        return model_graph(self, expanded=expanded)

    def plot_graph(self, path=None, expanded: bool = False, title=None):
        """Draw the definition; `expanded=True` shows named compounds as primitives."""
        from .graph import plot_graph
        return plot_graph(self, path=path, expanded=expanded, title=title)

    @property
    def param_table(self):
        """All parameters in the model, named by their path in the tree."""
        import pandas as pd
        rows = [(path, d.describe()) for path, d in named_params(self).items()]
        return pd.DataFrame(rows, columns=["parameter", "distribution"])


class Source(Block):
    """A block without input."""

    def own_type_rules(self, context):
        return context.unknown(sort="source")



class Transform(Block):
    """A block that maps an input sequence to an output sequence. Used on its own, its
    input is its own standard-normal noise."""

    def own_type_rules(self, context):
        return context.unknown(sort="filter")


    def __init__(self):
        from .blocks import noise
        self.source = noise()

    def apply(self, s, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_batch(self, b, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no fast (batched) implementation")

    def inputs(self):
        return [("input", self.source)]

    def run(self, s):
        return self.apply(s, s.value(self.source))

    def run_batch(self, b):
        return self.apply_batch(b, b.value(self.source))

    # A transform's own requirements, excluding its input; Feed combines them with the
    # requirements of whatever feeds it.
    def own_min_length(self, s) -> int:
        return 1

    def own_presample(self, s) -> int:
        return 0

    def own_min_length_batch(self, b) -> torch.Tensor:
        return torch.ones(b.n, device=b.device)

    def own_presample_batch(self, b) -> int:
        return 0

    def min_length(self, s):
        return max(s.min_length(self.source), self.own_min_length(s))

    def presample(self, s):
        return max(s.presample(self.source), self.own_presample(s))

    def min_length_batch(self, b):
        return torch.maximum(b.min_length(self.source), self.own_min_length_batch(b))

    def presample_batch(self, b):
        return max(b.presample(self.source), self.own_presample_batch(b))


class Feed(Block):
    """Nesting, `up >> down`: up's output is down's input."""

    def own_type_rules(self, context):
        return context.feed(self.up, self.down)


    def __init__(self, up: Block, down: Transform):
        if not isinstance(down, Transform):
            raise TypeError(f"cannot feed into {down.label()}: only transforms take an input "
                            "(sources and combinations do not). Mind the precedence: "
                            "`a >> b + c` is `a >> (b + c)`; write `(a >> b) + c`.")
        self.up, self.down = up, down
        self.name = down.name

    def label(self):
        return self.down.label()

    def params(self):
        return self.down.params()

    def inputs(self):
        return [("input", self.up)] + [io for io in self.down.inputs() if io[0] != "input"]

    def min_length(self, s):
        return max(s.min_length(self.up), self.down.own_min_length(s))

    def presample(self, s):
        return max(s.presample(self.up), self.down.own_presample(s))

    def min_length_batch(self, b):
        return torch.maximum(b.min_length(self.up), self.down.own_min_length_batch(b))

    def presample_batch(self, b):
        return max(b.presample(self.up), self.down.own_presample_batch(b))

    def run(self, s):
        return self.down.apply(s, s.value(self.up))

    def run_batch(self, b):
        return self.down.apply_batch(b, b.value(self.up))


class Scale(Block):
    """`c * a`: multiply a block's output by a parameter (fixed or random)."""

    def own_type_rules(self, context):
        return context.scale(self.block, self.c)

    def own_linear_terms(self):
        from .typecheck import _linear_terms
        if type(self) is Scale and isinstance(self.c, Fixed) and np.shape(self.c.value) == () and np.isfinite(self.c.value):
            return [(node, float(self.c.value) * c) for node, c in _linear_terms(self.block)]
        return [(self, 1.0)]


    name = "scale"

    def __init__(self, block: Block, c):
        self.block, self.c = block, as_dist(c)

    def label(self):
        return f"scale({self.c.describe()})"

    def params(self):
        return {"c": self.c}

    def inputs(self):
        return [("input", self.block)]

    def run(self, s):
        return s.draw(self.c) * s.value(self.block)

    def run_batch(self, b):
        c = b.draw(self.c)
        return c.unsqueeze(-1) * b.value(self.block) if c.dim() == 1 else c * b.value(self.block)


# ---------------------------------------------------------------------------------------
# combination operators: one Combine block, one schedule per operator
# ---------------------------------------------------------------------------------------
class Schedule:
    """Weights w_{k,t} of a Combine. `weights(s, K)` returns a list of K per-series
    weights (numbers, or arrays over time)."""

    def own_type_rules(self, context):
        return context.unknown(sort="schedule")


    name = "schedule"

    def params(self) -> dict[str, Distribution]:
        return {}

    def active(self, s, k: int) -> bool:
        """Whether operand k contributes to this series (lets selection skip operands)."""
        return True

    def describe(self) -> str:
        return self.name


class Ones(Schedule):

    def own_type_rules(self, context):
        return context.schedule("sum")

    name = "add"

    def weights(self, s, K):
        return [1.0] * K

    def weights_batch(self, b, K):
        return [1.0] * K


class Convex(Schedule):
    """Convex combination: weights fixed for the whole series, non-negative, summing to
    one. Two operands take a scalar weight w (giving w, 1 - w); K operands a weight
    vector (e.g. `tg.dirichlet(K)`)."""

    def own_type_rules(self, context):
        return context.schedule("convex", self.w)


    name = "convex"

    def __init__(self, w):
        self.w = as_dist(w)

    def params(self):
        return {"weight": self.w}

    def validate(self, K: int):
        """At construction: a scalar weight only for two operands, else a K-vector; fixed
        weights must lie in [0, 1] / be non-negative and sum to one."""
        shape = tuple(self.w.shape)
        if shape == () and K != 2:
            raise ValueError(f"a scalar convex weight needs exactly two operands, not {K}; "
                             f"give {K} weights (e.g. tg.dirichlet({K}))")
        if shape not in ((), (K,)):
            raise ValueError(f"convex weights must have shape ({K},), not {shape}")
        if isinstance(self.w, Fixed):
            _check_weights(np.asarray(self.w.value, float), scalar=shape == ())

    def weights(self, s, K):
        w = np.atleast_1d(np.asarray(s.draw(self.w), float))
        _check_weights(w, scalar=len(w) == 1)                # drawn weights too
        return [w[0], 1.0 - w[0]] if K == 2 and len(w) == 1 else list(w)

    def weights_batch(self, b, K):
        w = b.draw(self.w)
        _check_weights(w, scalar=w.dim() == 1)
        if w.dim() == 1:
            return [w.unsqueeze(-1), (1.0 - w).unsqueeze(-1)]
        return [w[:, k:k + 1] for k in range(K)]

    def describe(self):
        return f"convex(weight={self.w.describe()})"


def _check_weights(w, scalar: bool, tol: float = 1e-5):
    """Scalar weights in [0, 1], or weight vectors (last axis) non-negative summing to one."""
    lib = torch if torch.is_tensor(w) else np
    ok = bool(lib.isfinite(w).all()) and bool((w >= -tol).all())
    if not scalar:
        ok = ok and bool((abs(w.sum(-1) - 1) <= 1e-4).all())
    else:
        ok = ok and bool((w <= 1 + tol).all())
    if not ok:
        raise ValueError("convex weights must be non-negative and sum to one "
                         "(a scalar weight w must lie in [0, 1])")


class OneHot(Schedule):
    """Selection: one operand, drawn once per series, generates the whole series."""

    def own_type_rules(self, context):
        return context.schedule("select", self.index)


    name = "select"

    def __init__(self, probs):
        self.index = Choice(range(len(probs)), probs)

    def params(self):
        return {"which": self.index}

    def which(self, s) -> int:
        return int(s.draw(self.index))

    def active(self, s, k):
        return self.which(s) == k

    def weights(self, s, K):
        k = self.which(s)
        return [1.0 if j == k else 0.0 for j in range(K)]

    def weights_batch(self, b, K):
        k = b.draw(self.index)
        return [(k == j).float().unsqueeze(-1) for j in range(K)]

    def describe(self):
        from .dists import _fmt
        return f"select(probs={_fmt(self.index.probs)})"


class Combine(Block):
    """y_t = sum_k w_{k,t} y^(k)_t, with the weights given by a schedule."""

    def own_type_rules(self, context):
        return context.combine(self.operands, self.schedule)


    def __init__(self, operands, schedule: Schedule):
        self.operands = [_as_block(o) for o in operands]
        self.schedule = schedule
        self.name = schedule.name
        if hasattr(schedule, "validate"):
            schedule.validate(len(self.operands))

    def label(self):
        return self.schedule.describe()

    def params(self):
        return self.schedule.params()

    def inputs(self):
        return [(str(k), o) for k, o in enumerate(self.operands)]

    # structure of the realized series: only active operands count (selection)
    def min_length(self, s):
        return max(s.min_length(o) for k, o in enumerate(self.operands) if self.schedule.active(s, k))

    def presample(self, s):
        return max(s.presample(o) for k, o in enumerate(self.operands) if self.schedule.active(s, k))

    def run(self, s):
        w = self.schedule.weights(s, len(self.operands))
        # inactive operands are never simulated
        return sum(wk * s.value(o) for wk, o, k in zip(w, self.operands, range(len(w)))
                   if self.schedule.active(s, k))

    # fast backend. A selection simulates each operand only on the series that selected
    # it. If operands share blocks or parameters with each other or with the
    # rest of the model, every operand is simulated for every series instead and the
    # schedule picks: equally exact, just more work.
    def _subsets(self, b):
        if not isinstance(self.schedule, OneHot) or _shares_structure(b.root_model, self):
            return None
        k = b.draw(self.schedule.index)
        return [(j, (k == j).nonzero().squeeze(1)) for j in range(len(self.operands))]

    def min_length_batch(self, b):
        subsets = self._subsets(b)
        if subsets is None:
            vals = [b.min_length(o) for o in self.operands]
            if isinstance(self.schedule, OneHot):
                w = self.schedule.weights_batch(b, len(self.operands))
                return sum(wk.squeeze(-1) * v for wk, v in zip(w, vals))
            return torch.stack(vals).amax(0)
        out = torch.ones(b.n, device=b.device)
        for j, rows in subsets:
            if len(rows):
                out[rows] = b.subset(self, j, rows).min_length(self.operands[j])
        return out

    def presample_batch(self, b):
        subsets = self._subsets(b)
        if subsets is None:
            return max(b.presample(o) for o in self.operands)
        return max((b.subset(self, j, rows).presample(self.operands[j]) for j, rows in subsets if len(rows)),
                   default=0)

    def run_batch(self, b):
        subsets = self._subsets(b)
        if subsets is None:
            w = self.schedule.weights_batch(b, len(self.operands))
            return sum(wk * b.value(o) for wk, o in zip(w, self.operands))
        out = torch.zeros_like(b.t)
        for j, rows in subsets:
            if len(rows):
                out[rows] = b.subset(self, j, rows).value(self.operands[j])
        return out


class Product(Block):
    """y_t = a_t * b_t: the elementwise product of two processes, e.g. shocks times a
    volatility process (stochastic volatility) or a level times a seasonal factor."""

    name = "product"

    def own_type_rules(self, context):
        return context.product(self.operands)

    def __init__(self, a, b):
        self.operands = [_as_block(a), _as_block(b)]

    def label(self):
        return "product"

    def params(self):
        return {}

    def inputs(self):
        return [(str(k), o) for k, o in enumerate(self.operands)]

    def min_length(self, s):
        return max(s.min_length(o) for o in self.operands)

    def presample(self, s):
        return max(s.presample(o) for o in self.operands)

    def run(self, s):
        a, b = (s.value(o) for o in self.operands)
        return a * b

    def min_length_batch(self, b):
        return torch.stack([b.min_length(o) for o in self.operands]).amax(0)

    def presample_batch(self, b):
        return max(b.presample(o) for o in self.operands)

    def run_batch(self, b):
        x, y = (b.value(o) for o in self.operands)
        return x * y


class AtLeast(Block):
    """The same process with a minimum history length: shorter series are never sampled
    from it. Named models carry such floors (ETS, ARIMA); `at_least` gives one to any world,
    e.g. two full seasons plus the parameter count for a seasonal construction."""

    name = "at_least"

    def own_type_rules(self, context):
        return context.at_least(self.block, self.n)

    def __init__(self, block, n: int):
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError(f"at_least needs a positive integer length, not {n!r}")
        self.block, self.n = _as_block(block), int(n)

    def label(self):
        return f"at_least({self.n})"

    def params(self):
        return {}

    def inputs(self):
        return [("block", self.block)]

    def min_length(self, s):
        return max(self.n, s.min_length(self.block))

    def presample(self, s):
        return s.presample(self.block)

    def run(self, s):
        return s.value(self.block)

    def min_length_batch(self, b):
        return b.min_length(self.block).clamp_min(self.n)

    def presample_batch(self, b):
        return b.presample(self.block)

    def run_batch(self, b):
        return b.value(self.block)


def at_least(block, n: int) -> AtLeast:
    """`block` with a minimum history length of n points."""
    return AtLeast(block, n)


def product(a, b) -> Product:
    """Elementwise product of two blocks (`a * b`); `c * a` with a parameter c is scaling."""
    return Product(a, b)


def _reachable(root, stop=None) -> set[int]:
    """ids of the blocks and parameters reachable from `root`, not descending into `stop`."""
    seen, stack = set(), [root]
    while stack:
        x = stack.pop()
        if id(x) in seen or x is stop:
            continue
        seen.add(id(x))
        if isinstance(x, Block):
            stack.extend(c for _, c in x.inputs())
            stack.extend(x.params().values())
            if hasattr(x, "tree"):
                stack.append(x.tree)
            if isinstance(x, Combine):
                stack.extend(x.schedule.params().values())
        else:
            stack.extend(x.parents())
    return seen


_SHARING: dict[tuple[int, int], bool] = {}


def _shares_structure(root, select) -> bool:
    """Whether a selection's operands share any block or parameter with each other or
    with the rest of the model (then subsetting the batch would break the sharing)."""
    key = (id(root), id(select))
    if key not in _SHARING:
        outside = _reachable(root, stop=select) if root is not None else set()
        shared, union = False, set()
        for o in select.operands:
            r = _reachable(o)
            if r & union or r & outside:
                shared = True
                break
            union |= r
        _SHARING[key] = shared
    return _SHARING[key]


# ---------------------------------------------------------------------------------------
# operator functions
# ---------------------------------------------------------------------------------------
def _as_block(x) -> Block:
    if isinstance(x, Block):
        return x
    from .blocks import constant
    return constant(c=x)                       # `a + 3` adds a constant level


def compose(*blocks) -> Block:
    """Composition in signal order (`a >> b >> c`): each block's output is the next one's
    input (its innovations). Note the order: compose(a, b) applies a first, then b.

    * A process followed by transforms is a process: `tg.noise() >> tg.ar(1)`.
    * Transforms alone compose into a reusable transform with an open input:
      `filt = tg.ma(1) >> tg.ar(1)`, then `spiky >> filt`. Used on its own it is fed by
      its first stage's default noise. So composition is associative:
      `x >> (a >> b)` is `(x >> a) >> b`.

    Only the main input is composed. Driving a *parameter* with a process is a different
    operation (not available yet)."""
    if len(blocks) < 2:
        raise TypeError("compose needs at least two blocks")
    out = blocks[0]
    for b in blocks[1:]:
        out = _compose2(out, b)
    return out


def _compose2(a: Block, b: Block) -> Block:
    from .compounds import Composite
    if not isinstance(b, Transform):
        raise TypeError(f"cannot compose into {b.label()}: only transforms take an input (sources and "
                        "combinations do not). Mind the precedence: `a >> b + c` is `a >> (b + c)`; "
                        "write `(a >> b) + c`.")
    if isinstance(a, Transform):                      # transform >> transform: a new transform
        return Composite(*(a.stages if isinstance(a, Composite) else (a,)),
                         *(b.stages if isinstance(b, Composite) else (b,)))
    return Feed(a, b)


feed = compose      # earlier name, kept as an alias


def add(*blocks) -> Combine:
    """Addition (`a + b`); chained additions flatten into one n-ary sum."""
    ops = []
    for x in blocks:
        x = _as_block(x)
        if isinstance(x, Combine) and isinstance(x.schedule, Ones):
            ops.extend(x.operands)
        else:
            ops.append(x)
    return Combine(ops, Ones())


def scale(a: Block, c) -> Scale:
    """Scaling (`c * a`), c fixed or a parameter."""
    return Scale(a, c)


def convex(*blocks, weight=None) -> Combine:
    """Convex combination, weights fixed for the whole series: `convex(a, b, weight=w)` is
    w * a + (1 - w) * b (w ~ U(0, 1) by default); with K operands the weights are a
    probability vector (Dirichlet(1, ..., 1), uniform on the simplex, by default)."""
    from .dists import Dirichlet, Uniform
    if weight is None:
        weight = Uniform(0, 1) if len(blocks) == 2 else Dirichlet(len(blocks))
    return Combine(list(blocks), Convex(weight))


mix = convex        # earlier name, kept so existing experiment scripts still run


def select(*blocks, probs=None, flatten=False) -> Combine:
    """Selection (`a | b`): one operand, drawn once per series, generates the series.
    Chained `|` flattens: `a | b | c` is one selection with probability 1/3 each. An
    explicit `tg.select(s, b)` keeps a selection operand `s` whole (s and b equally likely)."""
    ops, ps = [], []
    given = None if probs is None else list(probs)
    for i, x in enumerate(blocks):
        p = 1.0 if given is None else given[i]
        if flatten and isinstance(x, Combine) and isinstance(x.schedule, OneHot):
            ops.extend(x.operands)
            ps.extend([1.0] * len(x.operands))
        else:
            ops.append(x)
            ps.append(p)
    return Combine(ops, OneHot(ps))


# ---------------------------------------------------------------------------------------
# display and parameter paths
# ---------------------------------------------------------------------------------------
def render_tree(root: Block) -> str:
    lines, seen = [], set()

    def walk(block, prefix, last, top):
        tag = " (shared)" if id(block) in seen else ""
        connector = "" if top else ("└── " if last else "├── ")
        lines.append(prefix + connector + block.label() + tag)
        if tag:
            return
        seen.add(id(block))
        kids = [b for _, b in block.inputs()]
        child_prefix = prefix + ("" if top else ("    " if last else "│   "))
        for i, kid in enumerate(kids):
            walk(kid, child_prefix, i == len(kids) - 1, False)

    walk(root, "", True, True)
    return "\n".join(lines)


def named_params(root: Block) -> dict[str, Distribution]:
    """{path: distribution} for every parameter, e.g. 'select.0.ets.alpha'. A parameter
    reached twice (shared) is listed once, at its first path."""
    out, seen_blocks, seen_params, keep = {}, set(), set(), []

    def as_chain(block):
        """A composite (`a >> b` on transforms) is named as the chain it stands for, so
        x >> (a >> b) and (x >> a) >> b get the same parameter paths, in the same order."""
        if getattr(block, "name", None) == "compose" and hasattr(block, "stages"):
            up, stages = block.source, block.stages
        elif isinstance(block, Feed) and getattr(block.down, "name", None) == "compose":
            up, stages = block.up, block.down.stages
        else:
            return block
        for stage in stages:
            up = Feed(up, stage)
            keep.append(up)                         # temporary nodes: keep their ids unique
        return up

    def walk(block, path):
        block = as_chain(block)
        if id(block) in seen_blocks:
            return
        seen_blocks.add(id(block))
        here = f"{path}.{block.name}" if path else block.name
        for pname, dist in block.params().items():
            if id(dist) not in seen_params:
                seen_params.add(id(dist))
                out[f"{here}.{pname}"] = dist
        for seg, kid in block.inputs():
            walk(kid, here if seg == "input" else f"{here}.{seg}")

    walk(root, "")
    return out
