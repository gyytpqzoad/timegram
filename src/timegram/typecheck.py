"""Opt-in, block-local structural checks; no simulation or training is performed.

Reports are scoped evidence, not validity or identification certificates. Unknown rules
remain unverified. Integration bounds describe conditional linear filter factors, not
exact finite-window stationarity or the effects of initialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    subject_ref: object
    claim: str
    status: str
    value: object = None
    scope: str = "structural"
    premises: tuple[str, ...] = ()
    rule: str = "timegram.typecheck/1"
    evidence: str = ""
    open_obligations: list[str] = field(default_factory=list)
    diagnostic_severity: str = "info"
    schema: str = "timegram.claim/1"


@dataclass(frozen=True)
class TypecheckReport:
    claims: tuple[ClaimRecord, ...]
    root_ref: object
    schema: str = "timegram.typecheck/1"

    @property
    def diagnostics(self):
        return tuple(c for c in self.claims if c.diagnostic_severity != "info" or c.status != "established")

    @property
    def has_errors(self):
        return any(c.diagnostic_severity == "error" for c in self.claims)

    @property
    def has_unverified(self):
        return any(c.status == "unverified" for c in self.claims)


@dataclass
class _Info:
    sort: str = "unknown"
    shape: tuple | None = None
    kind: str = "unknown"
    fixed: object = None
    interval: tuple | None = None
    entries: dict = field(default_factory=dict)
    iid: bool | None = None
    integer: bool | None = None
    bound: dict | None = None
    moment: ClaimRecord | None = None
    mode: str = "unknown"
    finite: bool | None = None


def implied_status(premises):
    """A failed sufficient premise blocks a proof; it does not refute its conclusion."""
    return "established" if all(p.status == "established" for p in premises) else "unverified"


class _Engine:
    def __init__(self, history_length, policies, identity_record):
        self.history_length, self.policies = history_length, policies
        self.identity_record = identity_record
        self.claims, self.refs, self.memo, self.active = [], {}, {}, set()

    def ref(self, obj):
        if id(obj) not in self.refs:
            ref = None
            if self.identity_record is not None:
                try:
                    ref = self.identity_record.subject_ref(obj)
                except (AttributeError, KeyError, ValueError):
                    pass
            self.refs[id(obj)] = ref if ref is not None else {
                "binding": f"local{len(self.refs)}", "provisional": True}
        return self.refs[id(obj)]

    def emit(self, obj, claim, status, **kw):
        record = ClaimRecord(f"claim{len(self.claims)}", self.ref(obj), claim, status, **kw)
        self.claims.append(record)
        return record

    def visit(self, obj, bindings=None, supplied=None, arity=None):
        bindings = bindings or {}
        if id(obj) in bindings:
            return bindings[id(obj)]
        key = (id(obj), tuple(sorted((k, id(v)) for k, v in bindings.items())), id(supplied), arity)
        if key in self.memo:
            return self.memo[key]
        if key in self.active:
            self.emit(obj, "acyclic", "refuted", evidence="dependency cycle", diagnostic_severity="error")
            return _Info()
        self.active.add(key)
        self.ref(obj)
        context = _Context(self, obj, bindings, supplied, arity)
        hook_name = "own_support" if hasattr(obj, "own_support") else "own_type_rules"
        hook = getattr(obj, hook_name, None)
        try:
            # A custom subclass does not inherit a built-in semantic certificate just
            # because it retains the method while overriding its sampler/recurrence.
            declared = hook_name in type(obj).__dict__
            builtin = type(obj).__module__.startswith("timegram.")
            info = hook(context) if callable(hook) and (declared or builtin) else context.unknown()
            if not isinstance(info, _Info):
                raise TypeError("rule must return a typecheck info record")
        except Exception as exc:
            self.emit(obj, "rule_execution", "unverified", evidence=f"{type(exc).__name__}: {exc}",
                      open_obligations=["repair or provide the extension rule"], diagnostic_severity="warning")
            info = _Info()
        self.active.remove(key)
        self.memo[key] = info
        return info


class _Context:
    """Helpers are chosen by local hooks; there is no central class-to-rule visitor."""
    def __init__(self, engine, subject, bindings, supplied, arity):
        self.engine, self.subject = engine, subject
        self.bindings, self.supplied, self.arity = bindings, supplied, arity
        self.shape_premises = []

    def claim(self, name, status="established", **kw):
        return self.engine.emit(self.subject, name, status, **kw)

    def visit(self, obj, **kw):
        return self.engine.visit(obj, bindings=self.bindings, **kw)

    def unknown(self, sort="unknown"):
        self.claim("semantic_rule", "unverified", evidence="no declared supported rule",
                   open_obligations=["supply a scoped local rule"])
        return _Info(sort=sort)

    def parameter(self, shape, **kw):
        shape = tuple(shape)
        self.claim("sort", value="parameter")
        self.claim("event_shape", value=shape, evidence="declared event shape; no sampler execution")
        return _Info(sort="parameter", shape=shape, **kw)

    def fixed(self, value, shape):
        v = np.asarray(value)
        finite = bool(np.issubdtype(v.dtype, np.number) and np.isfinite(v).all())
        self.claim("finite_parameter", "established" if finite else "refuted",
                   evidence="fixed value", diagnostic_severity="info" if finite else "error")
        numeric = v.astype(float) if finite else None
        integral = bool(finite and v.dtype.kind != "b" and np.equal(numeric, np.floor(numeric)).all())
        interval = (float(numeric.min()), float(numeric.max())) if finite and v.size else None
        return self.parameter(shape, kind="fixed", fixed=value, interval=interval, integer=integral, finite=finite)

    def real(self, *, interval=None, iid=True, finite_variance=True, integer=False):
        info = self.parameter((), kind="real", interval=interval, iid=iid, integer=integer, finite=True)
        info.moment = self.claim("finite_second_moment", "established" if finite_variance else "refuted",
                                scope="ideal_law", evidence="declared distribution family")
        return info

    def finite_choice(self, values):
        v = np.asarray(values)
        numeric = v.dtype.kind in "iuf" and bool(np.isfinite(v).all())
        integral = bool(numeric and np.equal(v, np.floor(v)).all())
        return self.parameter((), kind="finite_choice", iid=True, integer=integral, finite=bool(numeric),
                              interval=(float(v.min()), float(v.max())) if numeric and v.size else None)

    def region(self, shape, kind):
        self.claim("declared_region", value=kind, scope="ideal_law_almost_sure",
                   evidence="built-in sampler declaration; no uniform numerical margin or RNG endpoint guarantee")
        return self.parameter(shape, kind=kind, finite=True)

    def joint(self, entries=None):
        self.claim("sort", value="joint_parameter")
        return _Info(sort="parameter", kind="joint", entries=entries or {})

    def joint_key(self, joint, key):
        parent = self.visit(joint)
        if key not in joint.keys:
            self.claim("joint_key", "refuted", value=key, evidence="not among declared keys",
                       diagnostic_severity="error")
            return _Info(sort="parameter")
        shape = joint.shape_for(key)
        self.claim("joint_key", value=key, evidence="shape_for; parent draw remains shared")
        facts = parent.entries.get(key, {})
        return self.parameter(shape, **facts)

    def expression(self, args, shape):
        shapes = [self.visit(a).shape for a in args]
        if all(s is not None for s in shapes):
            try:
                actual = np.broadcast_shapes(*shapes)
                self.claim("broadcast_shape", value=actual)
            except ValueError:
                self.claim("broadcast_shape", "refuted", value=shapes, diagnostic_severity="error")
                actual = None
        else:
            actual = None
        self.claim("parameter_region", "unverified", evidence="Expr interval arithmetic deferred")
        return _Info(sort="parameter", shape=actual)

    def shape(self, parameter, allowed, name):
        info = self.visit(parameter)
        status = "unverified" if info.shape is None else (
            "established" if info.shape in allowed else "refuted")
        premise = self.claim(f"shape:{name}", status, value=info.shape,
                   evidence=f"consumer allows {allowed}",
                   diagnostic_severity="error" if status == "refuted" else "info")
        self.shape_premises.append(premise)
        return info

    def process(self, sort="source", bound=None, moment=None, inputs=(), finite=None):
        self.claim("sort", value=sort)
        structural = [c for c in self.engine.claims if c.diagnostic_severity == "error"]
        shape_ok = (not structural and all(p.shape == () for p in inputs)
                    and all(p.status == "established" for p in self.shape_premises))
        if not shape_ok:
            bound, finite = None, None
        self.claim("event_shape", "established" if shape_ok else "unverified", value=() if shape_ok else None,
                   evidence="scalar output requires compatible parameter and input shapes",
                   premises=tuple(c.claim_id for c in self.shape_premises),
                   open_obligations=[] if shape_ok else ["structural shape premises are not established"])
        self.claim("integration_upper_bound", "established" if bound is not None else "unverified", value=bound,
                   premises=tuple(c.claim_id for c in self.shape_premises),
                   scope="conditional_linear_stochastic_filter_factors",
                   evidence="upper bound only; deterministic terms, initialization and finite precision excluded",
                   open_obligations=[] if bound is not None else ["supported input and linear rule required"])
        return _Info(sort=sort, shape=() if shape_ok else None, bound=bound, moment=moment, finite=finite)

    def noise(self, law, scale):
        info = self.visit(law)
        scale_info = self.shape(scale, ((),), "scale")
        iid = self.claim("iid_sampler", "established" if info.iid else "unverified",
                         evidence="innovation sampling capability, separate from parameter shape")
        self.shape_premises.append(iid)
        return self.process(bound={} if info.iid else None, moment=info.moment,
                            inputs=(info,), finite=_finite_all((info, scale_info)))

    def deterministic(self, parameters, period=None):
        infos = [self.shape(p, allowed, name) for name, (p, allowed) in parameters.items()]
        if period is not None:
            self.period(period)
        # Only the stochastic transfer contribution is zero. This does not classify
        # a deterministic trend as a stationary observed process.
        return self.process(bound={}, finite=_finite_all(infos))

    def input(self):
        if self.supplied is not None:
            return self.supplied
        self.claim("input_binding", "unverified", evidence="open compound input")
        return _Info(sort="source", shape=())

    def incoming(self, source):
        return self.supplied if self.supplied is not None else self.visit(source)

    def filter(self, source, *, preserve=False):
        incoming = self.incoming(source)
        return self.process(sort="filter", bound=incoming.bound if preserve else None, moment=incoming.moment,
                            inputs=(incoming,), finite=incoming.finite if preserve else None)

    def feed(self, up, down):
        from .model import Transform
        incoming = self.visit(up)
        if not isinstance(down, Transform):
            self.claim("feed_sort", "refuted", evidence="right operand must accept an input",
                       diagnostic_severity="error")
            return _Info()
        if incoming.sort in ("parameter", "schedule"):
            self.claim("feed_sort", "refuted", evidence="left operand must be a process or filter",
                       diagnostic_severity="error")
            return _Info()
        self.claim("feed_sort", "established" if incoming.sort in ("source", "filter") else "unverified",
                   evidence="process into filter; unknown input semantics remain open")
        result = self.visit(down, supplied=incoming)
        return self.process(bound=result.bound, moment=result.moment, inputs=(result,), finite=result.finite)

    def compound(self, source, placeholder, tree):
        incoming = self.incoming(source)
        bindings = {**self.bindings, id(placeholder): incoming}
        result = self.engine.visit(tree, bindings=bindings)
        return self.process(sort="filter", bound=result.bound, moment=result.moment, inputs=(result,), finite=result.finite)

    def coefficients(self, source, parameter, order, period, role):
        p = self.shape(parameter, ((order,), ()) if order == 1 else ((order,),), role)
        self.period(period)
        wanted = "stationary" if role == "phi" else "invertible"
        status, evidence = "unverified", "support not known to lie in the required region"
        if p.kind in (wanted, "l1"):
            status, evidence = "established", "declared almost-sure region in matching sign convention"
        elif p.kind == "fixed":
            c = np.atleast_1d(np.asarray(p.fixed, float))
            if role == "theta":
                c = -c
            if len(c) == order and np.isfinite(c).all():
                roots = np.roots(np.r_[1., -c]) if order else np.array([])
                modulus = float(np.max(np.abs(roots))) if roots.size else 0.
                status = "established" if modulus < 1 - 1e-9 else ("refuted" if modulus >= 1 else "unverified")
                evidence = f"fixed float64 companion roots; max modulus={modulus:g}; tolerance=1e-9"
        elif order == 1 and p.interval is not None:
            lo, hi = p.interval
            if -1 < lo <= hi < 1:
                status, evidence = "established", "scalar interval strictly inside (-1,1)"
        if any(c.status != "established" for c in self.shape_premises):
            status, evidence = "unverified", "required coefficient shape is not established"
        region = self.claim(f"{wanted}_coefficients", status, evidence=evidence,
                            premises=tuple(c.claim_id for c in self.shape_premises),
                            scope="machine_predicate" if p.kind == "fixed" else "ideal_law_almost_sure",
                            diagnostic_severity="warning" if status == "refuted" else "info")
        incoming = self.incoming(source)
        # Stability is needed to retain an integration-factor summary for AR. A
        # finite MA does not need invertibility to retain that upper bound.
        bound = incoming.bound if role == "theta" or region.status == "established" else None
        return self.process(sort="filter", bound=bound, moment=incoming.moment, inputs=(incoming,),
                            finite=_finite_all((incoming, p)))

    def integer(self, parameter, name):
        p = self.shape(parameter, ((),), name)
        shape_premise = self.shape_premises[-1]
        valid = p.shape == () and p.integer is True and p.interval is not None and p.interval[0] >= 0
        false = p.shape == () and (p.integer is False or (p.interval is not None and p.interval[0] < 0))
        status = "established" if valid else "refuted" if false else "unverified"
        self.claim(f"nonnegative_integer:{name}", status, value=p.interval,
                   premises=(shape_premise.claim_id,),
                   evidence="declared or fixed integer support", diagnostic_severity="error" if false else "info")
        return int(p.interval[1]) if valid else None

    def integrate(self, source, order, period):
        upper = self.integer(order, "d")
        self.period(period)
        incoming = self.incoming(source)
        bound = None
        if incoming.bound is not None and upper is not None:
            bound = dict(incoming.bound)
            bound[period] = bound.get(period, 0) + upper
        return self.process(sort="filter", bound=bound, inputs=(incoming,),
                            finite=incoming.finite if upper is not None else None)

    def difference(self, source, order, period):
        self.period(period)
        incoming = self.incoming(source)
        if incoming.bound and incoming.bound.get(period, 0) > 0 and order > 0:
            self.claim("possible_cancellation", evidence="matching integration/difference factors; no rewrite",
                       scope="syntax_only", open_obligations=["boundary conventions and pathwise equality not checked"])
        # Retaining the original bound is conservative; transfer reduction is deferred.
        return self.process(sort="filter", bound=incoming.bound, inputs=(incoming,), finite=incoming.finite)

    def period(self, period):
        self.claim("declared_period", value=period,
                   evidence="declared lag/season period, not necessarily the fundamental period")
        e = self.engine
        if ("history_covers_declared_cycle" in e.policies and e.history_length is not None
                and period > e.history_length):
            self.claim("history_covers_declared_cycle", "refuted", value={"length": e.history_length, "period": period},
                       scope="policy", evidence="less than one declared cycle; not an impossibility theorem",
                       diagnostic_severity="warning")

    def scale(self, block, coefficient):
        p = self.shape(coefficient, ((),), "scale")
        incoming = self.visit(block)
        bound = {} if p.kind == "fixed" and np.asarray(p.fixed).shape == () and float(p.fixed) == 0 else incoming.bound
        return self.process(bound=bound, inputs=(incoming,), finite=_finite_all((incoming, p)))

    def schedule(self, mode, weight=None):
        self.claim("sort", value="schedule")
        if weight is not None:
            p = self.visit(weight)
            if mode == "convex":
                allowed = ((self.arity,), ()) if self.arity == 2 else ((self.arity,),)
                self.shape(weight, allowed, "weights")
                status = "unverified"
                if p.kind == "simplex" and p.shape == (self.arity,):
                    status = "established"
                elif p.kind == "fixed":
                    w = np.asarray(p.fixed, float)
                    ok = np.isfinite(w).all() and (w >= 0).all()
                    ok = ok and (float(w) <= 1 if w.shape == () else abs(w.sum() - 1) <= 1e-4)
                    status = "established" if ok else "refuted"
                elif p.shape == () and self.arity == 2 and p.interval is not None:
                    status = "established" if 0 <= p.interval[0] <= p.interval[1] <= 1 else "unverified"
                self.claim("simplex_weights", status, scope="parameter_region",
                           diagnostic_severity="error" if status == "refuted" else "info")
            elif mode == "select":
                self.shape(weight, ((),), "selection_index")
                ok = p.integer is True and p.interval is not None and 0 <= p.interval[0] <= p.interval[1] < self.arity
                self.claim("selection_index", "established" if ok else "unverified", value=p.interval)
        return _Info(sort="schedule", mode=mode)

    def product(self, operands):
        """Visit both factors (their diagnostics stay in the report); a product of processes
        inherits no linear or moment summary, so those claims stay unverified."""
        parts = [self.visit(o) for o in operands]
        known_bad = any(p.sort in ("parameter", "schedule") for p in parts)
        self.claim("product_sort", "refuted" if known_bad else
                   "established" if all(p.sort in ("source", "filter") for p in parts) else "unverified",
                   evidence="factors must be processes", diagnostic_severity="error" if known_bad else "info")
        return self.process(bound=None, inputs=parts, finite=_finite_all(parts))

    def at_least(self, block, n):
        """The wrapped process unchanged, with a declared minimum history length."""
        inner = self.visit(block)
        self.claim("minimum_history_length", value=n, scope="simulation_design",
                   evidence="declared floor: shorter histories are never sampled")
        return self.process(sort=inner.sort if inner.sort in ("source", "filter") else "source", bound=inner.bound,
                            moment=inner.moment, inputs=(inner,), finite=inner.finite)

    def combine(self, operands, schedule):
        parts = [self.visit(o) for o in operands]
        known_bad = any(p.sort in ("parameter", "schedule") for p in parts)
        self.claim("combination_sort", "refuted" if known_bad else
                   "established" if all(p.sort in ("source", "filter") for p in parts) else "unverified",
                   evidence="operands must be processes", diagnostic_severity="error" if known_bad else "info")
        sched = self.visit(schedule, arity=len(operands))
        bound = None
        if sched.mode in ("sum", "convex", "select") and all(p.bound is not None for p in parts):
            periods = set().union(*(p.bound for p in parts))
            bound = {period: max(p.bound.get(period, 0) for p in parts) for period in periods}
        if sched.mode == "sum":
            terms = {}
            for operand in operands:
                for node, coefficient in _linear_terms(operand):
                    terms[id(node)] = terms.get(id(node), 0) + coefficient
            if terms and all(c == 0 for c in terms.values()):
                finite = _finite_all(parts)
                premise = self.claim("finite_values", "established" if finite is True else
                                     "refuted" if finite is False else "unverified",
                                     scope="ideal_real_values", evidence="local finite-value declarations")
                status = implied_status([premise])
                self.claim("zero_witness", status, evidence="same realized node has coefficients summing exactly to zero",
                           scope="real_sequence_with_finite_values", premises=(premise.claim_id,),
                           open_obligations=[] if status == "established" else ["finite realized values required"])
                if status == "established":
                    bound = {}
        return self.process(bound=bound, inputs=parts, finite=_finite_all(parts))


def _finite_all(infos):
    values = [p.finite for p in infos]
    return False if False in values else True if all(v is True for v in values) else None


def _linear_terms(block):
    # Only the local built-in Scale rule is expanded. Unknown/custom recurrences
    # remain opaque node identities even if they inherit or override its metadata.
    from .model import Scale
    return block.own_linear_terms() if type(block) is Scale else [(block, 1.0)]


def typecheck(model, *, history_length=None, policies=("history_covers_declared_cycle",), identity_record=None):
    """Inspect without drawing. Unknown semantics return unverified claim records.

    Optional identity_record supplies canonical subject references. Otherwise references
    are deterministic within this report and explicitly provisional, never cache keys.
    """
    if history_length is not None and (isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 1):
        raise ValueError("history_length must be a positive integer or None")
    engine = _Engine(history_length, tuple(policies), identity_record)
    # Traverse only declared dependencies via local rules. The production validator
    # calls params(), whose lazy joint-key population would mutate the inspected graph.
    engine.visit(model)
    return TypecheckReport(tuple(engine.claims), engine.ref(model))
