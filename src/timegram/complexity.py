"""Opt-in descriptive grammar counts, discrete trace entropy and coordinate records.

These quantities are separate: no scalar score, training penalty or identification
claim is computed. Unsupported semantics leave totals unknown and retain partial facts.
The named-description convention counts declared definitions, not minimized expressions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from types import MappingProxyType

import numpy as np

from . import blocks, compounds, dists, families, identity, model

CONVENTION = "grammar-counts/1"
_COUNTS = ("blocks", "parameters", "schedules", "innovation_laws", "references",
           "reused_references", "numeric_atoms", "string_atoms", "null_atoms", "operator_atoms")
_NUMERIC_TYPES = (bool, int, float) + tuple(np.dtype(name).type for name in
    ("bool", "int8", "int16", "int32", "int64", "longlong", "uint8", "uint16", "uint32",
     "uint64", "ulonglong", "float16", "float32", "float64", "longdouble"))


def _numeric_scalar(value):
    return any(type(value) is cls for cls in _NUMERIC_TYPES)


def _inert_numeric(value):
    """Only built-in storage; never invoke extension conversion protocols."""
    if _numeric_scalar(value):
        return True
    if type(value) is np.ndarray:
        return value.dtype.kind in "biuf"
    if type(value) is tuple or type(value) is list:
        return all(_inert_numeric(item) for item in value)
    return False


@dataclass(frozen=True)
class CountRecord:
    status: str
    counts: object
    scope: str = "named declared definitions; known partial counts when status is unknown"


@dataclass(frozen=True)
class EntropyRecord:
    status: str
    bits: float | None
    known_subtotal_bits: float
    scope: str


@dataclass(frozen=True)
class DimensionCase:
    probability: float
    choices: tuple
    coordinates: tuple
    k: int | None
    known_coordinates: int
    unsupported: tuple[str, ...]


@dataclass(frozen=True)
class DimensionRecord:
    status: str
    cases: tuple[DimensionCase, ...]
    minimum: int | None
    maximum: int | None
    mean: float | None
    output_parameterization: tuple
    scope: str = "continuous generative coordinates conditional on discrete structure; not observation rank"


@dataclass(frozen=True)
class ComplexityRecord:
    structural: CountRecord
    trace_entropy: EntropyRecord
    structure_entropy: EntropyRecord
    dimensions: DimensionRecord
    world_key: str | None
    unsupported: tuple[str, ...]
    definitions: tuple
    identity_bindings: tuple = ()
    reference_scope: str = ("parameters:N and other count labels are local to this report; identity_bindings maps "
                            "the same objects to representative dag-v3 references when available, not unique "
                            "cross-document or contextual draw correspondences")
    convention: str = CONVENTION
    view: str = "named"
    schema: str = "timegram.complexity/1"
    L_G: None = None
    score_reason: str = "description-length score not implemented; raw counts only"
    weight_table: None = None
    assumptions: tuple[str, ...] = (
        "Exact built-in schemas only; custom attributes and extensions are not inferred.",
        "Named view excludes expansion trees, cached shapes and inactive default input sources.",
        "IID innovation-law definitions are counted separately; their per-time draws are not coordinates or structural choices.",
        "Simplex dimension is K-1 constrained coordinates, not gamma-sampler auxiliary draws.",
        "This record makes no identifiability, forecast-quality, coding-length or training-cost claim.",
    )

    @property
    def counts(self):
        return self.structural.counts


def _domain_issue(obj):
    """Small explicit preconditions for implemented dimension/entropy rules.

    This is not general constructor validation; unsupported semantics stay unknown.
    No sampler, user method or constructor is called to obtain these facts.
    """
    cls = type(obj)
    def finite(*values):
        if not all(_inert_numeric(value) for value in values):
            return False
        try:
            return bool(np.isfinite(np.asarray(values, dtype=float)).all())
        except (TypeError, ValueError):
            return False
    def integer(value, least=0):
        return (type(value) is not bool and type(value) is not np.bool_ and _numeric_scalar(value)
                and finite(value) and value == int(value) and value >= least)
    def scalar_finite(*values):
        return all(_numeric_scalar(value) for value in values) and finite(*values)
    def vector_shape(shape, k):
        return type(shape) is tuple and len(shape) == 1 and integer(shape[0]) and shape[0] == k
    try:
        ok = True
        if cls is dists.Fixed:
            if not _inert_numeric(obj.value):
                return "invalid declared Fixed parameter domain"
            a = np.asarray(obj.value)
            ok = a.dtype.kind in "biuf" and bool(np.isfinite(a).all())
        elif cls is dists.Uniform:
            ok = scalar_finite(obj.low, obj.high) and obj.low <= obj.high
        elif cls is dists.Normal:
            ok = scalar_finite(obj.mu, obj.sd) and obj.sd >= 0
        elif cls is dists.Laplace:
            ok = scalar_finite(obj.loc, obj.scale) and obj.scale >= 0
        elif cls is dists.StudentT:
            ok = scalar_finite(obj.df) and obj.df > 0
        elif cls is dists.Integers:
            ok = integer(obj.low, least=-math.inf) and integer(obj.high, least=-math.inf) and obj.low <= obj.high
        elif cls is dists.Choice:
            if not _inert_numeric(obj.probs) or not _inert_numeric(obj.values):
                return "invalid declared Choice parameter domain"
            probs = np.asarray(obj.probs, float)
            values = np.asarray(obj.values)
            ok = (values.ndim == 1 and len(values) > 0 and values.dtype.kind in "biuf" and bool(np.isfinite(values).all())
                  and probs.shape == (len(obj.values),) and bool(np.isfinite(probs).all())
                  and bool((probs >= 0).all()) and math.isclose(float(probs.sum()), 1., abs_tol=1e-12))
        elif cls in (dists.Stationary, dists.StationaryOrder, dists.Invertible, dists.L1Ball, dists.Dirichlet):
            ok = integer(obj.k, least=1 if cls is dists.Dirichlet else 0) and vector_shape(obj.shape, obj.k)
            if cls in (dists.Stationary, dists.StationaryOrder, dists.Invertible):
                ok = ok and scalar_finite(obj.pacf_max) and 0 < obj.pacf_max <= 1
            if cls is dists.StationaryOrder:
                ok = ok and scalar_finite(obj.last_min) and 0 < obj.last_min < obj.pacf_max
            if cls is dists.Dirichlet:
                ok = ok and scalar_finite(obj.concentration) and obj.concentration > 0
        elif cls is blocks.Centered:
            ok = integer(obj.k, 1) and vector_shape(obj.shape, obj.k)
        elif cls in (blocks.AR, blocks.MA):
            ok = integer(obj.p if cls is blocks.AR else obj.q) and integer(obj.period, 1)
            if cls is blocks.AR:
                ok = ok and type(obj.init) is str and obj.init in ("zero", "steady")
        elif cls is blocks.Difference:
            ok = integer(obj.d) and integer(obj.period, 1)
        elif cls in (blocks.Integrate, blocks.Pattern):
            ok = integer(obj.period, 1)
        elif cls is blocks.Lag:
            ok = integer(obj.k)
        elif cls is blocks.Normalize:
            ok = type(obj.kind) is str and obj.kind in ("minmax", "std", "robust")
        return None if ok else f"invalid declared {cls.__name__} parameter domain"
    except (ValueError, TypeError, AttributeError, OverflowError):
        return f"invalid declared {cls.__name__} parameter domain"


class _Limit(Exception):
    pass


class _Counts:
    def __init__(self):
        self.counts = dict.fromkeys(_COUNTS, 0)
        self.refs, self.definitions, self.edges, self.objects = {}, [], set(), {}
        self.contexts, self.active, self.issues = set(), set(), []

    def ref_by_id(self, token):
        return self.refs.get(("parameters", token), "unverified-draw")

    def checked(self, obj):
        try:
            names = identity._names(obj)
            issue = _domain_issue(obj)
            if issue:
                raise ValueError(issue)
            return names
        except (ValueError, TypeError, AttributeError) as exc:
            self.issues.append(str(exc))
            return None

    def ref(self, obj, kind="parameters"):
        return self.refs.get((kind, id(obj)), f"unverified:{type(obj).__name__}")

    def visit(self, obj, kind=None, parent=None, path=(), fed=False):
        names = self.checked(obj)
        if names is None:
            return
        if kind is None:
            kind = ("blocks" if isinstance(obj, model.Block) else
                    "schedules" if isinstance(obj, model.Schedule) else "parameters")
        token = (kind, id(obj))
        if token in self.active:
            self.issues.append("cyclic declared definition")
            return
        if token not in self.refs:
            self.refs[token] = f"{kind}:{self.counts[kind]}"
            self.counts[kind] += 1
            self.definitions.append((self.refs[token], type(obj).__name__))
            self.objects[self.refs[token]] = obj
        if parent is not None:
            self.edges.add((parent, path, token))
        context = (token, fed)
        if context in self.contexts:
            return
        self.contexts.add(context)
        self.active.add(token)
        # Public named configuration is a production. Its expansion and placeholder
        # are not counted a second time. Semantic constructor fields remain explicit.
        excluded = {"name"}
        if type(obj) in (compounds.ETS, compounds.ARIMA, compounds.Composite, families.ARFIMA):
            excluded |= {"tree", "input"}
        if isinstance(obj, dists.Distribution):
            excluded |= {"shape", "label"}  # derived shape and expression render label
        if fed:
            excluded.add("source")
        for name in names:
            if name in excluded:
                continue
            value = getattr(obj, name)
            # Same definition may be reached both as a fed filter and standalone.
            # Atom and edge occurrences are counted once by their defining field path.
            self.value(value, token, (name,), fed=(type(obj) is model.Feed and name == "down") or
                           (type(obj) is compounds.Composite and name == "stages"),
                       iid=(type(obj) is blocks.Noise and name == "shape") or
                           (type(obj) is blocks.Centered and name == "dist") or
                           (kind == "innovation_laws" and name == "components"))
        self.active.remove(token)

    def value(self, value, parent, path, *, fed=False, iid=False):
        if isinstance(value, (model.Block, model.Schedule, dists.Distribution)):
            self.visit(value, "innovation_laws" if iid else None, parent, path, fed)
            return
        atom_token = (parent, path, "atom")
        if not hasattr(self, "atoms"):
            self.atoms = set()
        if atom_token in self.atoms:
            return
        if value is None:
            category = "null_atoms"
        elif type(value) is str:
            category = "string_atoms"
        elif _numeric_scalar(value):
            category = "numeric_atoms"
            if not np.isfinite(value):
                self.issues.append("nonfinite numeric field")
        elif callable(value) and any(value is operator for operator in identity._OPS):
            category = "operator_atoms"
        elif type(value) is np.ndarray:
            if value.dtype.kind not in "biuf":
                self.issues.append(f"unsupported array dtype {value.dtype}")
                return
            self.value(value.tolist(), parent, path, iid=iid)
            return
        elif type(value) in (tuple, list):
            for i, item in enumerate(value):
                self.value(item, parent, path + (i,), iid=iid, fed=fed)
            return
        else:
            self.issues.append(f"unsupported declared value {type(value).__name__}")
            return
        self.atoms.add(atom_token)
        self.counts[category] += 1

    def finish(self):
        self.counts["references"] = len(self.edges)
        incoming = {}
        for _, _, target in self.edges:
            incoming[target] = incoming.get(target, 0) + 1
        self.counts["reused_references"] = sum(max(0, count - 1) for count in incoming.values())
        return CountRecord("unknown" if self.issues else "exact", MappingProxyType(dict(self.counts)))


@dataclass
class _Case:
    coordinates: dict = field(default_factory=dict)
    choices: dict = field(default_factory=dict)
    issues: tuple = ()
    trace_issues: tuple = ()


def _entropy(probabilities):
    return max(0., -sum(p * math.log2(p) for p in probabilities if p > 0))


class _Analysis:
    def __init__(self, counts, max_cases):
        self.counts, self.max_cases = counts, max_cases
        self.pmf, self.owners, self.active = {}, {}, set()
        self.global_issues, self.output_notes = [], {}

    def checked(self, obj):
        try:
            identity._names(obj)
            return _domain_issue(obj) is None
        except (ValueError, TypeError, AttributeError):
            return False

    def unknown(self, reason, *, trace=True):
        return [_Case(issues=(reason,), trace_issues=(reason,) if trace else ())]

    def coordinate(self, obj, k):
        return [_Case(coordinates={id(obj): (self.counts.ref(obj), k)} if k else {})]

    def product(self, pieces):
        result = [_Case()]
        for piece in pieces:
            out = {}
            for left, right in itertools.product(result, piece):
                if any(k in left.choices and left.choices[k] != v for k, v in right.choices.items()):
                    continue
                coordinates = left.coordinates | right.coordinates
                choices = left.choices | right.choices
                issues = tuple(dict.fromkeys(left.issues + right.issues))
                trace_issues = tuple(dict.fromkeys(left.trace_issues + right.trace_issues))
                case = _Case(coordinates, choices, issues, trace_issues)
                key = (tuple(sorted(coordinates)), tuple(sorted(choices.items())), issues, trace_issues)
                out[key] = case
                if len(out) > self.max_cases:
                    raise _Limit("conditional case budget exceeded")
            result = list(out.values())
        return result

    def parameter(self, obj):
        if not self.checked(obj):
            return self.unknown(f"unverified parameter {type(obj).__name__}", trace=False)
        token = ("parameter", id(obj))
        if token in self.active:
            return self.unknown("parameter dependency cycle", trace=False)
        self.active.add(token)
        try:
            return self._parameter(obj)
        finally:
            self.active.remove(token)

    def _parameter(self, obj):
        cls = type(obj)
        if cls is dists.Fixed:
            return [_Case()]
        if cls is dists.Uniform:
            return self.coordinate(obj, int(obj.low != obj.high))
        if cls is dists.Normal:
            return self.coordinate(obj, int(obj.sd > 0))
        if cls is dists.Laplace:
            return self.coordinate(obj, int(obj.scale > 0))
        if cls is dists.StudentT:
            return self.coordinate(obj, 1)
        if cls in (dists.Choice, dists.Integers):
            return [_Case()]  # no continuous coordinate; not a structural role here
        if cls in (dists.Stationary, dists.StationaryOrder, dists.Invertible, dists.L1Ball):
            return self.coordinate(obj, obj.k)
        if cls is dists.Dirichlet:
            return self.coordinate(obj, obj.k - 1)
        if cls is dists.Expr:
            if not any(obj.fn is operator for operator in identity._OPS):
                return self.unknown("custom parameter expression", trace=False)
            self.output_notes[id(obj)] = (self.counts.ref(obj), "unknown", None,
                                         "Expr adds no draw; output-map dimension is not inferred")
            return self.product([self.parameter(p) for p in obj.args])
        if cls is blocks.Centered:
            self.output_notes[id(obj)] = (self.counts.ref(obj), "upper_bound", max(0, obj.k - 1),
                                         "centered output; latent sampler coordinate convention deferred")
            return self.unknown("Centered generative-coordinate convention not implemented", trace=False)
        if cls is dists.JointKey:
            return self.unknown("joint-prior conditional dimension not implemented", trace=False)
        return self.unknown(f"parameter dimension not implemented for {cls.__name__}", trace=False)

    def choices(self, draw, owner, *, order=False, alternatives=None):
        if not self.checked(draw):
            return None
        if type(draw) is dists.Fixed:
            values, probs = [draw.value], [1.]
        elif type(draw) is dists.Choice:
            values, probs = draw.values, draw.probs
        elif type(draw) is dists.Integers:
            size = draw.high - draw.low + 1
            if size > self.max_cases:
                raise _Limit("discrete-choice support exceeds case budget")
            values, probs = range(draw.low, draw.high + 1), [1 / size] * size
        elif type(draw) is dists.Uniform and draw.low == draw.high:
            values, probs = [draw.low], [1.]
        else:
            return None
        grouped = {}
        for value, probability in zip(values, probs):
            if not np.isfinite(probability) or probability < 0:
                return None
            if probability == 0:
                continue
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or int(value) != value:
                return None
            value = int(value)
            if value < 0 or (alternatives is not None and value >= alternatives):
                return None
            grouped[value] = grouped.get(value, 0.) + float(probability)
        if not grouped or not math.isclose(sum(grouped.values()), 1., abs_tol=1e-12):
            return None
        key = id(draw)
        owners = self.owners.setdefault(key, set())
        owners.add(id(owner))
        if len(owners) > 1:
            self.global_issues.append("one structural draw is shared by distinct consumers; conditional dependence unverified")
        self.pmf[key] = grouped
        # A fixed single-valued structure has no random choice in its trace.
        return [(value, _Case(choices={key: value} if len(grouped) > 1 else {})) for value in sorted(grouped)]

    def block(self, obj, *, fed=False):
        if not self.checked(obj):
            return self.unknown(f"unverified block {type(obj).__name__}")
        token = ("block", id(obj), fed)
        if token in self.active:
            return self.unknown("block dependency cycle")
        self.active.add(token)
        try:
            return self._block(obj, fed=fed)
        finally:
            self.active.remove(token)

    def _block(self, obj, *, fed=False):
        cls = type(obj)
        if cls is model.Feed:
            if not isinstance(obj.down, model.Transform):
                return self.unknown("invalid feed target")
            return self.product([self.block(obj.up), self.block(obj.down, fed=True)])
        if cls is model.Scale:
            return self.product([self.block(obj.block), self.parameter(obj.c)])
        if cls is model.Combine:
            if not self.checked(obj.schedule):
                return self.unknown("unverified schedule")
            if type(obj.schedule) is model.OneHot:
                options = self.choices(obj.schedule.index, obj, alternatives=len(obj.operands))
                if options is None:
                    return self.unknown("selection law/dependence is unknown")
                out = []
                for index, event in options:
                    out.extend(self.product([[event], self.block(obj.operands[index])]))
                    if len(out) > self.max_cases:
                        raise _Limit("conditional case budget exceeded")
                return out
            if type(obj.schedule) in (model.Ones, model.Convex):
                pieces = [self.block(b) for b in obj.operands]
                if type(obj.schedule) is model.Convex:
                    pieces.append(self.parameter(obj.schedule.w))
                return self.product(pieces)
            return self.unknown("unsupported schedule semantics")
        if cls is blocks.Noise:
            pieces = [self.parameter(obj.scale)]
            if not self.iid_supported(obj.shape):
                pieces.append(self.unknown("innovation law has no supported IID contract"))
            return self.product(pieces)
        if cls in (blocks.Constant, blocks.Trend, blocks.DampedTrend, blocks.Pattern):
            fields = {blocks.Constant: ("c",), blocks.Trend: ("b",),
                      blocks.DampedTrend: ("b", "phi"), blocks.Pattern: ("values",)}[cls]
            return self.product([self.parameter(getattr(obj, name)) for name in fields])
        if cls in (blocks.AR, blocks.MA, blocks.Integrate, blocks.Difference, blocks.Lag, blocks.Normalize, blocks.Exp):
            pieces = [] if fed else [self.block(obj.source)]
            if cls in (blocks.AR, blocks.MA, blocks.Exp):
                field = {blocks.AR: "phi", blocks.MA: "theta", blocks.Exp: "rate"}[cls]
                pieces.append(self.parameter(getattr(obj, field)))
            if cls is blocks.Integrate:
                options = self.choices(obj.d, obj, order=True)
                pieces.append([event for _, event in options] if options is not None else
                              self.unknown("integration-order distribution/dependence is unknown"))
            return self.product(pieces)
        # Raw counts still describe supported named configurations. Their joint and
        # branch-sensitive parameter dimensions require separate rules.
        return self.unknown(f"generative analysis not implemented for {cls.__name__}")

    def iid_supported(self, law):
        if not self.checked(law):
            return False
        if type(law) in (dists.Uniform, dists.Normal, dists.StudentT, dists.Laplace, dists.Integers, dists.Choice):
            return True
        return type(law) is dists.Mixture and all(self.iid_supported(c) for c in law.components)

    def probability(self, case):
        return math.prod(self.pmf[token][value] for token, value in case.choices.items())

    def finish(self, cases):
        cases = self.product([cases])  # remove duplicate event/coordinate cases
        mass = sum(self.probability(c) for c in cases)
        if not math.isclose(mass, 1., abs_tol=1e-10):
            self.global_issues.append("conditional trace probability mass is not one")
        result = []
        for case in cases:
            issues = tuple(dict.fromkeys(case.issues + tuple(self.global_issues)))
            coords = tuple(sorted(case.coordinates.values()))
            k = sum(d for _, d in coords)
            choices = tuple(sorted((self.counts.ref_by_id(token), value) for token, value in case.choices.items()))
            result.append(DimensionCase(self.probability(case), choices, coords, None if issues else k, k, issues))
        complete = bool(result) and all(c.k is not None for c in result)
        dimension = DimensionRecord("exact" if complete else "unknown", tuple(result),
                                    min(c.k for c in result) if complete else None,
                                    max(c.k for c in result) if complete else None,
                                    sum(c.probability * c.k for c in result) if complete else None,
                                    tuple(self.output_notes.values()))
        # Known trace is the explicitly enumerated active choice outcomes; entropy of
        # unmodeled choices is not set to zero merely because no event was emitted.
        known = _entropy(c.probability for c in result)
        trace_complete = not self.global_issues and not any(c.trace_issues for c in cases)
        trace = EntropyRecord("exact" if trace_complete else "unknown", known if trace_complete else None, known,
                              "active discrete selection/order trace; no innovation or differential entropy")
        return dimension, trace


def _structure_entropy(world, analysis, trace):
    if trace.status == "exact" and trace.bits == 0:
        return EntropyRecord("exact", 0., 0., "no random structural choices")
    if type(world) is model.Combine and type(world.schedule) is model.OneHot and analysis.checked(world.schedule):
        options = analysis.choices(world.schedule.index, world, alternatives=len(world.operands))
        if options is not None:
            masses = {}
            for index, event in options:
                branch = world.operands[index]
                # Only a single outer choice with no further structural randomness.
                separate = _Analysis(analysis.counts, analysis.max_cases)
                _, child_trace = separate.finish(separate.block(branch))
                record = identity.world(branch)
                if child_trace.status != "exact" or child_trace.bits != 0 or record.status != "supported":
                    break
                masses[record.digest] = masses.get(record.digest, 0.) + analysis.probability(event)
            else:
                entropy = _entropy(masses.values())
                return EntropyRecord("exact", entropy, entropy,
                                     "direct selected branch definitions modulo supported dag-v3 identity")
    return EntropyRecord("unknown", None, 0., "no supported quotient calculation for this structure")


def complexity(world, *, convention=CONVENTION, view="named", max_cases=256):
    """Return deterministic descriptive records; never train, sample or modify a cache.

    Only named-description counts are implemented. Generative dimensions and entropy
    cover primitive sources/filters, fixed discrete choices and simple compositions;
    unsupported joint/named-family semantics retain unknown totals. max_cases bounds
    conditional enumeration without introducing a stochastic approximation.
    """
    if convention != CONVENTION:
        raise ValueError(f"unsupported count convention {convention!r}")
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1:
        raise ValueError("max_cases must be a positive integer")
    if view not in ("named", "expanded"):
        raise ValueError("view must be 'named' or 'expanded'")
    count = _Counts()
    if view == "named":
        try:
            count.visit(world)
        except RecursionError:
            count.issues.append("count traversal recursion limit; partial counts retained")
    else:
        count.issues.append("expanded view not implemented; named counts are not substituted")
    structural = count.finish()
    canonical = identity.world(world)
    analysis = _Analysis(count, max_cases)
    try:
        cases = analysis.block(world) if view == "named" else analysis.unknown("expanded view not implemented")
        dimension, trace = analysis.finish(cases)
        structure = _structure_entropy(world, analysis, trace) if view == "named" else EntropyRecord("unknown", None, 0., "expanded view not implemented")
    except (_Limit, RecursionError) as exc:
        reason = str(exc) or "recursive analysis limit"
        dimension, trace = analysis.finish(analysis.unknown(reason))
        structure = EntropyRecord("unknown", None, 0., reason)
    unsupported = tuple(dict.fromkeys(count.issues + [issue for case in dimension.cases for issue in case.unsupported]))
    bindings = tuple((label, canonical.subject_ref(obj)) for label, obj in count.objects.items())
    return ComplexityRecord(structural, trace, structure, dimension, canonical.digest, unsupported,
                            tuple(count.definitions), identity_bindings=bindings, convention=convention, view=view)
