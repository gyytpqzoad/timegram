"""Opt-in identity for a tested grammar fragment; legacy library keys are untouched.

``world`` identifies generating definitions, while ``problem`` also binds the resolved
task and, optionally, its key-relevant training configuration. This is not a complete
decision procedure for equality of laws or of floating-point training trajectories.
Unsupported extensions return an unverified record without a digest. Nothing is drawn,
trained, read from a cache, or written to model instances by these functions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping

import numpy as np

from . import blocks, compounds, dists, families, fractional, model, objective, priors, volatility
from .problem import Forecast, Loss, Missing, Params, Problem, Which

SCHEMA = "dag-v3"
RULE_VERSIONS = ("explicit-bindings/1", "composition/1", "disjoint-addition/1", "task-layout/1")
_INTEGER_TYPES = (int, np.int8, np.int16, np.int32, np.int64, np.longlong,
                  np.uint8, np.uint16, np.uint32, np.uint64, np.ulonglong)
_FLOAT_TYPES = (float, np.float16, np.float32, np.float64, np.longdouble)


@dataclass(frozen=True)
class ScopedRef:
    document_digest: str
    kind: str
    binding: str


@dataclass(frozen=True)
class CanonicalRecord:
    schema: str
    domain: str
    status: str
    canonical_json: str | None
    digest: str | None
    rule_versions: tuple[str, ...]
    binding_map: Mapping[int, ScopedRef]
    unsupported: tuple[str, ...] = ()
    automorphism_orbits: None = None  # no complete automorphism group is claimed
    correspondence_scope: str = "document-local; cross-document correspondence unverified"

    def subject_ref(self, obj) -> ScopedRef | None:
        """Representative first-encountered role, not a unique contextual correspondence.

        A transform can occur as an output, a filter definition and at several application
        sites. Those roles remain distinct in the document; callers needing a contextual
        proof must inspect those edges or retain a provisional contextual reference.
        Equal independent subgraphs may also have nonunique cross-document mappings.
        Object identities never enter canonical bytes.
        """
        return self.binding_map.get(id(obj))


class _Unsupported(ValueError):
    pass


# Exact classes and declared fields, not isinstance/subclass or repr-based codecs.
# New/changed semantic fields require a codec/ruleset review. Joint._keys is a cache.
_FIELDS = {
    blocks.Noise: "scale shape", blocks.Input: "", blocks.Constant: "c", blocks.Trend: "b",
    blocks.DampedTrend: "b phi", blocks.Centered: "dist k shape", blocks.Pattern: "period values",
    blocks.AR: "source p period phi init", blocks.MA: "source q period theta",
    blocks.Integrate: "source d period", blocks.Normalize: "source kind",
    blocks.Difference: "source d period", blocks.Exp: "source rate", blocks.Indicator: "source", blocks.Counts: "source", blocks.Lag: "source k",
    model.Feed: "up down name", model.Combine: "operands schedule name", model.Scale: "block c",
    model.Product: "operands", model.AtLeast: "block n",
    model.Ones: "", model.Convex: "w", model.OneHot: "index",
    compounds.ETS: "source input tree _floor spec period has_trend has_season damped alpha beta gamma phi level0 slope0 season0",
    compounds.ARIMA: "source input tree _floor p d q P D Q period prior phi theta seasonal_phi seasonal_theta d_param D_param mean level0 season0",
    compounds.Composite: "source input tree stages",
    compounds.EtsSmoothing: "trend seasonal period damped alpha_range frac_range phi_range",
    families.ARFIMA: "source input tree p q d phi theta history memory",
    fractional.FractionalFilter: "source d period memory history",
    volatility.GARCH: "source p q burn_in power absolute_moment omega alpha beta has_asymmetry gamma initial",
    dists.Fixed: "value shape", dists.Uniform: "low high", dists.Normal: "mu sd",
    dists.StudentT: "df", dists.Laplace: "loc scale", dists.Bernoulli: "p", dists.Mixture: "components weights",
    dists.Integers: "low high", dists.Choice: "values probs",
    dists.Stationary: "k pacf_max shape", dists.StationaryOrder: "k pacf_max shape last_min",
    dists.Invertible: "k pacf_max shape", dists.L1Ball: "k shape",
    dists.Dirichlet: "k concentration shape", dists.Expr: "fn args label shape",
    dists.JointKey: "joint key",
    priors.SarimaTree: "period max_order max_seasonal_order seasonal_prob D1_prob d_choices family_probs l1_prob sparse_prob pacf_max const_prob const_max warm_max",
    priors.Garch11: "variance persistence share",
}
_FIELDS = {cls: tuple(names.split()) for cls, names in _FIELDS.items()}
_VALUES = {Missing, Forecast, Params, Which, Loss, objective.Mean, objective.Worst,
           objective.Risk, objective.Bias, objective.Miscoverage, objective.Objective}
_OPS = {operator.add: "add", operator.sub: "subtract", operator.mul: "multiply",
        operator.truediv: "divide", operator.neg: "negate"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _check_composite_lowering(obj):
    """Only erase an expansion that is the declared chain with the same bindings.

    Inspect the stored tree without rebuilding it, drawing, or invoking extension
    hooks. Object identity here checks bindings; it never enters canonical bytes.
    The default source may be replaced independently and is encoded by the caller.
    """
    if (type(obj.stages) is not tuple or not obj.stages
            or type(obj.input) is not blocks.Input):
        raise _Unsupported("Composite requires nonempty tuple stages and a built-in Input")
    _names(obj.input)
    node = obj.tree
    for stage in reversed(obj.stages):
        if (not isinstance(stage, model.Transform) or type(node) is not model.Feed
                or node.down is not stage):
            raise _Unsupported("Composite expansion does not match its declared stage bindings")
        _names(node)
        node = node.up
    if node is not obj.input:
        raise _Unsupported("Composite expansion is not bound to its declared Input")


def _names(obj):
    cls = type(obj)
    if cls not in _FIELDS:
        raise _Unsupported(f"no codec for {cls.__module__}.{cls.__qualname__}")
    expected = set(_FIELDS[cls])
    extra = set(vars(obj)) - expected
    if cls is fractional.FractionalFilter and "name" in extra:
        # fractional_difference is a public constructor over the same signed-order
        # filter. Its known display name is not an additional stochastic setting.
        name = vars(obj)["name"]
        if type(name) is not str or name != "fractional_difference":
            raise _Unsupported("FractionalFilter has an unsupported instance name")
        extra.remove("name")
    if isinstance(obj, dists.Joint):
        extra.discard("_keys")
    if extra or expected - set(vars(obj)):
        raise _Unsupported(f"{cls.__name__} fields differ from codec: "
                           f"extra={sorted(extra)}, missing={sorted(expected - set(vars(obj)))}")
    if cls is compounds.Composite:
        _check_composite_lowering(obj)
    return _FIELDS[cls]


class _Encoder:
    def __init__(self, *, sort_add=True):
        self.definitions = []
        self.seen = {}
        self.subjects = {}
        self.sort_add = sort_add
        self.active = set()

    def bind(self, token, kind, tag, payload, obj=None):
        if token in self.active:
            raise _Unsupported("dependency cycle in identity graph")
        if token in self.seen:
            return {"ref": self.seen[token]}
        if len(self.definitions) >= 10000:
            raise _Unsupported("identity node budget exceeded (10000)")
        label = f"{kind}{len(self.definitions)}"
        self.seen[token] = label
        self.active.add(token)
        record = {"id": label, "kind": kind, "type": tag, "fields": None}
        self.definitions.append(record)
        if obj is not None:
            self.subjects.setdefault(id(obj), (kind, label))
        record["fields"] = payload()
        self.active.remove(token)
        return {"ref": label}

    def value(self, x):
        if isinstance(x, model.Block):
            return self.block(x)
        if isinstance(x, dists.Distribution):
            _names(x)
            def payload():
                attrs = self.attrs(x)
                if type(x) is dists.JointKey:
                    if x.key not in x.joint.keys:
                        raise _Unsupported(f"undeclared joint entry {x.key!r}")
                    attrs["event_shape"] = self.value(tuple(x.shape))
                return attrs
            return self.bind(("rv", id(x)), "rv", type(x).__name__, payload, x)
        if isinstance(x, model.Schedule):
            _names(x)
            return self.bind(("schedule", id(x)), "schedule", type(x).__name__, lambda: self.attrs(x), x)
        if x is None or type(x) in (str, bool):
            return x
        if type(x) in _INTEGER_TYPES:
            return {"integer": str(int(x))}
        if type(x) in _FLOAT_TYPES:
            if isinstance(x, np.floating) and x.dtype.itemsize > 8:
                raise _Unsupported("extended-precision floats need a separate exact codec")
            if not math.isfinite(x):
                raise _Unsupported("nonfinite numeric identity field")
            return {"float": float(x).hex()}
        if type(x) is np.bool_:
            return bool(x)
        if type(x) is np.ndarray:
            if x.dtype.kind not in "biuf" or (x.dtype.kind == "f" and x.dtype.itemsize > 8):
                raise _Unsupported(f"unsupported array dtype {x.dtype}")
            return {"array": self.value(x.tolist()), "shape": list(x.shape), "dtype": x.dtype.str}
        if type(x) in (tuple, list):
            return {type(x).__name__: [self.value(v) for v in x]}
        if type(x) is dict and all(type(k) is str for k in x):
            return {"mapping": {k: self.value(x[k]) for k in sorted(x)}}
        if type(x) in _VALUES:
            allowed = {f.name for f in fields(x)}
            if set(vars(x)) != allowed:
                raise _Unsupported(f"unexpected fields on {type(x).__name__}")
            return {"type": type(x).__name__, "fields": {k: self.value(getattr(x, k)) for k in sorted(allowed)}}
        for operation, name in _OPS.items():
            if x is operation:
                return {"operator": name}
        raise _Unsupported(f"unsupported identity value {type(x).__module__}.{type(x).__qualname__}")

    def attrs(self, obj, exclude=()):
        return {key: self.value(getattr(obj, key)) for key in sorted(_names(obj)) if key not in exclude}

    def definition(self, transform):
        _names(transform)
        # The caller supplies the input. Compound placeholders/trees remain definitions,
        # with each application represented separately rather than memoizing its output.
        return self.bind(("filter", id(transform)), "filter", type(transform).__name__,
                         lambda: self.attrs(transform, {"source"}), transform)

    def chain(self, up, stages, token, obj=None):
        if not stages:
            return self.block(up)
        def payload():
            return {"input": self.chain(up, stages[:-1], (token, "prefix")),
                    "filter": self.definition(stages[-1])}
        return self.bind(token, "block", "Apply", payload, obj)

    def block(self, b):
        _names(b)
        token = ("block", id(b))
        if type(b) is compounds.Composite:
            return self.chain(b.source, b.stages, token, b)
        if type(b) is model.Feed:
            _names(b.down)
            stages = b.down.stages if type(b.down) is compounds.Composite else (b.down,)
            return self.chain(b.up, stages, token, b)
        if isinstance(b, model.Transform):
            return self.chain(b.source, (b,), token, b)
        if type(b) is model.Combine:
            def payload():
                operands = list(b.operands)
                if type(b.schedule) is model.Ones and self.sort_add and self.disjoint(operands):
                    operands.sort(key=self.component_key)
                return {"operands": [self.block(x) for x in operands], "schedule": self.value(b.schedule)}
            return self.bind(token, "block", "Combine", payload, b)
        if type(b) is model.Product:                              # ordered: no commutation rule
            return self.bind(token, "block", "Product",
                             lambda: {"operands": [self.block(x) for x in b.operands]}, b)
        return self.bind(token, "block", type(b).__name__, lambda: self.attrs(b), b)

    def component_key(self, b):
        local = _Encoder(sort_add=self.sort_add)
        root = local.block(b)
        return _json({"definitions": local.definitions, "root": root})

    def disjoint(self, operands):
        seen = set()
        for b in operands:
            # Full local encoding certifies supported fields and discovers all bindings;
            # sharing fixed objects also conservatively blocks the rewrite.
            local = _Encoder(sort_add=False)
            local.block(b)
            deps = set(local.subjects)
            if seen & deps:
                return False
            seen |= deps
        return True


def _record(root, task, config, schema):
    domain = "world" if task is None else "problem" if config is None else "training-specification"
    try:
        if schema != SCHEMA:
            raise _Unsupported(f"unsupported identity schema {schema!r}")
        if not isinstance(root, model.Block):
            raise _Unsupported("world must be a Block")
        # Target-bearing tasks keep addition order. This avoids guessing a cross-graph
        # relabeling of equal-looking parameters/classes. Forecast-only tasks can sort.
        estimands = () if task is None else task.resolved_estimands(task.default_horizon())
        enc = _Encoder(sort_add=not any(type(e) in (Params, Which) for e in estimands))
        doc = {"schema": SCHEMA, "domain": domain, "rules": RULE_VERSIONS, "root": enc.block(root)}
        if task is not None:
            if type(task) is not Problem:
                raise _Unsupported("no codec for custom Problem")
            task._checked(complete=True)
            target_refs = []
            for e in estimands:
                if type(e) is Params:
                    target_refs.append([(path, enc.value(rv)) for path, rv in e.resolve(root).items()])
                elif type(e) is Which:
                    path, count = e.resolve(root)
                    target_refs.append({"path": path, "classes": count,
                                        "draw": enc.value(root.schedule.index),
                                        "operands": [enc.block(b) for b in root.operands]})
            doc["task"] = {"observation": enc.value(task.observation), "estimands": enc.value(estimands),
                           "targets": target_refs, "layout": enc.value(task.layout()),
                           "loss": enc.value(task.loss_fn), "objective": enc.value(task.objective_spec)}
        if config is not None:
            from .train import TrainConfig
            from .nets import ResMLP
            from .optim import AdamW, MuonAdamW
            if type(config) is not TrainConfig or type(config.net) is not ResMLP or type(config.optimizer) not in (AdamW, MuonAdamW):
                raise _Unsupported("unsupported training configuration type")
            doc["training"] = enc.value(config.key_dict())
        doc["definitions"] = enc.definitions
        canonical = _json(doc)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        bindings = MappingProxyType({oid: ScopedRef(digest, kind, name) for oid, (kind, name) in enc.subjects.items()})
        return CanonicalRecord(SCHEMA, domain, "supported", canonical, digest, RULE_VERSIONS, bindings)
    except (ValueError, TypeError, KeyError, AttributeError, NotImplementedError, RecursionError) as exc:
        return CanonicalRecord(schema, domain, "unverified", None, None, RULE_VERSIONS,
                               MappingProxyType({}), (str(exc),))


def world(model, *, schema=SCHEMA) -> CanonicalRecord:
    """Identify a supported world, without inventing an estimation task or training."""
    return _record(model, None, None, schema)


def problem(problem: Problem, config=None, *, schema=SCHEMA) -> CanonicalRecord:
    """Identify resolved targets/layout and optionally key-relevant training settings."""
    if type(problem) is not Problem:
        return CanonicalRecord(schema, "problem", "unverified", None, None, RULE_VERSIONS,
                               MappingProxyType({}), ("problem must be a built-in Problem",))
    # Resolving Params on ARIMA calls Joint.__getitem__, lazily caching key objects.
    # Keep that existing behavior off the caller's graph, then map correspondence back.
    # Unknown graph codecs are rejected before invoking any extension deepcopy hook.
    checked = world(problem.model, schema=schema)
    if checked.status != "supported":
        return CanonicalRecord(schema, "problem", "unverified", None, None, RULE_VERSIONS,
                               MappingProxyType({}), checked.unsupported)
    try:
        if set(vars(problem)) != {f.name for f in fields(problem)}:
            raise _Unsupported("unexpected fields on Problem")
        # Reject unknown task extension objects before calling their deepcopy hooks.
        metadata = _Encoder()
        for value in (problem.observation, problem.estimands, problem.loss_fn, problem.objective_spec):
            metadata.value(value)
        memo = {}
        cloned = copy.deepcopy(problem, memo)
    except (ValueError, TypeError, KeyError, AttributeError, NotImplementedError, RecursionError) as exc:
        return CanonicalRecord(schema, "problem", "unverified", None, None, RULE_VERSIONS,
                               MappingProxyType({}), (str(exc),))
    record = _record(cloned.model, cloned, config, schema)
    bindings = MappingProxyType({oid: ref for oid, clone in memo.items()
                                 if (ref := record.subject_ref(clone)) is not None})
    return CanonicalRecord(record.schema, record.domain, record.status, record.canonical_json,
                           record.digest, record.rule_versions, bindings, record.unsupported)
