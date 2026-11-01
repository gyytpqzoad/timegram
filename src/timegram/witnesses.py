"""Two opt-in, conditional parameter witnesses for complete raw observations.

This small catalogue proves level splitting and independent Gaussian variance splitting
only at a fully checked additive root. It is not a general identifiability engine.
Parameters are conditioned on, not marginalized over their Uniform sampling laws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite, sqrt

import numpy as np

from . import identity
from .blocks import Constant, Noise
from .dists import Distribution, Fixed, Normal, Uniform
from .model import Block, Combine, Ones, Scale
from .typecheck import ClaimRecord


@dataclass(frozen=True)
class ObservationDesign:
    history_length: int
    representation: str = "raw"
    missingness: str = "complete"
    time_origin: int = 1
    initialization: str = "source_t1"
    presample: int = 0


@dataclass(frozen=True)
class Target:
    """A declared function of actual parameter objects, never an arbitrary callable.

    A Distribution passed directly to witnesses is a ``parameter`` target. ``sum``
    is supported for level splitting; ``variance`` and ``sum_of_variances`` are
    supported for Gaussian scale splitting. Unchanged targets get no witness.
    """
    kind: str
    parameters: tuple[Distribution, ...]


@dataclass(frozen=True)
class WitnessRecord(ClaimRecord):
    targets: tuple[dict, ...] = ()
    coordinates: tuple[dict, ...] = ()
    observation: ObservationDesign | None = None
    conditioning: tuple[str, ...] = ()
    fragment: str = ""
    witness: dict = field(default_factory=dict)
    numerical_check: dict = field(default_factory=dict)
    schema: str = "timegram.parameter_witness/1"


@dataclass(frozen=True)
class WitnessReport:
    world_digest: str | None
    observation: ObservationDesign | None
    records: tuple[WitnessRecord, ...]
    premise_records: tuple[ClaimRecord, ...] = ()
    schema: str = "timegram.parameter_witness_report/1"

    @property
    def status(self):
        return "non_identified_witness" if any(r.status == "non_identified_witness" for r in self.records) else "unverified"

    @property
    def claims(self):
        return self.premise_records + self.records


class _Blocked(ValueError):
    pass


def _fraction(x):
    numeric_types = (int, float, Fraction, np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64)
    if type(x) not in numeric_types or not isfinite(x):
        raise _Blocked("a finite real scalar is required")
    if type(x) is Fraction:
        return x
    return Fraction(int(x)) if isinstance(x, (int, np.integer)) else Fraction.from_float(float(x))


def _fixed(x):
    if type(x) is not Fixed or x.shape != ():
        raise _Blocked("remainder parameters must be fixed scalars")
    return _fraction(x.value)


def _domain(x, positive=False):
    if type(x) is not Uniform or x.shape != ():
        raise _Blocked("changed coordinates must be distinct, nondegenerate scalar Uniform parameters")
    lo, hi = _fraction(x.low), _fraction(x.high)
    if not lo < hi or (positive and lo < 0):
        raise _Blocked("a nondegenerate domain is required; Gaussian scales require nonnegative support")
    return lo, hi


def _leaves(root):
    if type(root) is Combine and type(root.schedule) is Ones:
        return [leaf for child in root.operands for leaf in _leaves(child)]
    return [root]


def _gaussian(leaf):
    """Return the innovation object, its parameter scale and a fixed multiplier."""
    multiplier = Fraction(1)
    if type(leaf) is Scale:
        noise = leaf.block
        if type(noise) is not Noise:
            raise _Blocked("only a direct Gaussian noise may be scaled in this fragment")
        multiplier = _fixed(noise.scale)
        scale = leaf.c
    elif type(leaf) is Noise:
        noise, scale = leaf, leaf.scale
    else:
        raise _Blocked("root has an unsupported additive term")
    if type(noise.shape) is not Normal:
        raise _Blocked("Gaussian innovation law is required, not merely matching moments")
    mu, sd = _fraction(noise.shape.mu), _fraction(noise.shape.sd)
    if sd < 0:
        raise _Blocked("invalid Gaussian scale")
    return noise, scale, multiplier, mu, sd


def _refs(root):
    """Complete object dependencies for an identity-checked root, excluding caches."""
    seen, stack = set(), [root]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (Block, Distribution)):
            stack.extend(value for key, value in vars(item).items() if key != "_keys")
        elif type(item) in (list, tuple):
            stack.extend(item)
        elif type(item) is dict:
            stack.extend(item.values())
    return seen


def _exact_text(x, radical=False):
    text = str(x)
    return f"sqrt({text})" if radical else text


def _parameter_metadata(parameter, canonical):
    result = {"ref": canonical.subject_ref(parameter), "role": "unverified",
              "domain": None}
    try:
        if type(parameter) is Uniform and parameter.shape == ():
            lo, hi = _fraction(parameter.low), _fraction(parameter.high)
            result.update(role="declared Uniform; freedom and independence unverified",
                          domain={"lower": str(lo), "upper": str(hi)})
        elif type(parameter) is Fixed and parameter.shape == ():
            value = _fixed(parameter)
            result.update(role="fixed", domain={"lower": str(value), "upper": str(value)})
    except (_Blocked, ValueError, TypeError, OverflowError):
        pass
    return result


def _target_metadata(target, canonical):
    target = Target("parameter", (target,)) if isinstance(target, Distribution) else target
    if type(target) is not Target or type(target.parameters) not in (tuple, list):
        return {"kind": "unsupported", "parameters": (), "parameter_details": (), "changes": None}
    return {"kind": target.kind if type(target.kind) is str else "unsupported",
            "parameters": tuple(canonical.subject_ref(p) for p in target.parameters),
            "parameter_details": tuple(_parameter_metadata(p, canonical) for p in target.parameters),
            "changes": None}


def _target_record(target, coordinates, points, rule, canonical):
    target = Target("parameter", (target,)) if isinstance(target, Distribution) else target
    if (type(target) is not Target or type(target.kind) is not str
            or type(target.parameters) not in (tuple, list) or not target.parameters):
        raise _Blocked("targets must be parameter objects or supported Target functions")
    if any(id(p) not in coordinates for p in target.parameters):
        raise _Blocked("target is not a function of the two changed coordinates")
    indices = [coordinates[id(p)] for p in target.parameters]
    if target.kind == "parameter" and len(indices) == 1:
        values = [point[indices[0]] for point in points]
        radical = rule == "W2"
    elif target.kind == "sum" and rule == "W1":
        values = [sum((point[i] for i in indices), Fraction(0)) for point in points]
        radical = False
    elif target.kind == "variance" and rule == "W2" and len(indices) == 1:
        values = [point[indices[0]] for point in points]
        radical = False
    elif target.kind == "sum_of_variances" and rule == "W2":
        values = [sum((point[i] for i in indices), Fraction(0)) for point in points]
        radical = False
    else:
        raise _Blocked("unsupported target function for this witness rule")
    return {**_target_metadata(target, canonical),
            "point_values_exact": tuple(_exact_text(v, radical) for v in values),
            "point_values_numeric": tuple(sqrt(float(v)) if radical else float(v) for v in values),
            "changes": values[0] != values[1]}


def _moments(leaves, parameters, values, length):
    replacements = {id(p): v for p, v in zip(parameters, values)}
    mean, amplitudes = 0., {}
    for leaf in leaves:
        if type(leaf) is Constant:
            mean += replacements[id(leaf.c)] if id(leaf.c) in replacements else float(_fixed(leaf.c))
        else:
            source, parameter, multiplier, mu, sd = _gaussian(leaf)
            scale = replacements[id(parameter)] if id(parameter) in replacements else float(_fixed(parameter))
            mean += scale * float(multiplier * mu)
            amplitudes[id(source)] = amplitudes.get(id(source), 0.) + scale * float(multiplier * sd)
    # All supported shocks are IID; the complete covariance is diagonal. Store its
    # diagonal instead of allocating T**2 entries just to check identical zeros.
    return np.full(length, mean, dtype=np.float64), np.full(length, sum(a*a for a in amplitudes.values()), dtype=np.float64)


def witnesses(world, *, targets, observation, delta=None) -> WitnessReport:
    """Try W1 level splitting and W2 independent Gaussian variance splitting.

    ``targets`` is a nonempty sequence of actual parameter objects or Target records.
    ``observation`` must explicitly describe complete raw observations at t=1,...,T.
    Two points are generated in the interiors of the declared Uniform domains. For W2,
    ``delta`` is a variance displacement, not a scale displacement. An inadmissible
    delta, an unchanged target or an unsupported parent gives unverified, not identified.
    """
    canonical = identity.world(world)
    root_ref = canonical.subject_ref(world)
    design = observation if type(observation) is ObservationDesign else None
    target_list = tuple(targets) if isinstance(targets, (tuple, list)) else ()
    records, checks = [], []
    for rule in ("W1", "W2"):
        premise_ids = []
        target_records = tuple(_target_metadata(t, canonical) for t in target_list)
        coordinates = tuple({**detail, "coordinate": "scale" if rule == "W2" else "level",
                             "selection": "requested target parameter; witness direction unverified"}
                            for target in target_records for detail in target["parameter_details"])
        attempted = {}
        numerical = {"implementation": "timegram.witnesses.raw_iid_moments/1",
                     "rule_version": f"timegram.witnesses.{rule}/1", "numpy_version": np.__version__,
                     "status": "not_attempted"}

        def require(name, condition, evidence):
            claim_id = f"{rule}.premise{len(premise_ids)}"
            premise_ids.append(claim_id)
            checks.append(ClaimRecord(claim_id, root_ref, name, "established" if condition else "unverified",
                                      scope="conditional_ideal_law", rule=f"{rule}/1", evidence=evidence))
            if not condition:
                raise _Blocked(evidence)

        try:
            require("supported_identity", canonical.status == "supported", "whole root needs a supported dag-v3 identity")
            require("raw_complete_observation", design is not None
                    and all(type(value) is str for value in (design.representation, design.missingness, design.initialization))
                    and type(design.time_origin) is int and type(design.presample) is int
                    and design.representation == "raw"
                    and design.missingness == "complete" and design.time_origin == 1
                    and design.initialization == "source_t1" and design.presample == 0
                    and type(design.history_length) is int and 1 <= design.history_length <= 4096,
                    "v1 requires complete raw t=1..T source observations, no prehistory, 1<=T<=4096")
            require("explicit_targets", bool(target_list), "give at least one explicit parameter/function target")
            require("additive_root", type(world) is Combine and type(world.schedule) is Ones,
                    "witnesses apply only to the complete additive root, not an arbitrary ancestor")
            leaves = _leaves(world)
            candidates = []
            for index, leaf in enumerate(leaves):
                if rule == "W1" and type(leaf) is Constant and type(leaf.c) is Uniform:
                    candidates.append((index, leaf.c, None))
                if rule == "W2":
                    try:
                        source, parameter, multiplier, mu, sd = _gaussian(leaf)
                        if type(parameter) is Uniform and multiplier == 1 and mu == 0 and sd == 1:
                            candidates.append((index, parameter, source))
                    except _Blocked:
                        pass
            require("two_free_coordinates", len(candidates) == 2,
                    "exactly two eligible free level/standard-Gaussian scale terms are required")
            (i, a, source_a), (j, b, source_b) = candidates
            coordinates = tuple({**_parameter_metadata(p, canonical),
                                 "coordinate": "scale" if rule == "W2" else "level",
                                 "selection": "candidate coordinate"} for p in (a, b))
            require("untied_coordinates", a is not b and leaves[i] is not leaves[j],
                    "the changed parameters and terms must be distinct, not tied or repeated")
            domains = [_domain(p, positive=rule == "W2") for p in (a, b)]
            require("innovation_independence", rule == "W1" or source_a is not source_b,
                    "W2 requires distinct independent standard Gaussian innovation objects")
            protected = {id(a), id(b)} | ({id(source_a), id(source_b)} if rule == "W2" else set())
            remainders = []
            for index, leaf in enumerate(leaves):
                if index in (i, j):
                    continue
                require(f"root_dependency:{index}", not (_refs(leaf) & protected),
                        "an extra root term depends on a changed coordinate or innovation")
                if type(leaf) is Constant:
                    remainders.append({"kind": "fixed_constant", "value": str(_fixed(leaf.c))})
                else:
                    _, parameter, multiplier, mu, sd = _gaussian(leaf)
                    fixed = _fixed(parameter)
                    remainders.append({"kind": "fixed_Gaussian", "mean": str(fixed*multiplier*mu),
                                       "sd": str(abs(fixed*multiplier*sd)),
                                       "signed_amplitude": str(fixed*multiplier*sd)})
            require("whole_root_fragment", True,
                    "every root term checked; only independent fixed-law Gaussian/constant remainder allowed")
            working_domains = [(lo*lo, hi*hi) for lo, hi in domains] if rule == "W2" else domains
            start = tuple((lo+hi)/2 for lo, hi in working_domains)
            delta_bounds = (max(working_domains[0][0]-start[0], start[1]-working_domains[1][1]),
                            min(working_domains[0][1]-start[0], start[1]-working_domains[1][0]))
            attempted = {"coordinates_used": "variances" if rule == "W2" else "levels",
                         "allowable_delta": {"lower": str(delta_bounds[0]), "upper": str(delta_bounds[1]),
                                             "bounds": "open", "excludes": "0"},
                         "fixed_remainder": tuple(remainders)}
            step = min(hi-lo for lo, hi in working_domains)/8 if delta is None else _fraction(delta)
            end = (start[0]+step, start[1]-step)
            points = (start, end)
            attempted["delta_exact"] = str(step)
            coordinates = tuple({"ref": canonical.subject_ref(p), "role": "independently_drawn; conditioned_on",
                                 "domain": {"lower": str(lo), "upper": str(hi), "required_point_location": "strict interior"},
                                 "coordinate": "scale" if rule == "W2" else "level",
                                 "point_values_exact": tuple(_exact_text(point[k], rule == "W2") for point in points)}
                                for k, (p, (lo, hi)) in enumerate(zip((a, b), domains)))
            require("admissible_distinct_points", step != 0 and all(lo < v < hi for v, (lo, hi) in zip(end, working_domains)),
                    "nonzero displacement must keep both exact parameter points inside their domains")
            evaluated = list(target_records)
            for target_index, target in enumerate(target_list):
                evaluated[target_index] = _target_record(target, {id(a):0, id(b):1}, points, rule, canonical)
                target_records = tuple(evaluated)
            require("target_changes", any(t["changes"] for t in target_records),
                    "this pair leaves the requested target unchanged; no nonidentification witness for it")
            numeric = [tuple(sqrt(float(v)) if rule == "W2" else float(v) for v in point) for point in points]
            numerical["parameter_points"] = tuple(numeric)
            require("representable_points", all(isfinite(v) and (rule != "W2" or v > 0) for point in numeric for v in point),
                    "numerical replay needs finite representable points and positive Gaussian scales")
            require("numerically_distinct_points", all(x != y for x, y in zip(*numeric)),
                    "distinct exact coordinates collapse to the same float64 value; numerical witness is unverified")
            m0, c0 = _moments(leaves, (a,b), numeric[0], design.history_length)
            m1, c1 = _moments(leaves, (a,b), numeric[1], design.history_length)
            matched = bool(all(np.isfinite(v).all() for v in (m0,m1,c0,c1)) and
                           np.allclose(m0,m1,atol=1e-12,rtol=1e-12) and np.allclose(c0,c1,atol=1e-12,rtol=1e-12))
            numerical.update(status="matched" if matched else "failed", matches=matched)
            require("numerical_replay", matched, "float64 raw mean/covariance replay at the stated length must agree")
            argument = ("(a+delta)+(b-delta)=a+b, so the conditional deterministic mean and complete raw law agree"
                        if rule == "W1" else
                        "independent centered Gaussian noises have covariance (v1+v2)I; (v1+delta)+(v2-delta)=v1+v2")
            records.append(WitnessRecord(f"{rule}.witness", root_ref, "parameter_nonidentification",
                "non_identified_witness", scope="conditional ideal law; all positive raw history lengths for this fragment",
                premises=tuple(premise_ids), rule=f"timegram.witnesses.{rule}/1", evidence=argument,
                targets=target_records, coordinates=coordinates, observation=design,
                conditioning=("continuous parameter values fixed, not marginalized over priors", "discrete structure fixed",
                              "independent fixed-law additive remainder does not depend on changed coordinates"),
                fragment="additive constants and independent fixed-law Gaussian remainder" if rule=="W1" else
                         "additive independent standard Gaussian sources with free scales and fixed-law remainder",
                witness={**attempted, "argument":argument},
                numerical_check={**numerical, "history_length":design.history_length, "dtype":"float64", "atol":1e-12, "rtol":1e-12,
                                 "mean_max_abs":float(np.max(np.abs(m0-m1))),
                                 "covariance_max_abs":float(np.max(np.abs(c0-c1))), "matches":matched,
                                 "covariance_representation":"diagonal; all off-diagonal entries exactly zero in this fragment",
                                 "scope":"finite numerical replay; analytic argument supplies exact equality"}))
        except (_Blocked, ValueError, TypeError, OverflowError) as error:
            records.append(WitnessRecord(f"{rule}.witness", root_ref, "parameter_nonidentification", "unverified",
                scope="conditional ideal law; candidate rule not established", premises=tuple(premise_ids),
                rule=f"timegram.witnesses.{rule}/1", evidence="No claim that the target is identified.",
                open_obligations=[str(error)], observation=design, targets=target_records,
                coordinates=coordinates, witness=attempted, numerical_check=numerical))
    return WitnessReport(canonical.digest, design, tuple(records), tuple(checks))
