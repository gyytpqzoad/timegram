"""An opt-in, deterministic neighbourhood of a forecasting world.

Enumeration constructs and inspects graphs; it never simulates or trains them. The
``timegram.moves/1`` catalogue follows the reviewed research recipe, not a search
policy or a promise that any neighbour is a better forecaster. Its world digests
belong to ``dag-v3`` and are not estimator-library keys.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from typing import Any, Iterator

from .blocks import ar, damped_trend, exp, noise, normalize
from .compounds import ets
from .dists import student_t, uniform
from .families import arfima
from .fractional import fractional_integrate
from .model import Block, Combine, OneHot, Transform, convex, select
from .volatility import garch


SCHEMA = "timegram.moves/1"
DEFAULT_MOVE_IDS = (
    "innovations.ar1", "innovations.student", "innovations.garch11",
    "innovations.long_memory", "add.damped_trend", "add.long_memory",
    "combine.alternative.damped_ets", "combine.mix_self", "positive.exp",
)


@dataclass(frozen=True)
class MoveCandidate:
    move_ids: tuple[str, ...]
    world: Block
    report: Any
    world_key: str
    identity: Any
    parent_identity: Any
    constructor: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class SkippedMove:
    move_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class MoveSet:
    candidates: tuple[MoveCandidate, ...]
    skipped: tuple[SkippedMove, ...]
    parent_identity: Any
    schema: str = SCHEMA

    def __iter__(self) -> Iterator[MoveCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index]


def _accepts_input(world: Block, active=None) -> bool:
    """Only transforms and recursively well-formed selections can receive a feed."""
    if isinstance(world, Transform):
        return True
    if type(world) is not Combine or type(world.schedule) is not OneHot:
        return False
    active = set() if active is None else active
    if id(world) in active:
        return False
    active.add(id(world))
    result = bool(world.operands) and all(_accepts_input(w, active) for w in world.operands)
    active.remove(id(world))
    return result


def _feed_innovations(world: Block, factory) -> Block:
    """Lower a feed through selection only; ``world`` is already a private copy.

    Reuse its schedules, hence their choice RVs and parameter ties. Each leaf
    occurrence gets fresh innovations. Both addition and convex combination have
    multiple active operands, so this rule deliberately does not distribute there.
    """
    if isinstance(world, Transform):
        return factory() >> world
    if type(world) is Combine and type(world.schedule) is OneHot:
        return Combine([_feed_innovations(w, factory) for w in world.operands], world.schedule)
    raise ValueError("root_does_not_accept_input")


def _construct(move_id: str, w: Block):
    if move_id == "innovations.ar1":
        return (_feed_innovations(w, lambda: noise() >> ar(1, phi=uniform(0.3, 0.95))),
                "fresh AR(1) innovations >> each eligible leaf", {"phi_uniform": (0.3, 0.95)})
    if move_id == "innovations.student":
        return (_feed_innovations(w, lambda: noise(student_t(4))),
                "fresh Student-t innovations >> each eligible leaf", {"df": 4})
    if move_id == "innovations.garch11":
        return (_feed_innovations(w, lambda: noise() >> garch(1, 1, burn_in=200)),
                "fresh GARCH(1,1) innovations >> each eligible leaf", {"p": 1, "q": 1, "burn_in": 200})
    if move_id == "innovations.long_memory":
        return (_feed_innovations(w, lambda: noise() >> fractional_integrate(uniform(0.1, 0.45))),
                "fresh fractionally integrated innovations >> each eligible leaf",
                {"d_uniform": (0.1, 0.45), "presample": 0, "memory": None})
    if move_id == "add.damped_trend":
        return (w + damped_trend(b=uniform(-1, 1), phi=uniform(0.8, 0.98)),
                "w + damped_trend", {"b_uniform": (-1, 1), "phi_uniform": (0.8, 0.98)})
    if move_id == "add.long_memory":
        return (w + arfima(d=0.2, memory=128, presample=0),
                "w + arfima", {"d": 0.2, "memory": 128, "presample": 0})
    if move_id == "combine.alternative.damped_ets":
        alternative = select(ets("AAdN"), ets("AAdA", period=4), ets("AAdA", period=12))
        return (select(w, alternative), "select(w, select(AAdN, AAdA@4, AAdA@12))",
                {"alternative_specs": ("AAdN", "AAdA@4", "AAdA@12")})
    if move_id == "combine.mix_self":
        independent = w.copy()
        return (convex(w >> normalize("minmax"), independent >> normalize("minmax")),
                "convex(normalize(w), normalize(independent copy))", {"normalize": "minmax"})
    if move_id == "positive.exp":
        return (w >> normalize("std") >> exp(), "w >> normalize(std) >> exp", {"normalize": "std"})
    raise ValueError(f"unknown move ID {move_id!r}")


def moves(world: Block, *, history_length: int | None = 128,
          keep_parent: float | None = None, move_ids=None) -> MoveSet:
    """Return checked, keyed neighbours plus explicit skipped-move records.

    Nine fixed moves are considered in ``DEFAULT_MOVE_IDS`` order. ``move_ids``
    optionally selects/reorders that catalogue; repeated IDs are deduplicated by
    supported world identity. Every candidate gets a deep copy of the parent,
    preserving internal ties without sharing mutable nodes with other candidates.

    ``keep_parent=None`` or ``0`` leaves the moved world unwrapped. A probability
    in (0, 1] wraps it as ``select(parent.copy(), moved, probs=[p, 1-p])`` in that
    order. At p=1 only the unchanged parent is sampled. This option adds a new
    selection label; no existing parameter or ``which()`` target mapping is implied.

    Unverified type properties remain visible; structural errors or unsupported
    identities are skipped. Equality recognizes only the declared dag-v3 fragment.
    The selection-feed rule preserves the choice distribution and parameter ties,
    not seed-for-seed paths or a particular RNG-consumption order.
    """
    from . import identity

    if not isinstance(world, Block):
        raise TypeError("moves expects a world Block, not an estimation problem")
    if keep_parent is not None and (isinstance(keep_parent, bool) or not isinstance(keep_parent, Real)
                                   or not isfinite(keep_parent) or not 0 <= keep_parent <= 1):
        raise ValueError("keep_parent must be None or a finite probability in [0, 1]")
    selected = DEFAULT_MOVE_IDS if move_ids is None else tuple(move_ids)
    unknown = [name for name in selected if name not in DEFAULT_MOVE_IDS]
    if unknown:
        raise ValueError(f"unknown move IDs: {unknown!r}")
    parent = identity.world(world)
    if parent.status != "supported" or parent.digest is None:
        return MoveSet((), tuple(SkippedMove(name, "identity_unverified", str(parent.unsupported))
                                 for name in selected), parent)
    candidates, skipped, seen = [], [], {}
    for name in selected:
        if name.startswith("innovations.") and not _accepts_input(world):
            skipped.append(SkippedMove(name, "root_does_not_accept_input"))
            continue
        try:
            candidate, constructor, settings = _construct(name, world.copy())
            if keep_parent:
                candidate = select(world.copy(), candidate, probs=[keep_parent, 1 - keep_parent])
            settings = {**settings, "keep_parent": 0.0 if keep_parent is None else float(keep_parent)}
            report = candidate.typecheck(history_length=history_length)
        except (TypeError, ValueError) as error:
            skipped.append(SkippedMove(name, "construction_or_check_error", str(error)))
            continue
        if report.has_errors:
            skipped.append(SkippedMove(name, "typecheck_error", str(report.diagnostics)))
            continue
        record = identity.world(candidate)
        if record.status != "supported" or record.digest is None:
            skipped.append(SkippedMove(name, "identity_unverified", str(record.unsupported)))
            continue
        if record.digest in seen:
            index = seen[record.digest]
            old = candidates[index]
            candidates[index] = replace(old, move_ids=old.move_ids + (name,))
        else:
            seen[record.digest] = len(candidates)
            candidates.append(MoveCandidate((name,), candidate, report, record.digest, record,
                                            parent, constructor, settings))
    return MoveSet(tuple(candidates), tuple(skipped), parent)
