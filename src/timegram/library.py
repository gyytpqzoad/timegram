"""The library of pre-trained estimators, a local cache in this release.

A problem and its training configuration are reduced to a canonical form; its hash is
the library key. `pretrain()` looks the key up first. Equivalent specifications must get
the same key, so the canonical form

* numbers shared blocks and shared parameters by first appearance (graph, not tree);
* flattens and sorts the operands of addition (it is commutative);
* describes named compounds by their trees (plus their identifiability floor);
* writes every distribution with all its settings.

Entries live in `$TIMEGRAM_HOME` (default ~/.cache/timegram): one folder per key with the
weights, the canonical specification, the configuration, the training history and a
short model card. Names are aliases kept in `names.json`.

Legacy keys are retained for existing experiments. Because their canonical form can
alias different shared graphs, loading with a replacement problem and overwriting an
entry also check an order-sensitive representation of the stored problem. A mismatch
fails explicitly; it never silently retrains or replaces another world's weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import types
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch

from .dists import Distribution
from .model import Block, Combine, Feed, Ones

VERSION = "0.1"
SCHEMA = "2"       # canonical-form version: bump when the form changes, so old keys are not reused


def home() -> Path:
    return Path(os.environ.get("TIMEGRAM_HOME", Path.home() / ".cache" / "timegram"))


# ---------------------------------------------------------------------------------------
# canonical form
# ---------------------------------------------------------------------------------------
class _Canon:
    def __init__(self, *, strict: bool = False):
        self.ids: dict[int, str] = {}
        self.strict = strict

    def ref(self, obj, prefix) -> str | None:
        if id(obj) in self.ids:
            return self.ids[id(obj)]
        self.ids[id(obj)] = f"{prefix}{len(self.ids)}"
        return None

    def value(self, v) -> str:
        if isinstance(v, Distribution):
            return self.dist(v)
        if isinstance(v, Block):
            return self.block(v)
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}={self.value(x)}" for k, x in sorted(v.items())) + "}"
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(self.value(x) for x in v) + "]"
        if isinstance(v, np.ndarray):
            return self.value(v.tolist())
        if isinstance(v, float):
            return repr(v if self.strict else round(v, 12))
        if callable(v):
            return getattr(v, "__name__", repr(v))
        if hasattr(v, "__dict__"):                               # e.g. a schedule
            body = ",".join(f"{k}={self.value(x)}" for k, x in sorted(vars(v).items()) if not k.startswith("_"))
            return f"{type(v).__name__}({body})"
        return repr(v)

    def dist(self, d: Distribution) -> str:
        seen = self.ref(d, "rv")
        if seen:
            return seen
        binding = self.ids[id(d)]
        attrs = vars(d) if not is_dataclass(d) else {f.name: getattr(d, f.name) for f in fields(d)}
        body = ",".join(f"{k}={self.value(v)}" for k, v in sorted(attrs.items()) if not k.startswith("_"))
        value = f"{type(d).__name__}({body})"
        return f"{binding}={value}" if self.strict else value

    def block(self, b: Block) -> str:
        seen = self.ref(b, "node")
        if seen:
            return seen
        binding = self.ids[id(b)]
        value = self._block(b)
        return f"{binding}={value}" if self.strict else value

    def _block(self, b: Block) -> str:
        if getattr(b, "name", None) == "compose" and hasattr(b, "stages"):   # standalone composite
            return self.chain(b.source, b.stages)
        if isinstance(b, Feed) and getattr(b.down, "name", None) == "compose":
            return self.chain(b.up, b.down.stages)               # x >> (a >> b) is (x >> a) >> b
        if isinstance(b, Feed):                                  # the fed block's definition, then its input
            down = self.block(b.down) if hasattr(b.down, "tree") else self.definition(b.down)
            return f"feed({down}<-{self.block(b.up)})"
        if hasattr(b, "tree"):                                   # compound: its tree
            return f"{b.name}[floor={getattr(b, '_floor', 1)}]{{{self.block(b.tree)}}}"
        if isinstance(b, Combine) and isinstance(b.schedule, Ones):
            operands = [self.block(o) for o in b.operands]
            return "add(" + ",".join(operands if self.strict else sorted(operands)) + ")"
        kids = ",".join(self.block(c) for _, c in b.inputs())
        return f"{self.definition(b)}({kids})"

    def chain(self, up, stages) -> str:
        """Composition written as nested feeds, so associativity does not change the key."""
        out = self.block(up)
        for stage in stages:
            down = self.block(stage) if hasattr(stage, "tree") else self.definition(stage)
            out = f"feed({down}<-{out})"
        return out

    def definition(self, b: Block) -> str:
        """A block's own settings and parameters, fixed ones included, without its inputs."""
        inputs = {"source", "input", "tree", "up", "down", "operands", "block"}
        attrs = {k: v for k, v in vars(b).items() if k not in inputs and not k.startswith("_")}
        if hasattr(b, "canonical_fields"):              # extension blocks may declare their own
            attrs["fields"] = b.canonical_fields()
        return f"{type(b).__name__}[" + ",".join(f"{k}={self.value(v)}" for k, v in sorted(attrs.items())) + "]"


def canonical(problem) -> str:
    return _canonical(problem, _Canon())


def _canonical(problem, c: _Canon) -> str:
    from ._dependencies import validate_acyclic
    validate_acyclic(problem.model)
    from .problem import Params
    est = [repr(e) for e in problem.resolved_estimands(problem.default_horizon())]
    for e in problem.resolved_estimands(problem.default_horizon()):
        if isinstance(e, Params):                    # the head's actual layout (paths, in order)
            est.append("head=" + ",".join(e.resolve(problem.model)))
    parts = [f"schema={SCHEMA}", f"model={c.block(problem.model)}", f"observe={problem.observation!r}",
             f"estimate={est}", f"loss={problem.loss_fn!r}"]
    if not problem.objective_spec.is_default:        # ERM keys are unchanged from earlier versions
        parts.append(f"objective={problem.objective_spec!r}")
    return "|".join(parts)


def key(problem, config) -> str:
    text = canonical(problem) + "|train=" + json.dumps(config.key_dict(), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------------------
def exists(k: str) -> bool:
    return (home() / k / "estimator.pt").exists()


class CacheCollisionError(ValueError):
    """A legacy key refers to a different (or unverified) stored specification."""


def _check_compatible(k: str, stored: dict, problem, config=None):
    """Conservative guard, not general graph equivalence or a new key schema.

    Preserve operand order and explicitly bind reference IDs, so the known shared-add
    collision cannot pass. Equivalent reorderings may be rejected; callers can still
    load the original problem by its key/alias without attaching a replacement problem.
    """
    same = (_canonical(stored["problem"], _Canon(strict=True)) ==
            _canonical(problem, _Canon(strict=True)))
    if config is not None:
        same = same and stored["config"].key_dict() == config.key_dict()
    if not same:
        raise CacheCollisionError(
            f"library key {k!r} has a different or reordered stored specification; "
            "refusing to reuse or overwrite its weights. Load the original key/alias "
            "without a replacement problem to inspect it, or use a separate TIMEGRAM_HOME "
            "for the new problem. No automatic retraining or cache migration was performed.")


def check_compatible(k: str, problem, config=None):
    """Check an existing entry before an explicit retrain would spend any compute."""
    if exists(k):
        stored = _torch_load(home() / k / "estimator.pt")
        _check_compatible(k, stored, problem, config)


def save(estimator, name: str | None = None) -> Path:
    from .problem import Problem  # noqa: F401  (documents what is pickled)
    check_compatible(estimator.key, estimator.problem, estimator.config)
    d = home() / estimator.key
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"problem": estimator.problem, "config": estimator.config, "state": estimator.net.state_dict(),
                "horizon": estimator.horizon, "slots": estimator.slots, "history": estimator.history,
                "outputs": estimator.problem.layout()[2], "version": VERSION}, d / "estimator.pt")
    (d / "spec.txt").write_text(canonical(estimator.problem), encoding="utf-8")
    (d / "config.json").write_text(json.dumps(estimator.config.key_dict(), indent=1, default=str))
    (d / "card.md").write_text(card(estimator), encoding="utf-8")
    if name:
        alias(name, estimator.key)
    return d


# Classes that early checkpoints pickled under a script's __main__ and that now live in the
# package. Only these exact names are redirected; anything else resolves as usual.
_RELOCATED = {("__main__", "LogitNormalEts"): ("timegram.compounds", "LogitNormalEts")}


class _RelocatingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module, name = _RELOCATED.get((module, name), (module, name))
        return super().find_class(module, name)


_pickle = types.SimpleNamespace(Unpickler=_RelocatingUnpickler, load=pickle.load, loads=pickle.loads,
                                __name__="timegram_relocating_pickle")


def _torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle)


def load(k: str, problem=None):
    from .estimator import Estimator
    st = _torch_load(home() / k / "estimator.pt")
    if problem is not None:
        _check_compatible(k, st, problem)
    cfg = st["config"]
    h, slots = st["horizon"], st["slots"]
    out = st.get("outputs", h + sum(size for _, size in slots.values()))
    net = cfg.net.build(cfg.lengths[1], out)
    net.load_state_dict(st["state"])
    net.eval()
    return Estimator(problem or st["problem"], net, cfg, h, slots, st["history"], key=k)


def alias(name: str, k: str):
    path = home() / "names.json"
    names = json.loads(path.read_text()) if path.exists() else {}
    names[name] = k
    path.write_text(json.dumps(names, indent=1))


def resolve(name: str) -> str:
    """The key of a library entry, by name (alias) or key."""
    path = home() / "names.json"
    names = json.loads(path.read_text()) if path.exists() else {}
    return names.get(name, name)


def pretrained(name: str, problem=None):
    """Load a library entry by name (or key). `problem` attaches a current specification
    object only after a conservative identity check. Omit it to retain the stored
    specification, including when inspecting an old or differently ordered graph."""
    return load(resolve(name), problem=problem)


def list_entries():
    import pandas as pd
    names = json.loads((home() / "names.json").read_text()) if (home() / "names.json").exists() else {}
    by_key = {v: k for k, v in names.items()}
    rows = []
    for d in sorted(home().glob("*/estimator.pt")):
        cfg = json.loads((d.parent / "config.json").read_text())
        rows.append({"key": d.parent.name, "name": by_key.get(d.parent.name, ""),
                     "budget": cfg["budget"], "lengths": cfg["lengths"]})
    return pd.DataFrame(rows)


def card(estimator) -> str:
    cfg = estimator.config
    last = estimator.history[-1] if estimator.history else {}
    return (f"# timegram estimator {estimator.key}\n\n```\n{estimator.problem.describe()}\n```\n\n"
            f"- training: {cfg.budget:,} simulated series, lengths {cfg.lengths}, batch {cfg.batch_size}"
            f" ({cfg.batch_size // cfg.replicates} processes x {cfg.replicates} replicates, accumulate "
            f"{cfg.accumulate}), seed {cfg.seed}" + (f", from {cfg.init}" if cfg.init else "") + "\n"
            f"- network: {cfg.net.to_dict()}\n- optimizer: {cfg.optimizer.to_dict()}\n"
            f"- final training loss: {last.get('loss', float('nan')):.4f}\n- timegram {VERSION}\n"
            f"- inputs are standardized per series and forecasts mapped back: predictions are location-"
            f"scale equivariant (level of this world: {_level(estimator.problem.model)}; "
            f"see timegram.equivariance)\n")


def _level(model) -> str:
    from .equivariance import level_status
    try:
        return level_status(model)
    except Exception:                                # a card must never fail to write
        return "unverified"
