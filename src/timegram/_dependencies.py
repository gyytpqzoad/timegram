"""Dependency validation: reject cyclic model definitions before anything is drawn.

A model is a directed acyclic graph of blocks and parameters (temporal feedback lives
inside closed primitives such as `ar`). Sharing a node is fine; a node that depends on
itself is not. The check covers inactive alternatives of a selection and the trees of
named compounds. A fed transform's unused default source is not a dependency.

This is a structural check, not a statement about stationarity or identifiability.
(Adapted from the grammar_v2_20261028 prototype.)
"""
from __future__ import annotations


def _dependencies(obj, fed=False):
    from .dists import Distribution
    from .model import Block, Combine, Feed

    if isinstance(obj, Feed):
        yield obj.up, False
        yield obj.down, True
        return
    if isinstance(obj, Block):
        for port, child in obj.inputs():
            if not (fed and port == "input"):
                yield child, False
        yield from ((p, False) for p in obj.params().values())
        if isinstance(obj, Combine):
            yield from ((p, False) for p in obj.schedule.params().values())
        if hasattr(obj, "tree"):
            yield obj.tree, False
        if isinstance(getattr(obj, "shape", None), Distribution):
            yield obj.shape, False
    elif isinstance(obj, Distribution):
        yield from ((p, False) for p in obj.parents())
        for name, value in vars(obj).items():      # nested public distributions, e.g. mixture parts
            if name.startswith("_"):
                continue
            items = value if isinstance(value, (list, tuple)) else (value,)
            yield from ((v, False) for v in items if isinstance(v, (Distribution, Block)))


def _description(obj):
    return type(obj).__name__                     # never call label(): it may recurse on a cycle


def validate_acyclic(root):
    """Raise ValueError naming the cycle if the definition rooted at `root` has one."""
    done, active, path = set(), {}, []

    def enter(obj, fed):
        key = (id(obj), fed)
        if key in active:
            chain = path[active[key]:] + [obj]
            raise ValueError("dependency cycle: " + " -> ".join(map(_description, chain)))
        if key in done:
            return None
        active[key] = len(path)
        path.append(obj)
        return key, iter(_dependencies(obj, fed))

    stack = [enter(root, False)]
    while stack:
        key, children = stack[-1]
        try:
            child, fed = next(children)
        except StopIteration:
            stack.pop()
            path.pop()
            active.pop(key)
            done.add(key)
            continue
        frame = enter(child, fed)
        if frame is not None:
            stack.append(frame)
