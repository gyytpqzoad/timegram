"""Definition graphs: a model as a DAG, one node per shared realization.

`model.graph()` returns the graph data; `model.plot_graph(path)` draws it. With
`expanded=True`, named compounds are shown as the primitives they stand for, with their
input bound to the actual upstream process. Nothing is simulated or drawn. This is a
picture of the generating definition, not a time-unrolled or causal graph.
(Adapted from the grammar_v2_20261028 prototype.)
"""
from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ._dependencies import validate_acyclic
from .dists import Fixed
from .model import Combine, Convex, Feed, OneHot, Ones

MOTIVATIONS = {
    "noise": "New unpredictable shocks enter the process.",
    "constant": "A starting level or fixed baseline.",
    "trend": "A fixed slope carries growth or decline forward.",
    "damped_trend": "A trend gradually loses strength during extrapolation.",
    "pattern": "A fixed pattern repeats with the calendar.",
    "ar": "Inertia: previous departures feed into the current value.",
    "ma": "A shock leaves a finite trail over subsequent observations.",
    "integrate": "Accumulate shocks: a disturbance can permanently move the level.",
    "difference": "Compare changes across time to remove an accumulating component.",
    "lag": "Delay an effect before it reaches another component.",
    "normalize": "Put a process on a common scale, using only what is observed.",
    "scale": "Control the strength of a component or its response to a shared shock.",
    "add": "Several effects contribute to the same observed quantity.",
    "convex": "A weighted average of processes, weights fixed per series (convex combination).",
    "select": "Choose one generating model for the entire series: the world of model selection.",
    "ets": "One shock jointly moves the observation and its evolving components.",
    "arima": "Short-term dependence combined with accumulating changes.",
    "arfima": "Short-term dynamics combined with slowly decaying shock effects.",
    "fractional_integrate": "Shock effects fade as a power law rather than exponentially.",
    "fractional_difference": "Remove a fractional amount of slowly accumulating dependence.",
    "garch": "Past disturbance sizes and uncertainty set the scale of the next shock.",
}


@dataclass
class ModelGraph:
    nodes: list[dict]
    edges: list[tuple[str, str]]
    output: str
    expanded: bool

    def to_dict(self):
        return {"nodes": self.nodes, "edges": self.edges, "output": self.output,
                "expanded": self.expanded, "meaning": "generating definition"}


def _label(block) -> str:
    if block.name == "noise" and type(block.shape).__name__ == "Mixture":
        return f"Mixture innovations\nchoice per observation\nweights = {block.shape.describe()}"
    if isinstance(block, Combine):
        if isinstance(block.schedule, Convex):
            return f"Convex combination\nweights = {block.schedule.w.describe()}"
        if isinstance(block.schedule, OneHot):
            return "Model selection\none model per series"
        if isinstance(block.schedule, Ones):
            return "Add contributions"
    if block.name in ("ar", "ma"):
        name = "phi" if block.name == "ar" else "theta"
        parameter = block.params()[name]
        if isinstance(parameter, Fixed):
            return block.label() + f"\n{name} = {parameter.describe()}"
    return block.label()


def model_graph(model, expanded: bool = False) -> ModelGraph:
    """Graph of actual inputs, one node per shared realization."""
    validate_acyclic(model)
    nodes, edges, seen = [], [], {}

    def visit(block, bindings, scope):
        if id(block) in bindings:                       # a compound's input placeholder
            return bindings[id(block)]
        key = (id(block), scope)
        if key in seen:
            return seen[key]
        semantic = block.down if isinstance(block, Feed) else block
        # a composite (a >> b on transforms) is syntax, not a named model: always show its stages
        if hasattr(semantic, "tree") and (expanded or semantic.name == "compose"):
            upstream = block.up if isinstance(block, Feed) else semantic.source
            nested = {**bindings, id(semantic.input): visit(upstream, bindings, scope)}
            result = visit(semantic.tree, nested, scope + (id(block),))
            seen[key] = result
            return result
        result = f"n{len(nodes)}"
        seen[key] = result
        nodes.append({"id": result, "label": _label(semantic), "kind": semantic.name,
                      "motivation": MOTIVATIONS.get(semantic.name, "A named generative component."),
                      "parameters": {name: p.describe() for name, p in semantic.params().items()}})
        for _, child in block.inputs():
            edges.append((visit(child, bindings, scope), result))
        return result

    output = visit(model, {}, ())
    for node in nodes:
        node["shared"] = sum(a == node["id"] for a, _ in edges) > 1
    return ModelGraph(nodes, edges, output, bool(expanded))


def plot_graph(model, path=None, expanded: bool = False, title: str | None = None):
    """Draw the definition left to right (data flow); returns the matplotlib figure.
    Needs matplotlib and networkx, imported only here."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as DrawingPath

    data = model_graph(model, expanded=expanded)
    graph = nx.DiGraph()
    graph.add_nodes_from(node["id"] for node in data.nodes)
    graph.add_edges_from(data.edges)
    layers = [list(layer) for layer in nx.topological_generations(graph)]
    positions, ranks = {}, {}
    for x, layer in enumerate(layers):
        for i, node in enumerate(layer):
            positions[node] = (3.15 * x, (len(layer) - 1) / 2 - i)
            ranks[node] = x
    widest = max(map(len, layers))
    fig, ax = plt.subplots(figsize=(max(8.0, 2.5 * len(layers)), max(3.0, 1.1 * widest + 1.4)))
    low = min(y for _, y in positions.values())
    high = max(y for _, y in positions.values())
    top_lane = bottom_lane = 0
    for (source, target), multiplicity in Counter(data.edges).items():
        (x0, y0), (x1, y1) = positions[source], positions[target]
        if ranks[target] - ranks[source] > 1:            # route long edges around the layers
            if y0 >= y1:
                lane, top_lane = high + 0.7 + 0.3 * top_lane, top_lane + 1
            else:
                lane, bottom_lane = low - 0.7 - 0.3 * bottom_lane, bottom_lane + 1
            vertices = [(x0 + 1.22, y0), (x0 + 1.5, y0), (x0 + 1.5, lane),
                        (x1 - 1.5, lane), (x1 - 1.5, y1), (x1 - 1.22, y1)]
            route = DrawingPath(vertices, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * 5)
            arrow = FancyArrowPatch(path=route, arrowstyle="-|>", mutation_scale=13,
                                    linewidth=1.25, color="#6b7280", zorder=1)
        else:
            arrow = FancyArrowPatch((x0 + 1.22, y0), (x1 - 1.22, y1), arrowstyle="-|>", mutation_scale=13,
                                    linewidth=1.25, color="#6b7280", connectionstyle="arc3,rad=0.035", zorder=1)
        ax.add_patch(arrow)
        if multiplicity > 1:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, f"{multiplicity} uses", fontsize=8, ha="center")
    for node in data.nodes:
        x, y = positions[node["id"]]
        color = "#dbeafe" if node["shared"] else ("#dcfce7" if node["id"] == data.output else "#f3f4f6")
        ax.add_patch(FancyBboxPatch((x - 1.22, y - 0.32), 2.44, 0.64, boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor="#9ca3af", linewidth=0.8, zorder=2))
        label = "\n".join(textwrap.fill(line, 31) for line in node["label"].splitlines())
        if label.count("\n") > 3:
            label = "\n".join(label.splitlines()[:3]) + " ..."
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, zorder=3)
    ax.set_xlim(-1.55, 3.15 * (len(layers) - 1) + 1.55)
    ax.set_ylim(low - 0.7 - 0.3 * bottom_lane, high + 0.7 + 0.3 * top_lane)
    ax.axis("off")
    ax.set_title(title or ("Expanded model definition" if expanded else "Model definition"), fontsize=14, pad=18)
    fig.text(0.02, 0.025, "Arrows show generating flow. Blue nodes are shared realizations; green is the output.",
             fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.83)
    if path is not None:
        fig.savefig(Path(path), dpi=170, bbox_inches="tight")
    return fig
