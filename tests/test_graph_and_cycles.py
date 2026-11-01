"""Definition graphs and cycle rejection (ported from the grammar_v2_20261028 prototype)."""
import numpy as np
import pytest

import timegram as tg
from timegram import library
from timegram.dists import Fixed


# ---------------------------------------------------------------------------------------
# definition graphs
# ---------------------------------------------------------------------------------------
def test_expanded_ets_binds_one_actual_innovation_source():
    model = tg.noise(tg.student_t(8)) >> tg.ets("ANN", alpha=0.3, level0=2.0)
    assert len([n for n in model.graph().nodes if n["kind"] == "noise"]) == 1
    expanded = model.graph(expanded=True)
    noise = [n for n in expanded.nodes if n["kind"] == "noise"]
    assert len(noise) == 1 and "student_t" in noise[0]["label"] and noise[0]["shared"]
    assert not any(n["kind"] in ("input", "ets") for n in expanded.nodes)


def test_graph_distinguishes_shared_and_independent_sources():
    e = tg.noise()
    shared = (e + e).graph()
    assert len([n for n in shared.nodes if n["kind"] == "noise"]) == 1 and len(shared.edges) == 2
    independent = (tg.noise() + tg.noise()).graph()
    assert len([n for n in independent.nodes if n["kind"] == "noise"]) == 2


def test_graph_names_selection_mix_and_normalize():
    a, b = tg.ar(1), tg.ma(1)
    assert "one model per series" in (a | b).graph().nodes[0]["label"]
    assert "Convex combination" in tg.mix(a, b, tg.ets("ANN")).graph().nodes[0]["label"]
    kinds = {n["kind"] for n in (tg.ets("ANN") >> tg.normalize("std")).graph().nodes}
    assert "normalize" in kinds


def test_graph_draws_nothing_and_does_not_modify_the_model():
    model = tg.ets("AAA", period=4)
    before = str(model)
    assert model.graph(expanded=True).to_dict()["meaning"] == "generating definition"
    assert str(model) == before
    assert all(n["motivation"] for n in model.graph(expanded=True).nodes)


def test_plot_graph_saves_an_expanded_named_model(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = tmp_path / "ses.png"
    plt.close(tg.ets("ANN", alpha=0.3, level0=2.0).plot_graph(target, expanded=True))
    assert target.is_file() and target.stat().st_size > 1000


# ---------------------------------------------------------------------------------------
# cycles are rejected before anything is drawn
# ---------------------------------------------------------------------------------------
def _cyclic_sum():
    model = tg.add(tg.noise())
    model.operands.append(model)
    return model


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_simulation_rejects_block_cycles(backend):
    with pytest.raises(ValueError, match="cycle"):
        _cyclic_sum().simulate(length=10, backend=backend)


@pytest.mark.parametrize("backend", ["reference", "fast"])
def test_parameter_cycles_are_rejected_everywhere(backend):
    dist = tg.uniform() + 1
    dist.args = (dist, Fixed(1))
    model = tg.noise(scale=dist)
    with pytest.raises(ValueError, match="cycle"):
        model.simulate(length=10, backend=backend)
    with pytest.raises(ValueError, match="cycle"):
        model.graph()
    with pytest.raises(ValueError, match="cycle"):
        library.canonical(model.estimate(tg.forecast(h=2)))


def test_cycles_in_inactive_alternatives_and_compound_trees_are_rejected():
    with pytest.raises(ValueError, match="cycle"):
        tg.select(tg.noise(), _cyclic_sum(), probs=[1, 0]).simulate(length=10)
    named = tg.ets("ANN")
    named.tree = named
    for model in (named, tg.noise() >> named):
        with pytest.raises(ValueError, match="cycle"):
            model.simulate(length=16)


def test_unused_default_source_is_not_a_false_cycle():
    transform = tg.ar(1, phi=[0.5])
    transform.source = transform
    (tg.noise() >> transform).simulate(length=16)              # fine: the default source is unused
    with pytest.raises(ValueError, match="cycle"):
        transform.simulate(length=16)


# ---------------------------------------------------------------------------------------
# small ports
# ---------------------------------------------------------------------------------------
def test_mixture_weights_are_validated():
    with pytest.raises(ValueError):
        tg.mixture(tg.normal(), tg.normal(0, 5), weights=[0.5, -0.1])
    with pytest.raises(ValueError):
        tg.mixture(tg.normal(), weights=[0.5, 0.5])


def test_damped_trend_reference_equals_its_definition():
    from timegram.simulate import Series
    s = Series(np.random.default_rng(0))
    s.t = np.arange(-2, 30)
    y = s.value(tg.damped_trend(b=2.0, phi=0.9))
    expected = [0.0 if t < 1 else 2.0 * sum(0.9 ** k for k in range(1, t + 1)) for t in s.t]
    np.testing.assert_allclose(y, expected, rtol=1e-12)


def test_default_history_is_128():
    assert tg.TrainConfig().lengths == (16, 128)


def test_library_key_carries_a_schema_version():
    assert library.canonical(tg.ets("ANN").estimate(tg.forecast(h=2))).startswith(f"schema={library.SCHEMA}|")
