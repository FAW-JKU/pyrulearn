import itertools

import numpy as np
import pandas as pd
import pytest

from pyrulearn.data import DataSpec, DataSpecBuilder, HorizontalBitmapRepresentation, SparseDataRepresentation
from pyrulearn.data.io import build_dataspec, encode
from pyrulearn.learners.deep import (
    DeepGenetic, DeepLocalSearch, ModelInit, RandomInit, RuleSetInit, contradictions, nnf_forward,
)
from pyrulearn.models import ConceptModel, DeepModel
from pyrulearn.rule import Rule

SPEC = DataSpec(list("abcdefg"))
X = np.array(list(itertools.product([0, 1], repeat=7)), dtype=bool)
a, b, c, d, e, f, g = X.T
SEIP = (a | b) & (c | d) & (e | (f & g))
DATA = HorizontalBitmapRepresentation(SPEC, X, np.where(SEIP, "pos", "neg"))
# the same through `build_dataspec`: Boolean attributes, `x` and `not x`, mutually exclusive
FRAME = pd.DataFrame(X, columns=list("abcdefg"))
NEG_SPEC = build_dataspec(FRAME).build()
NEG_X = np.asarray(encode(NEG_SPEC, FRAME).X, dtype=bool)
NEG_DATA = HorizontalBitmapRepresentation(NEG_SPEC, NEG_X, DATA.y)


def _seip_weights():
    W1 = np.zeros((6, 7), dtype=int)
    for node, inputs in enumerate([[0], [1], [2], [3], [4], [5, 6]]):
        W1[node, inputs] = 1
    W2 = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    return [W1, W2, [[1, 1, 1]], [[1]]]


def test_nnf_forward_matches_the_model():
    dm = DeepModel(SPEC, ["pos"], weights=_seip_weights())
    values = nnf_forward(dm.weights, X)
    assert [v.shape for v in values] == [(6, 128), (3, 128), (1, 128), (1, 128)]
    assert values[-1][0].tolist() == SEIP.tolist()
    assert all((v == m).all() for v, m in zip(values, dm.node_values(DATA)))


def test_flip_errors_are_those_of_flipping():
    rng = np.random.default_rng(0)
    learner = DeepLocalSearch()
    for _ in range(10):
        shapes, below = [], 9
        for size in [int(x) for x in rng.integers(1, 6, size=rng.choice([1, 3]))] + [2]:
            shapes.append((size, below))
            below = size
        weights = [rng.random(s) < 0.4 for s in shapes]
        Xr, targets, sw = rng.random((40, 9)) < 0.5, rng.random((2, 40)) < 0.5, rng.random(40)
        conflicts = rng.random((9, 9)) < 0.3
        np.fill_diagonal(conflicts, False)
        values = nnf_forward(weights, Xr)
        for layer in range(len(weights)):
            cf = conflicts if layer == 0 else None
            got = learner._flip_errors(weights, values, Xr.T, layer, targets, sw, cf)
            for k, j in np.ndindex(weights[layer].shape):
                flipped = [w.copy() for w in weights]
                DeepLocalSearch._flip(flipped, layer, k, j, cf)
                want = ((nnf_forward(flipped, Xr)[-1] != targets) * sw).sum()
                assert got[k, j] == pytest.approx(want)


def test_local_search_learns_the_formula():
    m = DeepLocalSearch(layers=[6, 3, 1]).fit(NEG_DATA)
    assert isinstance(m, DeepModel) and m.layers == [6, 3, 1]
    assert m.outputs == ["pos"] and m.labels == ["neg", "pos"]  # two classes: the rarer one is the output
    assert (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.9
    assert m.provenance.learner == "DeepLocalSearch"


def test_two_classes_of_equal_size_default_to_the_other_one():
    y = np.where(a, "x", "y")  # 64 each
    m = DeepLocalSearch(layers=[2]).fit(HorizontalBitmapRepresentation(SPEC, X, y))
    assert m.outputs == ["x"] and m.default_prediction == "y"
    assert (m.predict(HorizontalBitmapRepresentation(SPEC, X)) == y).all()


def test_consistent_moves_never_build_a_contradictory_rule():
    m = DeepLocalSearch(layers=[8]).fit(NEG_DATA)
    for row in m.weights[0]:
        names = {NEG_SPEC.feature_names[j] for j in np.flatnonzero(row)}
        assert not any(f"not {x}" in names and x in names for x in "abcdefg")


def test_contradictions_come_from_the_constraints():
    b = DataSpecBuilder()
    b.add_numeric("x", [3, 5])
    b.add_nominal("c", ["r", "g", "b"])
    b.add_boolean("f")
    spec = b.build()
    pairs = {tuple(sorted((spec.feature_names[i], spec.feature_names[j])))
             for i, j in zip(*np.nonzero(contradictions(spec)))}
    assert {("x<3", "x>=3"), ("x<3", "x>=5"), ("x<5", "x>=5"), ("c=g", "c=r"), ("c!=r", "c=r"),
            ("f", "not f")} <= pairs
    assert ("c!=g", "c!=r") not in pairs and ("c=r", "f") not in pairs
    assert not contradictions(DataSpec(["f", "not f"])).any()  # bare names: no constraints


def test_local_search_keeps_a_perfect_start():
    initial = DeepModel(SPEC, ["pos"], weights=_seip_weights())
    m = DeepLocalSearch(layers=[6, 3, 1], init=ModelInit(initial)).fit(DATA)
    assert all((w == i).all() for w, i in zip(m.weights, initial.weights))
    with pytest.raises(ValueError, match="initial model has layers"):
        DeepLocalSearch(layers=[4, 3, 1], init=ModelInit(initial)).fit(DATA)


def test_rule_set_init_starts_as_the_rules_dnf():
    rules = ConceptModel([Rule([0, 2, 4], target="pos", dataspec=SPEC), Rule([1, 3, 5, 6], target="pos", dataspec=SPEC)])
    expected = (a & c & e) | (b & d & f & g)
    for layers in ([4], [4, 3, 2], [5, 2, 3, 2, 1]):
        start = RuleSetInit(rules).weights(DATA, ["pos"], [w.shape for w in DeepModel(SPEC, ["pos"], layers).weights],
                                           np.random.default_rng(1))
        assert nnf_forward(start, X)[-1][0].tolist() == expected.tolist()
    with pytest.raises(ValueError, match="first layer of 1"):
        DeepLocalSearch(layers=[1], init=RuleSetInit(rules)).fit(DATA)


def test_rule_set_init_takes_a_learner():
    from pyrulearn.learners.seco import Pypper

    m = DeepLocalSearch(layers=[12], init=RuleSetInit(Pypper()), n_epochs=0).fit(NEG_DATA)
    pypper = Pypper().fit(NEG_DATA)
    assert (m.predict(NEG_DATA) == pypper.predict(NEG_DATA)).mean() >= 0.99


def test_a_subclass_can_override_the_start():
    class Empty(DeepLocalSearch):
        def _initialize(self, data, labels, shapes, rng):
            return [np.zeros(s, dtype=bool) for s in shapes]

    m = Empty(layers=[3], n_epochs=0, max_flips=0).fit(DATA)
    assert not any(w.any() for w in m.weights)


def test_local_search_is_reproducible_and_takes_any_representation():
    learner = DeepLocalSearch(layers=[4, 2, 2], n_epochs=1, random_state=3)
    m1, m2 = learner.fit(DATA), learner.fit(SparseDataRepresentation(SPEC, X, DATA.y))
    assert all((w1 == w2).all() for w1, w2 in zip(m1.weights, m2.weights))


def test_local_search_on_several_classes():
    y = np.where(a & b, "ab", np.where(c, "c", "other"))
    assert DeepLocalSearch(n_epochs=1).fit(HorizontalBitmapRepresentation(SPEC, X, y)).layers == [3]  # ceil(sqrt(7))
    m = DeepLocalSearch(layers=[6]).fit(HorizontalBitmapRepresentation(NEG_SPEC, NEG_X, y))
    assert m.outputs == ["ab", "c", "other"]
    assert (m.predict(HorizontalBitmapRepresentation(NEG_SPEC, NEG_X)) == y).mean() > 0.9


def test_beck_start_and_moves_are_available():
    m = DeepLocalSearch(layers=[6, 3, 1], moves="attribute", init=RandomInit(min_rule_length=0, consistent=False)).fit(NEG_DATA)
    assert (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.8


def test_alignment_undoes_a_permutation_of_the_hidden_nodes():
    rng = np.random.default_rng(0)
    for sizes in ([6, 5, 4, 1], [7, 1]):
        shapes, below = [], 7
        for size in sizes:
            shapes.append((size, below))
            below = size
        a_ = [rng.random(s) < 0.4 for s in shapes]
        b_ = [w.copy() for w in a_]
        for layer in range(len(b_) - 1):
            perm = rng.permutation(b_[layer].shape[0])
            b_[layer], b_[layer + 1] = b_[layer][perm], b_[layer + 1][:, perm]
        back = DeepGenetic._aligned(a_, b_, X)
        assert all((x == y).all() for x, y in zip(nnf_forward(back, X), nnf_forward(a_, X)))


def test_genetic_learns_the_formula_and_is_reproducible():
    learner = DeepGenetic(layers=[6, 3, 1], population=8, generations=5)
    m = learner.fit(NEG_DATA)
    assert isinstance(m, DeepModel) and (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.9
    assert all((w1 == w2).all() for w1, w2 in zip(m.weights, learner.fit(NEG_DATA).weights))


def test_genetic_keeps_a_perfect_seed():
    seed = ModelInit(DeepModel(SPEC, ["pos"], weights=_seip_weights()))
    m = DeepGenetic(layers=[6, 3, 1], population=4, generations=3, seeds=[seed]).fit(DATA)
    assert (m.predict(DATA) == DATA.y).all()


@pytest.mark.parametrize("learner", [DeepLocalSearch(layers=[6, 3, 1], trace=True),
                                     DeepGenetic(layers=[6, 3, 1], population=6, generations=4, trace=True)])
def test_trace_records_every_new_best(learner):
    m = learner.fit(NEG_DATA)
    trace = m.trace
    assert len(trace) >= 2
    assert all(a.train_error > b.train_error for a, b in zip(trace, trace[1:]))
    assert all(a.evaluations <= b.evaluations and a.seconds <= b.seconds for a, b in zip(trace, trace[1:]))
    assert trace[0].evaluations == 128  # the first start, evaluated on every row
    assert all((w == v).all() for w, v in zip(trace[-1].weights, m.weights))  # the last point is the model
    first = m.trace_model(0)
    assert (first.predict(NEG_DATA) != NEG_DATA.y).sum() == trace[0].train_error
    assert DeepLocalSearch(layers=[3]).fit(NEG_DATA).trace is None


def test_parameters_are_checked():
    with pytest.raises(ValueError):
        DeepLocalSearch(moves="random")
    with pytest.raises(ValueError):
        DeepLocalSearch(max_flips=-1)
    with pytest.raises(ValueError):
        RandomInit(init_prob=1.5)
    with pytest.raises(ValueError):
        DeepGenetic(population=3, elite=3)
    with pytest.raises(ValueError, match="seeds"):
        DeepGenetic(population=2, elite=0, seeds=[RandomInit()] * 3)
    with pytest.raises(ValueError, match="two classes"):
        DeepLocalSearch().fit(HorizontalBitmapRepresentation(SPEC, X, np.array(["x"] * 128)))
