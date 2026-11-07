import itertools

import numpy as np
import pytest

from pyrulearn.data import DataSpec, HorizontalBitmapRepresentation, SparseDataRepresentation
from pyrulearn.learners.deep import DeepLocalSearch, nnf_forward
from pyrulearn.models import DeepModel

SPEC = DataSpec(list("abcdefg"))
X = np.array(list(itertools.product([0, 1], repeat=7)), dtype=bool)
a, b, c, d, e, f, g = X.T
SEIP = (a | b) & (c | d) & (e | (f & g))
DATA = HorizontalBitmapRepresentation(SPEC, X, np.where(SEIP, "pos", "neg"))


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
    model_values = dm.node_values(DATA)
    assert all((v == m).all() for v, m in zip(values, model_values))


def test_local_search_learns_the_formula():
    m = DeepLocalSearch(layers=[6, 3, 1], max_flips=20000, restarts=2).fit(DATA)
    assert isinstance(m, DeepModel) and m.layers == [6, 3, 1]
    assert m.outputs == ["pos"] and m.labels == ["neg", "pos"]  # two classes: the rarer one is the output
    assert (m.predict(DATA) == DATA.y).mean() > 0.95
    assert m.provenance.learner == "DeepLocalSearch"


def test_local_search_keeps_a_perfect_start():
    initial = DeepModel(SPEC, ["pos"], weights=_seip_weights())
    m = DeepLocalSearch(layers=[6, 3, 1], initial=initial).fit(DATA)
    assert all((w == i).all() for w, i in zip(m.weights, initial.weights))
    with pytest.raises(ValueError, match="initial has layers"):
        DeepLocalSearch(layers=[4, 3, 1], initial=initial).fit(DATA)


def test_local_search_is_reproducible_and_takes_any_representation():
    learner = DeepLocalSearch(layers=[4, 2, 2], max_flips=2000, random_state=3)
    m1, m2 = learner.fit(DATA), learner.fit(SparseDataRepresentation(SPEC, X, DATA.y))
    assert all((w1 == w2).all() for w1, w2 in zip(m1.weights, m2.weights))


def test_local_search_default_architecture_and_several_classes():
    y = np.where(a & b, "ab", np.where(c, "c", "other"))
    assert DeepLocalSearch(max_flips=10).fit(HorizontalBitmapRepresentation(SPEC, X, y)).layers == [3]
    # "other" is not (a & b) and not c: it needs the negated features
    spec = DataSpec(list("abcdefg") + [f"not {x}" for x in "abcdefg"])
    data = HorizontalBitmapRepresentation(spec, np.hstack([X, ~X]), y)
    m = DeepLocalSearch(layers=[6], max_flips=20000, restarts=2).fit(data)
    assert m.outputs == ["ab", "c", "other"]
    assert (m.predict(data) == y).mean() > 0.9


def test_local_search_checks_its_parameters():
    with pytest.raises(ValueError):
        DeepLocalSearch(noise=1.5)
    with pytest.raises(ValueError):
        DeepLocalSearch(max_flips=-1)
    with pytest.raises(ValueError, match="two classes"):
        DeepLocalSearch().fit(HorizontalBitmapRepresentation(SPEC, X, np.array(["x"] * 128)))
