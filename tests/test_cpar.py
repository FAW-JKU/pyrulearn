import numpy as np
import pandas as pd
import pytest

from pyrulearn.combiners import TopKMeanCombiner
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.data import HorizontalBitmapRepresentation, DataSpec, NListRepresentation, SparseDataRepresentation
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.heuristics import CoveredPositives, Gain, GeneralizedMEstimate, Laplace, Relative
from pyrulearn.learners.cpar import CPAR, DenseCPAR
from pyrulearn.learners.seco import (
    AdaBoostReweighting, AdditiveReweighting, PositiveWeightBelow, RemovalCovering, WeightedCovering,
)
from pyrulearn.models import ConceptCascade, ConceptSet, FlatRuleSet, annotate_rules
from pyrulearn.rule import Rule

from _negation_helpers import neg_spec, neg_X


def _data(n=600, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    flip = rng.random(n) < noise
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


# ------------------------------------------------------------ TopKMeanCombiner

def _topk_fixture():
    # one feature per rule: class a's rules cover 4, 1, 1 of its rows, b's 3 and 3
    spec = DataSpec([f"f{i}" for i in range(5)])
    X = np.zeros((12, 5), dtype=bool)
    for j, rows in enumerate([range(0, 4), [4], [5], range(6, 9), range(9, 12)]):
        X[list(rows), j] = True
    y = np.array(["a"] * 6 + ["b"] * 6)
    train = HorizontalBitmapRepresentation(spec, X, y)
    rules = annotate_rules([Rule([j], target=t, dataspec=spec)
                            for j, t in enumerate(["a", "a", "a", "b", "b"])], train)
    everything = HorizontalBitmapRepresentation(spec, np.ones((1, 5), dtype=bool))
    return rules, everything


def test_top_k_mean_depends_on_k():
    rules, row = _topk_fixture()
    best1 = FlatRuleSet(rules, combiner=TopKMeanCombiner(CoveredPositives(), k=1))
    best2 = FlatRuleSet(rules, combiner=TopKMeanCombiner(CoveredPositives(), k=2))
    assert best1.predict(row)[0] == "a"          # a's best rule (4) beats b's (3)
    assert best2.predict(row)[0] == "b"          # but a's best two average 2.5 < 3
    # a class with fewer than k covering rules averages what it has
    best5 = FlatRuleSet(rules, combiner=TopKMeanCombiner(CoveredPositives(), k=5))
    assert best5.predict(row)[0] == "b"          # a: 6/3 = 2, b: 6/2 = 3
    assert "best 2 rules" in best2.combiner.describe()
    with pytest.raises(ValueError):
        TopKMeanCombiner(k=0)


# ------------------------------------------------------------ CPAR

def test_cpar_builds_a_concept_set_predicted_by_top_k_expected_accuracy():
    data = _data()
    model = CPAR().fit(data)
    assert isinstance(model, ConceptSet)
    assert isinstance(model.combiner, TopKMeanCombiner) and model.combiner.k == 5
    h = model.combiner.heuristic
    assert isinstance(h, GeneralizedMEstimate) and (h.m, h.cost) == (2, 0.5)
    for concept in model.concepts:
        bodies = [tuple(sorted(l.feature for l in r.conditions)) for r in concept.rules]
        assert len(bodies) == len(set(bodies)) > 1
        assert all(r.target == concept.label for r in concept.rules)
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85
    assert model.to_string().splitlines()[0].startswith("% conflict resolution: mean GeneralizedMEstimate")


def test_one_search_copies_the_rule_at_equally_good_conditions():
    # y = f0 or f1, symmetric: f0 and f1 have the same gain, so one search yields both
    rng = np.random.default_rng(0)
    raw = rng.random((400, 3)) < 0.5
    y = np.where(raw[:, 0] | raw[:, 1], "pos", "neg")
    data = HorizontalBitmapRepresentation(neg_spec(["f0", "f1", "f2"]), neg_X(raw), y)
    X = data.X
    pos = y == "pos"
    w = np.ones(len(y))
    copying = DenseCPAR(gain_similarity=0.5)._grow(X, X.astype(float), pos, w)
    single = DenseCPAR(gain_similarity=1.0)._grow(X, X.astype(float), pos, w)
    assert sorted(copying) == [(0,), (2,)]                     # f0 and f1, from one search
    assert single == [(2,)]                                    # f1 is slightly better on this sample


def test_cpar_stops_once_the_positive_weight_has_decayed():
    data = _data()
    X, pos = data.X, data.y == "pos"
    bodies = DenseCPAR()._rules_for(X, X.astype(float), pos, np.ones(len(pos)))
    covered = np.zeros(len(pos), dtype=int)
    for b in bodies:
        covered += np.all(X[:, list(b)], axis=1)
    # every rule covers positives, and most positives were covered several times
    assert all(np.any(np.all(X[:, list(b)], axis=1) & pos) for b in bodies)
    assert np.mean(covered[pos] >= 2) > 0.5


def test_cpar_multiclass_and_data_weights():
    rng = np.random.default_rng(2)
    raw = rng.random((300, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = CPAR().fit(data)
    assert [c.label for c in model.concepts] == ["a", "b", "c"]
    assert model.combiner.heuristic.m == 3
    assert np.mean(np.asarray(model.predict(data)) == y) > 0.95
    weighted = CPAR().fit(data.with_weights(np.where(y == "c", 3.0, 1.0)))
    assert isinstance(weighted.concepts[0].rules[0].stats().confusion.rule_stats("a").tp, float)


@pytest.mark.parametrize("cls", [CPAR, DenseCPAR])
def test_cpar_rejects_bad_arguments(cls):
    with pytest.raises(ValueError):
        cls(gain_similarity=0.0)
    with pytest.raises(ValueError, match="gain over the rule"):
        cls(heuristic=Laplace())                       # not a gain: wrap it in Relative(Gain(...))
    with pytest.raises(ValueError, match="rule weights"):
        cls(covering=WeightedCovering(AdaBoostReweighting()))
    one = HorizontalBitmapRepresentation(DataSpec(["a"]), np.ones((3, 1), bool), np.array(["x"] * 3))
    with pytest.raises(ValueError, match="two classes"):
        cls().fit(one)


# ------------------------------------------------------------ CPAR vs. DenseCPAR

def _numeric_data(n=300, seed=3):
    # numeric thresholds: fixing one closes the attribute's others (propagation)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "c": rng.choice(list("pqr"), n)})
    y = np.where((df["x"] > 0.3) & (df["c"] != "q") | (df["z"] < -0.8), "pos", "neg")
    df["y"] = np.where(rng.random(n) < 0.1, np.where(y == "pos", "neg", "pos"), y)
    spec = build_dataspec(df, target="y", discretizer=TreeDiscretizer(max_intervals=6)).build()
    return HorizontalBitmapRepresentation(spec, binarize(spec, df), df["y"].to_numpy())


def _multiclass_data():
    rng = np.random.default_rng(2)
    raw = rng.random((300, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)


@pytest.mark.parametrize("make", [
    _data,
    _numeric_data,
    lambda: _data(noise=0.0).with_weights(np.linspace(0.5, 2.0, 600)),
    _multiclass_data,
])
def test_cpar_and_dense_cpar_learn_the_same_model(make):
    # an implied condition has FOIL gain 0 < min_gain, so DenseCPAR never picks
    # one; constraint propagation only spares CPAR from scoring it
    data = make()
    assert CPAR().fit(data).to_string() == DenseCPAR().fit(data).to_string()


@pytest.mark.parametrize("params", [
    dict(heuristic=Relative(Gain(Laplace())), min_gain=0.01),
    dict(covering=WeightedCovering(AdditiveReweighting(), PositiveWeightBelow(0.2), max_rounds=None)),
    dict(covering=RemovalCovering()),
    dict(combiner=TopKMeanCombiner(Laplace(), k=2)),
])
def test_cpar_components_are_exchangeable(params):
    data = _numeric_data()
    model = CPAR(**params).fit(data)
    assert model.to_string() == DenseCPAR(**params).fit(data).to_string()
    assert model.to_string() != CPAR().fit(data).to_string()       # the component actually changed something


def test_cpar_runs_on_every_representation():
    data = _numeric_data()
    expected = CPAR().fit(data).to_string()
    for rep in (NListRepresentation.from_boolean(data), SparseDataRepresentation.from_boolean(data)):
        assert CPAR().fit(rep).to_string() == expected


def test_cpar_decomposes_like_the_seco_learners():
    data = _multiclass_data()
    cascade = CPAR().fit(data, model=ConceptCascade)
    assert isinstance(cascade, ConceptCascade) and len(cascade.concepts) == 2
    assert np.mean(np.asarray(cascade.predict(data)) == data.y) > 0.9
