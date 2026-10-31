import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation, DataSpec, NListRepresentation, SparseDataRepresentation
from pyrulearn.learners.associative import generate_cars, sort_by_measured_precedence
from pyrulearn.learners.associative import CBA
from pyrulearn.models import (
    ConceptCascade, ConceptSet, DecisionList, ConceptModel, FlatRuleSet, PairwiseModel, annotate_rules,
)
from pyrulearn.rule import Rule


def _rep(X, y, names=None):
    X = np.asarray(X, dtype=bool)
    ds = DataSpec(names or [f"f{i}" for i in range(X.shape[1])])
    return HorizontalBitmapRepresentation(ds, X, np.asarray(y))


# -- CBA-RG / CBA-CB, hand-verified -----------------------------------------

def test_generate_cars_on_a_hand_worked_dataset():
    # f0=1 -> a for every row (5/5, confidence 1.0); f1=1 -> a for 3 of the
    # 5 rows it covers (confidence 0.6) -- both clear the default 0.5
    # min_confidence.
    X = [[1, 1]] * 3 + [[1, 0]] * 2 + [[0, 1]] * 2 + [[0, 0]] * 1
    y = ["a"] * 3 + ["a"] * 2 + ["b"] * 2 + ["b"] * 1
    data = NListRepresentation.from_boolean(_rep(X, y))

    cars = generate_cars(data, min_support=0.01, min_confidence=0.5, max_len=1)
    by_items = {c.items: c for c in cars}
    assert set(by_items) == {(0,), (1,)}  # (1,)->b never clears 0.5 confidence (2/5)
    assert by_items[(0,)].target == "a" and by_items[(0,)].confidence == pytest.approx(1.0)
    assert by_items[(1,)].target == "a" and by_items[(1,)].confidence == pytest.approx(0.6)


def test_cba_select_prunes_a_kept_rule_that_makes_things_worse():
    # Same dataset as above, restricted to the two 1-item CARs, wrapped
    # as Rules, annotated, and precedence-sorted -- CBA's own pipeline,
    # now built on already-annotated rules rather than raw CARs. Walking
    # them: rule (f0,)->a claims rows 0-4 perfectly (0 errors); rule
    # (f1,)->a then claims the two remaining f1=1 rows (5, 6) -- both
    # actually class b, so it is wrong on both and (CBA-CB "M1": a rule
    # is kept only if it correctly classifies >=1 unclaimed row) is
    # skipped outright. The classifier is rule 1 alone, default b.
    X = [[1, 1]] * 3 + [[1, 0]] * 2 + [[0, 1]] * 2 + [[0, 0]] * 1
    y = ["a"] * 3 + ["a"] * 2 + ["b"] * 2 + ["b"] * 1
    data = NListRepresentation.from_boolean(_rep(X, y))

    cars = generate_cars(data, min_support=0.01, min_confidence=0.5, max_len=1)
    rules = [Rule(c.items, target=c.target, dataspec=data.spec) for c in cars]
    ordered_rules = sort_by_measured_precedence(annotate_rules(rules, data))
    kept, default = CBA._select(ordered_rules, data)

    assert [r.conditions[0].feature for r in kept] == [0]
    assert kept[0].target == "a"
    assert default == "b"


# -- CBA end to end -----------------------------------------------------

def _separable_multiclass(n=180, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n, 5)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "x",
         np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    return _rep(raw, y, names=list("abcde")), raw, y


def test_cba_fit_returns_a_decision_list_that_predicts():
    rep, raw, y = _separable_multiclass()
    model = CBA(min_support=0.02, min_confidence=0.5, max_len=3).fit(rep)
    assert isinstance(model, DecisionList)
    preds = np.asarray(model.predict(rep))
    acc = float(np.mean(preds == y))
    assert acc > 0.7, f"CBA training accuracy only {acc:.3f}"


def test_cba_target_class_returns_a_concept_model():
    rep, raw, y = _separable_multiclass()
    model = CBA(min_support=0.02, min_confidence=0.5, target_class="x").fit(rep)
    assert isinstance(model, ConceptModel)
    assert all(r.target == "x" for r in model.rules)


def test_cba_produces_the_decomposition_types_too():
    caps = CBA().produces()
    assert {DecisionList, ConceptSet, ConceptCascade, PairwiseModel} <= caps


def test_cba_auto_converts_a_boolean_representation_transparently():
    rep, raw, y = _separable_multiclass(n=60)
    assert not isinstance(rep, NListRepresentation)
    model = CBA(min_support=0.05, min_confidence=0.5, max_len=2).fit(rep)
    assert isinstance(model, DecisionList)


def test_cba_raises_a_clear_error_past_the_auto_convert_threshold():
    rep, raw, y = _separable_multiclass(n=60)
    with pytest.raises(ValueError, match="max_auto_convert_cells"):
        CBA(max_auto_convert_cells=1).fit(rep)


# -- CBA consuming an externally-supplied rule pool (rules=) -----------------

def test_cba_with_an_externally_supplied_rules_pool_skips_mining_entirely():
    # Same hand-worked dataset as the CBA-CB tests above, but supplied as
    # a plain, already-annotated FlatRuleSet via rules= -- min_support/
    # min_confidence/max_len are irrelevant here (never consulted).
    X = [[1, 1]] * 3 + [[1, 0]] * 2 + [[0, 1]] * 2 + [[0, 0]] * 1
    y = ["a"] * 3 + ["a"] * 2 + ["b"] * 2 + ["b"] * 1
    data = NListRepresentation.from_boolean(_rep(X, y))
    rules = [Rule((0,), target="a", dataspec=data.spec), Rule((1,), target="a", dataspec=data.spec)]
    pool = FlatRuleSet(annotate_rules(rules, data))

    model = CBA(rules=pool).fit(data)
    assert isinstance(model, DecisionList)
    assert [r.conditions[0].feature for r in model.rules] == [0]
    assert model.default_prediction == "b"


def test_cba_distills_correctly_on_a_sparse_representation_not_just_boolean_or_nlist():
    # coverage_select/CBA._select only ever call the universal
    # data.coverage(rule)/.n_samples/.y -- confirm they actually work
    # (not just "don't crash") on a representation that's neither
    # NListRepresentation nor HorizontalBitmapRepresentation.
    X = [[1, 1]] * 3 + [[1, 0]] * 2 + [[0, 1]] * 2 + [[0, 0]] * 1
    y = ["a"] * 3 + ["a"] * 2 + ["b"] * 2 + ["b"] * 1
    boolean = _rep(X, y)
    sparse = SparseDataRepresentation.from_boolean(boolean)
    rules = [Rule((0,), target="a", dataspec=sparse.spec), Rule((1,), target="a", dataspec=sparse.spec)]
    pool = FlatRuleSet(annotate_rules(rules, sparse))

    model = CBA(rules=pool).fit(sparse)
    assert isinstance(model, DecisionList)
    assert [r.conditions[0].feature for r in model.rules] == [0]
    assert model.default_prediction == "b"


def test_cba_random_forest_rules_round_trip():
    from sklearn.ensemble import RandomForestClassifier

    from _negation_helpers import neg_spec, neg_X
    from pyrulearn.interfaces.sklearn import from_random_forest

    rep, raw, y = _separable_multiclass(n=200)
    ds = neg_spec(list("abcde"))
    data = HorizontalBitmapRepresentation(ds, neg_X(raw), y)

    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(neg_X(raw), y)
    forest_rules = from_random_forest(rf, dataspec=ds)
    pool = FlatRuleSet(annotate_rules(forest_rules.rules, data))

    model = CBA(rules=pool).fit(data)
    assert isinstance(model, DecisionList)
    assert len(model.rules) > 0
    preds = np.asarray(model.predict(data))
    assert preds.shape == y.shape
