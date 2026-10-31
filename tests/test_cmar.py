import numpy as np
import pytest

from pyrulearn.combiners import HeuristicVoteCombiner
from pyrulearn.data import HorizontalBitmapRepresentation, DataSpec, NListRepresentation
from pyrulearn.heuristics import ChiSquare
from pyrulearn.learners.associative import CMAR
from pyrulearn.models import ConceptCascade, ConceptSet, ConceptModel, FlatRuleSet, PairwiseModel

from _negation_helpers import neg_spec, neg_X


def _rep(X, y, names=None):
    X = np.asarray(X, dtype=bool)
    ds = DataSpec(names or [f"f{i}" for i in range(X.shape[1])])
    return HorizontalBitmapRepresentation(ds, X, np.asarray(y))


def test_cmar_drops_an_insignificant_car_that_would_pass_cbas_confidence_bar():
    # f0 covers 5 rows (3 'a', 2 'b') -- confidence 0.6 toward 'a',
    # comfortably over CBA's own 0.5 min_confidence bar -- but its
    # chi-square is exactly 0 (tp*tn == fp*fn: 3*2 == 2*3): its class
    # split among covered rows exactly matches the covered/uncovered
    # split overall, genuinely uncorrelated. f1 perfectly separates
    # a/b (confidence 1.0, chi-square 10, comfortably >= the default
    # 3.841 threshold). CMAR must keep f1's rule and drop f0's (and the
    # (f0,f1) combination, chi-square ~2.86, also insignificant) even
    # though CBA's own confidence bar alone would accept all of them.
    X = np.zeros((10, 2), dtype=bool)
    X[[0, 1, 2, 6, 7], 0] = True  # f0: 3 of the 6 'a' rows + 2 of the 4 'b' rows
    X[0:6, 1] = True              # f1: exactly the 6 'a' rows
    y = ["a"] * 6 + ["b"] * 4
    rep = _rep(X, y)

    model = CMAR(min_support=0.01, min_confidence=0.5, max_len=2,
                significance_threshold=3.841).fit(rep)
    used_features = {lit.feature for r in model.rules for lit in r.conditions}
    assert used_features == {1}


def test_cmar_significance_threshold_is_configurable():
    # same dataset -- lowering the bar below f0's own chi-square (0)
    # would need a negative threshold, which is nonsensical; instead
    # confirm raising it excludes even f1's rule (chi-square 10).
    X = np.zeros((10, 2), dtype=bool)
    X[[0, 1, 2, 6, 7], 0] = True
    X[0:6, 1] = True
    y = ["a"] * 6 + ["b"] * 4
    rep = _rep(X, y)

    model = CMAR(min_support=0.01, min_confidence=0.5, max_len=1,
                significance_threshold=50.0).fit(rep)
    assert len(model.rules) == 0


def _separable_multiclass(n=200, seed=0):
    # negation-enabled: the ground truth needs "NOT b" (x = a AND NOT b),
    # which only exists as a mineable item once negation features are
    # present -- without them, no CAR can ever target 'x' at all, no
    # matter how CMAR itself behaves (confirmed by direct comparison
    # while writing this test: CBA's own DecisionList happens to paper
    # over the same gap via its per-position majority default, but
    # CMAR's single global default can't, so this isn't a fair fixture
    # for either without negation).
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n, 5)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "x",
         np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    ds = neg_spec(list("abcde"))
    return HorizontalBitmapRepresentation(ds, neg_X(raw), y), raw, y


def test_cmar_fit_uses_a_heuristic_vote_combiner_weighted_by_chi_square():
    rep, raw, y = _separable_multiclass()
    model = CMAR(min_support=0.02, min_confidence=0.5, max_len=3).fit(rep)
    assert isinstance(model, FlatRuleSet)
    assert isinstance(model.combiner, HeuristicVoteCombiner)
    assert isinstance(model.combiner.heuristic, ChiSquare)
    preds = np.asarray(model.predict(rep))
    acc = float(np.mean(preds == y))
    assert acc > 0.6, f"CMAR training accuracy only {acc:.3f}"


def test_cmar_target_class_returns_a_concept_model():
    rep, raw, y = _separable_multiclass()
    model = CMAR(min_support=0.02, min_confidence=0.5, target_class="x").fit(rep)
    assert isinstance(model, ConceptModel)
    assert all(r.target == "x" for r in model.rules)


def test_cmar_produces_the_decomposition_types_too():
    caps = CMAR().produces()
    assert {FlatRuleSet, ConceptSet, ConceptCascade, PairwiseModel} <= caps


def test_cmar_auto_converts_a_boolean_representation_transparently():
    rep, raw, y = _separable_multiclass(n=60)
    assert not isinstance(rep, NListRepresentation)
    model = CMAR(min_support=0.05, min_confidence=0.5, max_len=2).fit(rep)
    assert isinstance(model, FlatRuleSet)


def test_cmar_raises_a_clear_error_past_the_auto_convert_threshold():
    rep, raw, y = _separable_multiclass(n=60)
    with pytest.raises(ValueError, match="max_auto_convert_cells"):
        CMAR(max_auto_convert_cells=1).fit(rep)


def test_cmar_prune_drops_a_rule_that_correctly_classifies_nothing_new():
    # Per-class pruning keeps a rule only if it correctly classifies >=1
    # still-unclaimed row of its class (CMAR's paper, same as CBA-CB) --
    # not merely covers something. r_wrong targets 'a' but covers only
    # 'b' rows; the old any-new-coverage condition kept it (55 vs. 8
    # rules on `vote`), the correct one drops it.
    from pyrulearn.models import annotate_rules
    from pyrulearn.rule import Rule

    X = [[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]]
    y = ["b", "b", "a", "a", "a", "b"]
    data = NListRepresentation.from_boolean(_rep(X, y))
    r_wrong = Rule((0,), target="a", dataspec=data.spec)   # covers rows 0,1: both 'b'
    r_right = Rule((1,), target="a", dataspec=data.spec)   # covers rows 2,3: both 'a'
    rules = annotate_rules([r_wrong, r_right], data)

    kept = CMAR()._prune(rules, data, ChiSquare())
    assert [r.conditions[0].feature for r in kept] == [1]


# -- CMAR consuming an externally-supplied rule pool (rules=) ----------------

def test_cmar_with_an_externally_supplied_rules_pool_skips_mining_entirely():
    rep, raw, y = _separable_multiclass()
    mined = CMAR(min_support=0.02, min_confidence=0.5, max_len=3).fit(rep)  # any FlatRuleSet works as rules=
    pool = FlatRuleSet(list(mined.rules))

    # min_support/min_confidence/max_len are irrelevant now -- rules= is given.
    model = CMAR(rules=pool, min_support=0.9, min_confidence=0.99).fit(rep)
    assert isinstance(model, FlatRuleSet)
    assert len(model.rules) > 0


def test_cmar_random_forest_rules_round_trip():
    from sklearn.ensemble import RandomForestClassifier

    from pyrulearn.interfaces.sklearn import from_random_forest
    from pyrulearn.models import annotate_rules

    rep, raw, y = _separable_multiclass(n=200)
    ds = neg_spec(list("abcde"))
    data = HorizontalBitmapRepresentation(ds, neg_X(raw), y)

    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(neg_X(raw), y)
    forest_rules = from_random_forest(rf, dataspec=ds)
    pool = FlatRuleSet(annotate_rules(forest_rules.rules, data))

    model = CMAR(rules=pool).fit(data)
    assert isinstance(model, FlatRuleSet)
    preds = np.asarray(model.predict(data))
    assert preds.shape == y.shape
