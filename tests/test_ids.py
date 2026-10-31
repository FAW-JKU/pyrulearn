import numpy as np
import pytest

from pyrulearn.combiners import HeuristicMaxCombiner
from pyrulearn.data import HorizontalBitmapRepresentation, DataSpec, NListRepresentation
from pyrulearn.heuristics import FBeta
from pyrulearn.learners.associative import generate_cars
from pyrulearn.learners.ids import (
    IDS, GreedySelection, SmoothLocalSearch, _Candidate, _candidates_from_rules, _greedy_select, _objective,
)
from pyrulearn.models import ConceptCascade, ConceptSet, ConceptModel, FlatRuleSet, PairwiseModel, annotate_rules
from pyrulearn.rule import Rule

from _negation_helpers import neg_spec, neg_X


def _rep(X, y, names=None):
    X = np.asarray(X, dtype=bool)
    ds = DataSpec(names or [f"f{i}" for i in range(X.shape[1])])
    return HorizontalBitmapRepresentation(ds, X, np.asarray(y))


def _hand_candidates():
    # 4 rows, 2 disjoint, individually-perfect rules: c0 -> 'a' on rows
    # 0-1, c1 -> 'b' on rows 2-3. Both length 1, both distinct itemsets,
    # 2 classes -- small enough to hand-compute every f1..f7 term.
    y = np.array(["a", "a", "b", "b"])
    c0 = _Candidate(rule=None, items=(0,), target="a",
                    mask=np.array([True, True, False, False]),
                    correct_mask=np.array([True, True, False, False]))
    c1 = _Candidate(rule=None, items=(1,), target="b",
                    mask=np.array([False, False, True, True]),
                    correct_mask=np.array([False, False, True, True]))
    return [c0, c1], y


def test_objective_matches_hand_computed_values_including_the_normalization():
    candidates, y = _hand_candidates()
    classes = ["a", "b"]
    n, s_size, lmax = 4, 2, 1
    weights = (1.0,) * 7

    # kept={}: f1=1 f2=1 f3=1 f4=1 f5=0 f6=1 f7=0 -> sum 5.0
    empty = _objective(set(), candidates, classes, n, s_size, lmax, weights)
    assert empty == pytest.approx(5.0)

    # kept={0}: f1=.5 f2=.5 f3=14/16 f4=1 f5=.5 f6=1 f7=.5 -> sum 4.875
    one = _objective({0}, candidates, classes, n, s_size, lmax, weights)
    assert one == pytest.approx(4.875)

    # kept={0,1}: f1=0 f2=0 f3=12/16 f4=1 f5=1 f6=1 f7=1 -> sum 4.75
    both = _objective({0, 1}, candidates, classes, n, s_size, lmax, weights)
    assert both == pytest.approx(4.75)

    # Both rules together classify every row correctly (f7=1, the max
    # possible) yet score *lower*, under naive equal weighting, than one
    # rule alone -- which itself scores lower than the empty decision
    # set. The interpretability terms (f1/f2/f3) tax every rule added,
    # even one that adds only new, non-overlapping, correct coverage,
    # more than precision/recall/class-coverage reward it at this toy
    # dataset's tiny scale. A real trade-off in the paper's own
    # objective, not a bug -- exactly why it fits lambdas by coordinate
    # ascent instead of shipping default weights at all.
    assert empty > one > both


def test_greedy_select_adds_both_rules_once_weights_favor_recall():
    candidates, y = _hand_candidates()
    classes = ["a", "b"]
    # weight only f7 (recall) -- now the union of both rules' correct
    # coverage (1.0) strictly beats either alone (0.5) or none (0.0),
    # so greedy must add both.
    weights = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    kept = _greedy_select(candidates, classes, n=4, s_size=2, lmax=1, weights=weights)
    assert kept == {0, 1}


def test_greedy_select_stays_empty_under_naive_equal_weights():
    candidates, y = _hand_candidates()
    classes = ["a", "b"]
    kept = _greedy_select(candidates, classes, n=4, s_size=2, lmax=1, weights=(1.0,) * 7)
    # matches the hand-computed comparison above: the empty set already
    # beats either single rule, so greedy never adds anything.
    assert kept == set()


def test_candidates_from_rules_caps_per_class_at_rule_cutoff():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, size=(120, 6)).astype(bool)
    y = np.where(raw[:, 0], "x", "y")
    ds = neg_spec(list("abcdef"))
    rep = HorizontalBitmapRepresentation(ds, neg_X(raw), y)
    nlist = NListRepresentation.from_boolean(rep)

    cars = generate_cars(nlist, min_support=0.01, min_confidence=0.5, max_len=2)
    rules = annotate_rules([Rule(c.items, target=c.target, dataspec=nlist.spec) for c in cars], nlist)

    uncapped = _candidates_from_rules(rules, nlist, rule_cutoff=10_000)
    capped = _candidates_from_rules(rules, nlist, rule_cutoff=3)
    assert len(capped) <= 3 * 2  # at most 3 per class, 2 classes
    assert len(capped) < len(uncapped)


def _separable_multiclass(n=250, seed=0):
    # negation-enabled, same reasoning as CMAR's own fixture: the ground
    # truth needs "NOT b", only mineable once negation features exist.
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n, 5)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "x",
         np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    ds = neg_spec(list("abcde"))
    return HorizontalBitmapRepresentation(ds, neg_X(raw), y), raw, y


# Most of the tests below deliberately pin optimizer=GreedySelection(),
# tune_lambdas=False -- IDS's own *default* now matches the paper as
# closely as possible (optimizer=SmoothLocalSearch(), tune_lambdas=True, max_len=10),
# which is real, deliberately-slower work (see the module docstring);
# these tests are exercising fit()/predict() shape and switcher
# behavior, not the optimizer/tuning machinery itself, so they pin the
# fast fallback to stay quick and deterministic. The default
# combination itself gets one dedicated, small-scale end-to-end test
# below (`test_ids_default_settings_match_the_paper_and_run_end_to_end`).
_FAST = dict(optimizer=GreedySelection(), tune_lambdas=False)


def test_ids_fit_returns_a_flat_rule_set_with_an_f1_tie_break_combiner():
    rep, raw, y = _separable_multiclass()
    # weight precision/recall/class-coverage over pure interpretability
    # so the fitted set is non-trivial (see the hand-worked objective
    # test above for why naive equal weights alone can stay minimal).
    weights = (0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 2.0)
    model = IDS(min_support=0.02, min_confidence=0.5, max_len=3, rule_cutoff=20,
               lambda_weights=weights, **_FAST).fit(rep)
    assert isinstance(model, FlatRuleSet)
    assert isinstance(model.combiner, HeuristicMaxCombiner)
    assert isinstance(model.combiner.heuristic, FBeta)
    preds = np.asarray(model.predict(rep))
    acc = float(np.mean(preds == y))
    baseline = float(np.mean(y == max(set(y.tolist()), key=list(y).count)))
    assert acc >= baseline, f"IDS train accuracy {acc:.3f} did not beat the majority baseline {baseline:.3f}"


def test_ids_target_class_returns_a_concept_model():
    rep, raw, y = _separable_multiclass()
    model = IDS(min_support=0.02, min_confidence=0.5, max_len=3, target_class="x",
               lambda_weights=(0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 2.0), **_FAST).fit(rep)
    assert isinstance(model, ConceptModel)
    assert all(r.target == "x" for r in model.rules)


def test_ids_produces_the_decomposition_types_too():
    caps = IDS().produces()
    assert {FlatRuleSet, ConceptSet, ConceptCascade, PairwiseModel} <= caps


def test_ids_auto_converts_a_boolean_representation_transparently():
    rep, raw, y = _separable_multiclass(n=60)
    assert not isinstance(rep, NListRepresentation)
    model = IDS(min_support=0.05, min_confidence=0.5, max_len=2, **_FAST).fit(rep)
    assert isinstance(model, FlatRuleSet)


def test_ids_raises_a_clear_error_past_the_auto_convert_threshold():
    rep, raw, y = _separable_multiclass(n=60)
    with pytest.raises(ValueError, match="max_auto_convert_cells"):
        IDS(max_auto_convert_cells=1, **_FAST).fit(rep)


def test_ids_rejects_an_unknown_optimizer_or_wrong_lambda_count():
    with pytest.raises(ValueError, match="optimizer"):
        IDS(optimizer="sls")                       # names aren't components
    with pytest.raises(ValueError, match="lambda_weights"):
        IDS(lambda_weights=(1.0,) * 5)


def test_ids_default_constructor_values_match_the_papers_reported_settings():
    ids = IDS()
    # max_len stays at 4 (AssociationRuleMiner's shared default), not
    # the paper's own max_len=10 -- measured directly to explode this
    # codebase's own pure-Python mining well before reaching 10 on any
    # dataset with more than a handful of features; a mining-cost limit,
    # not a paper-fidelity choice (see the module docstring).
    assert ids.max_len == 4
    assert ids.min_support == 0.01
    assert isinstance(ids.optimizer, SmoothLocalSearch)
    assert ids.tune_lambdas is True
    assert ids.validation_fraction == 0.05
    assert (ids.max_rules, ids.max_avg_length, ids.max_overlap, ids.max_uncovered) == (15, 10.0, 0.10, 0.15)


def test_ids_sls_optimizer_runs_and_returns_a_valid_model():
    rep, raw, y = _separable_multiclass(n=150)
    model = IDS(min_support=0.05, min_confidence=0.5, max_len=2, rule_cutoff=10,
               optimizer=SmoothLocalSearch(samples=5, max_restarts=25, final_samples=5),
               lambda_weights=(0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 2.0), tune_lambdas=False, seed=0).fit(rep)
    assert isinstance(model, FlatRuleSet)
    preds = np.asarray(model.predict(rep))
    assert preds.shape == y.shape


def test_ids_tune_lambdas_runs_and_returns_a_valid_model():
    rep, raw, y = _separable_multiclass(n=150)
    model = IDS(min_support=0.05, min_confidence=0.5, max_len=2, rule_cutoff=10,
               optimizer=GreedySelection(), tune_lambdas=True, validation_fraction=0.3, tune_passes=1,
               tune_grid=(0.2, 1.0, 3.0), seed=0).fit(rep)
    assert isinstance(model, FlatRuleSet)
    preds = np.asarray(model.predict(rep))
    assert preds.shape == y.shape


def test_ids_default_settings_match_the_paper_and_run_end_to_end():
    # the real default combination (optimizer=SmoothLocalSearch(), tune_lambdas=True)
    # on a small dataset with tiny sls_*/tune_* knobs, just to bound
    # runtime -- not a claim that these tiny knobs are themselves
    # paper-faithful, only that the *shape* of the defaults (which knobs
    # are on, which paper settings they carry) works end-to-end without
    # the caller overriding optimizer=/tune_lambdas=.
    rep, raw, y = _separable_multiclass(n=120)
    model = IDS(min_support=0.05, min_confidence=0.5, rule_cutoff=8,
               optimizer=SmoothLocalSearch(samples=3, max_restarts=15, final_samples=3),
               tune_passes=1, tune_grid=(0.5, 2.0), seed=0).fit(rep)
    assert isinstance(model, FlatRuleSet)
    preds = np.asarray(model.predict(rep))
    assert preds.shape == y.shape


# -- IDS consuming an externally-supplied rule pool (rules=) -----------------

def test_ids_with_an_externally_supplied_rules_pool_skips_mining_entirely():
    rep, raw, y = _separable_multiclass(n=150)
    mined = IDS(min_support=0.05, min_confidence=0.5, max_len=2, rule_cutoff=10,
               optimizer=GreedySelection(), tune_lambdas=False,
               lambda_weights=(0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 2.0)).fit(rep)
    pool = FlatRuleSet(list(mined.rules))

    # min_support/min_confidence/max_len are irrelevant now -- rules= is given.
    model = IDS(rules=pool, min_support=0.9, min_confidence=0.99, rule_cutoff=10,
               optimizer=GreedySelection(), tune_lambdas=False,
               lambda_weights=(0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 2.0)).fit(rep)
    assert isinstance(model, FlatRuleSet)


def test_ids_random_forest_rules_round_trip():
    from sklearn.ensemble import RandomForestClassifier

    from pyrulearn.interfaces.sklearn import from_random_forest

    rep, raw, y = _separable_multiclass(n=200)
    ds = neg_spec(list("abcde"))
    data = HorizontalBitmapRepresentation(ds, neg_X(raw), y)

    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(neg_X(raw), y)
    forest_rules = from_random_forest(rf, dataspec=ds)
    pool = FlatRuleSet(annotate_rules(forest_rules.rules, data))

    model = IDS(rules=pool, rule_cutoff=10, optimizer=GreedySelection(), tune_lambdas=False,
               lambda_weights=(0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 2.0)).fit(data)
    assert isinstance(model, FlatRuleSet)
    preds = np.asarray(model.predict(data))
    assert preds.shape == y.shape
