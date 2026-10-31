from pyrulearn.combiners import HeuristicMaxCombiner
import numpy as np
import pytest

from pyrulearn.data import (
    HorizontalBitmapRepresentation,
    DataSpec,
    NListRepresentation,
    SparseDataRepresentation,
)
from pyrulearn.learners.associative import (
    CARMiner,
    coverage_select,
    generate_cars,
    sort_by_measured_precedence,
)
from pyrulearn.models import ConceptModel, FlatRuleSet, annotate_rules
from pyrulearn.rule import Rule


def _rep(X, y, names=None):
    X = np.asarray(X, dtype=bool)
    ds = DataSpec(names or [f"f{i}" for i in range(X.shape[1])])
    return HorizontalBitmapRepresentation(ds, X, np.asarray(y))


# -- CARMiner.ensure_representation: the shared representation gate --------
# (pyrulearn.learners.base.NativeRuleLearner.ensure_representation,
# exercised here through CARMiner.NATIVE_REPRESENTATIONS = (NListRepresentation,))

def test_ensure_representation_passes_through_an_already_nlist_representation_unchanged():
    rep = NListRepresentation.from_boolean(_rep([[1, 0], [0, 1]], ["a", "b"]))
    miner = CARMiner(max_auto_convert_cells=1_000_000)
    assert miner.ensure_representation(rep, miner.max_auto_convert_cells) is rep


def test_ensure_representation_auto_converts_a_small_boolean_representation():
    rep = _rep([[1, 0], [0, 1], [1, 1]], ["a", "b", "a"])
    miner = CARMiner(max_auto_convert_cells=1_000_000)
    out = miner.ensure_representation(rep, miner.max_auto_convert_cells)
    assert isinstance(out, NListRepresentation)
    assert out.n_samples == rep.n_samples


def test_ensure_representation_raises_past_the_size_threshold():
    rep = _rep([[1, 0], [0, 1], [1, 1]], ["a", "b", "a"])
    miner = CARMiner(max_auto_convert_cells=1)
    with pytest.raises(ValueError, match="max_auto_convert_cells"):
        miner.ensure_representation(rep, miner.max_auto_convert_cells)


def test_ensure_representation_converts_a_non_boolean_non_nlist_representation_too(capsys):
    # Changed on purpose (2026-10-08): CARMiner used to run directly on
    # an unrecognized representation (correct, just slower, since
    # generate_cars only calls the universal data.coverage(rule)).
    # NativeRuleLearner.ensure_representation (shared with DenseENDER, in the
    # opposite direction) doesn't distinguish "needed for correctness"
    # from "just much faster", so a SparseDataRepresentation now gets
    # converted too, below the size threshold, like a Boolean one would.
    rep = _rep([[1, 0], [0, 1]], ["a", "b"])
    sparse = SparseDataRepresentation.from_boolean(rep)
    miner = CARMiner(max_auto_convert_cells=1_000_000)
    out = miner.ensure_representation(sparse, miner.max_auto_convert_cells, purpose="itemset mining")
    assert isinstance(out, NListRepresentation)
    assert out.n_samples == sparse.n_samples
    assert "auto-converting" in capsys.readouterr().out
    print("ensure_representation: converts a SparseDataRepresentation too, not just Boolean: OK")


def test_generate_cars_gives_the_same_cars_on_a_sparse_representation_as_on_nlist():
    # Same hand-worked dataset as the CBA tests -- confirms mining
    # directly on a non-NList representation is correct, not just
    # accepted without error.
    X = [[1, 1]] * 3 + [[1, 0]] * 2 + [[0, 1]] * 2 + [[0, 0]] * 1
    y = ["a"] * 3 + ["a"] * 2 + ["b"] * 2 + ["b"] * 1
    boolean = _rep(X, y)
    nlist = NListRepresentation.from_boolean(boolean)
    sparse = SparseDataRepresentation.from_boolean(boolean)

    from_nlist = {(c.items, c.target): c.confidence
                  for c in generate_cars(nlist, min_support=0.01, min_confidence=0.5, max_len=1)}
    from_sparse = {(c.items, c.target): c.confidence
                   for c in generate_cars(sparse, min_support=0.01, min_confidence=0.5, max_len=1)}
    assert from_nlist == from_sparse


# -- coverage_select: the shared database-coverage walk ---------------------

def _wrong_then_right():
    # Rows 0,1 are actually class 'b'; rule_high (precedence-first) covers
    # them under target 'a' -- wrong on both -- and rule_low covers the
    # same rows under the correct target 'b'.
    X = [[1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
    y = ["b", "b", "a", "a", "a", "a"]
    data = NListRepresentation.from_boolean(_rep(X, y))
    rule_high = Rule((0,), target="a", dataspec=data.spec)
    rule_low = Rule((1,), target="b", dataspec=data.spec)
    return data, rule_high, rule_low


def test_coverage_select_skips_a_rule_that_is_wrong_on_everything_it_newly_covers():
    # CBA-CB "M1" as printed and as pyarc implements it (cross-checked by
    # a rule-by-rule diff): a rule is kept only if it correctly
    # classifies >=1 still-unclaimed row ("marked"). rule_high is 0/2
    # correct, so it is skipped and claims NOTHING; rule_low then gets
    # rows 0,1 and classifies them correctly.
    data, rule_high, rule_low = _wrong_then_right()
    kept, trace = coverage_select([rule_high, rule_low], data)

    assert kept == [rule_low]
    assert len(trace) == 1
    errors_so_far, default_label = trace[0]
    assert errors_so_far == 0 and default_label == "a"


def test_coverage_select_with_keep_coverage_keeps_a_rule_on_any_new_coverage():
    # keep=Coverage() -- the *pre-fix* behavior, now opt-in (CMAR's
    # pruning passes it): rule_high is kept although wrong on both of its
    # rows, and claims them, so rule_low never gets a turn.
    from pyrulearn.heuristics import Coverage
    data, rule_high, rule_low = _wrong_then_right()
    kept, trace = coverage_select([rule_high, rule_low], data, keep=Coverage())

    assert kept == [rule_high]
    errors_so_far, default_label = trace[0]
    assert errors_so_far == 2 and default_label == "a"


def test_coverage_select_stops_once_everything_is_covered():
    X = [[1], [1], [0], [0]]
    y = ["a", "a", "b", "b"]
    data = NListRepresentation.from_boolean(_rep(X, y))
    rule_a = Rule((0,), target="a", dataspec=data.spec)
    catch_all = Rule((), target="b", dataspec=data.spec)  # empty body -- covers every row

    kept, trace = coverage_select([rule_a, catch_all], data)
    # rule_a claims rows 0,1 first (correctly); catch_all still gets a
    # turn at the remaining rows 2,3 (correctly, since they're 'b') --
    # confirm both end up kept and the walk ends error-free.
    assert len(kept) == 2
    assert trace[-1][0] == 0  # 0 total errors once both are applied


# -- generate_cars: support is the support of body + head -------------------

def test_generate_cars_thresholds_the_support_of_the_rule_not_of_the_antecedent():
    # f0 holds on 6 of 10 rows, split 3 'a' / 3 'b'. With min_support=0.4
    # (4 rows), the *itemset* {f0} is frequent (6 >= 4), but neither
    # (f0 -> a) nor (f0 -> b) has rule support >= 4 (each is 3), so no
    # CAR may be emitted -- the standard/CBA-RG/fim definition. (An
    # earlier version thresholded the antecedent alone and emitted both
    # at confidence 0.5.)
    X = [[1]] * 6 + [[0]] * 4
    y = ["a"] * 3 + ["b"] * 3 + ["a"] * 2 + ["b"] * 2
    data = NListRepresentation.from_boolean(_rep(X, y))

    assert generate_cars(data, min_support=0.4, min_confidence=0.5, max_len=1) == []
    lower = generate_cars(data, min_support=0.3, min_confidence=0.5, max_len=1)  # 3 rows: both qualify
    assert {(c.items, c.target) for c in lower} == {((0,), "a"), ((0,), "b")}


def test_generate_cars_only_class_restricts_which_itemsets_are_explored():
    # class-conditional support is anti-monotone, so with only_class an
    # itemset survives only if *that class* reaches min_support with it.
    X = [[1, 1]] * 4 + [[1, 0]] * 4 + [[0, 0]] * 2
    y = ["a"] * 4 + ["b"] * 4 + ["a"] * 2
    data = NListRepresentation.from_boolean(_rep(X, y))
    only_b = generate_cars(data, min_support=0.4, min_confidence=0.5, max_len=2, only_class="b")
    # f0 (rows 0-7) has 4 'b' rows -> qualifies at conf 4/8; f1 (rows 0-3) has 0 'b' rows -> pruned
    assert {(c.items, c.target) for c in only_b} == {((0,), "b")}


# -- stats from counts == stats from annotate --------------------------------

def _same_stats(a, b):
    ca, cb = a.stats().confusion, b.stats().confusion
    assert ca.labels == cb.labels, (ca.labels, cb.labels)
    assert np.array_equal(ca.counts, cb.counts), (ca.counts, cb.counts)
    assert (a.stats().n_rows, a.stats().n_rules, a.stats().n_conditions) == \
           (b.stats().n_rows, b.stats().n_rules, b.stats().n_conditions)


def test_set_stats_from_counts_equals_annotate_including_abstain_edge_cases():
    from pyrulearn.models import SingleRule
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, size=(90, 4)).astype(bool)
    y = np.array(["c", "a", "b"])[rng.integers(0, 3, size=90)]          # 3 classes, unsorted first-seen
    data = _rep(X, y)
    classes, tot = np.unique(y, return_counts=True)
    totals = dict(zip(classes.tolist(), tot.tolist()))

    rules = [
        Rule((0,), target="a", dataspec=data.spec),          # ordinary: covers some rows, abstains elsewhere
        Rule((0, 1, 2), target="b", dataspec=data.spec),      # narrow
        Rule((), target="c", dataspec=data.spec),              # covers EVERY row -> no ABSTAIN column
        Rule((1,), target="zzz", dataspec=data.spec),          # target label that never occurs in y
    ]
    for r in rules:
        expected = annotate_rules([Rule(r.conditions, target=r.target, dataspec=data.spec)], data)[0]
        mask = data.coverage(r)
        covered = {c: int(np.sum(mask & (y == c))) for c in totals}
        got = SingleRule(Rule(r.conditions, target=r.target, dataspec=data.spec)).set_stats_from_counts(covered, totals)
        _same_stats(expected, got)


def test_the_miners_count_based_stats_equal_a_fresh_annotate_for_every_rule():
    from _negation_helpers import neg_spec, neg_X
    _rep0, raw, y = _separable_multiclass()
    nl = NListRepresentation.from_boolean(HorizontalBitmapRepresentation(neg_spec(list("abcde")), neg_X(raw), y))
    model = CARMiner(min_support=0.02, min_confidence=0.5, max_len=3).fit(nl)
    assert len(model.rules) > 50
    for r in model.rules:
        fresh = annotate_rules([Rule(r.conditions, target=r.target, dataspec=nl.spec)], nl)[0]
        _same_stats(fresh, r)


# -- CARMiner: the raw, unpruned CAR pool -------------------

def _separable_multiclass(n=180, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n, 5)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "x",
         np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    return _rep(raw, y, names=list("abcde")), raw, y


def test_carminer_returns_the_raw_car_pool_as_a_flat_rule_set():
    rep, raw, y = _separable_multiclass()
    miner = CARMiner(min_support=0.02, min_confidence=0.5, max_len=3)
    model = miner.fit(rep)
    assert isinstance(model, FlatRuleSet)

    nlist = miner.ensure_representation(rep, miner.max_auto_convert_cells)
    cars = generate_cars(nlist, miner.min_support, miner.min_confidence, miner.max_len)
    assert len(model.rules) == len(cars)  # every mined CAR survives, unpruned


def test_carminer_defaults_to_the_same_combiner_as_a_random_forest_import():
    # No explicit combiner= on either side -- both fall back to
    # FlatRuleSet's own plain default ("max", HeuristicMaxCombiner) --
    # one convention for "a raw pool of many small rules", regardless of
    # whether it came from mining or from_random_forest.
    rep, raw, y = _separable_multiclass()
    model = CARMiner(min_support=0.02, min_confidence=0.5, max_len=3).fit(rep)
    assert isinstance(model.combiner, HeuristicMaxCombiner)


def test_carminer_runs_end_to_end_on_a_sparse_representation():
    # Confirms ensure_representation's auto-conversion works through the
    # whole fit() pipeline, not just generate_cars in isolation.
    rep, raw, y = _separable_multiclass()
    sparse = SparseDataRepresentation.from_boolean(rep)
    model = CARMiner(min_support=0.02, min_confidence=0.5, max_len=3).fit(sparse)
    assert isinstance(model, FlatRuleSet)
    assert len(model.rules) > 0


def test_carminer_target_class_returns_a_concept_model():
    rep, raw, y = _separable_multiclass()
    model = CARMiner(min_support=0.02, min_confidence=0.5, target_class="x").fit(rep)
    assert isinstance(model, ConceptModel)
    assert all(r.target == "x" for r in model.rules)


# -- sort_by_measured_precedence: the generalized (non-CAR) precedence sort --

def test_sort_by_measured_precedence_matches_confidence_then_support_order():
    # r_a: confidence 1.0 (3/3) -- highest confidence, sorts first.
    # r_b: confidence 0.5 (2/4), support 2.
    # r_c: confidence 0.5 (1/2), support 1 -- same confidence as r_b,
    # lower support, so sorts after it.
    X = [[1, 0, 0], [1, 0, 0], [1, 0, 0],   # rows 0-2: r_a covers, all 'x'
         [0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1]]  # rows 3-6: r_b covers all, r_c covers 5-6 only
    y = ["x", "x", "x", "x", "y", "x", "y"]
    data = _rep(X, y)
    r_a = Rule((0,), target="x", dataspec=data.spec)          # 3/3 = 1.0 confidence
    r_b = Rule((1,), target="x", dataspec=data.spec)          # covers rows 3-6: 2 'x' of 4 -> 0.5 confidence, support 2
    r_c = Rule((1, 2), target="x", dataspec=data.spec)        # covers rows 5-6: 1 'x' of 2 -> 0.5 confidence, support 1

    annotated = annotate_rules([r_b, r_c, r_a], data)  # deliberately unordered input
    ordered = sort_by_measured_precedence(annotated)
    assert [r.target for r in ordered] == ["x", "x", "x"]
    assert ordered[0].conditions == r_a.conditions  # highest confidence first
    assert ordered[1].conditions == r_b.conditions  # tie on confidence -> r_b: support 2 > r_c: support 1
    assert ordered[2].conditions == r_c.conditions


def test_sort_by_measured_precedence_breaks_a_confidence_and_support_tie_by_generality():
    # r_short and r_long cover exactly the same two rows (f3 is 1
    # wherever f0 is 1 in this data) -- identical confidence (1.0) and
    # support (2) -- so only length breaks the tie: the more general,
    # single-condition rule sorts first.
    X = [[1, 1], [1, 1], [0, 0]]
    y = ["x", "x", "y"]
    data = _rep(X, y)
    r_short = Rule((0,), target="x", dataspec=data.spec)
    r_long = Rule((0, 1), target="x", dataspec=data.spec)

    ordered = sort_by_measured_precedence(annotate_rules([r_long, r_short], data))
    assert ordered[0].conditions == r_short.conditions
    assert ordered[1].conditions == r_long.conditions


def test_sort_by_measured_precedence_raises_on_an_unannotated_rule():
    data = _rep([[1], [0]], ["a", "b"])
    unannotated = Rule((0,), target="a", dataspec=data.spec)
    with pytest.raises(ValueError, match="measured stats"):
        sort_by_measured_precedence([unannotated])
