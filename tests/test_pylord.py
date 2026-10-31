from pyrulearn.combiners import HeuristicMaxCombiner
import numpy as np
import pytest

from pyrulearn import HorizontalBitmapRepresentation, DataSpec
from pyrulearn.models import FlatRuleSet as RuleSet, SingleRule
from pyrulearn.heuristics import Laplace, MEstimate, RuleStats
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.rule import Rule
from pyrulearn.learners.seco import BeamSearch

from _negation_helpers import neg_spec, neg_X


def _noisy_multiclass(n=240, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n, 5)).astype(bool)  # a b c d e
    clean = np.where(raw[:, 0] & ~raw[:, 1], "x",
             np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    noise = rng.random(n) < 0.05
    y = np.where(noise, rng.permutation(clean), clean)
    ds = neg_spec(list("abcde"))
    return HorizontalBitmapRepresentation(ds, neg_X(raw), y), ds, y


def test_pylord_has_no_single_rule_producer():
    # PyLORD's induced pool comes from every training example essentially
    # in parallel (one seed each) -- there is no unforced "first rule" the
    # way a sequential-covering loop has one, only "best of the pool by
    # some score", which bakes in a selection criterion the algorithm
    # itself never makes. Deliberately not offered as a capability.
    assert SingleRule not in PyLORD().produces()


def test_pylord_fits_a_ruleset_with_a_default_rule():
    rep, ds, y = _noisy_multiclass()
    rules = PyLORD(m=0.1, random_state=0).fit(rep)
    assert isinstance(rules, RuleSet)
    assert rules.default_rule is not None and len(rules.default_rule.conditions) == 0
    values, counts = np.unique(y, return_counts=True)
    assert rules.default_rule.target == values[int(np.argmax(counts))]


def test_pylord_is_natively_multiclass():
    rep, ds, y = _noisy_multiclass()
    rules = PyLORD(m=0.1, random_state=0).fit(rep)
    assert set(r.target for r in rules.rules) == {"x", "y", "z"}


def test_target_class_learns_one_class_only():
    rep, ds, y = _noisy_multiclass()
    rules = PyLORD(m=0.1, random_state=0, target_class="x").fit(rep)

    assert set(r.target for r in rules.rules) == {"x"}  # no rules for y or z
    for r in rules.rules:                               # still wins for some x-row
        assert RuleStats.from_rule(r, rep, "x").tp >= 1


def test_default_is_always_training_majority():
    rep, ds, y = _noisy_multiclass()
    vals, cnts = np.unique(y, return_counts=True)
    majority = vals[int(np.argmax(cnts))]
    # None -> FlatRuleSet (combiner "max"); "x" -> ConceptModel (no combiner attr)
    flat = PyLORD(m=0.1, random_state=0).fit(rep)
    assert flat.default_rule.target == majority and isinstance(flat.combiner, HeuristicMaxCombiner)
    concept = PyLORD(m=0.1, random_state=0, target_class="x").fit(rep)
    assert concept.default_rule.target == majority


def test_every_kept_rule_wins_for_at_least_one_example_of_its_class():
    rep, ds, y = _noisy_multiclass()
    rules = PyLORD(m=0.1, random_state=0).fit(rep)
    for r in rules.rules:
        st = RuleStats.from_rule(r, rep, r.target)
        assert st.tp >= 1


def test_stats_match_the_full_training_data():
    # PyLORD stores no declarative weight or coverage anymore (its own
    # metric is used internally to grow/prune/filter, not stamped onto
    # the kept rules) -- the rule's own measured stats() is what
    # persists, matching an independently-computed RuleStats.
    rep, ds, y = _noisy_multiclass()
    rules = PyLORD(m=0.3, random_state=0).fit(rep)
    for r in rules.rules:
        st = RuleStats.from_rule(r, rep, r.target)
        measured = r.stats().confusion.rule_stats(r.target)
        assert (measured.tp, measured.fp) == (st.tp, st.fp)


def test_faithful_defaults():
    lord = PyLORD()
    assert lord.skip_covered is False       # seed every row (LORD, not a covering loop)
    assert lord.prune is True and lord.prune_fraction is None  # prune on the full training set
    assert lord.beam_width == 1             # LORD's greedy search


def test_prune_flag_toggles_the_grow_prune_phase():
    rep, ds, y = _noisy_multiclass()
    pruned = PyLORD(m=0.1, prune=True, random_state=0).fit(rep)
    unpruned = PyLORD(m=0.1, prune=False, random_state=0).fit(rep)
    pl = np.mean([r.length() for r in pruned.rules])
    ul = np.mean([r.length() for r in unpruned.rules])
    assert pl <= ul + 1e-9  # grow/prune only ever generalizes a rule


def test_prune_fraction_switches_to_a_ripper_style_holdout_split():
    rep, ds, y = _noisy_multiclass()
    train_only = PyLORD(m=0.1, random_state=0).fit(rep)                     # prune on full train
    holdout = PyLORD(m=0.1, prune_fraction=0.33, random_state=0).fit(rep)   # RIPPER/IREP split
    assert isinstance(holdout, RuleSet) and len(holdout.rules) > 0
    # a held-out pruning set generalizes more aggressively -> different (usually
    # shorter, fewer) rules than pruning on the same data the rule was grown on
    assert {frozenset(l.feature for l in r.conditions) for r in holdout.rules} != \
           {frozenset(l.feature for l in r.conditions) for r in train_only.rules}
    for r in holdout.rules:
        assert RuleStats.from_rule(r, rep, r.target).tp >= 1


def test_skip_covered_yields_a_smaller_pool():
    rep, ds, y = _noisy_multiclass()
    full = PyLORD(m=0.1, prune=False, random_state=0).fit(rep)
    skipped = PyLORD(m=0.1, prune=False, skip_covered=True, random_state=0).fit(rep)
    assert len(skipped.rules) <= len(full.rules)


def test_coverage_filter_drops_a_dominated_rule():
    X = np.array([
        [1, 1], [1, 1], [1, 1], [1, 0],
        [0, 0], [0, 1],
    ], dtype=bool)
    y = np.array(["p", "p", "p", "n", "n", "n"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    rules = PyLORD(m=0.5, prune=False, random_state=0).fit(rep)
    got = {(frozenset(l.feature for l in r.conditions), r.target) for r in rules.rules}
    assert (frozenset({0, 1}), "p") in got          # "a & b -> p", pure, covers 3 p-rows
    assert (frozenset({0}), "p") not in got         # bare "a -> p" is dominated on every p-row


def test_metric_and_search_overrides_are_honored():
    rep, ds, y = _noisy_multiclass()
    lord = PyLORD(metric=Laplace(), search=BeamSearch(beam_width=1, max_conditions=3), random_state=0)
    assert isinstance(lord.metric, Laplace)
    rules = lord.fit(rep)
    assert all(r.length() <= 3 for r in rules.rules)  # max_conditions honored via search=
    # metric= no longer leaves a stored weight (nothing does) -- kept
    # rules still carry measured stats matching an independently-computed
    # RuleStats, regardless of which metric drove the search/filter
    for r in rules.rules:
        st = RuleStats.from_rule(r, rep, r.target)
        measured = r.stats().confusion.rule_stats(r.target)
        assert (measured.tp, measured.fp) == (st.tp, st.fp)


def test_all_zero_row_is_skipped_not_crashed():
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=bool)  # row 0 all-zero
    y = np.array(["p", "p", "n", "n"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    rules = PyLORD(m=0.5, prune=False, random_state=0).fit(rep)  # must not raise
    assert isinstance(rules, RuleSet)


def test_induced_rules_and_default_rule_carry_their_own_training_stats():
    rep, ds, y = _noisy_multiclass()
    rules = PyLORD(m=0.1, random_state=0).fit(rep)
    for r in rules.rules:
        expected = RuleStats.from_rule(r.rule, rep, r.target)
        got = r.stats().confusion.rule_stats(r.target)
        assert (got.tp, got.fp, got.fn, got.tn) == (expected.tp, expected.fp, expected.fn, expected.tn)
    dr_stats = rules.default_rule.stats()
    assert dr_stats is not None and dr_stats.confusion is not None


def test_predict_is_best_rule_wins():
    rep, ds, y = _noisy_multiclass(seed=1)
    rules = PyLORD(m=0.1, prune=False, random_state=0).fit(rep)
    preds = np.asarray(rules.predict(rep))  # RuleSet.predict -> HeuristicMaxCombiner (Laplace on measured stats) by default
    assert np.mean(preds == y) > 0.9
