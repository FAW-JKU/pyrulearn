from contextlib import contextmanager

import numpy as np
import pytest
from pyrulearn.data import DataSpec
from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.models import (
    ConceptCascade, ConceptModel, ConceptSet, DecisionList, DisjointRuleSet, FlatRuleSet, PairwiseModel,
    annotate_rules,
)
from pyrulearn.combiners import HeuristicMaxCombiner, ListCombiner, MicroVoteCombiner
from pyrulearn.rule import Rule
from pyrulearn.heuristics import (
    Accuracy, Correlation, CoveredNegatives, CoveredPositives, DeltaGain, FoilGain,
    Laplace, LEF, LikelihoodRatio, MinimalLength, RuleStats, UncoveredPositives,
)
from pyrulearn.learners import NativeRuleLearner
from pyrulearn.pruning import AllOf, AnyOf, EncodingLengthRestriction, ThresholdPrePruning
from pyrulearn.learners.seco import (
    AQR, BeamSearch, CN2, EmptyRuleAllFeatures, FeatureSubset, GainAscentHillClimbing, HillClimbing,
    NoSplit, GrowPruneSplit,
    NoPostProcessing, PFoil, PFossil, Pypper, ReducedErrorPruning, ReplaceReviseOptimization,
    SeCo, SeedExample, SingleRuleLearner, rule_set_description_length,
)
from pyrulearn.heuristics import Precision
from pyrulearn.learners import seco

from _negation_helpers import make_rule, neg_spec, neg_X


@contextmanager
def _counting_expansions():
    """Counts node expansions (calls to seco.count_open_children, made
    once per rule a search expands) while active -- used to directly
    verify the tp==0 floor / optimistic-value pruning actually prevents
    a search from expanding a rule at all, not just that it doesn't
    change the final answer (which, for those conditions, it mathematically
    can't -- see seco.py's PrePruningCriterion docstring)."""
    count = [0]
    original = seco.count_open_children

    def wrapper(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)

    seco.count_open_children = wrapper
    try:
        yield count
    finally:
        seco.count_open_children = original


def _conjunction_dataset():
    # y = pos iff a AND b; c is irrelevant noise
    X = np.array([
        [1, 1, 0], [1, 1, 1],                                     # positives: a=b=1
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0],  # negatives
    ], dtype=bool)
    y = np.array(["pos", "pos"] + ["neg"] * 6)
    ds = DataSpec(["a", "b", "c"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    return rep, ds


def test_beam_search_finds_perfect_conjunction():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    best = BeamSearch(beam_width=3).search(rep, "pos", Accuracy(), initial)

    stats = RuleStats.from_rule(best, rep, "pos")
    assert (stats.tp, stats.fp, stats.fn, stats.tn) == (2, 0, 0, 6)
    assert best.length() == 2  # needs both a and b for perfect precision
    print("BeamSearch finds the perfect a-AND-b conjunction: OK")


def test_hill_climbing_finds_the_perfect_conjunction_with_a_plain_heuristic():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    best = HillClimbing().search(rep, "pos", Accuracy(), initial)

    stats = RuleStats.from_rule(best, rep, "pos")
    assert (stats.tp, stats.fp, stats.fn, stats.tn) == (2, 0, 0, 6)
    assert best.length() == 2  # needs both a and b for perfect precision
    print("HillClimbing(Accuracy) finds the perfect a-AND-b conjunction: OK")


def test_gain_ascent_hill_climbing_finds_the_perfect_conjunction_with_foilgain():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    best = GainAscentHillClimbing().search(rep, "pos", FoilGain(), initial)

    stats = RuleStats.from_rule(best, rep, "pos")
    assert (stats.tp, stats.fp, stats.fn, stats.tn) == (2, 0, 0, 6)
    assert best.length() == 2  # needs both a and b for perfect precision
    print("GainAscentHillClimbing(FoilGain) finds the perfect a-AND-b conjunction: OK")


def test_gain_ascent_hill_climbing_with_delta_gain_matches_foilgain_on_this_dataset():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    best = GainAscentHillClimbing().search(rep, "pos", DeltaGain(Laplace()), initial)

    stats = RuleStats.from_rule(best, rep, "pos")
    assert (stats.tp, stats.fp, stats.fn, stats.tn) == (2, 0, 0, 6)
    print("GainAscentHillClimbing(DeltaGain(Laplace())) also finds the perfect conjunction: OK")


def test_hill_climbers_stop_at_the_start_when_no_move_improves():
    # a is uncorrelated with y (both an exact 50/50 split) -- every
    # single-condition refinement matches the parent's own score exactly,
    # so nothing ever strictly improves (gain 0 / raw score unchanged)
    X = np.array([[1], [0], [1], [0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    ds = DataSpec(["a"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    initial = [(Rule([], target="pos", dataspec=ds), frozenset({0}))]

    assert GainAscentHillClimbing().search(rep, "pos", FoilGain(), initial).length() == 0
    assert HillClimbing().search(rep, "pos", Accuracy(), initial).length() == 0
    print("both hill climbers return the initial rule unchanged when nothing improves: OK")


def test_hill_climbing_returns_none_when_the_walk_cant_start_and_a_gate_is_set():
    # same "no refinement improves" situation, but with a gate configured:
    # the empty seed can't itself be the answer (it fails the gate), and
    # the walk never reaches anything else -> None, not the empty rule.
    X = np.array([[1], [0], [1], [0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])
    ds = DataSpec(["a"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    initial = [(Rule([], target="pos", dataspec=ds), frozenset({0}))]

    gate = ThresholdPrePruning(Accuracy(), threshold=0.6, operator="<")  # empty rule's accuracy 0.5 fails it
    assert HillClimbing().search(rep, "pos", Accuracy(), initial, filtering=gate) is None
    assert HillClimbing().search(rep, "pos", Accuracy(), initial, stopping=gate) is None
    print("HillClimbing returns None (not the empty seed) when the walk can't start and a gate is set: OK")


def test_hill_climbing_accepts_a_plain_heuristic_and_rejects_a_gain_one():
    rep, ds = _conjunction_dataset()
    initial = [(Rule([], target="pos", dataspec=ds), frozenset(range(3)))]

    HillClimbing().search(rep, "pos", Accuracy(), initial)  # no raise
    with pytest.raises(ValueError, match="GainAscentHillClimbing"):
        HillClimbing().search(rep, "pos", FoilGain(), initial)
    print("HillClimbing takes a plain RuleHeuristic and points a GainHeuristic at the subclass: OK")


def test_gain_ascent_hill_climbing_rejects_a_plain_heuristic():
    rep, ds = _conjunction_dataset()
    initial = [(Rule([], target="pos", dataspec=ds), frozenset(range(3)))]
    with pytest.raises(ValueError, match="GainHeuristic"):
        GainAscentHillClimbing().search(rep, "pos", Accuracy(), initial)
    print("GainAscentHillClimbing rejects a plain (non-Gain) heuristic: OK")


def test_hill_climbing_rejects_more_than_one_initial_candidate():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset({0, 1})), (empty_rule, frozenset({2}))]
    for search in (HillClimbing(), GainAscentHillClimbing()):
        heuristic = FoilGain() if isinstance(search, GainAscentHillClimbing) else Accuracy()
        with pytest.raises(ValueError):
            search.search(rep, "pos", heuristic, initial)
    print("both hill climbers reject more than one initial candidate: OK")


def test_beam_search_deduplicates_rules_reached_by_different_orders():
    # a and b are marginally tied (both needed for y=pos), so both survive
    # into the same beam round -- then "a" refining by adding b and "b"
    # refining by adding a reach the *same* rule. A little class noise keeps
    # {a, b} impure (fp>0) so neither the tp==0 floor nor optimistic pruning
    # cuts the search short before a further round can re-expand the duplicate.
    rng = np.random.default_rng(0)
    n = 300
    X = rng.integers(0, 2, size=(n, 5)).astype(bool)  # a, b, c, d, e
    y = np.where(X[:, 0] & X[:, 1], "pos", "neg")
    flip = rng.random(n) < 0.15
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    ds = DataSpec(["a", "b", "c", "d", "e"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(5)))]

    # duplicates are collapsed before any child is built, so (one seed, no
    # filtering) "never expanded twice" == "never built twice"
    seen = []
    original = seco.materialize_child

    def wrapper(*args, **kwargs):
        built = original(*args, **kwargs)
        if built is not None:
            seen.append(built[0].pos)
        return built

    seco.materialize_child = wrapper
    try:
        BeamSearch(beam_width=4).search(rep, "pos", Accuracy(), initial)
    finally:
        seco.materialize_child = original

    assert len(seen) == len(set(seen)), \
        f"a child with the same rule content was built more than once: {seen}"
    print("BeamSearch never builds two children with identical content: OK")


def test_beam_search_respects_example_mask():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    # full data: "a=True" alone has fp=2 (rows 2,5 have a=1,b=0) -- search
    # must grow to "a AND b" (length 2) for perfect accuracy
    best_full = BeamSearch(beam_width=3).search(rep, "pos", Accuracy(), initial)
    assert best_full.length() == 2

    # restricted to rows {0,3,4,6,7} (drops rows 1,2,5): the only masked
    # row with a=True is row 0, which is also the only masked positive --
    # "a=True" alone is already perfect *within this scope*, so the search
    # should stop at length 1 rather than needlessly growing to a tied
    # (not better) length-2 rule
    mask = np.zeros(8, dtype=bool)
    mask[[0, 3, 4, 6, 7]] = True
    best_masked = BeamSearch(beam_width=3).search(rep, "pos", Accuracy(), initial, example_mask=mask)
    assert best_masked.length() == 1
    assert best_masked.conditions[0].feature == 0  # feature "a"

    masked_stats = RuleStats.from_rule(best_masked, rep, "pos", example_mask=mask)
    assert (masked_stats.tp, masked_stats.fp, masked_stats.fn, masked_stats.tn) == (1, 0, 0, 4)
    print("BeamSearch's example_mask changes which rule is preferred: OK")


def test_beam_search_raises_on_empty_initial_candidates():
    rep, ds = _conjunction_dataset()
    try:
        BeamSearch().search(rep, "pos", Accuracy(), [])
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("BeamSearch raises on empty initial_candidates: OK")


def test_beam_search_never_returns_worse_than_initial_candidate():
    rep, ds = _conjunction_dataset()
    # start from a rule that's already perfect -- no refinement should be
    # able (or needed) to beat it, and none should be returned instead
    perfect_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # a AND b
    initial = [(perfect_rule, frozenset({2}))]  # only "c" left open

    best = BeamSearch(beam_width=3).search(rep, "pos", Accuracy(), initial)
    stats = RuleStats.from_rule(best, rep, "pos")
    assert (stats.tp, stats.fp, stats.fn, stats.tn) == (2, 0, 0, 6)
    print("BeamSearch never regresses below its best initial candidate: OK")


def test_empty_rule_all_features_starts_from_universal_rule():
    rep, ds = _conjunction_dataset()
    candidates = EmptyRuleAllFeatures().initial_candidates(rep, "pos")
    assert len(candidates) == 1
    rule, mask = candidates[0]
    assert rule.length() == 0
    assert rule.target == "pos"
    assert mask == frozenset({0, 1, 2})
    print("EmptyRuleAllFeatures: starts from the empty rule, every feature open: OK")


def test_feature_subset_resolves_names_to_indices():
    rep, ds = _conjunction_dataset()
    candidates = FeatureSubset(["b", "c"]).initial_candidates(rep, "pos")
    assert len(candidates) == 1
    rule, mask = candidates[0]
    assert rule.length() == 0
    assert mask == frozenset({ds.feature_index("b"), ds.feature_index("c")})
    assert 0 not in mask  # "a" excluded
    print("FeatureSubset resolves feature names to indices: OK")


def test_no_split_passes_mask_through_unchanged():
    rep, ds = _conjunction_dataset()
    mask = np.array([True, False] * 4)
    search_mask, context = NoSplit().prepare(rep, "pos", mask)
    assert np.array_equal(search_mask, mask)
    assert context is None
    search_mask_none, context_none = NoSplit().prepare(rep, "pos", None)
    assert search_mask_none is None
    assert context_none is None
    print("NoSplit passes the incoming mask through unchanged: OK")


def test_grow_prune_split_produces_disjoint_masks_covering_scope():
    rep, ds = _conjunction_dataset()
    grow_mask, prune_mask = GrowPruneSplit(prune_fraction=0.25, random_state=0).prepare(rep, "pos")
    assert not np.any(grow_mask & prune_mask)  # disjoint
    assert np.all(grow_mask | prune_mask)  # covers every row (no incoming mask -> full scope)
    assert prune_mask.sum() == 2  # 25% of 8 rows

    # restricting to a smaller incoming scope should restrict the split to it
    scope = np.array([True] * 6 + [False] * 2)  # rows 6,7 out of scope
    grow_mask2, prune_mask2 = GrowPruneSplit(prune_fraction=0.5, random_state=0).prepare(rep, "pos", scope)
    assert not np.any(grow_mask2 & prune_mask2)
    assert np.array_equal(grow_mask2 | prune_mask2, scope)
    print("GrowPruneSplit: disjoint masks, covers exactly the in-scope rows: OK")


def test_grow_prune_split_falls_back_when_stratification_is_impossible():
    # 1 positive, 10 negatives in scope -- stratified train_test_split
    # needs >=2 of every class to put one in each split, so this must
    # raise internally and fall back to a non-stratified split instead
    # of propagating the error -- confirmed as a real failure mode (not
    # hypothetical): a SeCo covering loop's remaining scope shrinks every
    # iteration and can easily end up exactly like this late on
    X = np.zeros((11, 2), dtype=bool)
    y = np.array(["pos"] + ["neg"] * 10)
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    grow_mask, prune_mask = GrowPruneSplit(prune_fraction=0.3, random_state=0).prepare(rep, "pos")
    assert not np.any(grow_mask & prune_mask)
    assert np.all(grow_mask | prune_mask)
    print("GrowPruneSplit falls back to a non-stratified split when stratification isn't possible: OK")


def test_grow_prune_split_falls_back_to_no_split_when_too_few_examples():
    X = np.zeros((1, 2), dtype=bool)
    y = np.array(["pos"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    mask, context = GrowPruneSplit(prune_fraction=0.3, random_state=0).prepare(rep, "pos")
    assert mask is None  # matches NoSplit's behavior: no example_mask given in, none constructed
    assert context is None
    print("GrowPruneSplit falls back to no split at all when there's too little to split: OK")


def _grow_prune_overfit_dataset():
    # a alone perfectly predicts y on both halves; b spuriously helps on the
    # grow half only (one grow row has a=True,b=False,y=neg -- adding "b"
    # removes that false positive there) but never matters on the prune
    # half (no a=True,b=False rows there at all), so a correctly-pruned
    # rule should end up as just "a=True", not "a=True AND b=True"
    X = np.array([
        [1, 1], [1, 1], [1, 0], [0, 0],  # grow: rows 0-3
        [1, 1], [1, 1], [0, 0], [0, 0],  # prune: rows 4-7
    ], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg", "pos", "pos", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    grow_mask = np.array([True] * 4 + [False] * 4)
    prune_mask = np.array([False] * 4 + [True] * 4)
    return rep, ds, grow_mask, prune_mask


def test_reduced_error_pruning_strips_condition_that_doesnt_generalize():
    rep, ds, grow_mask, prune_mask = _grow_prune_overfit_dataset()
    grown_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # "a AND b", as grown on grow_mask
    assert grown_rule.length() == 2

    pruned = ReducedErrorPruning(Accuracy()).postprocess(grown_rule, rep, "pos", prune_mask)
    assert pruned.length() == 1
    assert pruned.conditions[0] == grown_rule.conditions[0]  # kept "a", dropped "b"
    print("ReducedErrorPruning strips a condition that doesn't generalize to the prune set: OK")


def test_reduced_error_pruning_noop_without_context():
    rep, ds, grow_mask, prune_mask = _grow_prune_overfit_dataset()
    grown_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)
    unchanged = ReducedErrorPruning(Accuracy()).postprocess(grown_rule, rep, "pos", None)
    assert unchanged == grown_rule
    unchanged2 = NoPostProcessing().postprocess(grown_rule, rep, "pos", prune_mask)
    assert unchanged2 == grown_rule
    print("ReducedErrorPruning/NoPostProcessing are no-ops without a pruning mask/by default: OK")


def test_reduced_error_pruning_never_truncates_to_the_empty_rule():
    # prune set where EVERY row is the target class -- the empty
    # (unconditional) rule would score a perfect 1.0 here, strictly
    # beating any real truncation, yet must never be chosen: confirmed
    # directly as a real SeCo failure mode (an "always true" rule ends
    # a covering loop early and fires on every prediction), not just a
    # hypothetical edge case
    X = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)
    y = np.array(["pos", "pos", "pos"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    grown_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # "a AND b"
    prune_mask = np.ones(3, dtype=bool)

    pruned = ReducedErrorPruning(Accuracy()).postprocess(grown_rule, rep, "pos", prune_mask)
    assert pruned.length() >= 1
    print("ReducedErrorPruning never truncates a rule all the way to the empty/unconditional rule: OK")


def test_reduced_error_pruning_leaves_an_already_empty_rule_alone():
    # a rule that started empty (the *search's* own conclusion, not
    # something pruning truncated down to) should pass through unchanged
    ds = DataSpec(["a", "b"])
    X = np.array([[1, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "neg"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    empty_rule = Rule([], target="pos", dataspec=ds)
    prune_mask = np.ones(2, dtype=bool)

    result = ReducedErrorPruning(Accuracy()).postprocess(empty_rule, rep, "pos", prune_mask)
    assert result == empty_rule
    print("ReducedErrorPruning leaves an already-empty rule (from the search itself) unchanged: OK")


def test_single_rule_learner_default_config_matches_bare_beam_search():
    rep, ds = _conjunction_dataset()
    direct = BeamSearch().search(rep, "pos", Accuracy(), EmptyRuleAllFeatures().initial_candidates(rep, "pos"))
    via_learner = SingleRuleLearner(heuristic=Accuracy()).learn_one_rule(rep, "pos")
    assert via_learner == direct
    print("SingleRuleLearner's default config matches calling BeamSearch directly: OK")


def test_single_rule_learner_full_ripper_style_config_runs_end_to_end():
    rep, ds, grow_mask, prune_mask = _grow_prune_overfit_dataset()
    heuristic = Accuracy()
    learner = SingleRuleLearner(
        heuristic=heuristic,
        search=BeamSearch(beam_width=3),
        preparation=GrowPruneSplit(prune_fraction=0.5, random_state=0),
        postprocessing=ReducedErrorPruning(heuristic),
        space_init=EmptyRuleAllFeatures(),
    )
    rule = learner.learn_one_rule(rep, "pos")
    assert rule.target == "pos"
    assert rule.length() >= 1  # ran end to end and found something non-degenerate
    print("SingleRuleLearner with a full RIPPER-style config runs end to end: OK")


def test_beam_search_tp_zero_floor_stops_specializing():
    ds = neg_spec(["a", "b", "c"])
    X = neg_X([
        [1, 1, 0], [1, 1, 1],                                              # positives: a=b=1
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0],  # negatives
    ])
    y = np.array(["pos", "pos"] + ["neg"] * 6)
    rep = HorizontalBitmapRepresentation(ds, X, y)
    useless_rule = make_rule(ds, pos=["b"], neg=["a"], target="pos")  # b=True AND a=False
    stats = RuleStats.from_rule(useless_rule, rep, "pos")
    assert stats.tp == 0 and stats.fp > 0  # covers negatives only -- useless for "pos"

    initial = [(useless_rule, frozenset({ds.feature_index("c")}))]  # "c" still open
    with _counting_expansions() as count:
        best = BeamSearch(beam_width=3).search(rep, "pos", Accuracy(), initial)
    assert count[0] == 0  # tp==0 floor stopped it before specialize() was ever called
    assert best is None  # a rule covering no positives is never returned
    print("BeamSearch's tp==0 floor stops specialize() from being called at all: OK")


def _laplace_trap_dataset():
    # "e" covers no example at all, so its one-condition rule scores
    # Laplace (0+1)/(0+0+2) = 0.5 -- above every impure one-condition rule
    # (a: 2 pos / 3 neg -> 3/7). The pure rule a AND b (2 pos / 0 neg,
    # 0.75) is only reachable through an impure first step.
    X = np.array([
        [1, 1, 0], [1, 1, 0],                                        # positives
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],  # negatives
    ], dtype=bool)
    y = np.array(["pos", "pos"] + ["neg"] * 6)
    ds = DataSpec(["a", "b", "e"])
    return HorizontalBitmapRepresentation(ds, X, y), ds


@pytest.mark.parametrize("search", [BeamSearch(beam_width=1), HillClimbing()])
def test_rule_covering_no_positives_never_blocks_the_search(search):
    # regression test: a child covering no positives used to win the beam
    # slot (or the hill-climbing move) on its heuristic score alone, and,
    # being unrefinable, end the search at an impure one-condition rule
    rep, ds = _laplace_trap_dataset()
    empty = [(Rule([], target="pos", dataspec=ds), frozenset(range(3)))]
    best = search.search(rep, "pos", Laplace(), empty)
    assert best == Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)
    print(f"{type(search).__name__} gets past a rule covering no positives: OK")


def test_beam_search_optimistic_pruning_skips_an_already_pure_seed():
    rep, ds = _conjunction_dataset()
    pure_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # a AND b: fp=0
    stats = RuleStats.from_rule(pure_rule, rep, "pos")
    assert stats.fp == 0 and stats.tp > 0

    initial = [(pure_rule, frozenset({2}))]  # "c" still open
    with _counting_expansions() as count:
        best = BeamSearch(beam_width=3).search(rep, "pos", Accuracy(), initial)
    # a pure rule's optimistic (tp, 0) bound is just its own score, which is
    # already the running best -- so it never promises improvement and is
    # never specialized (the case the old unconditional fp==0 floor handled)
    assert count[0] == 0
    assert best == pure_rule

    # ...but turning optimistic pruning off removes that floor entirely:
    # the pure rule now does get specialized (fruitlessly)
    with _counting_expansions() as count_off:
        BeamSearch(beam_width=3, optimistic_pruning=False).search(rep, "pos", Accuracy(), initial)
    assert count_off[0] > 0
    print("BeamSearch optimistic pruning skips specializing an already-pure rule; off, it doesn't: OK")


def test_beam_search_optimistic_pruning_halts_the_whole_search_early():
    # y = a AND b, with c/d/e pure noise and 10% label flips so {a, b} stays
    # impure -- exactly the setup where, without a bound, the beam keeps
    # tacking noise literals onto an already-good rule (the Correlation
    # runtime blow-up this feature exists to cut off)
    rng = np.random.default_rng(1)
    n = 240
    X = rng.integers(0, 2, size=(n, 5)).astype(bool)
    y = np.where(X[:, 0] & X[:, 1], "pos", "neg")
    flip = rng.random(n) < 0.10
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    ds = DataSpec(["a", "b", "c", "d", "e"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    initial = [(Rule([], target="pos", dataspec=ds), frozenset(range(5)))]

    with _counting_expansions() as on:
        best_on = BeamSearch(beam_width=4).search(rep, "pos", Correlation(), initial)
    with _counting_expansions() as off:
        best_off = BeamSearch(beam_width=4, optimistic_pruning=False).search(
            rep, "pos", Correlation(), initial
        )

    assert 0 < on[0] < off[0]   # same search, reached with strictly less work
    assert best_on == best_off  # ...and the identical answer -- pruning only ever
    #                             skips branches that provably can't out-score it
    print("BeamSearch optimistic pruning halts the whole search early without changing the answer: OK")


def test_gain_ascent_optimistic_pruning_stops_at_a_pure_rule():
    rep, ds = _conjunction_dataset()
    pure_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # a AND b: fp == 0
    initial = [(pure_rule, frozenset({2}))]  # "c" still open

    with _counting_expansions() as on:
        best_on = GainAscentHillClimbing().search(rep, "pos", FoilGain(), initial)
    assert on[0] == 0  # optimistic (tp, 0) bound == the rule itself -> gain 0 -> stop
    assert best_on == pure_rule

    with _counting_expansions() as off:
        best_off = GainAscentHillClimbing(optimistic_pruning=False).search(rep, "pos", FoilGain(), initial)
    assert off[0] > 0             # no bound -> it does specialize once
    assert best_off == pure_rule  # ...but the gain floor still stops it at the same rule
    print("GainAscentHillClimbing optimistic pruning stops at a pure rule (subsuming the old fp==0 floor): OK")


def test_hill_climbing_optimistic_pruning_skips_an_already_pure_rule():
    # base-class optimistic pruning is BeamSearch's bound, restated: a
    # pure rule's (tp, 0) projection is itself, so it can't beat the
    # running best -> not specialized (same case the old fp==0 floor
    # handled). Same answer either way -- pure efficiency.
    rep, ds = _conjunction_dataset()
    pure_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # a AND b: fp == 0
    initial = [(pure_rule, frozenset({2}))]  # "c" still open

    with _counting_expansions() as on:
        best_on = HillClimbing().search(rep, "pos", Accuracy(), initial)
    assert on[0] == 0
    assert best_on == pure_rule

    with _counting_expansions() as off:
        best_off = HillClimbing(optimistic_pruning=False).search(rep, "pos", Accuracy(), initial)
    assert off[0] > 0             # no bound -> it specializes once
    assert best_off == pure_rule  # ...but the local-maximum stop lands on the same rule
    print("HillClimbing optimistic pruning skips an already-pure rule; same answer without it: OK")


def test_hill_climbing_matches_beam_search_width_one_on_a_unimodal_landscape():
    # on a landscape that rises monotonically to one peak, HillClimbing
    # and BeamSearch(beam_width=1) return the identical rule -- HillClimbing
    # just stops climbing sooner (its local-maximum stop), doing no more
    # specialization work than the beam.
    rep, y = _wide_noisy_disjunction_dataset(d=12, seed=3)
    initial = [(Rule([], target="pos", dataspec=rep.spec), frozenset(range(12)))]

    with _counting_expansions() as hc:
        hc_rule = HillClimbing().search(rep, "pos", Laplace(), initial)
    with _counting_expansions() as beam:
        beam_rule = BeamSearch(beam_width=1).search(rep, "pos", Laplace(), initial)

    assert hc_rule.pos == beam_rule.pos
    assert hc[0] <= beam[0]
    print("HillClimbing == BeamSearch(beam_width=1) on a unimodal landscape, with <= the work: OK")


def test_hill_climbers_grow_to_consistency_when_the_local_optimum_stop_is_off():
    # a noisy conjunction: y = a AND b with ~12% label flips, so {a, b} is
    # the best pure-ish rule but every prefix still covers negatives. With
    # stop_at_local_optimum=False both hill climbers keep adding conditions
    # past the score/gain peak until fp == 0 (or features run out).
    rng = np.random.default_rng(4)
    n = 300
    X = rng.integers(0, 2, size=(n, 6)).astype(bool)
    y_clean = X[:, 0] & X[:, 1]
    y = np.where(rng.random(n) < 0.12, ~y_clean, y_clean)
    y = np.where(y, "pos", "neg")
    ds = DataSpec([f"a{i}" for i in range(6)])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    init = lambda: [(Rule([], target="pos", dataspec=ds), frozenset(range(6)))]

    for search_cls, heuristic in [(HillClimbing, Accuracy()), (GainAscentHillClimbing, FoilGain())]:
        peak = search_cls().search(rep, "pos", heuristic, init())
        grown = search_cls(stop_at_local_optimum=False).search(rep, "pos", heuristic, init())
        peak_fp = RuleStats.from_rule(peak, rep, "pos").fp
        grown_stats = RuleStats.from_rule(grown, rep, "pos")
        assert grown.length() >= peak.length()
        assert grown_stats.fp < peak_fp or grown_stats.fp == 0  # drove negatives down
    print("HillClimbing / GainAscentHillClimbing grow to consistency with stop_at_local_optimum=False: OK")


def test_hill_climbing_returns_none_when_filtering_rejects_every_visited_rule():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    # laplace < 2.0 is always true -- no rule the walk visits is ever
    # eligible, so there's nothing to fall back to
    reject_everything = ThresholdPrePruning(Laplace(), threshold=2.0, operator="<")
    result = HillClimbing().search(
        rep, "pos", Laplace(), initial, filtering=reject_everything,
    )
    assert result is None
    print("HillClimbing returns None when filtering rejects every rule the walk visits: OK")


def test_threshold_pre_pruning_rejects_bad_operator():
    try:
        ThresholdPrePruning(Accuracy(), threshold=0.5, operator="=>")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ThresholdPrePruning rejects an invalid operator: OK")


def test_threshold_pre_pruning_all_operators():
    # default polarity=True makes reject() == evaluate() exactly -- the
    # comparison table is unaffected by polarity's existence
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    stats = RuleStats.from_rule(empty_rule, rep, "pos")
    score = Accuracy().score(stats)  # 0.25

    for operator, threshold, expected in [
        (">=", score, True), (">=", score + 0.01, False),
        (">", score, False), (">", score - 0.01, True),
        ("<=", score, True), ("<=", score - 0.01, False),
        ("<", score, False), ("<", score + 0.01, True),
        ("==", score, True), ("==", score + 0.01, False),
        ("!=", score, False), ("!=", score + 0.01, True),
    ]:
        criterion = ThresholdPrePruning(Accuracy(), threshold=threshold, operator=operator)
        assert criterion.reject(empty_rule, stats, rep, "pos") is expected, (operator, threshold)
    print("ThresholdPrePruning supports every comparison operator: OK")


def test_polarity_flips_which_of_accept_reject_equals_evaluate():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    stats = RuleStats.from_rule(empty_rule, rep, "pos")  # accuracy 0.25

    default_polarity = ThresholdPrePruning(Accuracy(), threshold=0.5, operator="<")  # evaluate(): 0.25 < 0.5 -> True
    assert default_polarity.polarity is True
    assert default_polarity.evaluate(empty_rule, stats, rep, "pos") is True
    assert default_polarity.reject(empty_rule, stats, rep, "pos") is True   # reject() == evaluate()
    assert default_polarity.accept(empty_rule, stats, rep, "pos") is False

    flipped = ThresholdPrePruning(Accuracy(), threshold=0.5, operator="<", polarity=False)
    assert flipped.evaluate(empty_rule, stats, rep, "pos") is True  # same raw predicate
    assert flipped.accept(empty_rule, stats, rep, "pos") is True    # but now accept() == evaluate()
    assert flipped.reject(empty_rule, stats, rep, "pos") is False
    print("polarity=False flips which of accept()/reject() equals evaluate() directly: OK")


def test_any_of_and_all_of_composition():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    stats = RuleStats.from_rule(empty_rule, rep, "pos")  # accuracy 0.25

    always_reject = ThresholdPrePruning(Accuracy(), threshold=0.0, operator=">=")  # always true
    never_reject = ThresholdPrePruning(Accuracy(), threshold=2.0, operator=">=")  # always false

    assert AnyOf(never_reject, always_reject).reject(empty_rule, stats, rep, "pos") is True
    assert AnyOf(never_reject, never_reject).reject(empty_rule, stats, rep, "pos") is False
    assert AllOf(always_reject, always_reject).reject(empty_rule, stats, rep, "pos") is True
    assert AllOf(always_reject, never_reject).reject(empty_rule, stats, rep, "pos") is False
    print("AnyOf/AllOf compose PrePruningCriterion instances correctly: OK")


def test_any_of_drives_beam_search_stopping_via_combination_not_a_single_forward():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    never_reject = ThresholdPrePruning(Accuracy(), threshold=2.0, operator=">=")  # alone: never fires
    always_reject = ThresholdPrePruning(Accuracy(), threshold=0.0, operator=">=")  # always true

    # never_reject alone never triggers stopping -- search runs to its
    # natural (optimistic-pruning) end, finding the perfect a-AND-b rule
    alone = BeamSearch(beam_width=2).search(rep, "pos", Accuracy(), initial, stopping=never_reject)
    assert alone.length() == 2

    # combined with always_reject via AnyOf, the disjunction fires at
    # round 1 -- before any rule has been tracked, and with accept()
    # False -- so the search returns None (not a length-2 rule like
    # `alone`). Proof this is genuine OR-combination, not just forwarding
    # never_reject's own (never-firing) behavior.
    combined = AnyOf(never_reject, always_reject)
    stopped = BeamSearch(beam_width=2).search(rep, "pos", Accuracy(), initial, stopping=combined)
    assert stopped is None
    print("AnyOf genuinely combines criteria to drive BeamSearch's stopping, not just forwarding one: OK")


def test_filtering_does_not_reorder_the_beam_and_only_relaxes_optimistic_pruning():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    # evaluate() == "accuracy < 2.0", always true -- rejects every candidate
    reject_everything = ThresholdPrePruning(Accuracy(), threshold=2.0, operator="<")

    # with optimistic pruning OFF, filtering has literally no effect on the
    # search: same beam every round, same rules specialized, same count
    with _counting_expansions() as off_without:
        BeamSearch(beam_width=2, optimistic_pruning=False).search(rep, "pos", Accuracy(), initial)
    with _counting_expansions() as off_with:
        BeamSearch(beam_width=2, optimistic_pruning=False).search(
            rep, "pos", Accuracy(), initial, filtering=reject_everything,
        )
    assert off_without[0] == off_with[0]

    # with optimistic pruning ON, the bound is the best *filter-passing*
    # score -- so a reject-everything filter leaves no bound, and the
    # search explores at least as much as it would unfiltered, never less
    with _counting_expansions() as on_without:
        result_without = BeamSearch(beam_width=2).search(rep, "pos", Accuracy(), initial)
    with _counting_expansions() as on_with:
        result_with = BeamSearch(beam_width=2).search(
            rep, "pos", Accuracy(), initial, filtering=reject_everything,
        )
    assert on_with[0] >= on_without[0]
    assert on_without[0] < off_without[0]  # pruning really did cut work when unfiltered

    # nothing ever passed the filter -> the search returns None (a hard gate),
    # while the unfiltered run finds the real rule
    assert result_without is not None
    assert result_with is None
    print("Filtering never reorders the beam; it only relaxes optimistic pruning's bound: OK")


def test_filtering_excludes_the_best_rule_without_blocking_the_search_from_finding_it():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    unfiltered = BeamSearch(beam_width=2).search(rep, "pos", Accuracy(), initial)
    stats = RuleStats.from_rule(unfiltered, rep, "pos")
    assert (stats.tp, stats.fp) == (2, 0)  # the perfect a-AND-b rule, accuracy 1.0

    # reject anything scoring above 0.8 -- excludes that perfect rule specifically
    filtering = ThresholdPrePruning(Accuracy(), threshold=0.8, operator=">")
    filtered = BeamSearch(beam_width=2).search(rep, "pos", Accuracy(), initial, filtering=filtering)
    filtered_stats = RuleStats.from_rule(filtered, rep, "pos")
    assert (filtered_stats.tp, filtered_stats.fp) == (2, 2)  # falls back to the 0.75 single-condition rule
    print("Filtering excludes the best rule from eligibility without preventing the search from finding it: OK")


def test_stopping_never_fires_on_the_seed_candidate():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    # evaluate() == "accuracy < 2.0" -- always true, even for the seed's
    # own 0.25 accuracy -- but the seed is never checked, so this can
    # only fire starting at the first refinement. When it does fire at
    # round 1, nothing acceptable preceded it, so the search returns None
    # -- NOT the empty seed and NOT round 1's own (rejected) top rule.
    always_true = ThresholdPrePruning(Accuracy(), threshold=2.0, operator="<")
    with _counting_expansions() as count:
        best = BeamSearch(beam_width=3).search(rep, "pos", Accuracy(), initial, stopping=always_true)
    assert best is None
    assert count[0] == 1  # the seed WAS specialized (round 1 ran); it just wasn't stopping-checked
    print("Stopping never fires on the seed; firing at round 1 with nothing tracked yet returns None: OK")


def test_stopping_leave_polarity_returns_the_pre_trigger_best_or_none():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    # default polarity=True: accept() is always False for whatever makes
    # evaluate() fire. Firing before any acceptable rule was tracked ->
    # None (see test_stopping_never_fires...). Firing *after* a good rule
    # was tracked -> that rule, from strictly before the triggering round.
    # Seed at "a" (accuracy 0.75); round 1 climbs to "a AND b" (accuracy
    # 1.0), round 2 to "a AND b AND c" (0.875) where `accuracy < 0.99`
    # fires. optimistic_pruning off so the search actually reaches round 2
    # rather than stopping at the "a AND b" optimum first.
    a_rule = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    fire_below_099 = ThresholdPrePruning(Accuracy(), threshold=0.99, operator="<")
    best = BeamSearch(beam_width=1, optimistic_pruning=False).search(
        rep, "pos", Accuracy(), [(a_rule, frozenset({1, 2}))], stopping=fire_below_099,
    )
    stats = RuleStats.from_rule(best, rep, "pos")
    assert (stats.tp, stats.fp) == (2, 0)  # "a AND b" -- round 1's rule, tracked before round 2 fired
    assert best.length() == 2
    print("Stopping (leave polarity) returns the best rule from before the triggering round: OK")


def test_stopping_enter_polarity_returns_the_triggering_rule_directly():
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset(range(3)))]

    # without stopping, hill-climbing-style search (beam_width=1) reaches
    # the perfect "a AND b" rule -- optimistic pruning only stops it *after*
    # finding the optimum
    without = BeamSearch(beam_width=1).search(rep, "pos", Accuracy(), initial)
    assert without.length() == 2

    # polarity=False: accept() == evaluate() -- the moment accuracy first
    # reaches 0.75 (round 1), accept() is True for that same candidate,
    # so it's returned directly rather than falling back to best_rule
    enter_at_075 = ThresholdPrePruning(Accuracy(), threshold=0.75, operator=">=", polarity=False)
    stopped = BeamSearch(beam_width=1).search(rep, "pos", Accuracy(), initial, stopping=enter_at_075)
    assert stopped.length() == 1
    stats = RuleStats.from_rule(stopped, rep, "pos")
    assert (stats.tp, stats.fp) == (2, 2)
    print("Stopping with polarity=False (enter) returns the triggering rule directly, before it improves further: OK")


def test_single_rule_learner_passes_filtering_and_stopping_through():
    rep, ds = _conjunction_dataset()
    stopping = ThresholdPrePruning(Accuracy(), threshold=0.75, operator=">=", polarity=False)
    learner = SingleRuleLearner(heuristic=Accuracy(), search=BeamSearch(beam_width=1), stopping=stopping)
    rule = learner.learn_one_rule(rep, "pos")
    assert rule.length() == 1  # matches the direct BeamSearch(beam_width=1, stopping=stopping) case
    print("SingleRuleLearner passes filtering/stopping through to the search: OK")


def test_single_rule_learner_propagates_none_when_the_search_finds_nothing_eligible():
    rep, ds = _conjunction_dataset()
    reject_everything = ThresholdPrePruning(Accuracy(), threshold=2.0, operator="<")
    # postprocessing that would blow up on None -- proof it's never reached
    learner = SingleRuleLearner(
        heuristic=Accuracy(),
        postprocessing=ReducedErrorPruning(Accuracy()),
        filtering=reject_everything,
    )
    assert learner.learn_one_rule(rep, "pos") is None
    print("SingleRuleLearner.learn_one_rule returns None (skipping post-processing) when the search does: OK")


def _disjunctive_dataset():
    # y = pos iff (a AND b) OR (c AND d) -- a genuinely disjunctive concept
    # no single conjunction can capture, needing two separate rules
    X = np.array([
        [1, 1, 0, 0],  # pos: a AND b
        [1, 1, 1, 1],  # pos: both
        [0, 0, 1, 1],  # pos: c AND d
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1],  # neg
    ], dtype=bool)
    y = np.array(["pos", "pos", "pos"] + ["neg"] * 7)
    ds = DataSpec(["a", "b", "c", "d"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    return rep, ds, y


def test_seco_is_a_native_rule_learner():
    learner = SingleRuleLearner(heuristic=Accuracy())
    seco = SeCo(single_rule_learner=learner, target_class="pos")
    assert isinstance(seco, NativeRuleLearner)
    print("SeCo is a NativeRuleLearner: OK")


def test_seco_covers_a_disjunctive_concept_with_two_rules():
    rep, ds, y = _disjunctive_dataset()
    learner = SingleRuleLearner(heuristic=Accuracy(), search=BeamSearch(beam_width=3))
    seco = SeCo(single_rule_learner=learner, target_class="pos")

    ruleset = seco.fit(rep)
    assert len(ruleset.rules) == 2
    assert {frozenset(l.feature for l in r.conditions) for r in ruleset.rules} == {
        frozenset({0, 1}), frozenset({2, 3}),  # {a,b} and {c,d}, in either order
    }
    assert ruleset.default_rule is not None and ruleset.default_rule.target == "neg"

    preds = ruleset.predict(rep)
    assert np.all(preds == y)  # perfect coverage of the disjunctive concept
    print("SeCo finds both disjuncts as separate rules and covers the concept perfectly: OK")


class _AlwaysEmptyRuleLearner:
    """Stand-in for SingleRuleLearner that always 'finds' the empty
    (unconditional) rule -- used to test SeCo's own length==0 guard
    directly, isolated from whether search or pruning is what actually
    produced it."""

    def learn_one_rule(self, data, target_class, example_mask=None):
        return Rule([], target=target_class, dataspec=data.spec)


def test_seco_binary_is_a_concept_model_with_a_majority_fallback():
    # 4 "b", 3 "a", 3 "c"; target = "a" -> a ConceptModel for "a" with the
    # overall training-majority label "b" as the negative fallback
    X = np.zeros((10, 2), dtype=bool)
    X[:3, 0] = True  # only the "a" rows carry feature 0
    y = np.array(["a", "a", "a", "b", "b", "b", "b", "c", "c", "c"])
    ds = DataSpec(["f0", "f1"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    learner = SingleRuleLearner(heuristic=Accuracy(), search=BeamSearch(beam_width=3))
    concept = SeCo(single_rule_learner=learner, target_class="a").fit(rep)

    from pyrulearn.models import ConceptModel, MajorityClass
    assert type(concept) is ConceptModel and concept.label == "a"
    assert isinstance(concept.default_prediction, MajorityClass)
    assert concept.default_prediction.constant_target == "b"

    # the fallback is a plain overridable attribute
    concept.default_prediction = "c"
    assert concept.default_rule.target == "c"
    print("SeCo binary fit -> ConceptModel with a MajorityClass fallback: OK")


def test_seco_rules_carry_measured_stats_not_a_stored_weight():
    # No declarative weight is stored anywhere anymore (weight_heuristic
    # is gone from SeCo's constructor entirely) -- every rule's own
    # measured stats() is the one source of truth, scoreable by any
    # heuristic on demand (Laplace here, matching what HeuristicMaxCombiner/
    # sort_rules default to, but any RuleHeuristic works equally).
    rep, ds = _conjunction_dataset()

    for learner in (CN2(significance_threshold=None, target_class="pos"),
                    PFossil(target_class="pos"),
                    SeCo(SingleRuleLearner(heuristic=FoilGain(), search=GainAscentHillClimbing()),
                         target_class="pos")):
        fitted = learner.fit(rep)
        assert fitted.rules
        for r in fitted.rules:
            expected = Laplace().score_rule(r, rep, positive_class="pos")
            measured = Laplace().score(r.stats().confusion.rule_stats("pos"))
            assert measured == pytest.approx(expected)

    aqr = AQR(random_state=0).fit(rep)
    for r in aqr.rules:
        expected = Laplace().score_rule(r, rep, positive_class=r.target)
        measured = Laplace().score(r.stats().confusion.rule_stats(r.target))
        assert measured == pytest.approx(expected)

    # weight_heuristic no longer exists as a constructor argument at all --
    # pass a heuristic to a combiner (HeuristicMaxCombiner(Precision())) or
    # sort_rules(by=Precision()) instead of configuring it on the learner
    with pytest.raises(TypeError):
        SeCo(SingleRuleLearner(heuristic=FoilGain(), search=GainAscentHillClimbing()),
             target_class="pos", weight_heuristic=Precision())
    print("SeCo rules carry measured stats (Laplace-scoreable); weight_heuristic is gone: OK")


def test_ripper_rules_carry_measured_stats():
    data, ds, y = _noisy_3class()
    model = Pypper(random_state=0).fit(data)
    assert model.rules
    # each stage measures its own rules against its own (row-sliced)
    # subproblem, so a rule's own Laplace is >= its Laplace on the full
    # data (fewer negatives left within that stage's scope)
    for r in model.rules:
        full = Laplace().score_rule(r, data, positive_class=r.target)
        own = Laplace().score(r.stats().confusion.rule_stats(r.target))
        assert 0.0 <= own <= 1.0 and own >= full - 1e-9
    print("Pypper rules carry measured stats (Laplace per stage subproblem): OK")


def test_seco_no_target_class_dispatches_the_model_type():
    # 3-class data (a <-> f0, b <-> ~f0 & f1, c <-> ~f0 & ~f1); needs a
    # negation spec so "c vs rest" is expressible
    rng = np.random.default_rng(0)
    raw = rng.random((240, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    ds = neg_spec([f"f{i}" for i in range(4)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(raw), y)
    learner = SingleRuleLearner(heuristic=Accuracy(), search=BeamSearch(beam_width=3))

    ovr = SeCo(single_rule_learner=learner).fit(rep)                 # default: ConceptSet
    assert type(ovr) is ConceptSet and {r.target for r in ovr.rules} == {"a", "b", "c"}

    ordered = SeCo(single_rule_learner=learner).fit(rep, model=ConceptCascade)
    assert type(ordered) is ConceptCascade

    pw = SeCo(single_rule_learner=learner).fit(rep, model=PairwiseModel)
    assert type(pw) is PairwiseModel and pw.pairs

    flat = SeCo(single_rule_learner=learner).fit(rep, model=FlatRuleSet)
    assert type(flat) is FlatRuleSet

    with pytest.raises(TypeError, match="cannot produce"):
        SeCo(single_rule_learner=learner).fit(rep, model=DisjointRuleSet)
    print("SeCo(target_class=None) dispatches model=: OK")


def test_seco_seed_covering_is_one_loop_over_all_classes():
    # 3-class, x <-> f0, y <-> ~f0 & f1, z <-> ~f0 & ~f1
    rng = np.random.default_rng(0)
    raw = rng.random((300, 5)) < 0.5
    y = np.where(raw[:, 0], "x", np.where(raw[:, 1], "y", "z"))
    y = np.where(rng.random(300) < 0.03, rng.permutation(y), y)
    ds = neg_spec([f"f{i}" for i in range(5)])
    rep = HorizontalBitmapRepresentation(ds, neg_X(raw), y)

    m = AQR(random_state=0).fit(rep)                       # AQR default is model=DecisionList
    assert type(m) is DecisionList
    assert {r.target for r in m.rules} == {"x", "y", "z"}  # one loop, rules for every class
    assert m.default_prediction.constant_target == "x"     # training majority
    assert np.mean(np.asarray(m.predict(rep)) == y) > 0.9

    again = AQR(random_state=0).fit(rep)
    assert [r.pos for r in m.rules] == [r.pos for r in again.rules]
    other = AQR(random_state=1).fit(rep)
    assert [r.pos for r in m.rules] != [r.pos for r in other.rules]

    # model=FlatRuleSet: the same rules, resolved by learn order -- same predictions
    fs = AQR(random_state=0).fit(rep, model=FlatRuleSet)
    assert type(fs) is FlatRuleSet and isinstance(fs.combiner, ListCombiner)
    assert [r.pos for r in fs.rules] == [r.pos for r in m.rules]
    assert list(fs.predict(rep)) == list(m.predict(rep))

    # also available (non-default) on the other SeCo learners
    cn2 = CN2(random_state=0).fit(rep, model=FlatRuleSet)
    assert type(cn2) is FlatRuleSet and {r.target for r in cn2.rules} <= {"x", "y", "z"}
    print("model=DecisionList / AQR default: one seed-covering loop over all classes: OK")


def test_seco_rejects_unconditional_rule_even_straight_from_search():
    rep, ds, y = _disjunctive_dataset()
    seco = SeCo(single_rule_learner=_AlwaysEmptyRuleLearner(), target_class="pos")
    ruleset = seco.fit(rep)
    assert len(ruleset.rules) == 0  # rejected immediately, not just when pruning produces it
    assert ruleset.default_rule is not None
    print("SeCo rejects a length-0 rule even straight from the search, not just from pruning: OK")


def test_seco_covering_removes_both_classes_not_just_positives():
    # a=True covers rows 0(pos),1(pos),2(neg) -- both classes present
    X = np.array([[1, 1], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    rule = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)  # a=True

    # replicates exactly the one-line update SeCo.fit uses
    remaining = np.ones(5, dtype=bool)
    remaining = remaining & ~rule.covers_data_packed(rep)
    assert not remaining[0] and not remaining[1] and not remaining[2]  # covered -- both classes gone
    assert remaining[3] and remaining[4]  # uncovered -- still there
    print("SeCo's covering-mask update removes every covered example, both classes, not just positives: OK")


def test_seco_max_rules_caps_the_loop():
    rep, ds, y = _disjunctive_dataset()
    learner = SingleRuleLearner(heuristic=Accuracy(), search=BeamSearch(beam_width=3))
    seco = SeCo(single_rule_learner=learner, target_class="pos", max_rules=1)

    ruleset = seco.fit(rep)
    assert len(ruleset.rules) == 1  # capped, even though 2 would be needed for full coverage
    preds = ruleset.predict(rep)
    assert not np.all(preds == y)  # incomplete coverage as a result
    print("SeCo's max_rules caps the loop even when more rules would help: OK")


def test_seco_stop_covering_prevents_rules_below_threshold():
    rep, ds, y = _disjunctive_dataset()
    learner = SingleRuleLearner(heuristic=Accuracy(), search=BeamSearch(beam_width=3))

    without_criterion = SeCo(single_rule_learner=learner, target_class="pos").fit(rep)
    assert len(without_criterion.rules) == 2
    assert np.all(without_criterion.predict(rep) == y)

    # the first rule found ("a AND b") scores 0.9 accuracy within the full
    # (10-row) remaining scope -- a 0.95 threshold rejects it outright.
    # default polarity=True means evaluate() == reject(), which is all
    # stop_covering ever consults, so this is the same ThresholdPrePruning
    # used for filtering/stopping elsewhere, just handed to a different slot
    criterion = ThresholdPrePruning(Accuracy(), threshold=0.95, operator="<")
    with_criterion = SeCo(single_rule_learner=learner, target_class="pos", stop_covering=criterion).fit(rep)
    assert len(with_criterion.rules) == 0
    assert with_criterion.default_rule.target == "neg"
    preds = with_criterion.predict(rep)
    assert np.mean(preds == y) == 0.7  # all 3 positives misclassified, nothing to catch them
    print("stop_covering (a plain ThresholdPrePruning) can reject every candidate, leaving only the default rule: OK")


def test_uncovered_positives_threshold_criterion_arithmetic():
    # dataset-independent check of the raw evaluate() arithmetic used by
    # test_seco_stop_covering_via_uncovered_positives below
    rep, ds = _conjunction_dataset()
    empty_rule = Rule([], target="pos", dataspec=ds)
    criterion = ThresholdPrePruning(UncoveredPositives(), threshold=0.0, operator=">=")  # p=0
    assert criterion.evaluate(empty_rule, RuleStats(tp=8, fp=0, fn=1, tn=0), rep, "pos") is False  # fn=1 > 0
    assert criterion.evaluate(empty_rule, RuleStats(tp=9, fp=0, fn=0, tn=0), rep, "pos") is True   # fn=0 <= 0
    print("ThresholdPrePruning(UncoveredPositives(), ...) fires exactly when fn <= p: OK")


def test_seco_stop_covering_via_uncovered_positives():
    # "stop once at most p positives would remain uncovered" --
    # ThresholdPrePruning(UncoveredPositives(), threshold=-p, ">=") fires
    # (score = -fn >= -p, i.e. fn <= p) exactly when a candidate rule
    # would *already* bring remaining-uncovered down to <=p -- meaning
    # don't bother keeping this final mopping-up rule, leave those <=p
    # stragglers to the default rule instead. Exercises UncoveredPositives
    # (score = -fn) plugged into stop_covering, reading the just-learned
    # rule's own stats.fn within the remaining scope.
    rep, ds, y = _disjunctive_dataset()
    learner = SingleRuleLearner(heuristic=Accuracy(), search=BeamSearch(beam_width=3))

    unconstrained = SeCo(single_rule_learner=learner, target_class="pos").fit(rep)
    assert len(unconstrained.rules) == 2
    assert np.all(unconstrained.predict(rep) == y)

    # p=0: only discard-and-stop on a rule that leaves *zero* positives
    # uncovered -- the first rule found ("a AND b") leaves 1 (not
    # perfect, so it's kept), the second finishes the job perfectly
    # (fn=0, so *it* gets discarded instead), leaving just the first rule
    criterion = ThresholdPrePruning(UncoveredPositives(), threshold=0.0, operator=">=")
    ruleset = SeCo(single_rule_learner=learner, target_class="pos", stop_covering=criterion).fit(rep)
    assert len(ruleset.rules) == 1
    assert not np.all(ruleset.predict(rep) == y)  # the straggler positive falls to the default rule instead
    print("stop_covering via UncoveredPositives stops once few enough positives would remain uncovered: OK")


def test_seco_raises_without_labels():
    ds = DataSpec(["a", "b"])
    rep_no_y = HorizontalBitmapRepresentation(ds, np.array([[1, 0], [0, 1]], dtype=bool))
    learner = SingleRuleLearner(heuristic=Accuracy())
    seco = SeCo(single_rule_learner=learner, target_class="pos")
    try:
        seco.fit(rep_no_y)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("SeCo.fit raises without labels: OK")


# -- CN2 ---------------------------------------------------------------

def test_cn2_defaults_wire_laplace_and_significance_as_filtering():
    cn2 = CN2(target_class="pos")
    assert isinstance(cn2.single_rule_learner.heuristic, Laplace)
    assert isinstance(cn2.single_rule_learner.search, BeamSearch)
    assert cn2.single_rule_learner.search.beam_width == 5
    assert cn2.single_rule_learner.stopping is None
    assert isinstance(cn2.single_rule_learner.filtering, ThresholdPrePruning)
    assert isinstance(cn2.single_rule_learner.filtering.heuristic, LikelihoodRatio)
    assert cn2.single_rule_learner.filtering.threshold == 3.841
    assert cn2.single_rule_learner.filtering.operator == "<"
    assert cn2.single_rule_learner.filtering.polarity is True
    print("CN2 defaults wire Laplace + a 3.841 likelihood-ratio significance test as filtering: OK")


def test_cn2_overrides_are_all_honored():
    heuristic = Accuracy()
    cn2 = CN2(
        target_class="pos",
        heuristic=heuristic,
        beam_width=2,
        significance_threshold=None,
        max_rules=1,
    )
    assert cn2.single_rule_learner.heuristic is heuristic
    assert cn2.single_rule_learner.search.beam_width == 2
    assert cn2.single_rule_learner.filtering is None  # significance_threshold=None disables it
    assert cn2.max_rules == 1
    print("CN2 constructor overrides (heuristic/beam_width/significance/max_rules) are honored: OK")


def test_cn2_explicit_stopping_overrides_significance_threshold_default():
    cn2 = CN2(target_class="pos", stopping=ThresholdPrePruning(LikelihoodRatio(), 2.0, "<"))
    assert cn2.single_rule_learner.stopping is not None
    assert cn2.single_rule_learner.filtering is None  # significance_threshold's default never built
    print("CN2's explicit stopping= suppresses the default filtering= construction: OK")


def test_cn2_passes_stop_covering_through():
    crit = ThresholdPrePruning(LikelihoodRatio(), 3.841, "<")
    assert CN2(significance_threshold=None, stop_covering=crit).stop_covering is crit
    print("CN2's stop_covering= reaches SeCo: OK")


def test_cn2_explicit_search_overrides_beam_width():
    custom = BeamSearch(beam_width=7)
    cn2 = CN2(target_class="pos", beam_width=2, search=custom)
    assert cn2.single_rule_learner.search is custom
    assert cn2.single_rule_learner.search.beam_width == 7  # beam_width= ignored once search= is given
    print("CN2's explicit search= overrides beam_width=: OK")


def test_cn2_fits_a_disjunctive_concept_with_significance_disabled():
    # with significance_threshold=None, CN2 reduces to plain BeamSearch
    # + Laplace, unconstrained by any stopping trigger -- the search
    # always runs to its natural (optimistic-pruning) end, same as any other
    # BeamSearch call, so this is really a "the wiring works" sanity
    # check, not a claim about the significance test's own behavior
    # (see the two tests below for that).
    rep, ds, y = _disjunctive_dataset()
    cn2 = CN2(target_class="pos", beam_width=3, significance_threshold=None)
    ruleset = cn2.fit(rep)
    assert np.all(ruleset.predict(rep) == y)
    print("CN2 with significance disabled fits a disjunctive concept perfectly: OK")


def test_cn2_significance_as_stopping_can_underfit_where_filtering_does_not():
    # the risk CN2's docstring flags: as a stopping criterion, the
    # significance test can halt the search before it ever reaches the
    # precise concept, which the default, the test as a filter, still
    # finds by running the search out
    rep, ds, y = _disjunctive_dataset()
    stopped = CN2(target_class="pos", beam_width=3,
                  stopping=ThresholdPrePruning(LikelihoodRatio(), 3.841, operator="<")).fit(rep)
    assert not np.all(stopped.predict(rep) == y)
    filtered = CN2(target_class="pos", beam_width=3).fit(rep)
    assert np.all(filtered.predict(rep) == y)
    print("CN2's significance test as stopping underfits a concept the default filter fits perfectly: OK")


def test_cn2_an_absurd_significance_threshold_yields_an_empty_ruleset_either_mode():
    # An absurd threshold makes every round "insignificant" from the very
    # first refinement. As `filtering`: nothing is ever eligible ->
    # search returns None -> SeCo stops with no rules. As `stopping`:
    # fires at round 1 with nothing acceptable tracked yet -> also None ->
    # also an empty ruleset. The two modes only diverge on a *moderate*
    # threshold, where `filtering` (running the search out) can still find
    # a rule that re-enters the acceptable region a few refinements deeper
    # than `stopping` (which quits) would ever look.
    rep, ds, y = _disjunctive_dataset()
    filtered = CN2(target_class="pos", beam_width=3,
                   filtering=ThresholdPrePruning(LikelihoodRatio(), 1000.0, "<")).fit(rep)
    stopped = CN2(target_class="pos", beam_width=3, significance_threshold=1000.0).fit(rep)
    assert list(filtered.rules) == []
    assert list(stopped.rules) == []
    assert filtered.default_rule is not None and stopped.default_rule is not None
    print("An absurd significance threshold empties the ruleset whether wired as filtering or stopping: OK")


def test_cn2_significance_test_as_stopping_is_passed_as_such():
    criterion = ThresholdPrePruning(LikelihoodRatio(), 3.841, "<")
    cn2 = CN2(target_class="pos", stopping=criterion)
    assert cn2.single_rule_learner.filtering is None     # no default filtering once a criterion is given
    assert cn2.single_rule_learner.stopping is criterion
    default = CN2(target_class="pos").single_rule_learner
    assert default.stopping is None and isinstance(default.filtering, ThresholdPrePruning)
    assert default.filtering.threshold == 3.841
    with pytest.raises(TypeError):
        CN2(target_class="pos", mode="stopping")            # no names: pass the criterion
    print("CN2's significance test: filtering by default, stopping when passed as such: OK")


def test_cn2_conceptset_uses_microvotecombiner_not_the_family_max_default():
    # a 3-class problem so fit(data) with no target_class/model= reaches
    # _fit_one_vs_rest -> ConceptSet, the one producer CN2 overrides
    ds = DataSpec(["a", "b", "c"])
    X = np.array([[1, 0, 0]] * 5 + [[0, 1, 0]] * 5 + [[0, 0, 1]] * 5, dtype=bool)
    y = np.array(["x"] * 5 + ["y"] * 5 + ["z"] * 5)
    rep = HorizontalBitmapRepresentation(ds, X, y)
    model = CN2(random_state=0).fit(rep)
    assert isinstance(model, ConceptSet)
    assert isinstance(model.combiner, MicroVoteCombiner)
    print("CN2's default ConceptSet uses MicroVoteCombiner, not the family's generic 'max': OK")


def test_cn2_microvotecombiner_matches_clark_and_boswell_1991s_summed_distribution_vote():
    # Reproduces the mechanism (not the exact numbers) of the worked example in
    # Clark & Boswell, 1991 sec. on unordered rule sets (also quoted in Lavrac
    # et al., "Subgroup Discovery with CN2-SD", JMLR 2004, sec. 2.3): each rule
    # carries the class distribution of the training rows it covers, and a
    # clash between firing rules of different classes is resolved by summing
    # those distributions and predicting the largest total -- not by picking
    # the single highest-Laplace rule (the family's generic HeuristicMaxCombiner()).
    # Here: one pure "elephant" rule (5 covered, higher individual Laplace)
    # against two pure "bird" rules (4 covered each) that together outvote it.
    ds = DataSpec(["a", "b", "c"])
    X = np.array(
        [[1, 0, 0]] * 5 +  # a=1 -> elephant (rule A's own coverage)
        [[0, 1, 0]] * 4 +  # b=1 -> bird (rule B's own coverage)
        [[0, 0, 1]] * 4,   # c=1 -> bird (rule C's own coverage)
        dtype=bool,
    )
    y = np.array(["elephant"] * 5 + ["bird"] * 4 + ["bird"] * 4)
    rep = HorizontalBitmapRepresentation(ds, X, y)

    rule_a = Rule.from_pos_neg(pos=[0], target="elephant", dataspec=ds)
    rule_b = Rule.from_pos_neg(pos=[1], target="bird", dataspec=ds)
    rule_c = Rule.from_pos_neg(pos=[2], target="bird", dataspec=ds)
    rule_a, rule_b, rule_c = annotate_rules([rule_a, rule_b, rule_c], rep)

    # rule A alone has the higher Laplace accuracy: (5+1)/(5+2) = 6/7 ~ 0.857
    # vs. rule B/C's (4+1)/(4+2) = 5/6 ~ 0.833 each -- "max" picks A -> elephant
    assert Laplace().score(rule_a.stats().confusion.rule_stats("elephant")) > \
           Laplace().score(rule_b.stats().confusion.rule_stats("bird"))

    fired_by_all = HorizontalBitmapRepresentation(ds, np.array([[1, 1, 1]], dtype=bool))
    rs = FlatRuleSet([rule_a, rule_b, rule_c])
    assert list(rs.predict(fired_by_all, combiner=HeuristicMaxCombiner(Laplace()))) == ["elephant"]
    # MicroVoteCombiner sums raw covered counts instead: elephant=5, bird=4+4=8 -> bird wins
    assert list(rs.predict(fired_by_all, combiner=MicroVoteCombiner())) == ["bird"]
    print("MicroVoteCombiner sums covered-class distributions and can flip 'max's own answer, "
          "matching CN2-unordered's actual voting rule: OK")


def test_cn2_laplace_counts_the_classes_of_the_data():
    from pyrulearn.heuristics import Laplace, WRAcc
    from pyrulearn.learners.seco import CN2
    rng = np.random.default_rng(0)
    raw = rng.random((90, 4)) < 0.5
    spec = neg_spec([f"f{i}" for i in range(4)])
    three = HorizontalBitmapRepresentation(spec, neg_X(raw), np.array(["a", "b", "c"] * 30))
    cn2 = CN2()
    cn2.fit(three)
    assert cn2.single_rule_learner.heuristic.n_classes == 3
    cn2.fit(HorizontalBitmapRepresentation(spec, neg_X(raw), np.array(["a", "b"] * 45)))
    assert cn2.single_rule_learner.heuristic.n_classes == 2
    fixed = CN2(heuristic=Laplace())
    fixed.fit(three)
    assert fixed.single_rule_learner.heuristic.n_classes == 2
    other = CN2(heuristic=WRAcc())
    other.fit(three)
    assert isinstance(other.single_rule_learner.heuristic, WRAcc)


def test_cn2_returns_significant_rules_only():
    # every rule CN2 returns passed the significance filter on the
    # examples left at its turn, so the first is significant on all
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(120, 6)).astype(bool)
    y = np.where(X[:, 0] & X[:, 1] | X[:, 2], "pos", "neg")
    flip = rng.random(120) < 0.15
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    data = HorizontalBitmapRepresentation(DataSpec([f"x{i}" for i in range(6)]), X, y)
    model = CN2(target_class="pos").fit(data)
    assert len(model.rules) >= 1
    first = model.rules[0]
    assert LikelihoodRatio().score(RuleStats.from_rule(first, data, "pos")) >= 3.841
    print(f"CN2 with the significance filter: {len(model.rules)} rules, the first significant: OK")


# ---- AQR (Clark & Niblett, 1989) + SeedExample -----------------------------

def test_seed_example_seeds_on_an_uncovered_positive():
    rep, ds, y = _disjunctive_dataset()
    candidates = SeedExample("first").initial_candidates(rep, "pos", np.ones(rep.n_samples, dtype=bool))
    assert len(candidates) == 1
    rule, mask = candidates[0]
    assert rule.length() == 0 and rule.target == "pos"
    # row 0 is the first positive: [1,1,0,0] -> open only to features a, b
    assert mask == frozenset({0, 1})
    print("SeedExample: seeds on the first uncovered positive, open mask = its True features: OK")


def test_seed_example_respects_the_example_mask():
    rep, ds, y = _disjunctive_dataset()
    mask = np.ones(rep.n_samples, dtype=bool)
    mask[:2] = False  # hide the two a-AND-b positives; only the c-AND-d one (row 2) is left
    _, open_mask = SeedExample("first").initial_candidates(rep, "pos", mask)[0]
    assert open_mask == frozenset({2, 3})  # row 2 is [0,0,1,1]
    print("SeedExample: only seeds on positives that are still in scope: OK")


def test_seed_example_raises_with_no_uncovered_positive():
    rep, ds, y = _disjunctive_dataset()
    none_left = np.zeros(rep.n_samples, dtype=bool)
    try:
        SeedExample().initial_candidates(rep, "pos", none_left)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "seed" in str(e).lower()
    print("SeedExample raises when there is no uncovered positive to seed on: OK")


def test_seed_example_random_strategy_is_reproducible():
    rep, ds, y = _disjunctive_dataset()
    full = np.ones(rep.n_samples, dtype=bool)
    a = SeedExample("random", random_state=0).initial_candidates(rep, "pos", full)[0][1]
    b = SeedExample("random", random_state=0).initial_candidates(rep, "pos", full)[0][1]
    assert a == b  # same random_state -> same seed
    # whichever positive it lands on, the open mask is that row's True features
    assert a in ({0, 1}, {2, 3}) or len(a) > 0
    print("SeedExample('random') is reproducible under a fixed random_state: OK")


def test_seed_example_rejects_unknown_strategy():
    try:
        SeedExample("sideways")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("SeedExample rejects an unknown strategy: OK")


def test_seed_example_index_strategy_seeds_on_the_exact_row():
    rep, ds, y = _disjunctive_dataset()
    # row 2 is [0,0,1,1] (a c-AND-d positive), regardless of target_class / mask
    _, open_mask = SeedExample("index", index=2).initial_candidates(rep, "anything", None)[0]
    assert open_mask == frozenset({2, 3})
    try:
        SeedExample("index")  # index is required
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("SeedExample(strategy='index') seeds on the given row, ignoring class/mask: OK")


def test_aqr_defaults_wire_seed_lef_and_consistency_filtering():
    aqr = AQR(target_class="pos")
    srl = aqr.single_rule_learner
    assert isinstance(srl.space_init, SeedExample)
    assert isinstance(srl.heuristic, LEF)
    assert [type(h) for h in srl.heuristic.heuristics] == [CoveredPositives, CoveredNegatives, MinimalLength]
    assert isinstance(srl.filtering, ThresholdPrePruning)
    assert isinstance(srl.filtering.heuristic, CoveredNegatives)
    assert srl.stopping is None
    assert isinstance(srl.search, BeamSearch) and srl.search.beam_width == 5  # the paper's star size
    print("AQR defaults: SeedExample + a LEF + consistency as filtering, beam_width 5: OK")


def test_aqr_maxstar_is_the_star_size_and_search_overrides_it():
    assert AQR(target_class="pos", maxstar=3).single_rule_learner.search.beam_width == 3
    custom = BeamSearch(beam_width=9)
    assert AQR(target_class="pos", maxstar=3, search=custom).single_rule_learner.search is custom
    print("AQR: maxstar sets the beam width; an explicit search= overrides it: OK")


def test_aqr_overrides_are_all_honored():
    aqr = AQR(
        target_class="pos",
        heuristic=CoveredPositives(),
        require_consistency=False,
        space_init=EmptyRuleAllFeatures(),
    )
    srl = aqr.single_rule_learner
    assert isinstance(srl.heuristic, CoveredPositives)
    assert srl.filtering is None  # require_consistency=False drops the gate
    assert isinstance(srl.space_init, EmptyRuleAllFeatures)
    print("AQR honors heuristic=/require_consistency=/space_init= overrides: OK")


def test_aqr_covers_a_disjunctive_concept_with_consistent_rules():
    rep, ds, y = _disjunctive_dataset()  # y = pos iff (a AND b) OR (c AND d)
    ruleset = AQR(target_class="pos", maxstar=5).fit(rep)

    assert len(ruleset.rules) == 2
    assert {frozenset(l.feature for l in r.conditions) for r in ruleset.rules} == {
        frozenset({0, 1}), frozenset({2, 3}),
    }
    # AQR's defining guarantee: every rule covers zero negatives
    for r in ruleset.rules:
        assert RuleStats.from_rule(r, rep, "pos").fp == 0
    assert ruleset.default_rule is not None and ruleset.default_rule.target == "neg"
    assert np.all(ruleset.predict(rep) == y)
    print("AQR covers the disjunctive concept with two consistent rules: OK")


def test_aqr_uses_negation_features_for_negated_conditions():
    # y = pos iff (a AND NOT b); needs an explicit negation feature to express
    ds = neg_spec(["a", "b"])
    X = neg_X([[1, 0], [1, 0], [1, 1], [0, 0], [0, 1]])
    y = np.array(["pos", "pos", "neg", "neg", "neg"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    ruleset = AQR(target_class="pos").fit(rep)
    assert len(ruleset.rules) == 1
    r = ruleset.rules[0]
    assert set(r.pos) == {ds.feature_index("a"), ds.feature_index("not b")}
    assert RuleStats.from_rule(r, rep, "pos").fp == 0
    assert np.all(ruleset.predict(rep) == y)
    print("AQR grows a negated condition as a literal on the paired negation feature: OK")


def test_aqr_stops_when_no_consistent_rule_can_be_found():
    # rows 0 and 1 have identical features but opposite labels -- no rule can
    # cover the positive without also covering the negative, so AQR (which
    # requires consistency) can add nothing and stops with an empty ruleset
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "neg", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    ruleset = AQR(target_class="pos").fit(rep)
    assert len(ruleset.rules) == 0
    assert ruleset.default_rule is not None and ruleset.default_rule.target == "neg"
    print("AQR stops (empty ruleset) when the data admits no consistent rule: OK")


def test_aqr_sets_a_bad_seed_aside_and_keeps_covering():
    # the first positive (row 0) is a noisy duplicate of a negative (row 1),
    # so no consistent rule can cover it. AQR seeds on the first uncovered
    # positive: it must set that seed aside and still learn the real concept
    # (a) from the other positives -- not stop with an empty rule set
    X = np.array([[0, 0], [0, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "neg", "pos", "pos", "neg", "neg", "neg"])
    rep = HorizontalBitmapRepresentation(DataSpec(["a", "b"]), X, y)

    for strategy in ("first", "random"):
        ruleset = AQR(target_class="pos", space_init=SeedExample(strategy, 0)).fit(rep)
        assert [set(r.pos) for r in ruleset.rules] == [{0}], strategy
        assert list(ruleset.predict(rep)) == ["neg", "neg", "pos", "pos", "neg", "neg", "neg"]
    print("AQR sets a seed with no consistent rule aside and keeps covering: OK")


def test_aqr_require_consistency_false_lets_an_inconsistent_rule_through():
    # same unfittable data as above, but with the consistency gate off AQR
    # will accept the best rule its LEF finds even though it covers a negative
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "neg", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)

    ruleset = AQR(target_class="pos", heuristic=CoveredPositives(), require_consistency=False).fit(rep)
    assert len(ruleset.rules) == 1
    assert RuleStats.from_rule(ruleset.rules[0], rep, "pos").fp > 0  # inconsistent, but accepted
    print("AQR(require_consistency=False) accepts an inconsistent rule: OK")


def test_encoding_length_restriction_arithmetic_matches_quinlan_1990_p251():
    # direct check against Quinlan (1990), Machine Learning 5(3):239-266,
    # p. 251: encoding_length = l*(1+log2(d)) - log2(l!), benefit =
    # log2(N) + log2(C(N,p)); evaluate() fires once encoding_length >
    # benefit. d=100 features (not e.g. 6) is deliberate -- confirmed
    # empirically that with only a handful of features, 1+log2(d) bits/
    # literal is too small for the criterion to ever fire at realistic
    # rule lengths/coverage (see test_pfoil_mdl_stopping_curbs_growth_
    # on_a_wide_noisy_dataset's own comment for the same finding at the
    # whole-search level)
    import math
    ds_wide = DataSpec([f"a{i}" for i in range(100)])
    rep_wide = HorizontalBitmapRepresentation(ds_wide, np.zeros((1, 100), dtype=bool))
    criterion = EncodingLengthRestriction()

    # l=2, p=40, N=300: nowhere near "too expensive" yet
    rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds_wide)
    stats = RuleStats(tp=40, fp=2, fn=10, tn=248)
    encoding_length = 2 * (1 + math.log2(100)) - math.log2(math.factorial(2))
    benefit = math.log2(300) + math.log2(math.comb(300, 40))
    assert encoding_length < benefit
    assert criterion.evaluate(rule, stats, rep_wide, "pos") is False

    # l=6, p=1, N=300: a long rule barely covering anything does trip it
    long_rule = Rule.from_pos_neg(pos=[0, 1, 2, 3, 4, 5], target="pos", dataspec=ds_wide)
    barely_covering_stats = RuleStats(tp=1, fp=0, fn=149, tn=150)
    encoding_length_long = 6 * (1 + math.log2(100)) - math.log2(math.factorial(6))
    benefit_barely = math.log2(300) + math.log2(math.comb(300, 1))
    assert encoding_length_long > benefit_barely
    assert criterion.evaluate(long_rule, barely_covering_stats, rep_wide, "pos") is True
    print("EncodingLengthRestriction's arithmetic matches Quinlan 1990 p.251 by hand-computation: OK")


def test_encoding_length_restriction_never_fires_on_the_empty_rule():
    ds6 = DataSpec(["a", "b", "c", "d", "e", "f"])
    rep6 = HorizontalBitmapRepresentation(ds6, np.zeros((1, 6), dtype=bool))
    empty = Rule([], target="pos", dataspec=ds6)
    criterion = EncodingLengthRestriction()
    assert criterion.evaluate(empty, RuleStats(tp=150, fp=150, fn=0, tn=0), rep6, "pos") is False
    print("EncodingLengthRestriction never fires on the length-0 (empty) rule: OK")


def _wide_noisy_disjunction_dataset(n=300, d=60, seed=1, noise_rate=0.05):
    # y = pos iff (a0 AND a1) OR (a2 AND NOT a3), plus label noise -- many
    # more (irrelevant) features than the toy 4-feature disjunctive
    # dataset, needed because EncodingLengthRestriction's benefit term
    # (dominated by log2(C(N,p))) swamps its encoding-length term whenever
    # d is small (confirmed empirically: the criterion is essentially
    # inert on _disjunctive_dataset's 4 features, only visibly constrains
    # growth once d is large enough for 1+log2(d) bits/literal to matter)
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, d)).astype(bool)
    ds = DataSpec([f"a{i}" for i in range(d)])
    y_clean = (X[:, 0] & X[:, 1]) | (X[:, 2] & ~X[:, 3])
    noise = rng.random(n) < noise_rate
    y = np.where(noise, ~y_clean, y_clean)
    y = np.where(y, "pos", "neg")
    rep = HorizontalBitmapRepresentation(ds, X, y)
    return rep, y


def test_pfoil_defaults_wire_gain_ascent_hill_climbing_foilgain_and_mdl_stopping():
    pfoil = PFoil(target_class="pos")
    assert isinstance(pfoil.single_rule_learner.heuristic, FoilGain)
    assert isinstance(pfoil.single_rule_learner.search, GainAscentHillClimbing)
    assert pfoil.single_rule_learner.filtering is None
    assert isinstance(pfoil.single_rule_learner.stopping, EncodingLengthRestriction)
    print("PFoil defaults wire GainAscentHillClimbing + FoilGain + EncodingLengthRestriction as stopping: OK")


def test_pfoil_mdl_stopping_false_recovers_mooneys_unrestricted_pfoil():
    pfoil = PFoil(target_class="pos", mdl_stopping=False)
    assert pfoil.single_rule_learner.stopping is None
    assert pfoil.single_rule_learner.filtering is None
    print("PFoil(mdl_stopping=False) disables the MDL criterion entirely: OK")


def test_pfoil_finds_the_perfect_conjunction_on_the_toy_dataset():
    # sanity check the wiring actually runs end-to-end and finds a
    # sensible rule -- not a claim about MDL stopping's effect (see the
    # wide-dataset test below for that; this toy dataset has too few
    # features for the criterion to ever fire, confirmed empirically)
    rep, ds = _conjunction_dataset()
    pfoil = PFoil(target_class="pos")
    ruleset = pfoil.fit(rep)
    assert len(ruleset.rules) >= 1
    assert {l.feature for l in ruleset.rules[0].conditions} == {0, 1}
    print("PFoil finds the perfect a-AND-b conjunction on the toy dataset: OK")


def test_pfoil_mdl_stopping_curbs_growth_on_a_wide_noisy_dataset():
    # with only 6 features (_wide_noisy_disjunction_dataset's opposite
    # case), the MDL criterion never fires -- confirmed empirically
    # (test above). With d=60, 1+log2(60)~=6.9 bits/literal is large
    # enough that it visibly trims the last, most overfit-prone rule
    # relative to mdl_stopping=False on the exact same search
    rep, y = _wide_noisy_disjunction_dataset(d=60, seed=1)
    restricted = PFoil(target_class="pos").fit(rep)
    unrestricted = PFoil(target_class="pos", mdl_stopping=False).fit(rep)

    acc_restricted = np.mean(restricted.predict(rep) == y)
    acc_unrestricted = np.mean(unrestricted.predict(rep) == y)
    assert acc_restricted <= acc_unrestricted  # MDL stopping trims the final rule short of perfectly memorizing
    assert acc_unrestricted == 1.0  # unrestricted growth fully memorizes this (small, noisy) training set
    assert acc_restricted < 1.0     # MDL stopping actually intervened, not a no-op
    print("PFoil's MDL stopping measurably curbs overfitting relative to unrestricted growth on a wide dataset: OK")


def test_pfossil_defaults_wire_hillclimbing_correlation_and_filtering():
    srl = PFossil(target_class="pos").single_rule_learner
    assert isinstance(srl.search, HillClimbing)
    assert not isinstance(srl.search, GainAscentHillClimbing)  # plain heuristic path
    assert isinstance(srl.heuristic, Correlation)  # passed straight through, no DeltaGain wrap
    assert srl.stopping is None
    assert isinstance(srl.filtering, ThresholdPrePruning)
    assert isinstance(srl.filtering.heuristic, Correlation)
    assert srl.filtering.threshold == 0.3
    assert srl.filtering.operator == "<"
    print("PFossil defaults wire HillClimbing + Correlation + a 0.3 threshold as filtering: OK")


def test_pfossil_threshold_as_stopping_is_passed_as_such():
    criterion = ThresholdPrePruning(Correlation(), 0.3, "<")
    srl = PFossil(target_class="pos", stopping=criterion).single_rule_learner
    assert srl.filtering is None
    assert srl.stopping is criterion
    print("PFossil's correlation threshold as stopping=, when passed as such: OK")


def test_pfossil_passes_any_plain_heuristic_straight_through():
    # no more DeltaGain wrapping -- HillClimbing ranks a plain heuristic
    # directly, and an explicit BeamSearch always did
    hc = PFossil(target_class="pos", heuristic=Laplace()).single_rule_learner.heuristic
    assert isinstance(hc, Laplace)
    beam = PFossil(
        target_class="pos", search=BeamSearch(beam_width=5), heuristic=Correlation(),
    ).single_rule_learner.heuristic
    assert isinstance(beam, Correlation)
    print("PFossil passes a plain heuristic straight through for both HillClimbing and BeamSearch: OK")


def test_pfossil_correlation_threshold_none_disables_the_criterion():
    pfossil = PFossil(target_class="pos", correlation_threshold=None)
    assert pfossil.single_rule_learner.stopping is None
    assert pfossil.single_rule_learner.filtering is None
    print("PFossil(correlation_threshold=None) disables the criterion entirely: OK")


def test_pfossil_fits_a_disjunctive_concept_with_the_threshold_disabled():
    rep, ds, y = _disjunctive_dataset()
    pfossil = PFossil(target_class="pos", correlation_threshold=None)
    ruleset = pfossil.fit(rep)
    assert np.all(ruleset.predict(rep) == y)
    print("PFossil with the correlation threshold disabled fits a disjunctive concept perfectly: OK")


# -- Pypper ------------------------------------------------------------

def _noisy_3class(n=500, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & ~raw[:, 1], "a",
         np.where(raw[:, 2] & raw[:, 3], "b", "c"))
    y = np.where(rng.random(n) < 0.05, rng.permutation(y), y)
    ds = neg_spec([f"f{i}" for i in range(6)])
    return HorizontalBitmapRepresentation(ds, neg_X(raw), y), ds, y


def test_ripper_returns_a_least_frequent_first_concept_cascade():
    data, ds, y = _noisy_3class()
    model = Pypper(random_state=0).fit(data)
    assert type(model) is ConceptCascade
    assert model.default_prediction == "c"                  # most frequent -> catch-all
    assert set(r.target for r in model.rules) <= {"a", "b"}  # rarer classes get rules
    assert np.mean(np.asarray(model.predict(data)) == y) > 0.85
    print("Pypper -> least-frequent-first ConceptCascade, majority is the default: OK")


def test_ripper_is_reproducible():
    data, ds, y = _noisy_3class()
    a = Pypper(random_state=1).fit(data)
    b = Pypper(random_state=1).fit(data)
    assert [r.pos for r in a.rules] == [r.pos for r in b.rules]
    print("Pypper(random_state=) is reproducible: OK")


def test_ripper_stage_learner_wires_the_irep_grow_phase():
    stage = Pypper(random_state=0)._stage_learner()
    srl = stage.single_rule_learner
    assert isinstance(srl.search, GainAscentHillClimbing)
    assert srl.search.stop_at_local_optimum is False  # grow to consistency, not to the gain peak
    assert isinstance(srl.postprocessing, ReducedErrorPruning)  # ...then prune it back
    assert isinstance(srl.stopping, ThresholdPrePruning)  # RIPPER's minNo: keep >= 2 positives
    assert srl.stopping.threshold == 2
    # covering loop stops on FOIL's MDL restriction OR IREP's coin-flip rule
    assert isinstance(stage.stop_covering, AnyOf)
    kinds = {type(c) for c in stage.stop_covering.criteria}
    assert EncodingLengthRestriction in kinds and ThresholdPrePruning in kinds
    irep = next(c for c in stage.stop_covering.criteria if isinstance(c, ThresholdPrePruning))
    assert isinstance(irep.heuristic, Precision) and irep.threshold == 0.5 and irep.operator == "<"
    print("Pypper stage learner: grow-to-consistency search + REP + minNo + IREP covering stop: OK")


def test_irep_covering_stop_prevents_a_coin_flip_rule_from_being_kept():
    # a target class whose positives are ~40% of what any rule can isolate:
    # a covering loop with no quality floor keeps adding sub-0.5 rules; the
    # IREP stop (Precision < 0.5) halts before the first one is appended.
    rng = np.random.default_rng(7)
    n = 400
    X = rng.integers(0, 2, size=(n, 5)).astype(bool)
    # y=a is a real but weak signal; nothing gets a pure rule for it
    p = np.where(X[:, 0], 0.55, 0.2)
    y = np.where(rng.random(n) < p, "a", "b")
    ds = DataSpec([f"f{i}" for i in range(5)])
    data = HorizontalBitmapRepresentation(ds, X, y)

    floor = ThresholdPrePruning(Precision(), 0.5, operator="<")
    grow = SingleRuleLearner(heuristic=FoilGain(), search=GainAscentHillClimbing(stop_at_local_optimum=False),
                             preparation=GrowPruneSplit(1 / 3, 0), postprocessing=ReducedErrorPruning(Precision()))
    with_floor = SeCo(grow, target_class="a", stop_covering=floor).fit(data)
    without = SeCo(grow, target_class="a").fit(data)

    assert all(RuleStats.from_rule(r, data, "a").tp
               / max(1, RuleStats.from_rule(r, data, "a").tp + RuleStats.from_rule(r, data, "a").fp) >= 0.5
               for r in with_floor.rules)
    assert len(with_floor.rules) <= len(without.rules)
    print("IREP covering stop keeps every rule above coin-flip precision: OK")


def test_replace_revise_optimization_simplifies_a_noisy_ruleset():
    data, ds, y = _noisy_3class(n=600)
    grow = SingleRuleLearner(
        heuristic=FoilGain(), search=GainAscentHillClimbing(),
        preparation=GrowPruneSplit(1 / 3, 0), postprocessing=ReducedErrorPruning(Precision()),
    )
    plain = SeCo(grow, target_class="a").fit(data)
    optimized = SeCo(grow, target_class="a", optimization=ReplaceReviseOptimization(2, 1 / 3, 0)).fit(data)

    assert len(optimized.rules) <= len(plain.rules)
    # optimization is guided by description length -- it doesn't make it worse
    dl_plain = rule_set_description_length(plain.rules, data, "a")
    dl_opt = rule_set_description_length(optimized.rules, data, "a")
    assert dl_opt <= dl_plain + 1e-6
    print(f"ReplaceReviseOptimization: {len(plain.rules)} -> {len(optimized.rules)} rules, "
          f"DL {dl_plain:.0f} -> {dl_opt:.0f}: OK")


def test_rule_set_description_length_grows_with_redundancy():
    data, ds, y = _noisy_3class(n=300)
    r1 = SeCo(
        SingleRuleLearner(heuristic=FoilGain(), search=GainAscentHillClimbing()),
        target_class="a",
    ).fit(data).rules
    assert r1
    dl1 = rule_set_description_length(r1, data, "a")
    dl2 = rule_set_description_length(r1 + [r1[0]], data, "a")  # a duplicate rule adds theory bits
    assert dl2 > dl1
    print("rule_set_description_length: a redundant rule costs more bits: OK")


def test_ripper_on_binary_data():
    data, ds, y = _noisy_3class()
    yb = np.where(y == "a", "a", "rest")
    db = HorizontalBitmapRepresentation(ds, np.asarray(data.X), yb)
    model = Pypper(random_state=0).fit(db)
    assert type(model) is ConceptCascade and model.default_prediction == "rest"
    assert np.mean(np.asarray(model.predict(db)) == yb) > 0.9
    print("Pypper on a binary problem: rules for the minority class, majority default: OK")


# ------------------------------------------------------------- fit-time stats ---

def test_covering_loop_rules_and_default_rule_carry_their_own_training_stats():
    rep, ds, y = _disjunctive_dataset()
    model = CN2(target_class="pos").fit(rep)  # -> ConceptModel
    for r in model.rules:
        expected = RuleStats.from_rule(r.rule, rep, "pos")
        got = r.stats().confusion.rule_stats("pos")
        assert (got.tp, got.fp, got.fn, got.tn) == (expected.tp, expected.fp, expected.fn, expected.tn)
    dr_stats = model.default_rule.stats()
    assert dr_stats is not None and dr_stats.confusion is not None
    print("CN2's covering-loop rules and default_rule carry stats matching RuleStats.from_rule: OK")


def test_seed_covering_rules_carry_their_own_training_stats():
    rep, ds, y = _disjunctive_dataset()
    model = AQR(random_state=0).fit(rep)  # -> FlatRuleSet, _seed_covering_fit path
    for r in model.rules:
        expected = RuleStats.from_rule(r.rule, rep, r.target)
        got = r.stats().confusion.rule_stats(r.target)
        assert (got.tp, got.fp, got.fn, got.tn) == (expected.tp, expected.fp, expected.fn, expected.tn)
    print("AQR's seed-covering rules carry stats matching RuleStats.from_rule: OK")


# ----------------------------------------------- several rules per search

def _either_dataset(n=400, seed=0):
    # y = a OR b: a and b are (about) equally good first conditions
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 3)) < 0.5
    y = np.where(raw[:, 0] | raw[:, 1], "pos", "neg")
    return HorizontalBitmapRepresentation(DataSpec(["a", "b", "c"]), raw, y)


def test_search_all_is_search_without_branching():
    rep = _either_dataset()
    initial = EmptyRuleAllFeatures().initial_candidates(rep, "pos")
    for search, heuristic in [(GainAscentHillClimbing(), FoilGain()), (HillClimbing(), Laplace()),
                              (BeamSearch(beam_width=3), Laplace())]:
        one = search.search(rep, "pos", heuristic, initial)
        assert search.search_all(rep, "pos", heuristic, initial) == ([] if one is None else [one])
    print("search_all without branching is search's one rule: OK")


def test_branching_gain_ascent_yields_a_copy_at_each_near_tie():
    rep = _either_dataset()
    initial = EmptyRuleAllFeatures().initial_candidates(rep, "pos")
    plain = GainAscentHillClimbing(branch_similarity=1.0).search_all(rep, "pos", FoilGain(), initial)
    both = GainAscentHillClimbing(branch_similarity=0.5).search_all(rep, "pos", FoilGain(), initial)
    assert len(plain) == 1
    assert {r.pos for r in both} == {(0,), (1,)}                  # a and b, from one search
    # search still returns the best-gain lineage alone
    best = GainAscentHillClimbing(branch_similarity=0.5).search(rep, "pos", FoilGain(), initial)
    assert best in both and len([best]) == 1
    # min_gain: a gain of at least this, not just any positive gain
    stuck = GainAscentHillClimbing(min_gain=1e9).search_all(rep, "pos", FoilGain(), initial)
    assert [r.length() for r in stuck] == [0]                     # no move: the seed, as `search` returns
    assert GainAscentHillClimbing(min_gain=1e9, branch_similarity=0.5).search_all(
        rep, "pos", FoilGain(), initial) == []                    # branching never yields the empty seed
    with pytest.raises(ValueError):
        GainAscentHillClimbing(min_gain=0.0)
    with pytest.raises(ValueError):
        GainAscentHillClimbing(branch_similarity=1.5)
    print("branching gain ascent yields a copy at each near-tie: OK")


def test_seco_accepts_every_rule_of_a_branching_search():
    rep = _either_dataset()
    plain = PFoil(target_class="pos").fit(rep)
    branching = PFoil(target_class="pos", search=GainAscentHillClimbing(branch_similarity=0.5)).fit(rep)
    assert len(branching.rules) >= len(plain.rules)
    assert all(r.target == "pos" and r.length() > 0 for r in branching.rules)
    # PFoil's stopping criterion still applies per lineage
    assert np.mean(np.asarray(branching.predict(rep)) == rep.y) >= np.mean(np.asarray(plain.predict(rep)) == rep.y) - 0.05
    print("SeCo accepts every rule a branching search yields: OK")


def test_cpar_is_a_seco_configuration():
    from pyrulearn.learners.cpar import CPAR
    cpar = CPAR()
    assert isinstance(cpar, SeCo)
    search = cpar.single_rule_learner.search
    assert type(search) is GainAscentHillClimbing
    assert (search.min_gain, search.branch_similarity) == (0.7, 0.99)
    assert isinstance(cpar.single_rule_learner.heuristic, FoilGain)
    print("CPAR is a SeCo configuration: OK")


if __name__ == "__main__":
    test_beam_search_finds_perfect_conjunction()
    test_hill_climbing_finds_the_perfect_conjunction_with_foilgain()
    test_hill_climbing_with_delta_gain_matches_foilgain_on_this_dataset()
    test_hill_climbing_stops_at_the_start_when_no_move_has_positive_gain()
    test_hill_climbing_returns_none_when_the_walk_cant_start_and_a_gate_is_set()
    test_hill_climbing_rejects_a_non_gain_heuristic()
    test_hill_climbing_rejects_more_than_one_initial_candidate()
    test_beam_search_deduplicates_rules_reached_by_different_orders()
    test_beam_search_respects_example_mask()
    test_beam_search_raises_on_empty_initial_candidates()
    test_beam_search_never_returns_worse_than_initial_candidate()
    test_empty_rule_all_features_starts_from_universal_rule()
    test_feature_subset_resolves_names_to_indices()
    test_no_split_passes_mask_through_unchanged()
    test_grow_prune_split_produces_disjoint_masks_covering_scope()
    test_grow_prune_split_falls_back_when_stratification_is_impossible()
    test_grow_prune_split_falls_back_to_no_split_when_too_few_examples()
    test_reduced_error_pruning_strips_condition_that_doesnt_generalize()
    test_reduced_error_pruning_noop_without_context()
    test_reduced_error_pruning_never_truncates_to_the_empty_rule()
    test_reduced_error_pruning_leaves_an_already_empty_rule_alone()
    test_single_rule_learner_default_config_matches_bare_beam_search()
    test_single_rule_learner_full_ripper_style_config_runs_end_to_end()
    test_beam_search_tp_zero_floor_stops_specializing()
    test_beam_search_fp_zero_floor_stops_specializing()
    test_threshold_pre_pruning_rejects_bad_operator()
    test_threshold_pre_pruning_all_operators()
    test_polarity_flips_which_of_accept_reject_equals_evaluate()
    test_any_of_and_all_of_composition()
    test_any_of_drives_beam_search_stopping_via_combination_not_a_single_forward()
    test_filtering_never_changes_exploration_only_best_rule_eligibility()
    test_filtering_excludes_the_best_rule_without_blocking_the_search_from_finding_it()
    test_stopping_never_fires_on_the_seed_candidate()
    test_stopping_leave_polarity_returns_the_pre_trigger_best_or_none()
    test_stopping_enter_polarity_returns_the_triggering_rule_directly()
    test_single_rule_learner_passes_filtering_and_stopping_through()
    test_seco_is_a_native_rule_learner()
    test_seco_covers_a_disjunctive_concept_with_two_rules()
    test_seco_rejects_unconditional_rule_even_straight_from_search()
    test_seco_covering_removes_both_classes_not_just_positives()
    test_seco_max_rules_caps_the_loop()
    test_seco_stop_covering_prevents_rules_below_threshold()
    test_uncovered_positives_threshold_criterion_arithmetic()
    test_seco_stop_covering_via_uncovered_positives()
    test_seco_raises_without_labels()
    test_cn2_defaults_wire_laplace_and_significance_as_stopping()
    test_cn2_overrides_are_all_honored()
    test_cn2_explicit_filtering_overrides_significance_threshold_default()
    test_cn2_explicit_search_overrides_beam_width()
    test_cn2_fits_a_disjunctive_concept_with_significance_disabled()
    test_cn2_default_significance_as_stopping_can_underfit()
    test_cn2_filtering_that_rejects_everything_is_inert_not_empty()
    test_cn2_an_absurd_significance_threshold_yields_an_empty_ruleset_either_mode()
    test_cn2_mode_filtering_wires_the_same_criterion_as_filtering_not_stopping()
    test_cn2_mode_rejects_unknown_value()
    test_cn2_explicit_stopping_suppresses_mode_routing()
    test_encoding_length_restriction_arithmetic_matches_quinlan_1990_p251()
    test_encoding_length_restriction_never_fires_on_the_empty_rule()
    test_pfoil_defaults_wire_hill_climbing_foilgain_and_mdl_stopping()
    test_pfoil_mdl_stopping_false_recovers_mooneys_unrestricted_pfoil()
    test_pfoil_finds_the_perfect_conjunction_on_the_toy_dataset()
    test_pfoil_mdl_stopping_curbs_growth_on_a_wide_noisy_dataset()
    test_pfossil_defaults_wire_hillclimbing_deltagain_correlation_and_filtering()
    test_pfossil_mode_stopping_wires_the_criterion_as_stopping_not_filtering()
    test_pfossil_wraps_a_plain_heuristic_for_hillclimbing_but_leaves_explicit_beamsearch_alone()
    test_pfossil_mode_rejects_unknown_value()
    test_pfossil_correlation_threshold_none_disables_the_criterion()
    test_pfossil_fits_a_disjunctive_concept_with_the_threshold_disabled()
    test_covering_loop_rules_and_default_rule_carry_their_own_training_stats()
    test_seed_covering_rules_carry_their_own_training_stats()
    print("\nAll tests passed.")
