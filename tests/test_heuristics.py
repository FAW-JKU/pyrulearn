import itertools
import math
import warnings

import numpy as np
import pytest
from pyrulearn import heuristics as H
from pyrulearn.data import DataSpec
from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.rule import Rule
from pyrulearn.heuristics import (
    Accuracy,
    ChiSquare,
    Correlation,
    Coverage,
    CoverageDifference,
    Entropy,
    FBeta,
    DeltaGain,
    FoilGain,
    GainHeuristic,
    GeneralizedMEstimate,
    GHeuristic,
    Laplace,
    LEF,
    LengthPenalized,
    MinimalLength,
    LikelihoodRatio,
    LinearCost,
    LinearCostRates,
    MEstimate,
    CoveredNegatives,
    CoveredPositives,
    UncoveredNegatives,
    UncoveredPositives,
    Precision,
    Recall,
    RuleHeuristic,
    RuleStats,
    Support,
    WRAcc,
    YoudenJ,
)


# -- heuristics --------------------------------------------------------

def test_precision_and_laplace_and_support_and_accuracy():
    stats = RuleStats(tp=3, fp=0, fn=3, tn=4)  # pure, small coverage; n_pos=6, n_neg=4
    assert stats.n_pos == 6 and stats.n_neg == 4
    assert Precision().score(stats) == 1.0
    assert abs(Laplace().score(stats) - 4 / 5) < 1e-12
    assert abs(Support().score(stats) - 0.3) < 1e-12
    assert abs(Accuracy().score(stats) - 0.7) < 1e-12

    stats2 = RuleStats(tp=4, fp=3, fn=2, tn=1)  # less pure, bigger coverage; n_pos=6, n_neg=4
    assert stats2.n_pos == 6 and stats2.n_neg == 4
    assert abs(Precision().score(stats2) - 4 / 7) < 1e-12
    assert abs(Laplace().score(stats2) - 5 / 9) < 1e-12
    assert abs(Support().score(stats2) - 0.7) < 1e-12
    assert abs(Accuracy().score(stats2) - 0.5) < 1e-12
    stats = RuleStats(tp=0, fp=0, fn=6, tn=4)
    assert Precision().score(stats) == 0.0


def test_tpminusfp_ranks_identically_to_accuracy():
    stats_a = RuleStats(tp=3, fp=0, fn=3, tn=4)
    stats_b = RuleStats(tp=4, fp=3, fn=2, tn=1)
    # same dataset (n_pos/n_neg equal for both) -> CoverageDifference and
    # Accuracy must agree on which one ranks higher
    assert (Accuracy().score(stats_a) > Accuracy().score(stats_b)) == \
           (CoverageDifference().score(stats_a) > CoverageDifference().score(stats_b))
    assert CoverageDifference().score(stats_a) == 3  # tp - fp = 3 - 0
    assert CoverageDifference().score(stats_b) == 1  # tp - fp = 4 - 3
    # exactly LinearCost(cost_ratio=1.0)
    assert CoverageDifference().score(stats_a) == LinearCost(cost_ratio=1.0).score(stats_a)


def test_mestimate_generalizes_precision_and_uses_prior():
    stats = RuleStats(tp=3, fp=0, fn=3, tn=4)  # prior p0 = 0.6
    assert MEstimate(0).score(stats) == Precision().score(stats)
    assert abs(MEstimate(2).score(stats) - 4.2 / 5) < 1e-12
    # as m grows, the score is pulled toward the prior p0 (0.6)
    assert abs(MEstimate(1_000_000).score(stats) - 0.6) < 1e-3


def test_generalized_mestimate_matches_mestimate_at_matching_cost():
    stats = RuleStats(tp=3, fp=0, fn=3, tn=4)  # prior p0 = 0.6
    p0 = stats.n_pos / (stats.n_pos + stats.n_neg)
    assert GeneralizedMEstimate(2, cost=p0).score(stats) == MEstimate(2).score(stats)
    # a cost far from the prior gives a different score than MEstimate
    assert GeneralizedMEstimate(2, cost=0.9).score(stats) != MEstimate(2).score(stats)


def test_generalized_mestimate_isometrics_are_a_pencil():
    # pivot at (-m*(1-cost), -m*cost) -- verify two different points on
    # the same line through that pivot score identically
    m, cost = 5.0, 0.3
    pivot_fp, pivot_tp = -m * (1 - cost), -m * cost

    def h(fp, tp):
        return GeneralizedMEstimate(m, cost).score(RuleStats(tp=tp, fp=fp, fn=0, tn=0))

    for slope in (0.3, 1.0, 2.7):
        fp1, fp2 = 10.0, 60.0
        tp1 = pivot_tp + slope * (fp1 - pivot_fp)
        tp2 = pivot_tp + slope * (fp2 - pivot_fp)
        assert abs(h(fp1, tp1) - h(fp2, tp2)) < 1e-9


def test_wracc_score():
    stats = RuleStats(tp=3, fp=0, fn=3, tn=4)
    # covered=3, p0=0.6 -> (3/10) * (3/3 - 0.6) = 0.3 * 0.4 = 0.12
    assert abs(WRAcc().score(stats) - 0.12) < 1e-12

    stats2 = RuleStats(tp=4, fp=3, fn=2, tn=1)
    # covered=7, p0=0.6 -> (7/10) * (4/7 - 0.6)
    expected = (7 / 10) * (4 / 7 - 0.6)
    assert abs(WRAcc().score(stats2) - expected) < 1e-12
    assert WRAcc().score(RuleStats(tp=0, fp=0, fn=6, tn=4)) == 0.0
    assert WRAcc().score(RuleStats(tp=0, fp=0, fn=0, tn=0)) == 0.0


def test_linear_cost():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)
    assert LinearCost().score(stats) == 1.0        # default cost_ratio=1: 4 - 1*3
    assert LinearCost(cost_ratio=2.0).score(stats) == -2.0  # 4 - 2*3


def test_youden_j_and_linear_cost_rates():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)  # n_pos=6, n_neg=4 -> tpr=2/3, fpr=3/4
    assert abs(YoudenJ().score(stats) - (-0.08333333333333337)) < 1e-9
    # YoudenJ is exactly LinearCostRates(cost_ratio=1.0)
    assert YoudenJ().score(stats) == LinearCostRates(cost_ratio=1.0).score(stats)
    assert abs(LinearCostRates(cost_ratio=2.0).score(stats) - (-0.8333333333333334)) < 1e-9
    assert YoudenJ().score(RuleStats(tp=0, fp=0, fn=0, tn=6)) == 0.0  # n_pos=0
    assert YoudenJ().score(RuleStats(tp=0, fp=0, fn=6, tn=0)) == 0.0  # n_neg=0
    assert LinearCostRates().score(RuleStats(tp=0, fp=0, fn=0, tn=6)) == 0.0


def test_covered_positives_and_negatives_heuristics():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)
    assert CoveredPositives().score(stats) == 4
    assert CoveredNegatives().score(stats) == -3
    # ignore the other side entirely: changing fp doesn't move
    # CoveredPositives, and vice versa
    assert CoveredPositives().score(RuleStats(tp=4, fp=100, fn=2, tn=1)) == 4
    assert CoveredNegatives().score(RuleStats(tp=100, fp=3, fn=2, tn=1)) == -3


def test_uncovered_positives_and_negatives_heuristics():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)
    assert UncoveredPositives().score(stats) == -2
    assert UncoveredNegatives().score(stats) == 1
    # ignore the other side entirely: changing tp doesn't move
    # UncoveredPositives, and vice versa
    assert UncoveredPositives().score(RuleStats(tp=100, fp=3, fn=2, tn=1)) == -2
    assert UncoveredNegatives().score(RuleStats(tp=4, fp=100, fn=2, tn=1)) == 1


def test_coverage_heuristic():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)  # n_pos=6, n_neg=4
    assert Coverage().score(stats) == 7
    # unnormalized Support, same relationship CoverageDifference has to Accuracy
    assert abs(Coverage().score(stats) - Support().score(stats) * (stats.n_pos + stats.n_neg)) < 1e-12


def test_g_heuristic_values_and_zero_guard():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)
    assert abs(GHeuristic(0).score(stats) - 4 / 3) < 1e-12
    assert GHeuristic(1).score(stats) == 1.0
    assert GHeuristic(5).score(stats) == 0.5
    # zero coverage and zero g together -> guarded, not a ZeroDivisionError
    assert GHeuristic(0).score(RuleStats(tp=0, fp=0, fn=6, tn=4)) == 0.0


def test_recall_heuristic():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)  # n_pos=6
    assert abs(Recall().score(stats) - 4 / 6) < 1e-12
    assert Recall().score(RuleStats(tp=0, fp=0, fn=0, tn=0)) == 0.0


def test_fbeta_heuristic():
    stats = RuleStats(tp=4, fp=3, fn=2, tn=1)  # precision=4/7, recall=4/6
    p, r = Precision().score(stats), Recall().score(stats)
    expected_f1 = 2 * p * r / (p + r)
    assert abs(FBeta().score(stats) - expected_f1) < 1e-12
    # beta -> favors recall more as beta grows
    assert FBeta(beta=0.0).score(stats) == p  # beta=0 collapses to precision
    assert FBeta(beta=1.0).score(stats) == expected_f1
    assert FBeta().score(RuleStats(tp=0, fp=0, fn=0, tn=0)) == 0.0


def test_fbeta_isometrics_are_a_pencil_through_fp_axis():
    # despite the P/R harmonic-mean formula not looking linear, FBeta's
    # isometrics are a straight-line pencil pivoting at
    # (-beta**2*n_pos, 0), the same fp-axis sub-family GHeuristic
    # belongs to -- verify two different points on the same line
    # through that pivot score identically.
    n_pos, beta = 100, 1.5
    pivot_fp = -(beta ** 2) * n_pos

    def h(fp, tp):
        return FBeta(beta).score(RuleStats(tp=tp, fp=fp, fn=n_pos - tp, tn=0))

    for slope in (0.3, 1.0, 2.7):
        fp1, fp2 = 10.0, 60.0
        tp1 = slope * (fp1 - pivot_fp)
        tp2 = slope * (fp2 - pivot_fp)
        assert abs(h(fp1, tp1) - h(fp2, tp2)) < 1e-9


def test_length_penalized_reads_length():
    stats = RuleStats(tp=3, fp=0, fn=3, tn=4, length=1)
    penalized = LengthPenalized(Precision(), penalty=0.05)
    assert abs(penalized.score(stats) - 0.95) < 1e-12
    # a heuristic that ignores length is unaffected by it
    assert Precision().score(stats) == Precision().score(RuleStats(tp=3, fp=0, fn=3, tn=4, length=99))


def test_correlation_extremes_and_uncorrelated():
    # perfect predictor: covers exactly the positives, nothing else
    perfect = RuleStats(tp=6, fp=0, fn=0, tn=4)
    assert abs(Correlation().score(perfect) - 1.0) < 1e-12

    # perfect anti-predictor: covers exactly the negatives, misses every positive
    anti = RuleStats(tp=0, fp=4, fn=6, tn=0)
    assert abs(Correlation().score(anti) - (-1.0)) < 1e-12

    # coverage proportional to the prior (3:2 pos:neg, same as covered) -> uncorrelated
    uncorrelated = RuleStats(tp=3, fp=2, fn=3, tn=2)
    assert abs(Correlation().score(uncorrelated) - 0.0) < 1e-12
    # never covers anything -> (tp+fp) term is 0 -> denominator 0
    assert Correlation().score(RuleStats(tp=0, fp=0, fn=6, tn=4)) == 0.0
    # covers everything -> (fp+tn) and (fn+tn) terms are 0 -> denominator 0
    assert Correlation().score(RuleStats(tp=6, fp=4, fn=0, tn=0)) == 0.0


def test_chi_square_matches_hand_computed_value_and_correlation_relation():
    # n=20; hand-verified: chi2 (uncorrected) == n * Correlation.score(stats)**2,
    # and matches a direct chi-square computation independently.
    stats = RuleStats(tp=6, fp=1, fn=4, tn=9)
    n = stats.tp + stats.fp + stats.fn + stats.tn
    uncorrected = ChiSquare(yates_correction=False).score(stats)
    assert abs(uncorrected - 5.4945054945054945) < 1e-9
    assert abs(uncorrected - n * Correlation().score(stats) ** 2) < 1e-9

    yates = ChiSquare(yates_correction=True).score(stats)
    assert abs(yates - 3.5164835164835164) < 1e-9
    assert yates < uncorrected  # Yates' correction is strictly more conservative here
    assert ChiSquare().score(RuleStats(tp=0, fp=0, fn=6, tn=4)) == 0.0
    assert ChiSquare().score(RuleStats(tp=6, fp=4, fn=0, tn=0)) == 0.0


def test_chi_square_yates_correction_floors_at_zero():
    # tp=fp=fn=tn=1 (n=4): diff = |tp*tn - fp*fn| = 0 -- no association at
    # all, chi-square must be exactly 0. Without max(0, diff - n/2)'s
    # floor, the corrected difference goes negative (0 - 2 = -2) and
    # squaring it would wrongly *manufacture* a nonzero, even larger
    # score (4) than the uncorrected one (0) -- the opposite of what a
    # continuity correction is supposed to do.
    stats = RuleStats(tp=1, fp=1, fn=1, tn=1)
    assert ChiSquare(yates_correction=False).score(stats) == 0.0
    assert ChiSquare(yates_correction=True).score(stats) == 0.0


def test_entropy_pure_and_even_split():
    # pure rule (only positives, or only negatives, among covered) -> best score, 0
    assert Entropy().score(RuleStats(tp=5, fp=0, fn=1, tn=4)) == 0.0
    assert Entropy().score(RuleStats(tp=0, fp=5, fn=6, tn=0)) == 0.0
    # even 50/50 split among covered -> worst score, -1 bit
    assert abs(Entropy().score(RuleStats(tp=5, fp=5, fn=0, tn=0)) - (-1.0)) < 1e-12
    # intermediate case, hand-verified: p=3/4 -> -H(3/4) ~= -0.8112781
    assert abs(Entropy().score(RuleStats(tp=3, fp=1, fn=0, tn=0)) - (-0.8112781244591328)) < 1e-9
    assert Entropy().score(RuleStats(tp=0, fp=0, fn=6, tn=4)) == -1.0


def test_likelihood_ratio_matches_hand_computed_value():
    # same stats as test_foil_gain's child rule: tp=6, fp=1, fn=4, tn=9
    # (n_pos=10, n_neg=10, p0=0.5, e_tp=e_fp=3.5) -- value cross-checked
    # against a direct G-statistic computation
    stats = RuleStats(tp=6, fp=1, fn=4, tn=9)
    assert abs(LikelihoodRatio().score(stats) - 3.9624320718015067) < 1e-9
    # zero coverage -> 0.0
    assert LikelihoodRatio().score(RuleStats(tp=0, fp=0, fn=6, tn=4)) == 0.0
    # degenerate prior (no negatives at all in the dataset) -> 0.0
    assert LikelihoodRatio().score(RuleStats(tp=6, fp=0, fn=4, tn=0)) == 0.0
    # a rule matching the prior exactly contributes no evidence -> ~0.0
    stats = RuleStats(tp=5, fp=5, fn=5, tn=5)  # p0=0.5, covered 5/5, exactly as expected
    assert abs(LikelihoodRatio().score(stats)) < 1e-9


def test_rule_heuristic_is_abstract():
    try:
        RuleHeuristic()
        assert False, "expected TypeError (can't instantiate an ABC)"
    except TypeError:
        pass


# -- RuleStats ---------------------------------------------------------

def _two_feature_data():
    X = np.array([
        [1, 1], [1, 1], [1, 0], [0, 1], [0, 1], [0, 0],  # positives (rows 0-5)
        [0, 1], [0, 1], [0, 1], [0, 0],                   # negatives (rows 6-9)
    ], dtype=bool)
    y = np.array(["pos"] * 6 + ["neg"] * 4)
    ds = DataSpec(["a", "b"])
    return HorizontalBitmapRepresentation(ds, X, y), ds


def test_rule_stats_universal_and_empty():
    universal = RuleStats.universal(n_pos=6, n_neg=4)
    assert (universal.tp, universal.fp, universal.fn, universal.tn) == (6, 4, 0, 0)
    assert universal.n_pos == 6 and universal.n_neg == 4
    assert Precision().score(universal) == 6 / 10  # covers everyone -> precision = prior

    empty = RuleStats.empty(n_pos=6, n_neg=4)
    assert (empty.tp, empty.fp, empty.fn, empty.tn) == (0, 0, 6, 4)
    assert empty.n_pos == 6 and empty.n_neg == 4

    # the universal rule is the natural FoilGain baseline: gain of the
    # universal rule relative to itself is exactly zero
    child = RuleStats(tp=6, fp=4, fn=0, tn=0)
    assert FoilGain().score(child, universal) == 0.0


def test_rule_stats_from_rule_and_score_rule():
    rep, ds = _two_feature_data()

    rule_a = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)  # a=True: rows 0,1,2 -> all pos
    stats = RuleStats.from_rule(rule_a, rep)
    assert (stats.tp, stats.fp, stats.fn, stats.tn, stats.length) == (3, 0, 3, 4, 1)
    assert Precision().score_rule(rule_a, rep) == 1.0

    rule_b = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)  # b=True: rows 0,1,3,4,6,7,8
    stats_b = RuleStats.from_rule(rule_b, rep)
    assert (stats_b.tp, stats_b.fp) == (4, 3)


def test_rule_stats_from_rule_with_example_mask():
    rep, ds = _two_feature_data()

    rule_a = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)  # a=True: rows 0,1,2
    full_stats = RuleStats.from_rule(rule_a, rep)
    assert (full_stats.tp, full_stats.fp, full_stats.fn, full_stats.tn) == (3, 0, 3, 4)

    # exclude row 2 (a covered positive -> drops tp) and row 6 (an uncovered
    # negative -> drops tn); everything else stays exactly as in full_stats
    mask = np.ones(10, dtype=bool)
    mask[[2, 6]] = False
    masked_stats = RuleStats.from_rule(rule_a, rep, example_mask=mask)
    assert (masked_stats.tp, masked_stats.fp, masked_stats.fn, masked_stats.tn) == (2, 0, 3, 3)
    assert masked_stats.tp + masked_stats.fp + masked_stats.fn + masked_stats.tn == mask.sum()


def test_from_rule_raises_without_labels_or_target():
    ds_no_y = DataSpec(["a", "b"])
    rep_no_y = HorizontalBitmapRepresentation(ds_no_y, np.array([[1, 0], [0, 1]], dtype=bool))  # no y
    r = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds_no_y)
    try:
        RuleStats.from_rule(r, rep_no_y)
        assert False, "expected ValueError"
    except ValueError:
        pass

    X = np.array([[1, 0], [0, 1]], dtype=bool)
    y = np.array(["pos", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    r_no_target = Rule.from_pos_neg(pos=[0], dataspec=ds)  # no target
    try:
        RuleStats.from_rule(r_no_target, rep)
        assert False, "expected ValueError"
    except ValueError:
        pass


# -- LEF and MinimalLength ---------------------------------------------

def test_minimal_length_scores_negated_length_and_breaks_lef_ties_toward_shorter_rules():
    assert MinimalLength().score(RuleStats(tp=5, fp=0, fn=0, tn=0, length=1)) == -1
    assert MinimalLength().score(RuleStats(tp=5, fp=0, fn=0, tn=0, length=4)) == -4

    # AQ's default-LEF shape: same positives and negatives, differ only in size
    short = RuleStats(tp=5, fp=1, fn=0, tn=9, length=2)
    long = RuleStats(tp=5, fp=1, fn=0, tn=9, length=5)
    lef = LEF(CoveredPositives(), CoveredNegatives(), MinimalLength())
    assert lef.score(short) > lef.score(long)  # tie on tp and fp -> shorter rule wins


def test_lef_score_returns_tuple_of_constituent_scores():
    stats = RuleStats(tp=3, fp=1, fn=1, tn=3)
    lef = LEF(Accuracy(), Precision())
    assert lef.score(stats) == (Accuracy().score(stats), Precision().score(stats))


def test_lef_lexicographic_tie_breaking():
    # a: tp=3,fp=1,fn=1,tn=3 -> accuracy 0.75, precision 0.75
    # b: tp=2,fp=0,fn=2,tn=4 -> accuracy 0.75 (tied with a), precision 1.0 (higher)
    # c: tp=4,fp=0,fn=0,tn=0 -> accuracy 1.0 (beats both outright on the primary)
    a = RuleStats(tp=3, fp=1, fn=1, tn=3)
    b = RuleStats(tp=2, fp=0, fn=2, tn=4)
    c = RuleStats(tp=4, fp=0, fn=0, tn=0)
    lef = LEF(Accuracy(), Precision())

    assert Accuracy().score(a) == Accuracy().score(b) == 0.75  # tied on the primary heuristic
    assert Precision().score(a) == 0.75 and Precision().score(b) == 1.0  # differ on the secondary

    # tied primary -> secondary (precision) breaks the tie: b ranks above a
    assert lef.score(b) > lef.score(a)
    # a differing primary always wins outright, regardless of the secondary
    assert lef.score(c) > lef.score(b) > lef.score(a)


def test_lef_works_with_beam_search():
    from pyrulearn.learners.seco import BeamSearch

    X = np.array([
        [1, 1], [1, 1], [1, 0], [0, 0],
        [0, 1], [0, 1], [0, 0], [0, 0],
    ], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg", "neg", "neg", "neg", "neg"])
    ds = DataSpec(["a", "b"])
    rep = HorizontalBitmapRepresentation(ds, X, y)
    empty_rule = Rule([], target="pos", dataspec=ds)
    initial = [(empty_rule, frozenset({0, 1}))]

    # BeamSearch only ever compares scores via > and sorted(key=...), both
    # of which already work correctly on LEF's tuple-valued score -- no
    # special-casing needed in BeamSearch itself for this to work
    best = BeamSearch(beam_width=3).search(rep, "pos", LEF(Accuracy(), Precision()), initial)
    assert isinstance(best, Rule)


# -- gain heuristics ---------------------------------------------------

def test_foil_gain():
    parent = RuleStats(tp=10, fp=10, fn=0, tn=0)     # precision 0.5
    child = RuleStats(tp=6, fp=1, fn=4, tn=9)  # precision 6/7
    expected = 6 * (math.log2(6 / 7) - math.log2(10 / 20))
    assert abs(FoilGain().score(child, parent) - expected) < 1e-6
    parent = RuleStats(tp=10, fp=10, fn=0, tn=0)
    assert FoilGain().score(RuleStats(tp=0, fp=5, fn=10, tn=5), parent) == 0.0
    child = RuleStats(tp=6, fp=1, fn=4, tn=9)
    assert FoilGain().score(child, RuleStats(tp=0, fp=0, fn=10, tn=10)) == 0.0


def test_foil_gain_raises_without_parent():
    stats = RuleStats(tp=6, fp=1, fn=4, tn=9)
    try:
        FoilGain().score(stats)
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_foil_gain_is_a_gain_heuristic():
    assert isinstance(FoilGain(), GainHeuristic)
    assert isinstance(FoilGain(), RuleHeuristic)
    assert not isinstance(Laplace(), GainHeuristic)


def test_gain_heuristic_score_rule_needs_parent_rule():
    rep, ds = _two_feature_data()

    parent_rule = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)  # b=True
    child_rule = Rule.from_pos_neg(pos=[0, 1], target="pos", dataspec=ds)  # a=True AND b=True

    stats = RuleStats.from_rule(child_rule, rep)
    parent_stats = RuleStats.from_rule(parent_rule, rep)
    assert (parent_stats.tp, parent_stats.fp) == (4, 3)  # b=True
    assert (stats.tp, stats.fp) == (2, 0)                # a=True AND b=True: rows 0,1 -> both pos
    gain = FoilGain().score_rule(child_rule, rep, parent_rule=parent_rule)
    assert gain == FoilGain().score(stats, parent_stats)
    assert gain > 0  # the refinement increased precision (0.571 -> 1.0)

    try:
        FoilGain().score_rule(child_rule, rep)
        assert False, "expected ValueError without parent_rule"
    except ValueError:
        pass


def test_delta_gain_is_the_wrapped_heuristics_own_difference():
    stats = RuleStats(tp=6, fp=1, fn=4, tn=9)
    parent_stats = RuleStats(tp=10, fp=10, fn=0, tn=0)
    dg = DeltaGain(Laplace())
    expected = Laplace().score(stats) - Laplace().score(parent_stats)
    assert dg.score(stats, parent_stats) == expected
    assert isinstance(dg, GainHeuristic)
    stats = RuleStats(tp=4, fp=2, fn=6, tn=8)
    assert DeltaGain(Accuracy()).score(stats, stats) == 0.0


# -- batch_score vs. score ---------------------------------------------

def _grid():
    """Every combination of small counts -- zeros included, so each
    heuristic's zero-denominator guards are hit -- plus weighted (float)
    counts, as flat arrays, and the same points as scalar RuleStats."""
    values = [0, 1, 2, 5, 13, 0.5, 2.25]
    rows = list(itertools.product(values, values, [0, 1, 7, 0.75], [0, 1, 7, 0.75], [0, 3]))
    cols = [np.array(c) for c in zip(*rows)]
    batch = H.RuleStats(tp=cols[0], fp=cols[1], fn=cols[2], tn=cols[3], length=cols[4].astype(int))
    single = [H.RuleStats(tp=r[0], fp=r[1], fn=r[2], tn=r[3], length=r[4]) for r in rows]
    return batch, single


#: arithmetic only (division, products of floats below 2**53): bit-identical
EXACT = [
    H.Precision(), H.Recall(), H.FBeta(), H.FBeta(beta=0.5), H.CoveredPositives(),
    H.CoveredNegatives(), H.UncoveredPositives(), H.UncoveredNegatives(), H.Laplace(),
    H.MEstimate(m=2.0), H.GeneralizedMEstimate(m=2.0, cost=0.3), H.GHeuristic(g=1.0),
    H.WRAcc(), H.YoudenJ(), H.LinearCostRates(cost_ratio=2.0), H.Accuracy(),
    H.CoverageDifference(), H.SlipperZ(), H.Support(), H.Coverage(), H.LinearCost(cost_ratio=0.5),
    H.LengthPenalized(H.Laplace(), penalty=0.1), H.MinimalLength(),
]
#: logs (numpy's may round differently from math's), or products the scalar
#: version computes in exact Python ints: equal up to rounding
CLOSE = [H.Correlation(), H.ChiSquare(), H.ChiSquare(yates_correction=False), H.Entropy(), H.LikelihoodRatio()]


def _no_warnings(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn()


@pytest.mark.parametrize("h", EXACT + CLOSE, ids=lambda h: repr(h))
def test_batch_score_agrees_with_score(h):
    batch, single = _grid()
    got = _no_warnings(lambda: h.batch_score(batch))
    want = np.array([h.score(s) for s in single], dtype=float)
    assert got.shape == want.shape
    if any(h is e for e in EXACT):
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("h", [H.FoilGain(), H.DeltaGain(H.Laplace()), H.DeltaGain(H.Entropy())],
                         ids=lambda h: repr(h))
def test_gain_batch_score_agrees_with_score(h):
    batch, single = _grid()
    for parent in [H.RuleStats(tp=13, fp=5, fn=0, tn=7), H.RuleStats(tp=0, fp=5, fn=2, tn=1),
                   H.RuleStats(tp=2.25, fp=0.5, fn=1, tn=0.75)]:
        got = _no_warnings(lambda: h.batch_score(batch, parent))
        want = np.array([h.score(s, parent) for s in single], dtype=float)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_lef_batch_score_is_one_array_per_criterion():
    batch, single = _grid()
    lef = H.LEF(H.Laplace(), H.MinimalLength())
    got = lef.batch_score(batch)
    assert isinstance(got, tuple) and len(got) == 2
    for k, column in enumerate(got):
        np.testing.assert_array_equal(column, [lef.score(s)[k] for s in single])


def test_default_batch_score_loops_over_a_custom_score():
    class Custom(H.RuleHeuristic):          # plain Python branches: not array-safe
        def score(self, stats):
            return stats.tp - 2 * stats.fp if stats.tp > stats.fp else -1.0

    class CustomPair(H.RuleHeuristic):      # a LEF-shaped tuple score
        def score(self, stats):
            return (stats.tp, -stats.length)

    batch, single = _grid()
    np.testing.assert_array_equal(Custom().batch_score(batch), [Custom().score(s) for s in single])
    pairs = CustomPair().batch_score(batch)
    assert pairs.shape == (len(single), 2)
    np.testing.assert_array_equal(pairs, [CustomPair().score(s) for s in single])


def test_batch_score_of_one_scalar_stats_is_zero_dimensional():
    s = H.RuleStats(tp=3, fp=1, fn=2, tn=4, length=2)
    for h in [H.Laplace(), H.Entropy(), H.MinimalLength()]:
        v = h.batch_score(s)
        assert np.ndim(v) == 0 and float(v) == pytest.approx(h.score(s), rel=1e-12)


def test_correlation_and_chi_square_take_a_rounding_error_below_zero_as_zero():
    # fractional row weights: with no positives at all, fn = P - tp can come out a hair below 0,
    # and with it the factor tp + fn under the root
    stats = RuleStats(tp=0.0, fp=2.5, fn=-1e-15, tn=3.0)
    for h in (Correlation(), ChiSquare()):
        assert h.score(stats) == 0.0
    batch = RuleStats(tp=np.array([0.0, 2.0]), fp=np.array([2.5, 1.0]),
                      fn=np.array([-1e-15, 1.0]), tn=np.array([3.0, 2.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = Correlation().batch_score(batch)
    assert scores[0] == 0.0 and scores[1] == pytest.approx(Correlation().score(RuleStats(tp=2.0, fp=1.0, fn=1.0, tn=2.0)))


# -- bounds on the refinements -----------------------------------------

def test_default_bound_is_the_tp_0_projection():
    stats = RuleStats(tp=4, fp=3, fn=6, tn=7, length=2)
    assert H.Laplace().bound(stats) == H.Laplace().score(RuleStats(tp=4, fp=0, fn=6, tn=10, length=2))
    penalized = H.LengthPenalized(LikelihoodRatio(), 0.5)
    assert abs(penalized.bound(stats) - (LikelihoodRatio().bound(stats) - 1.0)) < 1e-12


def test_likelihood_ratio_bound_covers_every_refinement():
    # the statistic isn't monotone in tp and fp, so its bound is the
    # maximum over the corners of the refinements' box; check it against
    # every refinement (1 <= tp' <= tp, 0 <= fp' <= fp) by brute force
    lr = LikelihoodRatio()
    for n_pos, n_neg in [(10, 30), (30, 10), (20, 20)]:
        for tp in range(1, n_pos + 1):
            for fp in range(n_neg + 1):
                stats = RuleStats(tp=tp, fp=fp, fn=n_pos - tp, tn=n_neg - fp)
                bound = lr.bound(stats)
                best = max(lr.score(RuleStats(tp=t, fp=f, fn=n_pos - t, tn=n_neg - f))
                           for t in range(1, tp + 1) for f in range(fp + 1))
                assert best <= bound + 1e-9
                assert abs(best - bound) < 1e-9   # a corner is itself a refinement
    batch = RuleStats(tp=np.array([1, 5]), fp=np.array([9, 0]), fn=np.array([9, 5]), tn=np.array([21, 30]))
    assert np.allclose(lr.batch_bound(batch), [lr.bound(H.RuleStats(tp=1, fp=9, fn=9, tn=21)),
                                               lr.bound(H.RuleStats(tp=5, fp=0, fn=5, tn=30))])
