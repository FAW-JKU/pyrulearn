"""Inverted heuristics and separate refinement / selection heuristics:
`pyrulearn.heuristics.Inverted`, and `selection=` in the searches and
`SingleRuleLearner` (the heuristic refines, the selection picks the rule
that is returned)."""

import warnings

import numpy as np
import pytest

from _negation_helpers import neg_spec, neg_X
from pyrulearn.data import DataSpec, HorizontalBitmapRepresentation
from pyrulearn.heuristics import (
    LEF, Accuracy, ChiSquare, Correlation, CoveredPositives, Entropy, FBeta, FoilGain, GHeuristic, Inverted, Laplace,
    LengthPenalized, LikelihoodRatio, LinearCost, MEstimate, MinimalLength, Precision, Recall, RuleStats, WRAcc,
    YoudenJ,
)
from pyrulearn.learners.seco import (
    BeamSearch, BranchAndBoundSearch, EmptyRuleAllFeatures, GainAscentHillClimbing, HeadPerRefinement, HillClimbing,
    SeCo, SingleRuleLearner,
)


def _grid(n_pos=12, n_neg=17):
    tp, fp = np.meshgrid(np.arange(n_pos + 1.0), np.arange(n_neg + 1.0))
    tp, fp = tp.ravel(), fp.ravel()
    return RuleStats(tp=tp, fp=fp, fn=n_pos - tp, tn=n_neg - fp, length=np.full(tp.shape, 2.0))


def _quiet(heuristic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Inverted(heuristic)


def _noisy(n=400, d=8, seed=3):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, d)) < 0.5
    y = np.where((raw[:, 0] & raw[:, 1]) | (raw[:, 2] & ~raw[:, 3]), "pos", "neg")
    y = np.where(rng.random(n) < 0.08, np.where(y == "pos", "neg", "pos"), y)
    return HorizontalBitmapRepresentation(neg_spec([f"a{i}" for i in range(d)]), neg_X(raw), y)


def _three_classes(n=500, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & ~raw[:, 1], "a", np.where(raw[:, 2] & raw[:, 3], "b", "c"))
    y = np.where(rng.random(n) < 0.05, rng.permutation(y), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


# ------------------------------------------------------------ Inverted ----

def test_inverted_precision_is_the_share_of_negatives_among_the_excluded_examples():
    # P = N = 10, a rule covering 5 positives and no negative: it excludes 5 positives and 10 negatives
    assert Inverted(Precision()).score(RuleStats(tp=5, fp=0, fn=5, tn=10)) == pytest.approx(10 / 15)


@pytest.mark.parametrize("heuristic", [
    Precision(), Laplace(), MEstimate(m=3), FBeta(), GHeuristic(g=2), Recall(), CoveredPositives(), Entropy(),
    LikelihoodRatio(), LinearCost(0.3), Correlation(), LengthPenalized(Precision(), 0.1),
])
def test_inverting_twice_gives_back_the_heuristic(heuristic):
    stats = _grid()
    twice = Inverted.__new__(Inverted)   # without the constructor's warnings
    twice.base = _quiet(heuristic)
    np.testing.assert_allclose(twice.batch_score(stats), heuristic.batch_score(stats))
    assert twice.score(RuleStats(tp=3, fp=4, fn=9, tn=13)) == pytest.approx(heuristic.score(RuleStats(3, 4, 9, 13)))


def test_inverting_twice_gives_back_a_lexicographic_heuristic():
    lef = LEF(CoveredPositives(), MinimalLength())
    twice = Inverted.__new__(Inverted)
    twice.base = Inverted(lef)
    stats = RuleStats(tp=3, fp=4, fn=9, tn=13, length=2)
    assert twice.score(stats) == lef.score(stats)


@pytest.mark.parametrize("heuristic", [Accuracy(), WRAcc(), Correlation(), ChiSquare(), YoudenJ(), LinearCost(),
                                       MinimalLength()])
def test_a_symmetric_heuristic_warns_and_ranks_as_before(heuristic):
    with pytest.warns(UserWarning, match="changes nothing"):
        inverted = Inverted(heuristic)
    a, b = np.asarray(heuristic.batch_score(_grid())), np.asarray(inverted.batch_score(_grid()))
    assert np.array_equal(np.sign(a[:, None] - a[None, :]).round(9), np.sign(b[:, None] - b[None, :]).round(9))


@pytest.mark.parametrize("heuristic", [Precision(), Laplace(), MEstimate(m=3), LinearCost(0.3), Recall()])
def test_an_asymmetric_heuristic_does_not_warn(heuristic):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Inverted(heuristic)


def test_a_class_neutral_heuristic_inverts_its_two_class_score():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inverted = Inverted(Entropy())
    # the uncovered examples' purity, not the covered ones': a pure rule covering half the positives
    stats = RuleStats(tp=5, fp=0, fn=5, tn=10)
    assert Entropy().score(stats) == 0.0
    assert inverted.score(stats) == pytest.approx(Entropy().score(RuleStats(tp=10, fp=5, fn=0, tn=0)))
    # a rule of a fixed class is learned
    assert _fit(SingleRuleLearner(inverted, selection=Laplace()))


@pytest.mark.parametrize("heuristic", [Entropy(), LengthPenalized(Entropy(), 0.1)])
def test_a_class_neutral_heuristic_has_no_inverted_class_distribution_score(heuristic):
    with pytest.raises(ValueError, match="class distribution"):
        _fit(SingleRuleLearner(Inverted(heuristic)), HeadPerRefinement(), _three_classes())


def test_a_gain_heuristic_cannot_be_inverted():
    with pytest.raises(TypeError, match="gain heuristic"):
        Inverted(FoilGain())


@pytest.mark.parametrize("heuristic", [Precision(), Laplace(), MEstimate(m=3), LengthPenalized(Laplace(), 0.05)])
def test_the_inverted_bound_bounds_every_refinement(heuristic):
    inverted = _quiet(heuristic)
    n_pos, n_neg = 12, 17
    for tp in range(n_pos + 1):
        for fp in range(n_neg + 1):
            bound = inverted.bound(RuleStats(tp=tp, fp=fp, fn=n_pos - tp, tn=n_neg - fp, length=2))
            sub_tp, sub_fp = np.meshgrid(np.arange(tp + 1.0), np.arange(fp + 1.0))
            sub = RuleStats(tp=sub_tp.ravel(), fp=sub_fp.ravel(), fn=n_pos - sub_tp.ravel(),
                            tn=n_neg - sub_fp.ravel(), length=2)
            assert np.all(np.asarray(inverted.batch_score(sub)) <= bound + 1e-12)


def test_a_heuristic_with_its_own_bound_has_none_inverted():
    with pytest.raises(ValueError, match="optimistic_pruning=False"):
        Inverted(LikelihoodRatio()).bound(RuleStats(tp=5, fp=3, fn=5, tn=7))


# ---------------------------------------------------- selection: same h ----

def _fit(srl, target_class="pos", data=None):
    data = data if data is not None else _noisy()
    return str(SeCo(srl, target_class=target_class).fit(data))


@pytest.mark.parametrize("search", [BeamSearch(beam_width=3), BeamSearch(beam_width=1), HillClimbing()])
@pytest.mark.parametrize("heuristic", [Laplace(), MEstimate(m=5), Correlation(), Precision()])
def test_selecting_by_the_heuristic_itself_changes_nothing(search, heuristic):
    plain = _fit(SingleRuleLearner(heuristic, search=search))
    assert _fit(SingleRuleLearner(heuristic, search=search, selection=heuristic)) == plain


@pytest.mark.parametrize("search", [BeamSearch(beam_width=3), HillClimbing()])
def test_selecting_by_the_heuristic_itself_changes_nothing_for_a_decision_list(search):
    data = _three_classes()
    plain = _fit(SingleRuleLearner(Laplace(), search=search), HeadPerRefinement(), data)
    assert _fit(SingleRuleLearner(Laplace(), search=search, selection=Laplace()), HeadPerRefinement(), data) == plain


def test_a_learner_without_selection_has_the_parameters_it_had_before():
    # the demos' cache keys are built from the public attributes
    assert "selection" not in vars(SingleRuleLearner(Laplace()))
    assert SingleRuleLearner(Laplace()).selection is None
    selection = Laplace()
    assert vars(SingleRuleLearner(Inverted(Precision()), selection=selection))["selection"] is selection


# --------------------------------------------------- selection: separate ----

def _search(search, data, heuristic, selection=None, target="pos"):
    initial = EmptyRuleAllFeatures().initial_candidates(data, target, None)
    kwargs = {} if selection is None else {"selection": selection}
    return search.search(data, target, heuristic, initial, **kwargs)


def _score(heuristic, rule, data, target="pos"):
    return heuristic.score(RuleStats.from_rule(rule, data, target))


def test_hill_climbing_returns_the_best_rule_on_its_walk_by_the_selection():
    data = _noisy()
    refine = Inverted(Precision())
    walk = HillClimbing(stop_at_local_optimum=False)
    end = _search(walk, data, refine)
    prefixes = [type(end)(end.conditions[:k], target=end.target, dataspec=end.dataspec, n_features=end.n_features)
                for k in range(1, end.length() + 1)]
    scores = [_score(Laplace(), r, data) for r in prefixes]
    best = prefixes[int(np.argmax(scores))]
    for optimistic in (True, False):
        found = _search(HillClimbing(stop_at_local_optimum=False, optimistic_pruning=optimistic), data, refine,
                        Laplace())
        assert found == best
    assert _score(Laplace(), best, data) >= _score(Laplace(), end, data)


@pytest.mark.parametrize("beam_width", [1, 3, 5])
def test_a_beam_selects_at_least_as_good_a_rule_by_the_selection(beam_width):
    # without optimistic pruning both searches explore the same rules: the beam follows the refinement
    data = _noisy()
    search = BeamSearch(beam_width=beam_width, optimistic_pruning=False)
    refine = Inverted(Precision())
    plain = _search(search, data, refine)
    selected = _search(search, data, refine, Laplace())
    assert _score(Laplace(), selected, data) >= _score(Laplace(), plain, data)
    # and optimistic pruning by the selection's bound doesn't change the result
    assert _search(BeamSearch(beam_width=beam_width), data, refine, Laplace()) == selected


def test_gain_ascent_with_a_selection():
    data = _noisy()
    rule = _search(GainAscentHillClimbing(), data, FoilGain(), Laplace())
    assert rule is not None and rule.length() > 0


def test_a_learner_refines_by_one_heuristic_and_selects_by_another():
    data = _noisy()
    model = SeCo(SingleRuleLearner(Inverted(Precision()), search=BeamSearch(beam_width=3), selection=Laplace()),
                 target_class="pos").fit(data)
    assert len(model.rules) > 0
    decision_list = SeCo(SingleRuleLearner(Inverted(Precision()), selection=Laplace()),
                         target_class=HeadPerRefinement()).fit(_three_classes())
    assert {r.target for r in decision_list.rules} <= {"a", "b", "c"}


def test_an_exhaustive_search_has_no_separate_selection():
    with pytest.raises(ValueError, match="exhaustively"):
        _search(BranchAndBoundSearch(max_conditions=2), _noisy(), Precision(), Laplace())


def test_a_selection_cannot_be_a_gain_heuristic():
    with pytest.raises(ValueError, match="gain heuristic"):
        _search(BeamSearch(), _noisy(), Laplace(), FoilGain())


def test_branching_has_no_selection():
    data = _noisy()
    initial = EmptyRuleAllFeatures().initial_candidates(data, "pos", None)
    with pytest.raises(ValueError, match="no selection"):
        GainAscentHillClimbing(branch_similarity=0.9).search_all(data, "pos", FoilGain(), initial,
                                                                 selection=Laplace())
