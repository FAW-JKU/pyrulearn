"""Choosing a rule's head during the search: `RuleHeuristic.best_head`
and its mixins, `HeadObjective`, and the decision lists built on them
(`model=DecisionList` for a learner that doesn't seed its search)."""

import numpy as np
import pytest

from _negation_helpers import neg_spec, neg_X
from pyrulearn.data import (
    HorizontalBitmapRepresentation, NListRepresentation, SparseDataRepresentation, VerticalBitmapRepresentation,
)
from pyrulearn.heuristics import (
    LEF, CoveredPositives, Correlation, Entropy, FoilGain, GHeuristic, Laplace, LengthPenalized, MEstimate,
    MinimalLength, Precision, RuleHeuristic, RuleStats, WRAcc,
)
from pyrulearn.learners.seco import (
    AQR, CN2, HeadPerRefinement, HeadPerRule, EmptyRuleAllFeatures, FirstSeed, FixedClass, PFossil, Pypper, RandomSeed,
    HeadPerSeedExample, SeedExample,
)
from pyrulearn.models import ConceptCascade, ConceptModel, ConceptSet, DecisionList, FlatRuleSet


def _three_classes(n=600, seed=0, weights=None):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 5)) < 0.5
    y = np.where(raw[:, 0], "x", np.where(raw[:, 1], "y", "z"))
    y = np.where(rng.random(n) < 0.03, rng.permutation(y), y)
    spec = neg_spec([f"f{i}" for i in range(5)])
    return HorizontalBitmapRepresentation(spec, neg_X(raw), y, weights=weights), y


# ------------------------------------------------------------ best_head ----

def test_the_majority_is_not_always_the_best_head():
    # 1,000 examples, 800 of A and 200 of B; a rule covering 60 A and 40 B
    covered, totals = np.array([60.0, 40.0]), np.array([800.0, 200.0])
    assert Precision().best_head(covered, totals)[1] == 0          # the majority
    assert WRAcc().best_head(covered, totals)[1] == 1              # B: twice its prior share
    assert Correlation().best_head(covered, totals)[1] == 1
    # the m-estimate's virtual examples pull towards the frequent class
    assert MEstimate(m=20).best_head(np.array([45.0, 55.0]), np.array([900.0, 100.0]))[1] == 0


def test_the_default_scores_each_class_against_the_rest():
    covered, totals = np.array([60.0, 30.0, 10.0]), np.array([500.0, 300.0, 200.0])
    scores = WRAcc().head_scores(covered, totals)
    for c in range(3):
        tp, fp = covered[c], covered.sum() - covered[c]
        expected = WRAcc().score(RuleStats(tp=tp, fp=fp, fn=totals[c] - tp, tn=1000 - totals[c] - fp))
        assert scores[c] == pytest.approx(expected)
    score, head = WRAcc().best_head(covered, totals)
    assert head == int(np.argmax(scores)) and score == pytest.approx(scores.max())


@pytest.mark.parametrize("heuristic", [Precision(), Laplace(n_classes=4), GHeuristic(g=2.0), CoveredPositives()])
def test_majority_head_is_only_a_speedup(heuristic):
    rng = np.random.default_rng(1)
    covered = rng.integers(0, 40, size=(4, 300)).astype(float)
    totals = covered.sum(axis=1) + rng.integers(0, 100, size=4)
    fast, slow = heuristic.best_head(covered, totals), RuleHeuristic.best_head(heuristic, covered, totals)
    assert np.allclose(fast[0], slow[0]) and (fast[1] == slow[1]).all()


def test_a_head_must_cover_something_and_ties_go_to_more_covered():
    _, head = RuleHeuristic.best_head(Precision(), np.array([0.0, 1.0, 0.0]), np.array([10.0, 10.0, 10.0]))
    assert head == 1                                   # the only class covered
    _, head = RuleHeuristic.best_head(Precision(), np.array([[2.0], [2.0]]), np.array([10.0, 10.0]))
    assert head[0] == 0                                # a full tie: the first class
    score, _ = Precision().best_head(np.zeros(3), np.array([10.0, 10.0, 10.0]))
    assert score == -np.inf                            # covers nothing


def test_entropy_is_class_neutral():
    covered, totals = np.array([60.0, 40.0]), np.array([800.0, 200.0])
    score, head = Entropy().best_head(covered, totals)
    assert head == 0
    assert score == pytest.approx(Entropy().score(RuleStats(tp=60, fp=40, fn=740, tn=160)))  # two classes
    three = np.array([60.0, 39.0, 1.0])
    p = three / three.sum()
    assert Entropy().best_head(three, np.array([500.0, 300.0, 200.0]))[0] == pytest.approx((p * np.log2(p)).sum())
    assert Entropy().head_bound(three, np.array([500.0, 300.0, 200.0])) == 0.0   # a pure refinement


def test_a_gain_is_taken_over_the_parent_for_the_same_head():
    covered, parent, totals = np.array([[30.0], [10.0]]), np.array([40.0, 30.0]), np.array([100.0, 100.0])
    scores = FoilGain().head_scores(covered, totals, parent_covered=parent)
    for c, other in ((0, 1), (1, 0)):
        expected = FoilGain().score(RuleStats(tp=covered[c, 0], fp=covered[other, 0], fn=0, tn=0),
                                    RuleStats(tp=parent[c], fp=parent[other], fn=0, tn=0))
        assert scores[c, 0] == pytest.approx(expected)
    # no parent: the gain over the universal rule, the class's share of the scope
    unparented = FoilGain().head_scores(covered, totals)
    for c, other in ((0, 1), (1, 0)):
        expected = covered[c, 0] * (np.log2(covered[c, 0] / covered.sum()) - np.log2(totals[c] / totals.sum()))
        assert unparented[c, 0] == pytest.approx(expected)
    # the universal rule itself gains nothing for any head
    assert np.allclose(FoilGain().head_scores(totals[:, None], totals), 0.0)


def test_wrappers_and_lexicographic_scores():
    covered, totals = np.array([60.0, 40.0]), np.array([800.0, 200.0])
    base, head = Entropy().best_head(covered, totals)
    score, head2 = LengthPenalized(Entropy(), 0.1).best_head(covered, totals, length=2)
    assert head2 == head and score == pytest.approx(base - 0.2)
    with pytest.raises(TypeError, match="lexicographic"):
        LEF(Precision(), MinimalLength()).best_head(covered, totals)


# -------------------------------------------------------- decision lists ----

@pytest.mark.parametrize("learner", [CN2(), CN2(heuristic=Entropy()), CN2(heuristic=WRAcc()), PFossil(), Pypper()],
                         ids=["CN2", "CN2-entropy", "CN2-WRAcc", "PFossil", "Pypper"])
def test_a_decision_list_has_rules_of_several_classes(learner):
    rep, y = _three_classes()
    model = learner.fit(rep, model=DecisionList)
    assert type(model) is DecisionList
    assert len({r.target for r in model.rules}) >= 2
    assert np.mean(np.asarray(model.predict(rep)) == y) > 0.9
    # the default is the majority of what no rule covers
    covered = np.zeros(len(y), dtype=bool)
    for r in model.rules:
        covered |= np.asarray(r.rule.covers_data_packed(rep))
    if (~covered).any():
        values, counts = np.unique(y[~covered], return_counts=True)
        assert model.default_prediction.constant_target == values[np.argmax(counts)]


def test_each_rule_is_the_best_for_its_head_on_what_remains():
    rep, y = _three_classes()
    model = CN2(significance_threshold=None).fit(rep, model=DecisionList)
    remaining = np.ones(len(y), dtype=bool)
    for r in model.rules:
        stats = RuleStats.from_rule(r.rule, rep, r.target, example_mask=remaining)
        assert stats.tp > 0
        in_scope = y[remaining & np.asarray(r.rule.covers_data_packed(rep))]
        values, counts = np.unique(in_scope, return_counts=True)
        assert r.target == values[np.argmax(counts)]        # Laplace: the covered majority
        remaining &= ~np.asarray(r.rule.covers_data_packed(rep))


def test_the_rule_set_has_the_same_rules_and_predictions():
    rep, _ = _three_classes()
    dl = CN2().fit(rep, model=DecisionList)
    fs = CN2().fit(rep, model=FlatRuleSet)
    assert [r.rule for r in fs.rules] == [r.rule for r in dl.rules]
    assert list(fs.predict(rep)) == list(dl.predict(rep))


def test_every_representation_learns_the_same_decision_list():
    rep, _ = _three_classes()
    expected = CN2().fit(rep, model=DecisionList).to_string()
    for kind in (VerticalBitmapRepresentation, NListRepresentation, SparseDataRepresentation):
        assert CN2().fit(kind.from_boolean(rep), model=DecisionList).to_string() == expected, kind.__name__


def test_unit_weights_learn_the_same_decision_list():
    # weights go through the weighted counting path; 1.0 (not 2.0, which
    # changes Laplace and the significance test) must change nothing
    rep, _ = _three_classes()
    weighted, _ = _three_classes(weights=np.ones(600))
    assert [r.rule for r in CN2().fit(weighted, model=DecisionList).rules] == \
        [r.rule for r in CN2().fit(rep, model=DecisionList).rules]


def test_a_gain_heuristic_chooses_the_head_once():
    from pyrulearn.learners.seco import HeadObjective, handle_for
    from pyrulearn.rule import Rule
    rep, _ = _three_classes()
    objective = HeadObjective(FoilGain())
    objective.begin(rep, None, None)
    root = Rule([], target=None, dataspec=rep.spec)
    handle = handle_for(rep, root, None)
    root_stats = objective.rule_stats(rep, None, root, handle)
    features, batch, _, _ = objective.children(rep, None, frozenset(range(rep.spec.n_features)), handle, root_stats)
    assert len(set(np.asarray(batch.head).tolist())) > 1      # from the root, every head competes
    i = int(np.argmax(batch.score))
    child = Rule([int(features[i])], target=None, dataspec=rep.spec)
    child_handle = handle_for(rep, child, None)
    child_stats = objective.at(batch, i)
    rest = frozenset(range(rep.spec.n_features)) - {int(features[i])}
    _, grandchildren, _, _ = objective.children(rep, None, rest, child_handle, child_stats)
    assert set(np.asarray(grandchildren.head).tolist()) == {child_stats.head}   # then it stays


def test_a_seeded_learner_keeps_its_seed_covering():
    rep, _ = _three_classes()
    model = AQR(random_state=0).fit(rep, model=DecisionList)
    assert [r.rule for r in model.rules] == [r.rule for r in AQR(random_state=0).fit(rep).rules]


# ------------------------------------------------------ seeds and heads ----

def _rules(model):
    return [r.rule for r in model.rules]


def test_target_class_chooses_the_model():
    rep, _ = _three_classes()
    assert type(CN2(target_class=HeadPerRefinement()).fit(rep)) is DecisionList
    assert type(CN2(target_class=HeadPerRule()).fit(rep)) is DecisionList
    assert type(CN2(seed=RandomSeed()).fit(rep)) is DecisionList
    fixed = CN2(target_class=FixedClass("x")).fit(rep)
    assert type(fixed) is ConceptModel and fixed.label == "x"
    assert _rules(fixed) == _rules(CN2(target_class="x").fit(rep))
    assert type(CN2().fit(rep)) is ConceptSet                       # the multi-class default, as before


def test_the_head_options_also_go_to_fit():
    rep, _ = _three_classes()
    assert _rules(CN2().fit(rep, model=DecisionList, target_class=HeadPerRule())) == \
        _rules(CN2(target_class=HeadPerRule()).fit(rep))
    assert _rules(CN2().fit(rep, model=DecisionList)) == _rules(CN2(target_class=HeadPerRefinement()).fit(rep))
    assert _rules(CN2().fit(rep, model=DecisionList, seed=RandomSeed())) == _rules(CN2(seed=RandomSeed()).fit(rep))


def test_a_seed_implies_its_generalizations_and_its_class():
    rep, y = _three_classes()
    # the seed covering, asked for by seed= or (as before) by a SeedExample search space
    seeded = CN2(seed=RandomSeed()).fit(rep)
    assert _rules(seeded) == _rules(CN2(space_init=SeedExample()).fit(rep, model=DecisionList))
    assert _rules(seeded) == _rules(CN2(seed=RandomSeed(), target_class=HeadPerSeedExample(),
                                        space_init=SeedExample()).fit(rep))
    assert _rules(seeded) != _rules(CN2(seed=RandomSeed(), random_state=1).fit(rep))
    assert np.mean(np.asarray(seeded.predict(rep)) == y) > 0.9


def test_a_seed_can_leave_the_search_space_or_the_head_free():
    rep, y = _three_classes()
    for learner in (CN2(seed=RandomSeed(), space_init=EmptyRuleAllFeatures()),    # the seed only picks the class
                    CN2(seed=RandomSeed(), target_class=HeadPerRefinement()),              # covers the seed, any head
                    CN2(seed=FirstSeed())):
        model = learner.fit(rep)
        assert type(model) is DecisionList and len({r.target for r in model.rules}) >= 2
        assert np.mean(np.asarray(model.predict(rep)) == y) > 0.9


def test_each_class_takes_the_best_of_one_search_per_class():
    rep, y = _three_classes()
    model = CN2(target_class=HeadPerRule()).fit(rep)
    assert len({r.target for r in model.rules}) >= 2
    assert np.mean(np.asarray(model.predict(rep)) == y) > 0.9
    assert type(Pypper(target_class=HeadPerRule()).fit(rep)) is DecisionList


def test_impossible_combinations_are_refused():
    rep, _ = _three_classes()
    with pytest.raises(ValueError, match="needs a seed"):
        CN2(target_class=HeadPerSeedExample()).fit(rep)
    with pytest.raises(ValueError, match="several classes"):
        CN2().fit(rep, model=DecisionList, target_class=FixedClass("x"))
    with pytest.raises(ValueError, match="no single class"):
        CN2(target_class=HeadPerRefinement()).fit(rep, model=ConceptModel)


def test_pypper_takes_seeds_too():
    rep, y = _three_classes()
    model = Pypper(seed=RandomSeed()).fit(rep)
    assert type(model) is DecisionList
    assert np.mean(np.asarray(model.predict(rep)) == y) > 0.9
    assert type(Pypper().fit(rep)) is ConceptCascade                # the default, as before


def test_pypper_takes_a_search_space():
    rep, y = _three_classes()
    model = Pypper(seed=RandomSeed(), space_init=EmptyRuleAllFeatures()).fit(rep)   # the seed only picks the class
    assert type(model) is DecisionList and len({r.target for r in model.rules}) >= 2
    assert np.mean(np.asarray(model.predict(rep)) == y) > 0.9
    assert _rules(model) != _rules(Pypper(seed=RandomSeed()).fit(rep))


def test_opus_and_cpar_take_seeds():
    from pyrulearn.learners.cpar import CPAR
    from pyrulearn.learners.opus import Opus
    rep, y = _three_classes()
    model = Opus(max_conditions=2, seed=RandomSeed()).fit(rep)
    assert type(model) is DecisionList and np.mean(np.asarray(model.predict(rep)) == y) > 0.9
    # CPAR reweights, so a seed only seeds its class loops; its model stays a rule set
    model = CPAR(seed=FirstSeed()).fit(rep)
    assert type(model) is ConceptSet and np.mean(np.asarray(model.predict(rep)) == y) > 0.9


def test_a_seed_not_given_stays_out_of_the_parameters():
    # cache keys are built from the parameters: a new, unset seed= or space_init= must not change them
    from pyrulearn.learners.cpar import CPAR
    from pyrulearn.learners.opus import Opus
    for learner in (CN2(), Pypper(), Opus(), CPAR()):
        params = learner._provenance_params()
        assert "seed" not in params and "space_init" not in params, type(learner).__name__
    assert isinstance(CN2(seed=RandomSeed())._provenance_params()["seed"], RandomSeed)
    assert isinstance(Pypper(space_init=EmptyRuleAllFeatures())._provenance_params()["space_init"],
                      EmptyRuleAllFeatures)
