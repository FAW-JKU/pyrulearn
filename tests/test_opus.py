import itertools

import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation, DataSpec, NListRepresentation, SparseDataRepresentation
from pyrulearn.heuristics import Accuracy, FoilGain, Laplace, Precision, RuleStats, WRAcc
from pyrulearn.learners.opus import Opus, OpusTopK
from pyrulearn.learners.seco import BranchAndBoundSearch, CN2, EmptyRuleAllFeatures, SeCo
from pyrulearn.models import ConceptModel, ConceptSet
from pyrulearn.rule import Rule

from _negation_helpers import neg_spec, neg_X


def _data(n=200, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 5)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & ~raw[:, 3], "pos", "neg")
    y = np.where(rng.random(n) < noise, np.where(y == "pos", "neg", "pos"), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(5)]), neg_X(raw), y)


def _brute_force(data, target, heuristic, max_len, k):
    """The k best scores, one per distinct coverage, over every consistent
    conjunction of up to max_len features."""
    w = np.ones(data.n_samples) if data.weights is None else data.weights
    pos = np.asarray(data.y) == target
    best = {}
    for length in range(1, max_len + 1):
        for feats in itertools.combinations(range(data.spec.n_features), length):
            rule = Rule(list(feats), target=target, dataspec=data.spec)
            if not rule.is_consistent():
                continue
            cov = data.coverage(rule)
            tp, fp = float(w[cov & pos].sum()), float(w[cov & ~pos].sum())
            if tp == 0:
                continue
            score = heuristic.score(RuleStats(tp=tp, fp=fp, fn=float(w[pos].sum()) - tp,
                                              tn=float(w[~pos].sum()) - fp, length=length))
            best[cov.tobytes()] = max(best.get(cov.tobytes(), -np.inf), score)
    return sorted(best.values(), reverse=True)[:k]


@pytest.mark.parametrize("heuristic", [Laplace(), WRAcc(), Precision(), Accuracy()], ids=lambda h: repr(h))
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("weighted", [False, True])
def test_branch_and_bound_finds_the_optimal_rules(heuristic, k, weighted):
    data = _data()
    if weighted:
        data = data.with_weights(np.linspace(0.5, 2.0, data.n_samples))
    initial = EmptyRuleAllFeatures().initial_candidates(data, "pos")
    rules = BranchAndBoundSearch(max_conditions=3, k=k).search_all(data, "pos", heuristic, initial)
    scores = [heuristic.score(RuleStats.from_rule(r, data, "pos")) for r in rules]
    np.testing.assert_allclose(scores, _brute_force(data, "pos", heuristic, 3, k), rtol=1e-12)
    assert len({tuple(data.coverage(r)) for r in rules}) == len(rules)      # distinct coverage
    assert all(r.length() <= 3 for r in rules)


def test_branch_and_bound_is_the_same_on_every_representation():
    data = _data()
    initial = EmptyRuleAllFeatures().initial_candidates(data, "pos")
    want = BranchAndBoundSearch(max_conditions=3, k=4).search_all(data, "pos", WRAcc(), initial)
    for rep in (NListRepresentation.from_boolean(data), SparseDataRepresentation.from_boolean(data)):
        assert BranchAndBoundSearch(max_conditions=3, k=4).search_all(rep, "pos", WRAcc(), initial) == want


def test_branch_and_bound_refuses_a_gain_heuristic_and_bad_k():
    data = _data()
    initial = EmptyRuleAllFeatures().initial_candidates(data, "pos")
    with pytest.raises(ValueError, match="relative to"):
        BranchAndBoundSearch().search(data, "pos", FoilGain(), initial)
    with pytest.raises(ValueError):
        BranchAndBoundSearch(k=0)


def test_branch_and_bound_beats_or_ties_a_beam_search():
    data = _data(noise=0.2)
    for heuristic in (Laplace(), WRAcc()):
        optimal = SeCo(_learner(heuristic, BranchAndBoundSearch(max_conditions=3)), target_class="pos")
        rule = optimal.single_rule_learner.learn_one_rule(data, "pos")
        beam = CN2(heuristic=heuristic, target_class="pos").single_rule_learner.learn_one_rule(data, "pos")
        best = heuristic.score(RuleStats.from_rule(rule, data, "pos"))
        assert best >= heuristic.score(RuleStats.from_rule(beam, data, "pos")) - 1e-12 or beam.length() > 3


def _learner(heuristic, search):
    from pyrulearn.learners.seco import SingleRuleLearner
    return SingleRuleLearner(heuristic, search=search)


def test_opus_is_seco_with_branch_and_bound():
    opus = Opus(max_conditions=2)
    assert isinstance(opus, SeCo)
    assert isinstance(opus.single_rule_learner.search, BranchAndBoundSearch)
    assert isinstance(opus.single_rule_learner.heuristic, Laplace)
    data = _data()
    model = opus.fit(data)
    assert isinstance(model, ConceptSet)
    assert all(r.length() <= 2 for r in model.rules)
    assert np.mean(np.asarray(model.predict(data)) == data.y) > 0.8
    assert str(model) == str(Opus(max_conditions=2).fit(NListRepresentation.from_boolean(data)))


def test_opus_top_k_finds_the_k_best_rules_per_class():
    data = _data()
    model = OpusTopK(k=3, max_conditions=3, productive=False).fit(data)
    assert isinstance(model, ConceptSet)
    for concept in model.concepts:
        assert len(concept.rules) == 3
        scores = [WRAcc().score(RuleStats.from_rule(r, data, concept.label)) for r in concept.rules]
        assert scores == sorted(scores, reverse=True)
        np.testing.assert_allclose(scores, _brute_force(data, concept.label, WRAcc(), 3, 3), rtol=1e-12)
    one = OpusTopK(k=3, max_conditions=3, productive=False, target_class="pos").fit(data)
    assert isinstance(one, ConceptModel) and one.label == "pos" and len(one.rules) == 3
    with pytest.raises(ValueError):
        OpusTopK(k=0)


def _precision(data, target, feats):
    cov = data.coverage(Rule(list(feats), target=target, dataspec=data.spec))
    pos = np.asarray(data.y) == target
    return pos[cov].sum() / cov.sum() if cov.any() else -np.inf


def _productive_brute_force(data, target, heuristic, max_len, k):
    """Like `_brute_force`, restricted to rules with a strictly higher
    precision than each of their generalizations (the empty rule included)."""
    pos = np.asarray(data.y) == target
    best = {}
    for length in range(1, max_len + 1):
        for feats in itertools.combinations(range(data.spec.n_features), length):
            rule = Rule(list(feats), target=target, dataspec=data.spec)
            if not rule.is_consistent():
                continue
            cov = data.coverage(rule)
            tp, fp = float((cov & pos).sum()), float((cov & ~pos).sum())
            if tp == 0:
                continue
            own = tp / (tp + fp)
            if any(own <= _precision(data, target, sub)
                   for size in range(length) for sub in itertools.combinations(feats, size)):
                continue
            score = heuristic.score(RuleStats(tp=tp, fp=fp, fn=pos.sum() - tp, tn=(~pos).sum() - fp, length=length))
            best[cov.tobytes()] = max(best.get(cov.tobytes(), -np.inf), score)
    return sorted(best.values(), reverse=True)[:k]


def test_productive_rule_rejects_a_rule_no_better_than_a_generalization():
    from pyrulearn.pruning import ProductiveRule
    data = _data()
    pos = "pos"
    criterion = ProductiveRule()
    for feats in itertools.combinations(range(data.spec.n_features), 2):
        rule = Rule(list(feats), target=pos, dataspec=data.spec)
        if not rule.is_consistent() or not data.coverage(rule).any():
            continue
        own = _precision(data, pos, feats)
        productive = all(own > _precision(data, pos, sub) for sub in [(), feats[:1], feats[1:]])
        assert criterion.evaluate(rule, RuleStats.from_rule(rule, data, pos), data, pos) == (not productive)
    assert criterion.evaluate(Rule([], target=pos, dataspec=data.spec), RuleStats.from_rule(
        Rule([], target=pos, dataspec=data.spec), data, pos), data, pos) is False


@pytest.mark.parametrize("heuristic", [WRAcc(), Laplace()], ids=lambda h: repr(h))
def test_opus_top_k_finds_the_k_best_productive_rules(heuristic):
    data = _data(noise=0.2)
    model = OpusTopK(k=5, max_conditions=3, heuristic=heuristic).fit(data)
    for concept in model.concepts:
        scores = [heuristic.score(RuleStats.from_rule(r, data, concept.label)) for r in concept.rules]
        np.testing.assert_allclose(scores, _productive_brute_force(data, concept.label, heuristic, 3, 5), rtol=1e-12)
        for r in concept.rules:      # each beats every generalization
            feats = [lit.feature for lit in r.conditions]
            own = _precision(data, concept.label, feats)
            assert all(own > _precision(data, concept.label, sub)
                       for size in range(len(feats)) for sub in itertools.combinations(feats, size))


def test_laplace_counts_the_classes():
    stats = RuleStats(tp=6, fp=2, fn=4, tn=8, length=1)
    assert Laplace().score(stats) == 7 / 10
    assert Laplace(n_classes=3).score(stats) == 7 / 11
    assert repr(Laplace()) == "Laplace" and repr(Laplace(n_classes=3)) == "Laplace(n_classes=3)"
    np.testing.assert_allclose(Laplace(n_classes=4).batch_score(RuleStats(tp=np.array([1.0, 3.0]), fp=np.array([0.0, 1.0]),
                                                                           fn=np.zeros(2), tn=np.zeros(2), length=1)),
                               [2 / 5, 4 / 8])
    with pytest.raises(ValueError):
        Laplace(n_classes=0)


@pytest.mark.parametrize("heuristic", [Laplace(), Laplace(n_classes=3), WRAcc(), Precision(), Accuracy()],
                         ids=lambda h: repr(h))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dominance_pruning_keeps_the_optimum(heuristic, seed):
    data = _data(noise=0.2, seed=seed)
    initial = EmptyRuleAllFeatures().initial_candidates(data, "pos")
    scores = []
    for dominance in (False, True):
        rule = BranchAndBoundSearch(max_conditions=4, dominance_pruning=dominance).search(data, "pos", heuristic, initial)
        scores.append(heuristic.score(RuleStats.from_rule(rule, data, "pos")))
    assert scores[0] == pytest.approx(scores[1], rel=1e-12)
    assert scores[1] == pytest.approx(_brute_force(data, "pos", heuristic, 4, 1)[0], rel=1e-12)


def test_opus_laplace_counts_the_classes_of_the_data():
    data = _data()
    y = np.where(np.arange(data.n_samples) % 5 == 0, "other", np.asarray(data.y))
    three = HorizontalBitmapRepresentation(data.spec, data.X, y)
    opus = Opus(max_conditions=2)
    opus.fit(three)
    assert opus.single_rule_learner.heuristic.n_classes == 3
    opus.fit(data)
    assert opus.single_rule_learner.heuristic.n_classes == 2
    assert Opus(heuristic=WRAcc()).fit(three) is not None
