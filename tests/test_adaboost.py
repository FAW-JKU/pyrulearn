import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.learners.boosting import AdaBoost, samme_weight
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.multiclass import OneVsRest
from pyrulearn.learners.seco import PFoil, Pypper
from pyrulearn.models import ConceptSet, EnsembleModel, FlatRuleSet
from pyrulearn.rule import Rule

from _negation_helpers import neg_spec, neg_X


def _data(n=400, seed=0, noise=0.15):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", np.where(raw[:, 4], "neg", "mid"))
    flip = rng.random(n) < noise
    y = np.where(flip, rng.choice(["pos", "neg", "mid"], n), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


class _Recorder(NativeRuleLearner):
    """Predicts the class of feature 0 ("pos" where it holds, else "neg"),
    and records the row weights it was fitted with."""

    def __init__(self):
        self.seen = []

    def fit(self, data):
        self.seen.append(None if data.weights is None else data.weights.copy())
        return FlatRuleSet([Rule([0], target="pos", dataspec=data.spec)], default_prediction="neg")


def test_samme_weight_is_adaboost_for_two_classes_and_grows_with_the_classes():
    assert samme_weight(0.25, 2) == pytest.approx(np.log(3))
    assert samme_weight(0.25, 3) == pytest.approx(np.log(3) + np.log(2))
    assert samme_weight(0.0, 2) > 20   # no training errors: a large, finite weight


def test_adaboost_learns_weighted_members_on_reweighted_data():
    data = _data()
    model = AdaBoost(OneVsRest(PFoil()), n_estimators=4).fit(data)
    assert isinstance(model, EnsembleModel) and 1 <= len(model.members) <= 4
    assert all(isinstance(m, ConceptSet) for m in model.members)
    assert len(model.member_weights) == len(model.members) and np.all(model.member_weights > 0)
    assert np.mean(model.predict(data) == data.y) > 0.8


def test_adaboost_upweights_the_misclassified_examples():
    raw = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg", "neg", "pos"])   # the last two are misclassified
    data = HorizontalBitmapRepresentation(neg_spec(["a", "b"]), neg_X(raw), y)
    learner = _Recorder()
    model = AdaBoost(learner, n_estimators=2).fit(data)
    first, second = learner.seen
    assert np.allclose(first, 1.0)
    assert second.sum() == pytest.approx(6.0)                 # rescaled to the number of rows
    assert second[4] == second[5] > second[0] == second[1]     # the errors weigh more
    assert model.member_weights[0] == pytest.approx(samme_weight(2 / 6, 2))


def test_adaboost_stops_at_a_member_no_better_than_guessing():
    raw = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])   # feature 0 is right on half of them
    data = HorizontalBitmapRepresentation(neg_spec(["a", "b"]), neg_X(raw), y)
    model = AdaBoost(_Recorder(), n_estimators=5).fit(data)
    assert len(model.members) == 1 and model.member_weights[0] == 1.0   # kept, as it is the first


def test_without_early_stopping_adaboost_restarts_from_a_bootstrap_sample():
    raw = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    y = np.array(["pos", "pos", "neg", "neg"])   # feature 0 is right on half of them
    data = HorizontalBitmapRepresentation(neg_spec(["a", "b"]), neg_X(raw), y)
    learner = _Recorder()
    model = AdaBoost(learner, n_estimators=3, early_stopping=False).fit(data)
    # the first member is no better than guessing: left out, and boosting goes on with redrawn weights
    assert len(learner.seen) > 1 and not np.allclose(learner.seen[1], 1.0)
    assert np.isclose(learner.seen[1].sum(), 4.0) and np.all(learner.seen[1] == learner.seen[1].round())
    assert 1 <= len(model.members) <= 3
    full = AdaBoost(OneVsRest(PFoil()), n_estimators=6, early_stopping=False).fit(_data())
    assert len(full.members) == 6


def test_abstain_leaves_out_the_members_defaults():
    data = _data()
    model = AdaBoost(OneVsRest(Pypper()), n_estimators=6, early_stopping=False).fit(data)
    assert any(rule is None for member in model.members for rule in member.deciding_rules(data))
    out = model.predict(data)
    for j in range(data.n_samples):
        tally = {}
        for member, weight in zip(model.members, model.member_weights):
            if member.deciding_rules(data)[j] is not None:
                label = member.predict(data)[j]
                tally[label] = tally.get(label, 0.0) + weight
        assert not tally or tally[out[j]] == max(tally.values())


def test_abstain_reaches_the_ensemble_and_its_stages():
    model = AdaBoost(OneVsRest(PFoil()), n_estimators=3, early_stopping=False).fit(_data())
    assert model.abstain and model.staged(2).abstain
    voting = AdaBoost(OneVsRest(PFoil()), n_estimators=3, abstain=False).fit(_data())
    assert not voting.abstain and "weighted vote of members and defaults" in voting.to_string()
