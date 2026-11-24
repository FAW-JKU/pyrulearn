"""`staged(n)`: the model after its first n rounds is the model a fit with
n rounds builds -- for the ensembles (`Bagging`, `AdaBoost`) and the rule
boosters (`Slipper`, `ENDER`, `Boomer`)."""

import numpy as np
import pytest

from pyrulearn.learners.boosting import AdaBoost
from pyrulearn.learners.bagging import Bagging
from pyrulearn.learners.boosting import ENDER, Boomer, Slipper
from pyrulearn.learners.multiclass import OneVsRest
from pyrulearn.learners.seco import PFoil
from pyrulearn.data import HorizontalBitmapRepresentation

from _negation_helpers import neg_spec, neg_X


def _data(n=300, seed=0, noise=0.15, classes=3):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", np.where(raw[:, 4], "neg", "mid"))
    if classes == 2:
        y = np.where(y == "mid", "neg", y)
    labels = np.unique(y)
    flip = rng.random(n) < noise
    y = np.where(flip, rng.choice(labels, n), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


def _rules(model):
    return sorted((str(r.target), tuple(c.feature for c in r.conditions), round(float(r.weight), 9))
                  for r in model.rules)


def _same_ensemble(a, b, data):
    assert len(a.members) == len(b.members)
    assert [m.to_string() for m in a.members] == [m.to_string() for m in b.members]
    if a.member_weights is not None:
        assert np.allclose(a.member_weights, b.member_weights)
    assert list(a.predict(data)) == list(b.predict(data))


def test_the_first_members_of_bagging_are_bagging_with_fewer_members():
    data = _data()
    full = Bagging(OneVsRest(PFoil()), n_estimators=6, random_state=4).fit(data)
    _same_ensemble(full.staged(3), Bagging(OneVsRest(PFoil()), n_estimators=3, random_state=4).fit(data), data)
    assert full.staged(6).predict(data).tolist() == full.predict(data).tolist()


def test_the_first_members_of_adaboost_are_adaboost_with_fewer_rounds():
    data = _data()
    full = AdaBoost(OneVsRest(PFoil()), n_estimators=5).fit(data)
    n = min(3, len(full.members))
    _same_ensemble(full.staged(n), AdaBoost(OneVsRest(PFoil()), n_estimators=n).fit(data), data)


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("make", [lambda n: Slipper(n_rounds=n), lambda n: ENDER(n_rules=n),
                                  lambda n: Boomer(n_rules=n)], ids=["Slipper", "ENDER", "Boomer"])
def test_the_first_rounds_of_a_rule_booster_are_the_booster_with_fewer_rounds(make, classes):
    data = _data(classes=classes)
    full = make(20).fit(data)
    assert full.n_stages == 20
    for n in (0, 1, 7, 20):
        staged, refit = full.staged(n), make(n).fit(data) if n else None
        if refit is not None:
            assert _rules(staged) == _rules(refit)
            assert list(staged.predict(data)) == list(refit.predict(data))
        assert staged.n_stages == n
    assert _rules(full.staged(20)) == _rules(full)


def test_staged_keeps_the_rules_training_stats():
    data = _data()
    staged = ENDER(n_rules=10).fit(data).staged(5)
    assert all(r.stats() is not None for r in staged.rules)


def test_staged_needs_recorded_rounds_and_a_valid_n():
    data = _data()
    with pytest.raises(ValueError):
        ENDER(n_rules=5).fit(data).staged(6)
    with pytest.raises(ValueError):
        Bagging(PFoil(), n_estimators=2).fit(data).staged(0)
    from pyrulearn.models import LinearRuleModel
    from pyrulearn.rule import WeightedRule
    plain = LinearRuleModel([WeightedRule([0], target="pos", dataspec=data.spec, weight=1.0)], classes=["neg", "pos"])
    with pytest.raises(ValueError):
        plain.staged(0)
