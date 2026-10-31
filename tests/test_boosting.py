import math

import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation, NListRepresentation
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.evaluation import RuleStats
from pyrulearn.heuristics import SlipperZ
from pyrulearn.learners.boosting import Slipper
from pyrulearn.models import LinearRuleModel

from _negation_helpers import neg_spec, neg_X


def _data(n=600, seed=0, noise=0.1, n_features=6):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, n_features)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    flip = rng.random(n) < noise
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(n_features)]), neg_X(raw), y)


def _bodies(model):
    return [tuple(sorted(l.feature for l in r.conditions)) for r in model.rules]


def test_slipper_z_heuristic():
    assert SlipperZ().score(RuleStats(tp=9, fp=4, fn=0, tn=0)) == pytest.approx(1.0)
    assert SlipperZ().score(RuleStats(tp=2.25, fp=0.0, fn=0, tn=0)) == pytest.approx(1.5)


def test_slipper_learns_a_linear_model_that_finds_the_concept():
    data = _data()
    model = Slipper(n_rounds=20, random_state=0).fit(data)
    assert isinstance(model, LinearRuleModel)
    assert model.labels == ["neg", "pos"]
    # the less frequent class is the target; the intercept comes first
    assert {r.target for r in model.rules} == {"pos"}
    assert model.rules[0].conditions == () or len(model.rules[0].conditions) == 0
    bodies = _bodies(model)
    assert len(bodies) == len(set(bodies))                          # repeated rules merged
    assert (0, 2) in bodies and (4, 6) in bodies                    # f0 & f1, f2 & f3
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85
    assert model.to_string().splitlines()[0] == "% conflict resolution: sum of rule weights per class, highest wins"


def test_slipper_prediction_is_the_sign_of_the_summed_confidences():
    data = _data()
    model = Slipper(n_rounds=10, random_state=0).fit(data)
    scores = model.scores(data)
    np.testing.assert_array_equal(np.asarray(model.predict(data)) == "pos", scores[:, 1] > 0)


def test_slipper_is_reproducible_and_representation_independent():
    data = _data()
    a = Slipper(n_rounds=8, random_state=3).fit(data)
    b = Slipper(n_rounds=8, random_state=3).fit(data)
    c = Slipper(n_rounds=8, random_state=3).fit(NListRepresentation.from_boolean(data))
    assert [r.to_string() for r in a.rules] == [r.to_string() for r in b.rules] == [r.to_string() for r in c.rules]


def test_slipper_target_class_and_multiclass():
    data = _data()
    neg = Slipper(n_rounds=5, target_class="neg", random_state=0).fit(data)
    assert {r.target for r in neg.rules} == {"neg"}
    rng = np.random.default_rng(2)
    raw = rng.random((400, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    multi = HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = Slipper(n_rounds=5, random_state=0).fit(multi)
    assert model.labels == ["a", "b", "c"]
    assert {r.target for r in model.rules} == {"a", "b", "c"}
    assert np.mean(np.asarray(model.predict(multi)) == y) > 0.95


def test_slipper_uses_the_data_weights_as_initial_boosting_weights():
    data = _data()
    heavy_neg = np.where(data.y == "neg", 10.0, 1.0)
    plain = Slipper(n_rounds=5, random_state=0).fit(data)
    weighted = Slipper(n_rounds=5, random_state=0).fit(data.with_weights(heavy_neg))
    assert [r.to_string() for r in plain.rules] != [r.to_string() for r in weighted.rules]
    assert isinstance(weighted.rules[1].stats().confusion.rule_stats("pos").tp, float)


def test_slipper_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Slipper(n_rounds=0)
    one_class = HorizontalBitmapRepresentation(neg_spec(["a"]), neg_X(np.ones((3, 1), bool)), np.array(["x"] * 3))
    with pytest.raises(ValueError, match="two classes"):
        Slipper().fit(one_class)


# ------------------------------------------------------------------- ENDER

from pyrulearn.learners.boosting import (  # noqa: E402
    ENDER, ConstantStep, ExponentialLoss, Gradient, GradientBoosting, LogisticLoss, Newton, SigmoidLoss,
    Simultaneous,
)


@pytest.mark.parametrize("loss", [LogisticLoss(), ExponentialLoss(), SigmoidLoss()])
def test_loss_derivatives_match_finite_differences(loss):
    rng = np.random.default_rng(0)
    n, K = 20, 2
    F = rng.normal(size=(n, K))
    Y = np.eye(K)[rng.integers(0, K, n)]
    d = rng.random(n) + 0.5
    G, H = loss.derivatives(F, Y, d)
    eps = 1e-5
    for k in range(K):
        for i in (0, 7, 13):
            step = np.zeros_like(F)
            step[i, k] = eps
            g_num = (loss.value(F + step, Y, d) - loss.value(F - step, Y, d)) / (2 * eps)
            h_num = (loss.value(F + step, Y, d) - 2 * loss.value(F, Y, d) + loss.value(F - step, Y, d)) / eps ** 2
            assert G[i, k] == pytest.approx(g_num, rel=1e-4, abs=1e-6)
            assert H[i, k] == pytest.approx(h_num, rel=1e-3, abs=1e-4)


def test_ender_learns_positive_class_votes_that_find_the_concept():
    data = _data()
    model = ENDER(n_rules=100, random_state=0).fit(data)
    assert isinstance(model, LinearRuleModel)
    assert all(r.weight > 0 for r in model.rules)                 # every rule votes for its class
    assert sum(len(r.conditions) == 0 for r in model.rules) == 1  # one default rule
    bodies = {(r.target, tuple(sorted(l.feature for l in r.conditions))) for r in model.rules}
    assert ("pos", (0, 2)) in bodies and ("pos", (4, 6)) in bodies
    assert len(bodies) == len(model.rules)                        # repeated rules merged
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85
    scores = model.scores(test)
    np.testing.assert_array_equal(np.asarray(model.predict(test)) == "pos", scores[:, 1] > scores[:, 0])


@pytest.mark.parametrize("kw", [
    dict(method=Gradient()), dict(method=GradientBoosting()), dict(method=Newton(), subsample=0.5),
    dict(loss=ExponentialLoss()), dict(loss=ExponentialLoss(), method=Simultaneous()),
    dict(loss=SigmoidLoss(), shrinkage=1.0, subsample=0.5), dict(shrinkage=1.0), dict(method=ConstantStep(0.6)),
])
def test_ender_variants(kw):
    data = _data()
    model = ENDER(n_rules=40, random_state=0, **kw).fit(data)
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85


def test_ender_multiclass_and_loss_restrictions():
    rng = np.random.default_rng(2)
    raw = rng.random((400, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = ENDER(n_rules=60, random_state=0).fit(data)
    assert model.labels == ["a", "b", "c"]
    assert np.mean(np.asarray(model.predict(data)) == y) > 0.95
    with pytest.raises(ValueError, match="two classes only"):
        ENDER(loss=ExponentialLoss()).fit(data)
    with pytest.raises(ValueError):
        ENDER(method="exact")                      # names aren't components


def test_ender_early_stopping_ends_on_noise():
    rng = np.random.default_rng(3)
    raw = rng.random((400, 6)) < 0.5
    noise = HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw),
                                      rng.choice(["x", "y"], 400))
    full = ENDER(n_rules=200, random_state=0).fit(noise)
    stopped = ENDER(n_rules=200, early_stopping=True, random_state=0).fit(noise)
    assert len(stopped.rules) < 0.75 * len(full.rules)


def test_ender_is_reproducible_and_uses_data_weights():
    data = _data()
    a = ENDER(n_rules=30, random_state=5).fit(data)
    b = ENDER(n_rules=30, random_state=5).fit(data)
    assert [r.to_string() for r in a.rules] == [r.to_string() for r in b.rules]
    w = ENDER(n_rules=30, random_state=5).fit(data.with_weights(np.where(data.y == "pos", 4.0, 1.0)))
    assert [r.to_string() for r in w.rules] != [r.to_string() for r in a.rules]


def test_ender_defaults_are_the_papers_constant_step_logit_setting():
    e = ENDER()
    assert isinstance(e.method, ConstantStep) and e.method.beta == 0.2
    assert (e.shrinkage, e.subsample, e.n_rules) == (0.1, 0.25, 500)
    assert isinstance(e.loss, LogisticLoss)


def test_exponential_response_is_the_smoothed_exact_minimizer():
    loss = ExponentialLoss()
    loss.eps = 0.5
    w_pos, w_neg = 6.0, 2.0                        # g = W- - W+, h = W+ + W-
    assert loss.response(w_neg - w_pos, w_pos + w_neg) == pytest.approx(0.5 * np.log(6.5 / 2.5))
    assert SigmoidLoss(beta=0.7).response(-3.0, 1.0) == 0.7


def test_a_larger_constant_step_gives_rules_covering_less():
    data = _data()
    X = data.X

    def mean_coverage(beta):
        model = ENDER(n_rules=60, method=ConstantStep(beta), random_state=0).fit(data)
        body = [r for r in model.rules if r.conditions]
        return np.mean([np.all(X[:, [l.feature for l in r.conditions]], axis=1).sum() for r in body])

    assert mean_coverage(0.05) > mean_coverage(2.0)


def test_invalid_method_and_loss_combinations_are_refused():
    with pytest.raises(ValueError, match="exponential"):
        ENDER(method=Simultaneous())
    with pytest.raises(ValueError, match="convex"):
        ENDER(loss=SigmoidLoss(), method=Newton())
    with pytest.raises(ValueError):
        ConstantStep(0.0)
    with pytest.raises(ValueError):
        SigmoidLoss(beta=0.0)
    with pytest.raises(ValueError):
        ENDER(loss="logistic")                     # names aren't components


def test_ender_l2_regularization_shrinks_the_weights():
    data = _data()
    plain = ENDER(n_rules=20, method=Newton(), subsample=0.5, random_state=0).fit(data)
    reg = ENDER(n_rules=20, method=Newton(), subsample=0.5, l2_regularization=50.0, random_state=0).fit(data)
    assert np.mean([r.weight for r in reg.rules]) < np.mean([r.weight for r in plain.rules])
    with pytest.raises(ValueError):
        ENDER(l2_regularization=-1.0)


# ------------------------------------------------------- optimal rule boosting

from itertools import combinations  # noqa: E402

from pyrulearn.learners.boosting import DenseORB, SquaredMarginLoss  # noqa: E402


def test_branch_and_bound_finds_the_best_conjunction():
    data = _data(n=200)
    X = data.X
    y = np.where(data.y == "pos", 1.0, -1.0)
    learner = DenseORB(reg=1.0)
    g, h = learner._derivatives(y, np.zeros(len(y)))
    body = learner._best_query(X, g, h)
    best = learner._objective(g, h, np.all(X[:, list(body)], axis=1))
    for size in (1, 2, 3):                                  # brute force over short conjunctions
        for combo in combinations(range(X.shape[1]), size):
            cov = np.all(X[:, list(combo)], axis=1)
            if cov.any():
                assert learner._objective(g, h, cov) <= best + 1e-9
    greedy = DenseORB(search="greedy")._best_query(X, g, h)
    assert learner._objective(g, h, np.all(X[:, list(greedy)], axis=1)) <= best + 1e-9


def test_optimal_rule_boosting_model():
    data = _data()
    model = DenseORB(n_rules=5).fit(data)
    assert isinstance(model, LinearRuleModel) and model.labels == ["neg", "pos"]
    assert all(r.weight > 0 for r in model.rules)          # negative weights vote for "neg"
    bodies = {(r.target, tuple(sorted(l.feature for l in r.conditions))) for r in model.rules}
    assert ("pos", (0, 2)) in bodies and ("pos", (4, 6)) in bodies
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85
    with_offset = DenseORB(n_rules=3, offset=True).fit(data)
    assert any(len(r.conditions) == 0 for r in with_offset.rules)
    squared = DenseORB(n_rules=5, loss=SquaredMarginLoss()).fit(data)
    assert np.mean(np.asarray(squared.predict(test)) == test.y) > 0.85


def test_optimal_rule_boosting_rejects_bad_input():
    rng = np.random.default_rng(2)
    raw = rng.random((60, 3)) < 0.5
    three = HorizontalBitmapRepresentation(neg_spec(["a", "b", "c"]), neg_X(raw), rng.choice(["x", "y", "z"], 60))
    with pytest.raises(ValueError, match="binary"):
        DenseORB().fit(three)
    with pytest.raises(ValueError):
        DenseORB(search="beam")


def _best_gain_brute_force(X, g, h, reg, sign=None, max_len=3):
    """The best XGBoost gain over every conjunction of up to `max_len` features."""
    from pyrulearn.learners.boosting import XGBGain
    objective = XGBGain(g, h, reg=reg, sign=sign)
    best = -math.inf
    for size in range(1, max_len + 1):
        for combo in combinations(range(X.shape[1]), size):
            cov = np.all(X[:, list(combo)], axis=1)
            if cov.any():
                G, H = g[cov].sum(), h[cov].sum()
                best = max(best, float(objective._gain(np.array([G]), np.array([H]))[0]))
    return best


@pytest.mark.parametrize("reg", [1.0, 0.0])
@pytest.mark.parametrize("sign", [None, 1, -1])
def test_orb_branch_and_bound_finds_the_best_gain(reg, sign):
    from pyrulearn.learners.boosting import XGBGain
    from pyrulearn.learners.seco import BranchAndBoundSearch, EmptyRuleAllFeatures
    data = _data(n=200)
    X = data.X
    y = np.where(data.y == "pos", 1.0, -1.0)
    s = np.random.default_rng(3).normal(0, 0.5, len(y))          # a later round: varied g and h
    g, h = DenseORB()._derivatives(y, s)
    objective = XGBGain(g, h, reg=reg, sign=sign)
    initial = EmptyRuleAllFeatures().initial_candidates(data, None)
    rule = BranchAndBoundSearch(max_conditions=3).search(data, None, objective, initial)
    cov = np.all(X[:, [l.feature for l in rule.conditions]], axis=1)
    found = float(objective._gain(np.array([g[cov].sum()]), np.array([h[cov].sum()]))[0])
    assert found == pytest.approx(_best_gain_brute_force(X, g, h, reg, sign), rel=1e-12)


@pytest.mark.parametrize("search", ["exhaustive", "greedy"])
def test_orb_matches_dense_orb(search):
    from pyrulearn.learners.boosting import ORB
    from pyrulearn.learners.seco import BranchAndBoundSearch, HillClimbing
    rule_search = {"exhaustive": BranchAndBoundSearch, "greedy": HillClimbing}[search]
    data = _data()
    for kw in ({}, {"offset": True}, {"loss": SquaredMarginLoss()}, {"max_length": 2}):
        dense = DenseORB(n_rules=5, search=search, **kw).fit(data)
        orb_kw = {k: v for k, v in kw.items() if k != "max_length"}
        modular = ORB(n_rules=5, search=rule_search(max_conditions=kw.get("max_length")), **orb_kw).fit(data)
        assert str(modular) == str(dense)


def test_orb_is_the_same_on_every_representation():
    from pyrulearn.data import SparseDataRepresentation
    from pyrulearn.learners.boosting import ORB
    from pyrulearn.learners.seco import BeamSearch
    data = _data()
    for make in (lambda: ORB(n_rules=4), lambda: ORB(n_rules=4, search=BeamSearch(beam_width=3))):
        want = str(make().fit(data))
        for rep in (NListRepresentation.from_boolean(data), SparseDataRepresentation.from_boolean(data)):
            assert str(make().fit(rep)) == want


def test_orb_rejects_bad_input():
    from pyrulearn.learners.boosting import ORB
    from pyrulearn.learners.seco import BeamSearch
    rng = np.random.default_rng(2)
    raw = rng.random((60, 3)) < 0.5
    three = HorizontalBitmapRepresentation(neg_spec(["a", "b", "c"]), neg_X(raw), rng.choice(["x", "y", "z"], 60))
    with pytest.raises(ValueError, match="binary"):
        ORB().fit(three)
    with pytest.raises(ValueError):
        ORB(search="beam")
    assert isinstance(ORB().search, BeamSearch)

def test_boomer_is_ender_with_boomers_defaults():
    from pyrulearn.learners.boosting import Boomer
    b = Boomer()
    assert (b.n_rules, b.shrinkage, b.l2_regularization, b.subsample) == (1000, 0.3, 1.0, 1.0)
    assert isinstance(b.method, Newton)
    assert isinstance(b.loss, LogisticLoss)
    data = _data()
    model = Boomer(n_rules=50).fit(data)
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85


# ------------------------------------------------- ENDER vs. DenseENDER

from pyrulearn.learners.boosting import (  # noqa: E402
    Boomer, ConstantStep, DenseBoomer, DenseENDER, Gradient, GradientBoosting, Newton, Simultaneous,
)


def _numeric_data(n=300, seed=3):
    import pandas as pd
    from pyrulearn.data.io import binarize, build_dataspec
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "c": rng.choice(list("pqr"), n)})
    y = np.where((df["x"] > 0.3) & (df["c"] != "q") | (df["z"] < -0.8), "pos", "neg")
    df["y"] = np.where(rng.random(n) < 0.1, np.where(y == "pos", "neg", "pos"), y)
    spec = build_dataspec(df, target="y", discretizer=TreeDiscretizer(max_intervals=6)).build()
    return HorizontalBitmapRepresentation(spec, binarize(spec, df), df["y"].to_numpy())


ENDER_CONFIGS = [
    dict(),
    dict(method=Gradient()),
    dict(method=GradientBoosting()),
    dict(method=Newton(), l2_regularization=1.0, subsample=1.0),
    dict(method=Simultaneous(), loss=ExponentialLoss()),
    dict(loss=ExponentialLoss()),
    dict(loss=SigmoidLoss()),
    dict(max_length=2, early_stopping=True),
]


@pytest.mark.parametrize("kw", ENDER_CONFIGS, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "default")
@pytest.mark.parametrize("make", [_data, _numeric_data, lambda: _data().with_weights(np.linspace(0.5, 2, 600))],
                         ids=["boolean", "numeric", "weighted"])
def test_ender_and_dense_ender_learn_the_same_model(kw, make):
    # an implied condition never lowers the impurity strictly, so DenseENDER
    # never picks one either; constraint propagation only spares ENDER from scoring it
    data = make()
    assert str(ENDER(n_rules=40, **kw).fit(data)) == str(DenseENDER(n_rules=40, **kw).fit(data))


def test_impurity_criteria_are_components():
    data = _numeric_data()
    assert str(ENDER(n_rules=30, method=ConstantStep(0.2)).fit(data)) == str(ENDER(n_rules=30).fit(data))
    other_beta = ENDER(n_rules=30, method=ConstantStep(1.0)).fit(data)
    assert str(other_beta) != str(ENDER(n_rules=30).fit(data))       # the criterion's own beta is used
    with pytest.raises(ValueError, match="exponential"):
        ENDER(method=Simultaneous())
    with pytest.raises(ValueError, match="convex"):
        DenseENDER(method=Newton(), loss=SigmoidLoss())


def test_boomer_and_dense_boomer_learn_the_same_model():
    data = _numeric_data()
    assert str(Boomer(n_rules=40).fit(data)) == str(DenseBoomer(n_rules=40).fit(data))
    assert DenseBoomer().NATIVE_REPRESENTATIONS == (HorizontalBitmapRepresentation,)
    assert Boomer().NATIVE_REPRESENTATIONS is None
