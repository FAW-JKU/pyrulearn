import numpy as np
import pytest

from pyrulearn.data import HorizontalBitmapRepresentation, DataSpec
from pyrulearn.data import NListRepresentation, SparseDataRepresentation
from pyrulearn.learners.boosting import LRI, DenseLRI
from pyrulearn.objectives.losses import LRIError
from pyrulearn.learners.seco import (
    BeamSearch, CoveringState, EmptyRuleAllFeatures, HillClimbing, LRIReweighting,
)
from pyrulearn.models import ConceptModel, EnsembleModel

from _negation_helpers import neg_spec, neg_X


def _data(n=600, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    flip = rng.random(n) < noise
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


def _make(cls, max_length=None, **kw):
    """`DenseLRI(max_length=...)` / `LRI(search=HillClimbing(max_conditions=...))`."""
    if cls is DenseLRI:
        return DenseLRI(**({"max_length": max_length} if max_length is not None else {}), **kw)
    search = HillClimbing(max_conditions=max_length, stop_at_local_optimum=False) if max_length is not None else None
    return LRI(search=search, **kw)


both = pytest.mark.parametrize("cls", [DenseLRI, LRI], ids=lambda c: c.__name__)


@both
def test_lri_builds_an_equal_number_of_dnf_rules_per_class(cls):
    data = _data()
    model = _make(cls, max_length=4, n_rules=7, max_terms=3).fit(data)
    assert isinstance(model, EnsembleModel)
    assert all(isinstance(m, ConceptModel) and m.default_prediction is None for m in model.members)
    per_class = {}
    for m in model.members:
        per_class[m.label] = per_class.get(m.label, 0) + 1
        assert 1 <= len(m.rules) <= 3
        assert all(1 <= len(r.conditions) <= 4 for r in m.rules)
    assert per_class == {"neg": 7, "pos": 7}                      # both classes, equal counts
    assert model.to_string().splitlines()[0] == "% conflict resolution: vote of members"


@both
def test_lri_predicts_by_counting_satisfied_rules(cls):
    data = _data()
    model = _make(cls, n_rules=10, max_terms=2).fit(data)
    votes = {c: np.zeros(data.n_samples) for c in ("neg", "pos")}
    for m in model.members:
        votes[m.label] += np.asarray(m.predict(data)) == m.label
    pred = np.asarray(model.predict(data))
    clear = votes["pos"] != votes["neg"]
    np.testing.assert_array_equal(pred[clear], np.where(votes["pos"] > votes["neg"], "pos", "neg")[clear])


@both
def test_lri_is_accurate_on_a_noisy_concept(cls):
    model = _make(cls, n_rules=50, max_terms=4).fit(_data())
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85


@both
def test_terms_keep_a_true_positive_and_stop_once_they_cover_no_negative(cls):
    data = _data()
    X = data.X
    pos = data.y == "pos"
    model = _make(cls, max_length=5, n_rules=5, max_terms=1).fit(data)
    for m in model.members:
        cov = np.all(X[:, [l.feature for l in m.rules[0].conditions]], axis=1)
        own = data.y == m.label
        assert np.any(cov & own)                                  # TP > 0
        if len(m.rules[0].conditions) < 5:
            assert not np.any(cov & ~own)                         # stopped early only at FP = 0
    assert pos.any()


@both
def test_freezing_restricts_later_rules_to_the_features_used_so_far(cls):
    data = _data()
    model = _make(cls, n_rules=6, max_terms=1, freeze_features_after=1).fit(data)
    for label in ("neg", "pos"):
        members = [m for m in model.members if m.label == label]
        first = {l.feature for l in members[0].rules[0].conditions}
        for m in members[1:]:
            assert {l.feature for r in m.rules for l in r.conditions} <= first


@both
def test_lri_multiclass_and_data_weights(cls):
    rng = np.random.default_rng(2)
    raw = rng.random((300, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = _make(cls, n_rules=5, max_terms=2).fit(data)
    assert sorted({m.label for m in model.members}) == ["a", "b", "c"] and len(model.members) == 15
    assert np.mean(np.asarray(model.predict(data)) == y) > 0.95
    weighted = _make(cls, n_rules=5, max_terms=2).fit(data.with_weights(np.where(y == "c", 5.0, 1.0)))
    assert len(weighted.members) == 15


def test_lri_reweighting_halves_the_error_counts_past_the_cap():
    pos = np.array([True, False])
    st = CoveringState(np.ones(2), pos)
    rw = LRIReweighting(max_errors=2)
    wrong = np.array([False, True])                               # errs on both rows
    for _ in range(3):
        st.record(wrong)
        w = rw.weights(st, wrong)
    np.testing.assert_array_equal(st.extra["lri_errors"], [1, 1])  # 3 > 2 -> 3 // 2
    np.testing.assert_allclose(w, [2.0, 2.0])
    np.testing.assert_array_equal(st.errors, [3, 3])              # the plain counts


@both
def test_lri_rejects_bad_arguments(cls):
    with pytest.raises(ValueError):
        cls(n_rules=0)
    one = HorizontalBitmapRepresentation(DataSpec(["a"]), np.ones((3, 1), bool), np.array(["x"] * 3))
    with pytest.raises(ValueError, match="two classes"):
        cls().fit(one)


# -- the modular LRI ------------------------------------------------------------

def _plain(n=300, k=12, seed=0, weights=False, classes=2):
    """Plain Boolean features: no constraints among them."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n, k)) < rng.uniform(0.2, 0.7, size=k)
    if classes == 2:
        y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & ~raw[:, 3] | raw[:, 4], "pos", "neg")
        y = np.where(rng.random(n) < 0.1, np.where(y == "pos", "neg", "pos"), y)
    else:
        y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = HorizontalBitmapRepresentation(DataSpec([f"f{i}" for i in range(k)]), raw, y)
    return data.with_weights(rng.integers(1, 4, n).astype(float)) if weights else data


@pytest.mark.parametrize("kw", [{}, {"weights": True}, {"classes": 3}], ids=["binary", "weighted", "three-class"])
@pytest.mark.parametrize("params", [{}, {"max_terms": 2, "n_rules": 8}, {"n_rules": 10, "freeze_features_after": 3}],
                         ids=["default", "short", "frozen"])
def test_lri_learns_the_dense_model_on_unconstrained_features(kw, params):
    # without constraints among the features (no negations, no thresholds) the term search
    # is the same walk: the weights are integers, so every count is exact on both sides
    data = _plain(**kw)
    params = {"n_rules": 10, **params}
    want = str(DenseLRI(**params).fit(data))
    assert str(LRI(**params).fit(data)) == want
    assert str(LRI(**params).fit(NListRepresentation.from_boolean(data))) == want
    assert str(LRI(**params).fit(SparseDataRepresentation.from_boolean(data))) == want


def test_lri_with_constraints_is_the_same_on_every_representation_and_never_adds_an_implied_condition():
    data = _data()
    model = LRI(n_rules=10).fit(data)
    assert str(LRI(n_rules=10).fit(NListRepresentation.from_boolean(data))) == str(model)
    for member in model.members:
        for term in member.rules:
            feats = [l.feature for l in term.conditions]
            for f in feats:                                   # nothing is determined by the others
                closure = data.spec.propagate({g: True for g in feats if g != f})
                assert f not in closure or closure[f] is not True


def test_lri_takes_any_rule_search_and_any_objective():
    data = _data()
    beam = LRI(n_rules=5, search=BeamSearch(beam_width=3, optimistic_pruning=False)).fit(data)
    assert isinstance(beam, EnsembleModel) and len(beam.members) == 10
    fixed = LRI(n_rules=5, heuristic=LRIError(k=2.0)).fit(data)      # a fixed error weight
    assert len(fixed.members) == 10
    custom = LRI(n_rules=5, reweighting=LRIReweighting(power=2.0, max_errors=None)).fit(data)
    assert len(custom.members) == 10
    with pytest.raises(ValueError):
        LRI(search="greedy")
    with pytest.raises(ValueError):
        LRIError(k=0)


def test_lri_error_is_the_weighted_error_with_the_papers_adaptive_k():
    # 2 positives and 3 negatives; condition a covers both positives and one negative, b one
    # positive and no negative, c two negatives and no positive (dead)
    X = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
    y = np.array(["p", "p", "n", "n", "n"])
    data = HorizontalBitmapRepresentation(DataSpec(["a", "b", "c"]), X, y)
    obj = LRIError()
    from pyrulearn.objectives import handle_for
    rule = EmptyRuleAllFeatures().initial_candidates(data, "p")[0][0]
    handle = handle_for(data, rule, None)
    root = obj.rule_stats(data, "p", rule, handle)
    obj.begin(data, "p", None)
    features, batch, same, dead = obj.children(data, "p", frozenset({0, 1, 2}), handle, root)
    assert features.tolist() == [0, 1] and dead == {2}
    # k = 1: a has fp 1, fn 0 (error 1), b has fp 0, fn 1 (error 1); the dead c has fp 2,
    # fn 2 (error 4), no cheaper than either, so k stays 1
    assert batch.k == 1.0
    np.testing.assert_allclose(obj.batch_score(batch), [-1.0, -1.0])
    assert obj.consistent(obj.at(batch, 1)) and not obj.consistent(obj.at(batch, 0))
    assert LRIError(k=3.0).children(data, "p", frozenset({0, 1, 2}), handle, root)[1].k == 3.0
