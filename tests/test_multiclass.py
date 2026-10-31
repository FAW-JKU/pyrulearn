import numpy as np
import pytest

from pyrulearn.models import MajorityVote, WeightedVote
from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.data import DataSpec
from pyrulearn.models import (
    AccuracyWeightedVote, ConceptCascade, ConceptSet, MajorityClass, MajorityVote,
    PairwiseModel, WeightedVote,
)
from pyrulearn.learners.multiclass import OneVsRest, OrderedOneVsRest, Pairwise
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import CN2, PFossil

from _negation_helpers import neg_spec, neg_X


def _three_class(n=400, seed=0):
    # x = f0 ; y = ~f0 & f1 ; z = ~f0 & ~f1  (f2..f5 noise). Negation spec so
    # "z vs rest" (~f0 & ~f1) is expressible.
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0], "x", np.where(raw[:, 1], "y", "z"))
    y = np.where(rng.random(n) < 0.04, rng.permutation(y), y)
    ds = neg_spec([f"f{i}" for i in range(6)])
    return HorizontalBitmapRepresentation(ds, neg_X(raw), y), ds, y


def _cn2():
    return CN2(significance_threshold=None)


# -- OneVsRest : sugar for fit(model=ConceptSet) --------------------------

def test_one_vs_rest_is_sugar_for_model_conceptset():
    data, ds, y = _three_class()
    m = OneVsRest(CN2()).fit(data)
    assert type(m) is ConceptSet
    assert {r.target for r in m.rules} == {"x", "y", "z"}
    assert isinstance(m.default_prediction, MajorityClass)
    assert m.default_prediction.constant_target == "x"       # training majority
    for c in ("x", "y", "z"):
        assert m.concept_for(c).rules
    assert np.mean(m.predict(data) == y) > 0.85


def test_base_learner_template_not_mutated():
    base = CN2()
    OneVsRest(base).fit(_three_class()[0])
    Pairwise(base).fit(_three_class()[0])
    assert base.target_class is None


def test_one_vs_rest_with_a_pylord_base():
    data, ds, y = _three_class(n=350, seed=2)
    m = OneVsRest(PyLORD(m=0.1, random_state=0)).fit(data)
    assert type(m) is ConceptSet and {r.target for r in m.rules} == {"x", "y", "z"}
    assert np.mean(m.predict(data) == y) > 0.9


# -- OrderedOneVsRest : sugar for fit(model=ConceptCascade) --------------

def test_ordered_returns_a_concept_cascade_last_label_is_the_default():
    data, ds, y = _three_class()
    least = OrderedOneVsRest(CN2()).fit(data)
    most = OrderedOneVsRest(CN2(), order="most_frequent").fit(data)
    assert type(least) is ConceptCascade and type(most) is ConceptCascade
    assert least.default_prediction == "x"                   # rarest peeled first
    assert most.default_prediction in ("y", "z")
    assert least.default_prediction not in least.concept_labels
    assert np.mean(np.asarray(least.predict(data)) == y) > 0.85


def test_ordered_explicit_and_bad_orders():
    data, ds, y = _three_class()
    m = OrderedOneVsRest(CN2(), order=["y", "z", "x"]).fit(data)
    assert m.default_prediction == "x" and m.concept_labels[0] == "y"
    with pytest.raises(ValueError, match="permutation"):
        OrderedOneVsRest(CN2(), order=["x", "y"]).fit(data)


def test_ordered_random_is_reproducible_via_the_wrappers_random_state():
    data, ds, y = _three_class()
    a = OrderedOneVsRest(CN2(), order="random", random_state=7).fit(data)
    b = OrderedOneVsRest(CN2(), order="random", random_state=7).fit(data)
    assert a.concept_labels == b.concept_labels


# -- Pairwise : sugar for fit(model=PairwiseModel) + a vote combiner ----

def test_pairwise_one_member_per_pair_smaller_target():
    data, ds, y = _three_class()
    counts = {c: int((y == c).sum()) for c in ("x", "y", "z")}
    m = Pairwise(_cn2(), random_state=0).fit(data)
    assert type(m) is PairwiseModel and len(m.members) == 3
    assert {frozenset(p) for p in m.pairs} == {frozenset({"x", "y"}), frozenset({"x", "z"}),
                                               frozenset({"y", "z"})}
    for pos, neg in m.pairs:
        assert counts[pos] <= counts[neg]                    # "smaller" default
    assert isinstance(m.default_prediction, MajorityClass)
    assert isinstance(m.combiner, MajorityVote)
    larger = Pairwise(_cn2(), positive="larger", random_state=0).fit(data)
    for pos, neg in larger.pairs:
        assert counts[pos] >= counts[neg]


def test_pairwise_both_is_two_members_per_pair():
    data, ds, y = _three_class(n=400)
    m = Pairwise(_cn2(), positive="both", random_state=0).fit(data)
    assert len(m.members) == 6
    assert sorted(m.pairs) == sorted([(a, b) for a in "xyz" for b in "xyz" if a != b])
    assert np.mean(np.asarray(m.predict(data)) == y) > 0.85


def test_pairwise_callable_positive():
    data, ds, y = _three_class()
    m = Pairwise(_cn2(), positive=lambda a, b: "y" if "y" in (a, b) else a, random_state=0).fit(data)
    for pos, neg in m.pairs:
        if "y" in (pos, neg):
            assert pos == "y"


def test_pairwise_combiner_is_swappable_no_refit():
    data, ds, y = _three_class(n=500, seed=1)
    m = Pairwise(_cn2(), random_state=0).fit(data)
    assert np.mean(np.asarray(m.predict(data)) == y) > 0.9
    for combiner in (MajorityVote(tie_break="prior"), WeightedVote(), AccuracyWeightedVote()):
        m.combiner = combiner
        assert set(np.unique(np.asarray(m.predict(data)))) <= {"x", "y", "z"}


def test_pairwise_weighted_vote_end_to_end():
    data, ds, y = _three_class(n=500, seed=1)
    m = Pairwise(_cn2(), combiner=WeightedVote(), random_state=0).fit(data)
    assert isinstance(m.combiner, WeightedVote)
    for _, _, sub in m._triples:
        # WeightedVote scores each deciding rule from its own measured
        # stats now (Laplace, computed on the fly), not a stored weight
        assert all(r.stats() is not None for r in sub.rules)
        assert sub.default_rule.stats() is not None
    assert np.mean(np.asarray(m.predict(data)) == y) > 0.9


def test_pairwise_weighted_vote_helps_an_imprecise_base():
    data, ds, y = _three_class(n=500, seed=3)
    hard = Pairwise(PyLORD(m=0.1, random_state=0), combiner=MajorityVote(), random_state=0).fit(data)
    soft = Pairwise(PyLORD(m=0.1, random_state=0), combiner=WeightedVote(), random_state=0).fit(data)
    assert np.mean(np.asarray(soft.predict(data)) == y) >= np.mean(np.asarray(hard.predict(data)) == y)


def test_pairwise_records_member_accuracies():
    data, ds, y = _three_class(n=500, seed=1)
    m = Pairwise(_cn2(), random_state=0).fit(data)
    assert m.member_weights is not None and m.member_weights.shape == (len(m._triples),)
    assert np.all((m.member_weights >= 0.0) & (m.member_weights <= 1.0))
    for (pos, neg, sub), w in zip(m._triples, m.member_weights):
        rows = (np.asarray(y) == pos) | (np.asarray(y) == neg)
        stage = data.select_rows(rows)
        assert w == pytest.approx(np.mean(np.asarray(sub.predict(stage)) == np.asarray(stage.y)))


def test_pairwise_with_a_pylord_base():
    data, ds, y = _three_class(n=300, seed=2)
    m = Pairwise(PyLORD(m=0.1, random_state=0), random_state=0).fit(data)
    assert type(m) is PairwiseModel
    assert set(np.unique(np.asarray(m.predict(data)).astype(str))) <= {"x", "y", "z"}


# -- learner defaults ---------------------------------------------------

def test_seco_family_multiclass_defaults():
    data, ds, y = _three_class(n=500, seed=1)
    assert type(CN2().fit(data)) is ConceptSet
    assert type(PFossil().fit(data)) is ConceptSet
    assert type(CN2().fit(data, model=ConceptCascade)) is ConceptCascade
    assert type(CN2().fit(data, model=PairwiseModel)) is PairwiseModel
    assert np.mean(np.asarray(CN2().fit(data).predict(data)) == y) > 0.9
    # the learner's own config flows into each per-class sub-fit
    shallow = CN2(significance_threshold=None).fit(data)
    deep = CN2(significance_threshold=50.0).fit(data)
    assert len(deep.rules) < len(shallow.rules)


def test_target_class_still_does_one_binary_problem():
    data, ds, y = _three_class(n=400)
    from pyrulearn.models import ConceptModel
    binary = CN2(target_class="x").fit(data)
    assert type(binary) is ConceptModel and {r.target for r in binary.rules} == {"x"}
    assert "x" in set(np.unique(np.asarray(binary.predict(data)).astype(str)))


def test_every_decomposition_gives_its_default_rule_training_stats():
    # the default rule covers every training row, so its frozen stats are
    # the class distribution of the entire training data
    data, ds, y = _three_class()
    for model in (ConceptSet, ConceptCascade, PairwiseModel):
        m = _cn2().fit(data, model=model)
        dr = m.default_rule
        st = dr.stats()
        assert st is not None, model.__name__
        assert st.n_rows == data.n_samples
        rs = st.confusion.rule_stats(dr.target)
        assert rs.tp == int(np.sum(y == dr.target)) and rs.tp + rs.fp == data.n_samples
        default_line = m.to_string(fmt="prolog").split("% default")[-1]
        assert "% (" in default_line or "% [" in default_line, model.__name__  # (tp/fp), or CN2's distribution


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("\nAll multiclass tests passed.")
