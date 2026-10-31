"""Pairwise voting layer + `PairwiseModel` (lives in pyrulearn.models)."""

import numpy as np
import pytest

from pyrulearn.models import AccuracyWeightedVote, WeightedVote
from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.data import DataSpec
from pyrulearn.models import (
    AccuracyWeightedVote, CompositeModel, FlatRuleSet, MajorityVote, PairwiseCombiner,
    PairwiseModel, PairwiseVote, WeightedVote, _resolve_pairwise_combiner,
)
from pyrulearn.rule import Rule, WeightedRule

LABELS = np.array(["a", "b", "c"], dtype=object)


def _votes(*triples):
    return [PairwiseVote(*t) for t in triples]


# -- combiner scores / decide -----------------------------------------

def test_majority_vote_scores_and_decide():
    v = _votes(("a", "a", "b"), ("a", "a", "c"), ("c", "b", "c"))
    assert list(MajorityVote().scores(v, LABELS)) == [2.0, 0.0, 1.0]
    assert MajorityVote().decide(v, LABELS) == "a"
    assert MajorityVote().decide(_votes((None, "a", "b")), LABELS) is None
    # abstain / unknown label don't score
    assert list(MajorityVote().scores(_votes((None, "a", "b"), ("z", "a", "c")), LABELS)) == [0, 0, 0]


def test_tie_breaks():
    v = _votes(("a", "a", "b"), ("c", "b", "c"), ("a", "a", "c"))    # a,c tied; a won a-vs-c
    assert MajorityVote(tie_break="direct").decide(v, LABELS) == "a"
    v2 = _votes(("a", "a", "b"), ("c", "b", "c"))                    # tied, no direct duel
    assert MajorityVote(tie_break="prior").decide(v2, LABELS, {"a": 1, "c": 9}) == "c"
    assert MajorityVote(tie_break="first").decide(v2, LABELS) == "a"
    assert MajorityVote(tie_break=lambda t, v, p: sorted(t)[-1]).decide(v2, LABELS) == "c"
    with pytest.raises(ValueError):
        MajorityVote(tie_break="nope")


def test_weighted_vote_splits_the_vote():
    wv = WeightedVote()
    assert list(wv.scores(_votes(("a", "a", "b"), ("c", "b", "c")), LABELS)) == [1.0, 0.0, 1.0]
    vw = [PairwiseVote("a", "a", "b", 0.8), PairwiseVote("c", "b", "c", 0.6),
          PairwiseVote("a", "a", "c", 0.7)]
    assert list(wv.scores(vw, LABELS)) == pytest.approx([1.5, 0.6, 0.9])
    assert list(wv.scores([PairwiseVote("a", "a", "b", 5.0)], LABELS)) == [1.0, 0.0, 0.0]   # clamp
    assert list(wv.scores([PairwiseVote("a", "a", "b", float("nan"))], LABELS)) == [0.5, 0.5, 0.0]
    assert wv.needs_weights and not MajorityVote().needs_weights
    assert WeightedVote.weight_source == "rule" and AccuracyWeightedVote.weight_source == "member"


def test_resolve_combiner_takes_objects_only():
    assert isinstance(_resolve_pairwise_combiner(None), MajorityVote)
    mv = MajorityVote()
    assert _resolve_pairwise_combiner(mv) is mv
    for name in ("vote", "weighted_vote", "nope"):                 # names aren't components
        with pytest.raises(ValueError, match="PairwiseCombiner"):
            _resolve_pairwise_combiner(name)


# -- PairwiseModel mechanics -----------------------------------------

def _toy():
    ds = DataSpec(["p", "q"])
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    data = HorizontalBitmapRepresentation(ds, X, np.array(["a", "b", "c", "a"], dtype=object))
    ab = FlatRuleSet([Rule([0], target="a", dataspec=ds)], default_prediction="b")
    ac = FlatRuleSet([Rule([0], target="a", dataspec=ds)], default_prediction="c")
    bc = FlatRuleSet([Rule([1], target="b", dataspec=ds)], default_prediction="c")
    m = PairwiseModel([("a", "b", ab), ("a", "c", ac), ("b", "c", bc)],
                      default_prediction="a", labels=["a", "b", "c"],
                      label_priors={"a": 2, "b": 1, "c": 1})
    return data, ds, m


def test_pooled_rules_and_repr():
    _, _, m = _toy()
    assert len(m.rules) == 3 and isinstance(m, CompositeModel)
    with pytest.raises(TypeError):
        m.add(Rule([0], target="a"))
    assert "3 members" in repr(m)


def test_predict_is_a_pairwise_vote():
    data, ds, m = _toy()
    # row0 [p]: ab->a ac->a bc->c(dflt) => a(2)
    # row1 [q]: ab->b ac->c bc->b       => b(2)
    # row2 [p,q]: ab->a ac->a bc->b     => a
    # row3 []: b,c,c defaults => tie b/c; direct b-vs-c -> c
    assert list(np.asarray(m.predict(data))) == ["a", "b", "a", "c"]


def test_weighted_vote_reads_the_deciding_rules_weight():
    ds = DataSpec(["p", "q"])
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    data = HorizontalBitmapRepresentation(ds, X, np.array(["a", "b", "c", "a"], dtype=object))
    ab = FlatRuleSet([WeightedRule([0], target="a", dataspec=ds, weight=0.95)], default_prediction="b")
    ac = FlatRuleSet([WeightedRule([0], target="a", dataspec=ds, weight=0.55)], default_prediction="c")
    bc = FlatRuleSet([WeightedRule([1], target="b", dataspec=ds, weight=0.90)], default_prediction="c")
    m = PairwiseModel([("a", "b", ab), ("a", "c", ac), ("b", "c", bc)],
                      combiner=WeightedVote(), default_prediction="a", labels=["a", "b", "c"])
    assert np.asarray(m.predict(data))[0] == "a"          # a: 0.95 + 0.55 = 1.5
    ac.rules[0].weight = 0.05
    assert set(np.unique(m.predict(data))) <= {"a", "b", "c"}


def test_accuracy_weighted_vote_uses_member_weights():
    data, ds, _ = _toy()
    ab = FlatRuleSet([Rule([0], target="a", dataspec=ds)], default_prediction="b")
    ac = FlatRuleSet([Rule([0], target="a", dataspec=ds)], default_prediction="c")
    bc = FlatRuleSet([Rule([1], target="b", dataspec=ds)], default_prediction="c")
    members = [("a", "b", ab), ("a", "c", ac), ("b", "c", bc)]
    # row0 [p]: ab->a ac->a bc->c ; accs (0.9, 0.1, 0.5):
    #   a: 0.9 + 0.1 = 1.0 ; b: 0.1 + 0.5 = 0.6 ; c: 0.9 + 0.5 = 1.4  -> c
    m = PairwiseModel(members, combiner=AccuracyWeightedVote(), default_prediction="a",
                      labels=["a", "b", "c"], member_weights=[0.9, 0.1, 0.5])
    assert np.asarray(m.predict(data))[0] == "c"
    m.member_weights = np.array([0.9, 0.95, 0.5])
    assert np.asarray(m.predict(data))[0] == "a"
    with pytest.raises(ValueError):
        PairwiseModel(members, member_weights=[0.5, 0.5])
    with pytest.raises(ValueError):
        PairwiseModel(members, combiner=AccuracyWeightedVote()).predict(data)


def test_filter_and_remap_recurse_into_members():
    data, ds, m = _toy()
    only_a = m.filter("a")
    assert type(only_a) is PairwiseModel and only_a.pairs == m.pairs and len(only_a.members) == 3
    assert all(all(r.target == "a" for r in mem.rules) for mem in only_a.members)
    ds2 = DataSpec(["p", "q", "extra"])
    remapped = m.remap(ds2)
    assert type(remapped) is PairwiseModel
    assert np.array_equal(np.asarray(remapped.predict(
        HorizontalBitmapRepresentation(ds2, np.array([[1, 0, 0], [0, 1, 1]], dtype=bool), None))),
        np.array(["a", "b"], dtype=object))
